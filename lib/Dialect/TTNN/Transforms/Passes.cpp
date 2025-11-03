// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNCREATEINPUTGENERATORS
#define GEN_PASS_DEF_TTNNLOADINPUTTENSORS
#define GEN_PASS_DEF_TTNNDEALLOCATE
#define GEN_PASS_DEF_TTNNTUPLIFYTENSORS
#define GEN_PASS_DEF_TTNNEMPYWORKAROUNDS
#define GEN_PASS_DEF_TTNNPRETTIFYFORCODEGEN
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNDeallocate : public impl::TTNNDeallocateBase<TTNNDeallocate> {

public:
  using impl::TTNNDeallocateBase<TTNNDeallocate>::TTNNDeallocateBase;

  Operation *getLastValueUsageOp(const LivenessBlockInfo *livenessInfo,
                                 Value value) {
    Operation *startOp = livenessInfo->getStartOperation(value);
    Operation *endOp = livenessInfo->getEndOperation(value, startOp);
    auto *opOperandIter =
        llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
          return opOperand.is(value);
        });

    // In case of DPS op keep going until we find the last usage of the tensor.
    //
    while (
        opOperandIter != endOp->getOpOperands().end() &&
        isa<DestinationStyleOpInterface>(endOp) &&
        cast<DestinationStyleOpInterface>(endOp).isDpsInit(&(*opOperandIter))) {
      OpResult result =
          cast<DestinationStyleOpInterface>(endOp).getTiedOpResult(
              &(*opOperandIter));
      endOp = livenessInfo->getEndOperation(result, endOp);
      opOperandIter =
          llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
            return opOperand.is(result);
          });
    }

    return endOp;
  }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp->walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return;
      }
      assert(func.getBody().hasOneBlock() &&
             "found func that didn't have one block!");
      Liveness liveness(func.getOperation());
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(&func.getBody().front());

      // Const eval subgraphs and trace functions may not dealloc their params
      // since they don't own them.
      if (!ttmlir::utils::isConstEvalFunc(func) &&
          !utils::isTTNNTraceFunc(func)) {
        // Handle func op input parameters
        for (BlockArgument arg : func.getArguments()) {
          if (!isa<RankedTensorType>(arg.getType())) {
            continue;
          }
          Operation *lastOp = getLastValueUsageOp(livenessInfo, arg);

          if (isa<func::ReturnOp>(lastOp)) {
            continue;
          }

          rewriter.setInsertionPointAfter(lastOp);
          rewriter.create<DeallocateOp>(lastOp->getLoc(), arg);
        }
      }

      // Handle non DPS ops which do not store function result and are used to
      // allocate tensors. DPS ops are handled via ttnn::EmptyOp.
      //
      func->walk([&](Operation *op) {
        if (isa<DestinationStyleOpInterface>(op)) {
          return;
        }

        // Skip ops which do not have results.
        //
        if (op->getNumResults() == 0) {
          return;
        }

        // Iterate over all results of the op.
        //
        for (OpResult result : op->getResults()) {
          // Check if result is ranked tensor type.
          //
          if (!isa<RankedTensorType>(result.getType())) {
            continue;
          }

          RankedTensorType resultTy =
              mlir::cast<RankedTensorType>(result.getType());
          assert(resultTy.getEncoding());

          Operation *lastOp = getLastValueUsageOp(livenessInfo, result);

          if (isa<func::ReturnOp>(lastOp)) {
            continue;
          }

          // Don't deallocate the activation after conv2d op if
          // 'deallocate_activation' in Conv2dConfig is set to true.
          if (auto conv2dOp = mlir::dyn_cast<ttnn::Conv2dOp>(lastOp)) {
            if (conv2dOp.getInput() == result &&
                conv2dOp.getConv2dConfigAttr() &&
                conv2dOp.getConv2dConfigAttr().getDeallocateActivation() &&
                conv2dOp.getConv2dConfigAttr()
                    .getDeallocateActivation()
                    .getValue()) {
              continue;
            }
          }

          rewriter.setInsertionPointAfter(lastOp);
          rewriter.create<DeallocateOp>(lastOp->getLoc(), result);
        }
      });
    });
  }
};

template <typename Derived>
class TTNNInputFunctionCreatorBase {
protected:
  void runOnOperationImpl(ModuleOp moduleOp, IRRewriter &rewriter,
                          const std::string &functionPrefix) {
    // Ensure that the module has a single region and a single block within that
    // region.
    //
    assert(moduleOp->getRegions().size() == 1);
    assert(moduleOp->getRegion(0).hasOneBlock());

    // Get the only existing block.
    //
    Block *block = moduleOp.getBody(0);

    // Find all the func.func ops in the module that are "forward" functions.
    //
    SmallVector<func::FuncOp, 1> forwardFuncOps;
    block->walk([&](func::FuncOp funcOp) {
      // Rename the function if it's named `main` to `_main`. This is done to
      // avoid name conflicts.
      //
      if (funcOp.getName() == "main") {
        rewriter.modifyOpInPlace(funcOp, [&]() { funcOp.setSymName("_main"); });
      }

      if (funcOp.isPrivate() || ttmlir::utils::isConstEvalFunc(funcOp)) {
        return mlir::WalkResult::skip();
      }

      forwardFuncOps.push_back(funcOp);
      return mlir::WalkResult::advance();
    });

    // Iterate over all func ops and add input tensor functions if needed.
    //
    llvm::SmallVector<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 1>
        forwardAndInputFuncOps;
    for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
      rewriter.setInsertionPointToEnd(block);
      // Check if the forward function has any arguments. If it doesn't, we
      // do not create an input function for it.
      mlir::func::FuncOp inputFuncOp =
          forwardFuncOp.getNumArguments() == 0
              ? nullptr
              : createInputFunctionImpl(rewriter, forwardFuncOp.getLoc(),
                                        forwardFuncOp, functionPrefix);
      forwardAndInputFuncOps.emplace_back(forwardFuncOp, inputFuncOp);
    }

    // Create a main function to call input functions and forward funcs.
    //
    createMainFunction(moduleOp, rewriter, forwardAndInputFuncOps);
  }

  func::FuncOp createInputFunctionImpl(IRRewriter &rewriter, Location loc,
                                       func::FuncOp forwardFuncOp,
                                       const std::string &functionPrefix) {
    MLIRContext *ctx = rewriter.getContext();

    // Create a new function that will handle the input tensors.
    //
    std::string inputFuncName = functionPrefix + forwardFuncOp.getName().str();

    // Create the function type.
    //
    llvm::ArrayRef<mlir::Type> returnTypes =
        forwardFuncOp.getFunctionType().getInputs();
    FunctionType functionType = mlir::FunctionType::get(ctx, {}, returnTypes);

    // Create the function.
    //
    func::FuncOp inputFuncOp =
        rewriter.create<mlir::func::FuncOp>(loc, inputFuncName, functionType);

    // Add a Block to func op and set insertion point to the beginning of the
    // Block.
    //
    rewriter.modifyOpInPlace(inputFuncOp, [&]() {
      rewriter.setInsertionPointToStart(inputFuncOp.addEntryBlock());
    });

    // Create/load input tensors.
    //
    assert(
        returnTypes.size() == 1 && mlir::isa<TupleType>(returnTypes.front()) &&
        "Expected input function to return a single tuple of input tensors!");

    SmallVector<Value> tensors;
    size_t argIndex = 0;
    for (const Type &type :
         mlir::cast<mlir::TupleType>(returnTypes[0]).getTypes()) {
      // Ensure that the type is a RankedTensorType.
      //
      RankedTensorType rankedTensorType =
          mlir::dyn_cast<RankedTensorType>(type);
      assert(rankedTensorType &&
             "Expected input tensor to be of type RankedTensorType!");

      tensors.push_back(static_cast<Derived *>(this)->createTensor(
          rewriter, loc, rankedTensorType, argIndex));
      argIndex++;
    }

    // Create a tuple from the tensors.
    //
    ttcore::TupleOp tuple =
        rewriter.create<ttcore::TupleOp>(loc, returnTypes, tensors);

    // Create ReturnOp.
    //
    rewriter.create<func::ReturnOp>(forwardFuncOp.getLoc(),
                                    tuple->getResults());

    return inputFuncOp;
  }

private:
  void createMainFunction(
      ModuleOp moduleOp, IRRewriter &rewriter,
      llvm::SmallVector<std::pair<mlir::func::FuncOp, mlir::func::FuncOp>, 1>
          forwardAndInputFuncOps) {
    std::string mainFuncName = "main";

    // Create a function type.
    //
    FunctionType functionType = mlir::FunctionType::get(
        rewriter.getContext(), {}, rewriter.getI32Type());

    // Set insertion point to end of the block.
    //
    Block *block = moduleOp.getBody(0);
    rewriter.setInsertionPointToEnd(block);

    // Create the main function.
    //
    func::FuncOp mainFuncOp = rewriter.create<mlir::func::FuncOp>(
        moduleOp.getLoc(), mainFuncName, functionType);

    // Set insertion point to the start of the main function.
    //
    rewriter.modifyOpInPlace(mainFuncOp, [&]() {
      rewriter.setInsertionPointToStart(mainFuncOp.addEntryBlock());
    });

    for (auto [forwardFuncOp, inputFuncOp] : forwardAndInputFuncOps) {

      llvm::SmallVector<Value> operands;
      // Generate/load the input tensors for a forwardFuncOp if needed.
      //
      if (inputFuncOp) {
        func::CallOp tensors = rewriter.create<mlir::func::CallOp>(
            forwardFuncOp.getLoc(), inputFuncOp,
            /*operands=*/ValueRange());
        operands = tensors->getResults();
      }

      // Call a forward function. If there are input tensors, pass them as
      // operands.
      //
      rewriter.create<mlir::func::CallOp>(forwardFuncOp.getLoc(), forwardFuncOp,
                                          operands);
    }

    // Return 0
    //
    // func::ReturnOp requires a Value to be returned, which means that an SSA
    // needs to be returned, hence create a constant 0 via arith::ConstantOp.
    //
    Value constantZero = rewriter.create<arith::ConstantOp>(
        rewriter.getUnknownLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(0));
    rewriter.create<func::ReturnOp>(mainFuncOp->getLoc(), constantZero);
  }
};

class TTNNCreateInputGenerators
    : public impl::TTNNCreateInputGeneratorsBase<TTNNCreateInputGenerators>,
      public TTNNInputFunctionCreatorBase<TTNNCreateInputGenerators> {

public:
  using impl::TTNNCreateInputGeneratorsBase<
      TTNNCreateInputGenerators>::TTNNCreateInputGeneratorsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());
    runOnOperationImpl(moduleOp, rewriter, "create_inputs_for_");
  }

  mlir::Value createTensor(IRRewriter &rewriter, Location loc, Type type,
                           size_t argIndex) {
    return generateTensor(rewriter, loc, type);
  }

private:
  // Currently only supports generating tensors of ones.
  // TODO(azecevic): Support generating other types of tensors that has a
  // `TTCore_CreationOpTrait`.
  // https://github.com/tenstorrent/tt-mlir/issues/3261
  //
  static mlir::Value generateTensor(IRRewriter &rewriter, Location loc,
                                    Type type) {
    MLIRContext *ctx = rewriter.getContext();
    RankedTensorType tensorType = llvm::cast<mlir::RankedTensorType>(type);

    // Get the layout attribute.
    //
    ttnn::TTNNLayoutAttr layoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());

    // Get the shape of the tensor, tensor layout, and data type.
    //
    ShapeAttr shapeAttr = ttnn::ShapeAttr::get(ctx, tensorType.getShape());
    ttnn::LayoutAttr tensorLayoutAttr =
        ttnn::LayoutAttr::get(ctx, layoutAttr.getLayout());
    ttcore::DataTypeAttr dTypeAttr =
        ttcore::DataTypeAttr::get(ctx, layoutAttr.getDataType());

    mlir::Value device;
    if (layoutAttr.isDeviceBufferType()) {
      device = ttnn::utils::getOrInsertDevice(rewriter,
                                              rewriter.getInsertionBlock());
    }
    // Create a new tensor of ones.
    //
    ttnn::OnesOp onesOp = rewriter.create<ttnn::OnesOp>(
        loc, tensorType, device, shapeAttr, dTypeAttr, tensorLayoutAttr,
        /*memory_config=*/nullptr);

    return onesOp;
  }
};

class TTNNLoadInputTensors
    : public impl::TTNNLoadInputTensorsBase<TTNNLoadInputTensors>,
      public TTNNInputFunctionCreatorBase<TTNNLoadInputTensors> {

public:
  using impl::TTNNLoadInputTensorsBase<
      TTNNLoadInputTensors>::TTNNLoadInputTensorsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());
    runOnOperationImpl(moduleOp, rewriter, "load_inputs_for_");
  }

  mlir::Value createTensor(IRRewriter &rewriter, Location loc, Type type,
                           size_t argIndex) {
    return loadTensor(rewriter, loc, type, argIndex, this->tensorLoadDirectory,
                      this->tensorLoadFilePrefix);
  }

private:
  static mlir::Value loadTensor(IRRewriter &rewriter, Location loc, Type type,
                                size_t argIndex,
                                std::string tensorLoadDirectory,
                                std::string tensorLoadFilePrefix) {
    RankedTensorType tensorType = llvm::cast<mlir::RankedTensorType>(type);

    // Get the layout attribute.
    //
    ttnn::TTNNLayoutAttr layoutAttr =
        mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());

    // Create filename, defaults are:
    // tensorsLoadDirectory = "" (current directory)
    // tensorLoadFilePrefix = "arg"
    // For example: arg0.tensorbin, arg1.tensorbin, etc.
    //
    std::string filename;
    if (tensorLoadDirectory.empty()) {
      filename = tensorLoadFilePrefix + std::to_string(argIndex) + ".tensorbin";
    } else {
      filename = tensorLoadDirectory + "/" + tensorLoadFilePrefix +
                 std::to_string(argIndex) + ".tensorbin";
    }
    StringAttr filePathAttr = rewriter.getStringAttr(filename);

    mlir::Value device;
    if (layoutAttr.isDeviceBufferType()) {
      device = ttnn::utils::getOrInsertDevice(rewriter,
                                              rewriter.getInsertionBlock());
    }
    // Create LoadTensorOp to load tensor from disk.
    //
    ttnn::LoadTensorOp loadTensorOp = rewriter.create<ttnn::LoadTensorOp>(
        loc, tensorType, filePathAttr, device);

    return loadTensorOp;
  }
};

class TTNNTuplifyTensors
    : public impl::TTNNTuplifyTensorsBase<TTNNTuplifyTensors> {

public:
  using impl::TTNNTuplifyTensorsBase<
      TTNNTuplifyTensors>::TTNNTuplifyTensorsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Ensure that the module has a single region and a single block within that
    // region.
    //
    assert(moduleOp->getRegions().size() == 1);
    assert(moduleOp->getRegion(0).getBlocks().size() == 1);

    Block *block = moduleOp.getBody(0);

    // Rename `main` to `_main` as `main` is a reserved name in C/C++.
    //
    // TODO (svuckovic): Move this to its own pass.
    //
    block->walk([&](func::FuncOp funcOp) {
      // Rename the function if it's named `main` to `_main`. This is done
      // as compiler will complain that `main` must return `int` and that
      // it's first parameter must also be an `int`.
      if (funcOp.getName() == "main") {
        rewriter.modifyOpInPlace(funcOp, [&]() { funcOp.setSymName("_main"); });
      }
    });

    // Find all the func.func ops in the module that are target functions for
    // both input tuplification, and result tuplification.
    //
    SmallVector<func::FuncOp, 1> targetFuncOpsInput;
    SmallVector<func::FuncOp, 1> targetFuncOpsResult;
    block->walk([&](func::FuncOp funcOp) {
      // Skip private functions.
      //
      if (funcOp.isPrivate()) {
        return mlir::WalkResult::skip();
      }

      mlir::FunctionType functionType = funcOp.getFunctionType();

      // Check that input is not empty and that all args are of type
      // RankedTensorType.
      //
      // If `tuplifyInputIfEmpty` option is set, tuplify the input even if the
      // function has no inputs.
      //
      if ((tuplifyInputIfEmpty || !functionType.getInputs().empty()) &&
          llvm::all_of(functionType.getInputs(),
                       [](Type t) { return mlir::isa<RankedTensorType>(t); })) {
        targetFuncOpsInput.push_back(funcOp);
      }

      // Check that results are not empty and that all args are of type
      // RankedTensorType.
      //
      if (!functionType.getResults().empty() &&
          llvm::all_of(functionType.getResults(),
                       [](Type t) { return mlir::isa<RankedTensorType>(t); })) {
        targetFuncOpsResult.push_back(funcOp);
      }
      return mlir::WalkResult::advance();
    });

    // Iterate over all the input target func ops and modify their signatures.
    //
    for (mlir::func::FuncOp targetFuncOpInput : targetFuncOpsInput) {
      // Replace the signature of the target function so that all the tensor
      // arguments are packed into a single tuple.
      //
      mlir::FunctionType originalFuncType = targetFuncOpInput.getFunctionType();

      // Create TupleType object containing all input tensors.
      //
      mlir::TupleType tuplifiedInputTensors =
          mlir::TupleType::get(&getContext(), originalFuncType.getInputs());

      // Create modified function type (signature) that takes the input tuple
      // as an operand.
      //
      FunctionType modifiedFuncType = originalFuncType.clone(
          tuplifiedInputTensors, originalFuncType.getResults());

      rewriter.modifyOpInPlace(targetFuncOpInput,
                               [&targetFuncOpInput, &modifiedFuncType]() {
                                 targetFuncOpInput.setType(modifiedFuncType);
                               });

      // First block of the function (often referred to as "entry block") needs
      // its arguments updated as well - the args need to match the containing
      // func's arguments; this is implemented here by first inserting the tuple
      // as the first argument of the block, inserting GetTupleElementOp ops to
      // start of the block in order to unpack tuple elements, and then
      // replacing all uses of the original block arguments with the
      // GetTupleElementOp results - after this it's finally safe to remove
      // original block arguments as they have no live uses anymore.
      //
      Block &entryBlock = targetFuncOpInput.getBlocks().front();
      constexpr size_t paramOffset = 1;
      entryBlock.insertArgument(/*index=*/0u, tuplifiedInputTensors,
                                targetFuncOpInput.getLoc());

      // Add GetTupleElementOp ops to unpack the tuple elements.
      //
      rewriter.setInsertionPointToStart(&entryBlock);
      for (size_t idx = 0; idx < originalFuncType.getNumInputs(); idx++) {
        ttcore::GetTupleElementOp getTupleElementOp =
            rewriter.create<ttcore::GetTupleElementOp>(
                targetFuncOpInput.getLoc(), targetFuncOpInput.getArgument(0),
                idx);

        // Replace all uses of the original tensor arguments with the
        // GetTupleElementOp results.
        rewriter.replaceAllUsesWith(entryBlock.getArgument(paramOffset + idx),
                                    getTupleElementOp);
      }

      // Erase original arguments.
      //
      entryBlock.eraseArguments(paramOffset,
                                originalFuncType.getInputs().size());
    }

    // Iterate over all the result target func ops and modify their signatures.
    //
    for (mlir::func::FuncOp targetFuncOpResult : targetFuncOpsResult) {
      // Replace the signature of the target function so that all the return
      // value tensors are packed into a tuple.
      //
      mlir::FunctionType originalFuncType =
          targetFuncOpResult.getFunctionType();

      // Create TupleType object containing all result tensors.
      //
      mlir::TupleType tuplifiedOutputTensors =
          mlir::TupleType::get(&getContext(), originalFuncType.getResults());

      // Create modified function type (signature) that takes the result tuple
      // as the return value.
      //
      FunctionType modifiedFuncType = originalFuncType.clone(
          originalFuncType.getInputs(), tuplifiedOutputTensors);

      rewriter.modifyOpInPlace(targetFuncOpResult,
                               [&targetFuncOpResult, &modifiedFuncType]() {
                                 targetFuncOpResult.setType(modifiedFuncType);
                               });

      // Find return statement and replace with result tuple.
      //
      targetFuncOpResult.walk<WalkOrder::PostOrder, ReverseIterator>(
          [&](mlir::func::ReturnOp returnOp) {
            rewriter.setInsertionPoint(returnOp);
            ttcore::TupleOp tupleOp = rewriter.create<ttcore::TupleOp>(
                returnOp.getLoc(), returnOp.getOperands());
            rewriter.modifyOpInPlace(returnOp, [&]() {
              returnOp.getOperandsMutable().assign(tupleOp);
            });
          });
    }
  }
};

class TTNNPrettifyForCodegen
    : public impl::TTNNPrettifyForCodegenBase<TTNNPrettifyForCodegen> {

private:
  struct PyLoc {
    Location loc;
    Operation *op;
  };

  void printFnInfo(func::FuncOp funcOp) {
    llvm::outs() << "Fn: " << funcOp.getName() << "\n";
    std::cout << "  Inputs count: "
              << funcOp.getFunctionType().getInputs().size() << "\n";
    llvm::outs() << "  Results count: "
                 << funcOp.getFunctionType().getResults().size() << "\n";
    llvm::outs() << "  Is private: " << funcOp.isPrivate() << "\n";
    llvm::outs() << "  Is const-eval: "
                 << ttmlir::utils::isConstEvalFunc(funcOp) << "\n";
    llvm::outs() << "  Is candidate: " << isCandidateFn(funcOp) << "\n";
  }

  bool isCandidateFn(func::FuncOp funcOp) {
    return !funcOp.isPrivate() && !ttmlir::utils::isConstEvalFunc(funcOp) &&
           !funcOp.getFunctionType().getInputs().empty();
  }

  bool isCandidateOp(Operation *op) {
    // Check if ttnn op
    return isa<ttnn::TTNNDialect>(op->getDialect());
  }

  static std::string locationToStr(const mlir::Location &loc) {
    std::string locStr;
    llvm::raw_string_ostream(locStr) << loc;
    return locStr;
  }

  PyLoc parseIRLocation(Operation *op) {
    llvm::outs() << "Op: " << op->getName() << "\n";
    llvm::outs() << "  Loc: " << op->getLoc() << "\n";

    // Early exit if unknown location
    if (op->getLoc() == UnknownLoc::get(op->getContext())) {
      return PyLoc{op->getLoc(), op};
    }

    // Example location:
    // loc("Tail[tail]/ReLU[tail.relu]/forward(test.py:110)/aten__relu")

    std::string locStr = locationToStr(op->getLoc());

    // Break locStr by the "/" character
    std::vector<std::string> locParts;
    for (llvm::StringRef part : llvm::split(locStr, "/")) {
      locParts.push_back(part.str());
    }

    // The last part of the locParts is the operation name
    std::string opName = locParts.back();
    locParts.pop_back();
    // Remove trailing ")" or `")` if present
    if (!opName.empty() && opName.back() == ')') {
      opName.pop_back();
    }
    if (!opName.empty() && opName.back() == '"') {
      opName.pop_back();
    }

    // Next is function name and file line number
    std::string funcNameAndFileLineNum = locParts.back();
    locParts.pop_back();

    // Break funcNameAndFileLineNum by the "(" and ")" characters
    size_t openParen = funcNameAndFileLineNum.find("(");
    size_t closeParen = funcNameAndFileLineNum.find(")");
    std::string funcName = funcNameAndFileLineNum.substr(0, openParen);
    std::string fileLineNum = funcNameAndFileLineNum.substr(
        openParen + 1, closeParen - openParen - 1);

    // Next are Py op type + name, like this: ReLU[tail.relu]
    std::string pyOpTypeAndName = locParts.back();
    locParts.pop_back();
    // Remove leading "loc(" if present (when there's no class name)
    if (pyOpTypeAndName.find("loc(") == 0) {
      pyOpTypeAndName = pyOpTypeAndName.substr(4); // Remove "loc("
    }
    // Remove leading quote if present (from loc("...)
    if (!pyOpTypeAndName.empty() && pyOpTypeAndName[0] == '"') {
      pyOpTypeAndName = pyOpTypeAndName.substr(1);
    }
    size_t openBracket = pyOpTypeAndName.find("[");
    size_t closeBracket = pyOpTypeAndName.find("]");
    std::string pyOpType = pyOpTypeAndName.substr(0, openBracket);
    std::string pyOpName =
        pyOpTypeAndName.substr(openBracket + 1, closeBracket - openBracket - 1);

    // Next, there MIGHT be a class name, and the object name, like this:
    // Tail[tail] So first check if locParts is not empty, use placeholder
    // `None` if empty
    std::string className = "None";
    std::string objectName = "None";
    if (!locParts.empty()) {
      std::string classNameAndObjectName = locParts.back();
      locParts.pop_back();
      // Remove leading "loc(" if present
      if (classNameAndObjectName.find("loc(") == 0) {
        classNameAndObjectName =
            classNameAndObjectName.substr(4); // Remove "loc("
      }
      // Remove leading quote if present (from loc("...)
      if (!classNameAndObjectName.empty() && classNameAndObjectName[0] == '"') {
        classNameAndObjectName = classNameAndObjectName.substr(1);
      }
      size_t classOpenBracket = classNameAndObjectName.find("[");
      size_t classCloseBracket = classNameAndObjectName.find("]");
      className = classNameAndObjectName.substr(0, classOpenBracket);
      objectName = classNameAndObjectName.substr(
          classOpenBracket + 1, classCloseBracket - classOpenBracket - 1);
    }

    llvm::outs() << "    Op: " << opName << "\n";
    llvm::outs() << "    Func: " << funcName << "\n";
    llvm::outs() << "    File: " << fileLineNum << "\n";
    llvm::outs() << "    Py Op Type: " << pyOpType << "\n";
    llvm::outs() << "    Py Op Name: " << pyOpName << "\n";
    llvm::outs() << "    Class: " << className << "\n";
    llvm::outs() << "    Object: " << objectName << "\n";

    return PyLoc{op->getLoc(), op};
  }

  SmallVector<func::FuncOp> findCandidateFns(ModuleOp moduleOp) {
    SmallVector<func::FuncOp, 1> candidateFns;
    moduleOp->walk([&](func::FuncOp funcOp) {
      // printFnInfo(funcOp);
      if (isCandidateFn(funcOp)) {
        candidateFns.push_back(funcOp);
      }
    });
    return candidateFns;
  }

  SmallVector<PyLoc> parseOpsAndGatherLocations(func::FuncOp funcOp) {
    SmallVector<PyLoc> locations;

    funcOp.walk([&](Operation *op) {
      // llvm::outs() << "Op: " << op->getName() << "\n";
      if (!isCandidateOp(op)) {
        return WalkResult::advance();
      }

      PyLoc pyLoc = parseIRLocation(op);
      locations.push_back(pyLoc);
      return WalkResult::advance();
    });

    return locations;
  }

  void buildFunctree(SmallVector<func::FuncOp> candidateFns,
                     SmallVector<PyLoc> locations) {
    for (PyLoc &pyLoc : locations) {
      (void)pyLoc;
    }
  }

public:
  using impl::TTNNPrettifyForCodegenBase<
      TTNNPrettifyForCodegen>::TTNNPrettifyForCodegenBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Need a couple passes
    // 1. Find candidate fns
    // 2. Parse and gather IR locations (remember ops)
    // 3. Build functree
    // 4. Analyze functree
    // 5. Move from original IR to functree-aligned IR

    // Open questions:
    // - What do we do with const-eval fns?
    // - What makes a fn a candidate fn?

    SmallVector<func::FuncOp> candidateFns = findCandidateFns(moduleOp);
    SmallVector<PyLoc> locations;
    for (func::FuncOp funcOp : candidateFns) {
      locations = parseOpsAndGatherLocations(funcOp);
    }
    buildFunctree(candidateFns, locations);
  }
};
} // namespace mlir::tt::ttnn
