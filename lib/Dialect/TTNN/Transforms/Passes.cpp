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
  // @dataclass
  // class LocationModuleCodegen:
  //     module_class: str
  //     module_name: str

  // @dataclass
  // class LocationCodegen:
  //     modules: list[LocationModuleCodegen]
  //     func_path: str
  //     func_name: str
  //     op_line_num: int
  //     op_name: str
  struct PyLoc {
    struct Module {
      std::string moduleClass;
      std::string moduleName;
    };

    llvm::SmallVector<Module> modules;
    std::string funcPath;
    std::string funcName;
    int opLineNum;
    std::string opName;

    Operation *op;

    PyLoc(Operation *op) {
      this->op = op;

      // Get location without "loc(" and ")" characters.
      std::string locStr = locationToStr(op->getLoc());

      // Split locStr by "|" character.
      // For example, given:
      //   "Tail[tail]|ReLU[tail.relu]|/localdev/.../test.py:106|forward|107|aten__relu"
      // Return:
      //   ["Tail[tail]", "ReLU[tail.relu]", "/localdev/.../test.py:106",
      //   "forward", "107", "aten__relu"]
      llvm::SmallVector<llvm::StringRef, 5> locParts;
      llvm::StringRef(locStr).split(locParts, "|", -1, false);

      // Fill in fields from back of locParts.
      size_t n = locParts.size();
      this->opName = locParts[n - 1].str();
      this->opLineNum = std::stoi(locParts[n - 2].str());
      this->funcName = locParts[n - 3].str();
      this->funcPath = locParts[n - 4].str();
      this->modules = llvm::SmallVector<Module>();
      for (size_t i = 0; i < n - 4; i++) {
        // Split each module into class and name.
        // For example, given:
        //   "Tail[tail]"
        // Return:
        //   ["Tail", "tail"]
        llvm::SmallVector<llvm::StringRef, 2> moduleParts;
        locParts[i].split(moduleParts, "[", -1, false);
        this->modules.push_back(
            Module{/* moduleClass= */ moduleParts[0].str(),
                   // Remove trailing "]" from module name.
                   /* moduleName= */ moduleParts[1].str().substr(
                       0, moduleParts[1].str().size() - 1)});
      }
    }
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

    // Remove the loc(" and ") characters
    if (locStr.find("loc(\"") == 0) {
      locStr = locStr.substr(5);
    }
    if (locStr.find("\")") == locStr.size() - 2) {
      locStr = locStr.substr(0, locStr.size() - 2);
    }

    return locStr;
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

      PyLoc pyLoc(op);
      locations.push_back(pyLoc);
      return WalkResult::advance();
    });

    return locations;
  }

  void printPyLocs(func::FuncOp candidateFn, SmallVector<PyLoc> locations) {
    for (PyLoc &pyLoc : locations) {
      llvm::outs() << "PyLoc: " << pyLoc.op->getName() << "\n";
      llvm::outs() << "  Loc: " << pyLoc.op->getLoc() << "\n";
      llvm::outs() << "  Func path: " << pyLoc.funcPath << "\n";
      llvm::outs() << "  Func name: " << pyLoc.funcName << "\n";
      llvm::outs() << "  Op line num: " << pyLoc.opLineNum << "\n";
      llvm::outs() << "  Op name: " << pyLoc.opName << "\n";
      llvm::outs() << "  Modules: " << pyLoc.modules.size() << "\n";
      for (PyLoc::Module &module : pyLoc.modules) {
        llvm::outs() << "    Module: " << module.moduleClass << "["
                     << module.moduleName << "]\n";
      }
    }
  }

  std::string validateLocations(SmallVector<PyLoc> locations) {
    llvm::StringMap<PyLoc> funcPathMap;
    for (PyLoc &pyLoc : locations) {
      auto [it, success] = funcPathMap.try_emplace(pyLoc.funcPath, pyLoc);
      if (!success && it->second.funcName != pyLoc.funcName) {
        return "Found different func names at the same path: " +
               pyLoc.funcPath + " (first func name: " + it->second.funcName +
               ", second func name: " + pyLoc.funcName + ")";
      }
    }
    return "";
  }

  // Group operations by their funcPath.
  // Returns a map: funcPath -> (funcName, vector of PyLocs)
  llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>>
  groupOperationsByFunction(SmallVector<PyLoc> &locations) {
    llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>> funcGroups;

    for (PyLoc &pyLoc : locations) {
      auto it = funcGroups.find(pyLoc.funcPath);
      if (it == funcGroups.end()) {
        // First time seeing this funcPath
        SmallVector<PyLoc> ops;
        ops.push_back(pyLoc);
        funcGroups[pyLoc.funcPath] = {pyLoc.funcName, std::move(ops)};
      } else {
        // Add to existing group
        it->second.second.push_back(pyLoc);
      }
    }

    return funcGroups;
  }

  void printFunctionGroups(
      const llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>>
          &funcGroups) {
    llvm::outs() << "\n=== Function Groups ===\n";
    for (const auto &[funcPath, funcInfo] : funcGroups) {
      const std::string &funcName = funcInfo.first;
      const SmallVector<PyLoc> &locations = funcInfo.second;

      llvm::outs() << "\nFunction: " << funcName << "\n";
      llvm::outs() << "  Path: " << funcPath << "\n";
      llvm::outs() << "  Operations: " << locations.size() << "\n";
      for (const PyLoc &pyLoc : locations) {
        llvm::outs() << "    - " << pyLoc.op->getName() << " (line "
                     << pyLoc.opLineNum << ")\n";
      }
    }
    llvm::outs() << "\n";
  }

  // Information about data flow for a function group
  struct FunctionBoundaryInfo {
    std::string funcName;
    std::string funcPath;
    SmallVector<PyLoc> locations;

    // Values that flow INTO this function (used but not defined here)
    SmallVector<Value> inputValues;

    // Values that flow OUT of this function (defined here, used elsewhere)
    SmallVector<Value> outputValues;

    // Values that are internal (defined and used only within this function)
    SmallVector<Value> internalValues;
  };

  // Analyze data flow for each function group
  llvm::StringMap<FunctionBoundaryInfo> analyzeFunctionBoundaries(
      const llvm::StringMap<std::pair<std::string, SmallVector<PyLoc>>>
          &funcGroups) {
    llvm::StringMap<FunctionBoundaryInfo> boundaryInfos;

    for (const auto &[funcPath, funcInfo] : funcGroups) {
      const std::string &funcName = funcInfo.first;
      const SmallVector<PyLoc> &locations = funcInfo.second;

      FunctionBoundaryInfo info;
      info.funcName = funcName;
      info.funcPath = funcPath.str();
      info.locations = locations;

      // Build a set of operations in this group for fast lookup
      llvm::DenseSet<Operation *> opsInGroup;
      for (const PyLoc &pyLoc : locations) {
        opsInGroup.insert(pyLoc.op);
      }

      // Track all values defined within this group
      llvm::DenseSet<Value> valuesDefinedInGroup;
      for (const PyLoc &pyLoc : locations) {
        for (Value result : pyLoc.op->getResults()) {
          valuesDefinedInGroup.insert(result);
        }
      }

      // Analyze each operation's operands and results
      llvm::DenseSet<Value> inputValuesSet;
      llvm::DenseSet<Value> outputValuesSet;
      llvm::DenseSet<Value> internalValuesSet;

      for (const PyLoc &pyLoc : locations) {
        // Check operands (inputs to the op)
        for (Value operand : pyLoc.op->getOperands()) {
          // If this value is not defined in this group, it's an input
          if (!valuesDefinedInGroup.contains(operand)) {
            inputValuesSet.insert(operand);
          }
        }

        // Check results (outputs from the op)
        for (Value result : pyLoc.op->getResults()) {
          bool usedOutside = false;
          bool usedInside = false;

          // Check if this result is used outside this group
          for (Operation *user : result.getUsers()) {
            if (opsInGroup.contains(user)) {
              usedInside = true;
            } else {
              usedOutside = true;
            }
          }

          if (usedOutside) {
            outputValuesSet.insert(result);
          } else if (usedInside) {
            internalValuesSet.insert(result);
          }
          // Note: if a value is not used at all, we don't track it
        }
      }

      // Convert sets to vectors
      info.inputValues.assign(inputValuesSet.begin(), inputValuesSet.end());
      info.outputValues.assign(outputValuesSet.begin(), outputValuesSet.end());
      info.internalValues.assign(internalValuesSet.begin(),
                                 internalValuesSet.end());

      boundaryInfos[funcPath] = std::move(info);
    }

    return boundaryInfos;
  }

  void printFunctionBoundaries(
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos) {
    llvm::outs() << "\n=== Function Boundary Analysis ===\n";
    for (const auto &[funcPath, info] : boundaryInfos) {
      llvm::outs() << "\nFunction: " << info.funcName << "\n";
      llvm::outs() << "  Path: " << info.funcPath << "\n";
      llvm::outs() << "  Input values: " << info.inputValues.size() << "\n";
      for (Value input : info.inputValues) {
        llvm::outs() << "    - " << input << " (type: " << input.getType()
                     << ")\n";
      }
      llvm::outs() << "  Output values: " << info.outputValues.size() << "\n";
      for (Value output : info.outputValues) {
        llvm::outs() << "    - " << output << " (type: " << output.getType()
                     << ")\n";
      }
      llvm::outs() << "  Internal values: " << info.internalValues.size()
                   << "\n";
    }
    llvm::outs() << "\n";
  }

  // Create new function declarations based on boundary information
  llvm::StringMap<func::FuncOp> createNewFunctions(
      IRRewriter &rewriter, ModuleOp moduleOp,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos) {
    llvm::StringMap<func::FuncOp> newFunctions;

    for (const auto &[funcPath, info] : boundaryInfos) {
      // Collect input types from input values
      SmallVector<Type> inputTypes;
      for (Value input : info.inputValues) {
        inputTypes.push_back(input.getType());
      }

      // Collect output types from output values
      SmallVector<Type> outputTypes;
      for (Value output : info.outputValues) {
        outputTypes.push_back(output.getType());
      }

      // Create function type
      FunctionType funcType =
          FunctionType::get(rewriter.getContext(), inputTypes, outputTypes);

      // Set insertion point to end of module
      rewriter.setInsertionPointToEnd(moduleOp.getBody());

      // Create the new function
      func::FuncOp newFunc = rewriter.create<func::FuncOp>(
          moduleOp.getLoc(), info.funcName, funcType);

      // Mark as private for now (we'll make the entry point public later)
      newFunc.setPrivate();

      // Store the function
      newFunctions[funcPath] = newFunc;
    }

    return newFunctions;
  }

  void printNewFunctions(const llvm::StringMap<func::FuncOp> &newFunctions) {
    llvm::outs() << "\n=== Created Functions ===\n";
    for (const auto &entry : newFunctions) {
      llvm::StringRef funcPath = entry.getKey();
      func::FuncOp funcOp = entry.getValue();

      llvm::outs() << "\nFunction: " << funcOp.getName() << "\n";
      llvm::outs() << "  Path: " << funcPath << "\n";
      llvm::outs() << "  Signature: " << funcOp.getFunctionType() << "\n";
      llvm::outs() << "  Input types: "
                   << funcOp.getFunctionType().getInputs().size() << "\n";
      for (Type inputType : funcOp.getFunctionType().getInputs()) {
        llvm::outs() << "    - " << inputType << "\n";
      }
      llvm::outs() << "  Output types: "
                   << funcOp.getFunctionType().getResults().size() << "\n";
      for (Type outputType : funcOp.getFunctionType().getResults()) {
        llvm::outs() << "    - " << outputType << "\n";
      }
    }
    llvm::outs() << "\n";
  }

  // Populate function bodies by cloning operations from the original function
  void populateFunctionBodies(
      IRRewriter &rewriter,
      const llvm::StringMap<FunctionBoundaryInfo> &boundaryInfos,
      llvm::StringMap<func::FuncOp> &newFunctions) {

    // For each function, we need to:
    // 1. Create entry block with arguments
    // 2. Clone operations in order
    // 3. Map old values to new values
    // 4. Add return statement

    for (const auto &entry : newFunctions) {
      llvm::StringRef funcPath = entry.getKey();
      func::FuncOp funcOp = entry.getValue();
      const FunctionBoundaryInfo &info = boundaryInfos.lookup(funcPath);

      // Create entry block
      Block *entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);

      // Create value mapping: old values -> new values
      IRMapping valueMapping;

      // Map input values to block arguments
      for (size_t i = 0; i < info.inputValues.size(); i++) {
        valueMapping.map(info.inputValues[i], entryBlock->getArgument(i));
      }

      // Clone operations in order
      for (const PyLoc &pyLoc : info.locations) {
        Operation *oldOp = pyLoc.op;
        Operation *newOp = rewriter.clone(*oldOp, valueMapping);

        // Update the value mapping with the results of the cloned operation
        for (size_t i = 0; i < oldOp->getNumResults(); i++) {
          valueMapping.map(oldOp->getResult(i), newOp->getResult(i));
        }
      }

      // Collect output values (now mapped to new function's values)
      SmallVector<Value> returnValues;
      for (Value oldOutput : info.outputValues) {
        Value newOutput = valueMapping.lookup(oldOutput);
        returnValues.push_back(newOutput);
      }

      // Add return statement
      rewriter.create<func::ReturnOp>(funcOp.getLoc(), returnValues);
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
    // -----
    // 3. Build functree
    // 4. Analyze functree
    // 5. Move from original IR to functree-aligned IR

    // Open questions:
    // - What do we do with const-eval fns?
    // - What makes a fn a candidate fn?

    // For simplicity, supporting only one for now, but can support multiple.
    SmallVector<func::FuncOp> candidateFns = findCandidateFns(moduleOp);
    assert(candidateFns.size() == 1 &&
           "Only one candidate fn is supported now");

    func::FuncOp candidateFn = candidateFns.front();

    llvm::SmallVector<PyLoc> locations =
        parseOpsAndGatherLocations(candidateFn);

    // Debug prints
    printPyLocs(candidateFn, locations);

    // Validate locations
    std::string errorMessage = validateLocations(locations);
    if (!errorMessage.empty()) {
      llvm::outs() << "PrettifyForCodegen error: " << errorMessage << "\n";
      signalPassFailure();
    }

    // Group operations by function
    auto funcGroups = groupOperationsByFunction(locations);
    printFunctionGroups(funcGroups);

    // Analyze function boundaries and data flow
    auto boundaryInfos = analyzeFunctionBoundaries(funcGroups);
    printFunctionBoundaries(boundaryInfos);

    // Create new functions based on boundary information
    auto newFunctions = createNewFunctions(rewriter, moduleOp, boundaryInfos);
    printNewFunctions(newFunctions);

    // Populate function bodies with operations
    populateFunctionBodies(rewriter, boundaryInfos, newFunctions);
  }
};
} // namespace mlir::tt::ttnn
