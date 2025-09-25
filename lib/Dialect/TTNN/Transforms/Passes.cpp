// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
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
#define GEN_PASS_DEF_LOCTEST
#define GEN_PASS_DEF_TTNNCREATEINPUTGENERATORS
#define GEN_PASS_DEF_TTNNDEALLOCATE
#define GEN_PASS_DEF_TTNNTUPLIFYTENSORS
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

class TTNNCreateInputGenerators
    : public impl::TTNNCreateInputGeneratorsBase<TTNNCreateInputGenerators> {

public:
  using impl::TTNNCreateInputGeneratorsBase<
      TTNNCreateInputGenerators>::TTNNCreateInputGeneratorsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

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

    // Iterate over all func ops and add input tensor generator functions.
    //
    llvm::SmallVector<mlir::func::FuncOp, 1> inputGenFuncOps;
    for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
      rewriter.setInsertionPointToEnd(block);
      inputGenFuncOps.emplace_back(createInputGeneratorFunction(
          rewriter, forwardFuncOp.getLoc(), forwardFuncOp));
    }

    // Create a main function to call input generators and forward funcs.
    //
    {
      std::string mainFuncName = "main";

      // Create a function type.
      //
      FunctionType functionType =
          mlir::FunctionType::get(&getContext(), {}, rewriter.getI32Type());

      // Set insertion point to end of the block.
      //
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

      for (auto [forwardFuncOp, inputGenFuncOp] :
           llvm::zip_equal(forwardFuncOps, inputGenFuncOps)) {

        // Generate the input tensors for a forwardFuncOp.
        //
        func::CallOp generatedTensors = rewriter.create<mlir::func::CallOp>(
            forwardFuncOp.getLoc(), inputGenFuncOp, /*operands=*/ValueRange());

        // Call a forward function with the generated tensors.
        //
        rewriter.create<mlir::func::CallOp>(forwardFuncOp.getLoc(),
                                            forwardFuncOp,
                                            generatedTensors->getResults());
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
  }

private:
  static func::FuncOp createInputGeneratorFunction(IRRewriter &rewriter,
                                                   Location loc,
                                                   func::FuncOp forwardFuncOp) {
    MLIRContext *ctx = rewriter.getContext();

    // Create a new function that will generate the input tensors.
    //
    std::string inputGenFuncName =
        "create_inputs_for_" + forwardFuncOp.getName().str();

    // Create the function type.
    //
    llvm::ArrayRef<mlir::Type> returnTypes =
        forwardFuncOp.getFunctionType().getInputs();
    FunctionType functionType = mlir::FunctionType::get(ctx, {}, returnTypes);

    // Create the function.
    //
    func::FuncOp inputGenFuncOp = rewriter.create<mlir::func::FuncOp>(
        loc, inputGenFuncName, functionType);

    // Add a Block to func op and set insertion point to the beginning of the
    // Block.
    //
    rewriter.modifyOpInPlace(inputGenFuncOp, [&]() {
      rewriter.setInsertionPointToStart(inputGenFuncOp.addEntryBlock());
    });

    // Create input tensors. Currently, we only create tensors of ones.
    //
    assert(
        returnTypes.size() == 1 && mlir::isa<TupleType>(returnTypes.front()) &&
        "Expected input generator to return a single tuple of input tensors!");

    SmallVector<Value> generatedTensors;
    for (const Type &type :
         mlir::cast<mlir::TupleType>(returnTypes[0]).getTypes()) {
      // Ensure that the type is a RankedTensorType.
      //
      RankedTensorType rankedTensorType =
          mlir::dyn_cast<RankedTensorType>(type);
      assert(rankedTensorType &&
             "Expected input tensor to be of type RankedTensorType!");

      generatedTensors.push_back(
          generateTensor(rewriter, loc, rankedTensorType));
    }

    // Create a tuple from the generated tensors.
    //
    ttcore::TupleOp tuple =
        rewriter.create<ttcore::TupleOp>(loc, returnTypes, generatedTensors);

    // Create ReturnOp.
    //
    rewriter.create<func::ReturnOp>(forwardFuncOp.getLoc(),
                                    tuple->getResults());

    return inputGenFuncOp;
  }

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

    // Create a new tensor of ones.
    //
    ttnn::OnesOp onesOp = rewriter.create<ttnn::OnesOp>(
        loc, tensorType, /*device=*/nullptr, shapeAttr, dTypeAttr,
        tensorLayoutAttr, /*memory_config=*/nullptr);

    // If tensor is meant to be on device, add ToDevice op.
    //
    if (layoutAttr.isDeviceBufferType()) {
      ttnn::GetDeviceOp device =
          ttnn::utils::getOrInsertDevice(rewriter, onesOp);

      mlir::Value tensorOnDevice = rewriter.create<ttnn::ToDeviceOp>(
          loc, tensorType, onesOp, device, nullptr);

      return tensorOnDevice;
    }
    return onesOp;
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

class LocTest : public impl::LocTestBase<LocTest> {

public:
  using impl::LocTestBase<LocTest>::LocTestBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // ttir::AddOp addOp;

    moduleOp.walk([&](Operation *op) {
      // llvm::outs() << op->getName() << "\n";
      // if (auto target = mlir::dyn_cast<ttir::AddOp>(op)) {
      //   addOp = target;
      //   // llvm::outs() << "Found add op: " << target << "\n";
      // }

      llvm::outs() << "Printing location for op " << op->getName() << "\n";
      printLocationStack(op->getLoc(), llvm::outs());
      llvm::outs() << "\n\n\n\n";
    });

    // llvm::outs() << "\n\nSTART\n\n";
    // printLocationStack(addOp.getLoc(), llvm::outs());
    // llvm::outs() << "\nEND\n\n";

    // mlir::OpPrintingFlags flags;
    // flags.enableDebugInfo(
    //     /*prettyForm=*/true); // show locations, expand callsite locs
    // llvm::outs() << "\n\n\n\n\n";
    // addOp->print(llvm::outs(), flags);
    // llvm::outs() << "\n\n\n\n\n";
    // // addOp.dump();

    // // Print loc on last add op
    // printExpandedLoc(addOp->getLoc());
    // llvm::outs() << "\n\n\n\n\n";
  }

  void printExpandedLoc(mlir::Location loc) {
    if (auto callLoc = mlir::dyn_cast<mlir::CallSiteLoc>(loc)) {
      llvm::outs() << "Call @ "
                   << mlir::dyn_cast<FileLineColLoc>(callLoc.getCaller())
                   << "\n";
      printExpandedLoc(callLoc.getCallee());
    } else if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
      fileLoc.print(llvm::outs());
      llvm::outs() << "\n";
    } else {
      loc.print(llvm::outs());
      llvm::outs() << "\n";
    }
  }

  void printLocationStack(Location loc, raw_ostream &os, int indent = 0) {
    if (auto callLoc = dyn_cast<CallSiteLoc>(loc)) {
      os.indent(indent) << "CallSite:\n";
      os.indent(indent + 2) << "Callee:\n";
      printLocationStack(callLoc.getCallee(), os, indent + 4);
      os.indent(indent + 2) << "Caller:\n";
      printLocationStack(callLoc.getCaller(), os, indent + 4);
      return;
    }

    if (auto nameLoc = dyn_cast<NameLoc>(loc)) {
      os.indent(indent) << "NameLoc: " << nameLoc.getName() << "\n";
      printLocationStack(nameLoc.getChildLoc(), os, indent + 2);
      return;
    }

    if (auto fileLoc = dyn_cast<FileLineColLoc>(loc)) {
      os.indent(indent) << fileLoc.getFilename().str() << ":"
                        << fileLoc.getLine() << ":" << fileLoc.getColumn()
                        << "\n";
      return;
    }

    if (auto fused = dyn_cast<FusedLoc>(loc)) {
      os.indent(indent) << "FusedLoc:\n";
      for (Location sub : fused.getLocations()) {
        printLocationStack(sub, os, indent + 2);
      }
      return;
    }

    if (isa<UnknownLoc>(loc)) {
      os.indent(indent) << "<unknown>\n";
      return;
    }

    // Generic fallback (covers other future Location kinds).
    os.indent(indent) << loc << "\n";
  }
};

} // namespace mlir::tt::ttnn
