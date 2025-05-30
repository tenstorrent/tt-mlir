// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
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
#define GEN_PASS_DEF_TTNNDEALLOCATE
#define GEN_PASS_DEF_TTNNCREATEINPUTGENERATORS
#define GEN_PASS_DEF_TTNNMODIFYSIGNATURESFORDYLIB
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

      // Const eval subgraphs may not dealloc their params since they don't own
      // them.
      if (!ttmlir::utils::isConstEvalFunc(func)) {
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

      if (!funcOp->getUses().empty()) {
        mlir::WalkResult::skip();
      }
      forwardFuncOps.push_back(funcOp);
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
    // Get all the input tensors for the current forward func.
    //
    llvm::SmallVector<mlir::RankedTensorType, 2> inputTensors(
        llvm::map_to_vector(forwardFuncOp.getFunctionType().getInputs(),
                            [](Type type) {
                              return llvm::cast<mlir::RankedTensorType>(type);
                            }));

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
    SmallVector<Value, 2> generatedTensors(
        llvm::map_to_vector(returnTypes, [&](Type type) {
          return generateTensor(rewriter, loc, type);
        }));

    rewriter.create<func::ReturnOp>(forwardFuncOp.getLoc(), generatedTensors);
    return inputGenFuncOp;
  }

  // Currently only supports generating tensors of ones.
  // TODO(azecevic): Support generating other types of tensors that has a
  // `TT_CreationOpTrait`.
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
    DataTypeAttr dTypeAttr = DataTypeAttr::get(ctx, layoutAttr.getDataType());

    // Create a new tensor of ones.
    //
    ttnn::OnesOp onesOp =
        rewriter.create<ttnn::OnesOp>(loc, tensorType, shapeAttr, dTypeAttr,
                                      tensorLayoutAttr, nullptr, nullptr);

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

class TTNNModifySignaturesForDylib
    : public impl::TTNNModifySignaturesForDylibBase<
          TTNNModifySignaturesForDylib> {

public:
  using impl::TTNNModifySignaturesForDylibBase<
      TTNNModifySignaturesForDylib>::TTNNModifySignaturesForDylibBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // If we have a nested module structure, we want to use nested module inside
    // DeviceModule.
    tt::DeviceModuleOp deviceModule;
    for (auto &op : moduleOp.getBody()->getOperations()) {
      deviceModule = llvm::dyn_cast<tt::DeviceModuleOp>(op);
      if (deviceModule) {
        break;
      }
    }
    if (deviceModule) {
      moduleOp = dyn_cast_if_present<mlir::ModuleOp>(
          deviceModule.getBodyRegion().front().front());
      assert(moduleOp &&
             "Found tt::DeviceModuleOp but it didn't contain a single "
             "mlir::ModuleOp!");
    }
    IRRewriter rewriter(&getContext());

    // Ensure that the module has a single region and a single block within that
    // region.
    //
    assert(moduleOp->getRegions().size() == 1);
    assert(moduleOp->getRegion(0).getBlocks().size() == 1);

    // Get the the only existing block.
    //
    Block *block = moduleOp.getBody(0);

    // Find all the func.func ops in the module that are "forward" functions.
    //
    SmallVector<func::FuncOp, 1> forwardFuncOps;
    block->walk([&](func::FuncOp funcOp) {
      // Rename the function if it's named `main` to `_main`. This is done
      // as compiler will complain that `main` must return `int` and that
      // it's first parameter must also be an `int`.
      if (funcOp.getName() == "main") {
        rewriter.modifyOpInPlace(funcOp, [&]() { funcOp.setSymName("_main"); });
      }

      if (!funcOp->getUses().empty()) {
        mlir::WalkResult::skip();
      }

      forwardFuncOps.push_back(funcOp);
    });

    // Iterate over all the func ops and modify the signatures.
    //
    for (mlir::func::FuncOp forwardFuncOp : forwardFuncOps) {
      // Replace the signature of the forward function so that all the tensor
      // arguments are packed into a single tuple, and device type is appended.
      //
      mlir::FunctionType originalFuncType = forwardFuncOp.getFunctionType();
      assert(
          llvm::all_of(originalFuncType.getInputs(),
                       [](Type t) { return mlir::isa<RankedTensorType>(t); }) &&
          "Expected all inputs must be of type RankedTensorType");

      // Create Type objects for modified function signature:
      // 1. tuplifiedInputTensors: TupleType of all input tensors
      // 2. tuplifiedOutputTensors: TupleType of all output tensors
      //
      mlir::TupleType tuplifiedInputTensors =
          mlir::TupleType::get(&getContext(), originalFuncType.getInputs());
      mlir::TupleType tuplifiedOutputTensors =
          mlir::TupleType::get(&getContext(), originalFuncType.getResults());

      // Create modified function type (signature) that takes the input tuple
      // as an operand, and returns the output tuple.
      //
      FunctionType modifiedFuncType =
          originalFuncType.clone(tuplifiedInputTensors, tuplifiedOutputTensors);

      rewriter.modifyOpInPlace(forwardFuncOp,
                               [&forwardFuncOp, &modifiedFuncType]() {
                                 forwardFuncOp.setType(modifiedFuncType);
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
      // Additionally, the Device is added as the second argument, and the
      // GetDeviceOp that creates Device is removed.
      //
      // The return statement is modified to return a tuple.
      //
      Block &entryBlock = forwardFuncOp.getBlocks().front();
      constexpr size_t paramOffset = 1;
      entryBlock.insertArgument(/*index=*/0u, tuplifiedInputTensors,
                                forwardFuncOp.getLoc());

      rewriter.setInsertionPointToStart(&entryBlock);
      for (size_t idx = 0; idx < originalFuncType.getNumInputs(); idx++) {
        tt::GetTupleElementOp getTupleElementOp =
            rewriter.create<tt::GetTupleElementOp>(
                forwardFuncOp.getLoc(), forwardFuncOp.getArgument(0), idx);

        rewriter.replaceAllUsesWith(entryBlock.getArgument(paramOffset + idx),
                                    getTupleElementOp);
      }

      // Erase original arguments.
      //
      entryBlock.eraseArguments(paramOffset,
                                originalFuncType.getInputs().size());

      // Find return statement and replace with tuple.
      //
      forwardFuncOp.walk<WalkOrder::PostOrder, ReverseIterator>(
          [&](mlir::func::ReturnOp returnOp) {
            rewriter.setInsertionPoint(returnOp);
            TupleOp tupleOp = rewriter.create<mlir::tt::TupleOp>(
                returnOp.getLoc(), returnOp.getOperands());
            rewriter.modifyOpInPlace(returnOp, [&]() {
              returnOp.getOperandsMutable().assign(tupleOp);
            });
          });
    }
  }
};

} // namespace mlir::tt::ttnn
