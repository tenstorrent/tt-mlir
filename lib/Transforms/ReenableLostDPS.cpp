// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreTraits.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::transforms {

#define GEN_PASS_DEF_REENABLELOSTDPS
#include "ttmlir/Transforms/Passes.h.inc"

// Function to transform defining ops of return values s.t. they use original
// DPS parameter (stashed in return_to_output_mapping attr) instead of empty ops
// created during lowering.
static LogicalResult reenableDpsFromAttr(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  bool failed = false;
  // Find all functions with the return_to_output_mapping attribute
  moduleOp.walk([&](func::FuncOp funcOp) {
    // Only process functions that have the mapping attribute
    if (!funcOp->hasAttr(ttir::ReturnToOutputMappingAttr::name)) {
      return;
    }

    // Get the return_to_output_mapping attribute
    auto mappingAttr = funcOp->getAttrOfType<IntegerAttr>(
        ttir::ReturnToOutputMappingAttr::name);
    if (!mappingAttr) {
      funcOp->emitError() << "Function has ttir.return_to_output_mapping "
                             "attribute but it's not an IntegerAttr";
      failed = true;
      return;
    }

    // Get the output operand index
    unsigned outputArgIdx = mappingAttr.getInt();
    if (outputArgIdx >= funcOp.getNumArguments()) {
      funcOp->emitError() << "Output argument index " << outputArgIdx
                          << " is out of range (function has "
                          << funcOp.getNumArguments() << " arguments)";
      failed = true;
      return;
    }

    // Check if the output parameter is actually used in the function
    // If it's already being used, then DPS is already enabled and we don't
    // need to do anything. This is a more reliable check than relying on
    // pass-specific attributes.
    BlockArgument outputArg = funcOp.getArgument(outputArgIdx);
    if (!outputArg.use_empty()) {
      // The output argument is already being used, so DPS is already enabled
      // Remove the mapping attribute and skip this function
      funcOp->removeAttr(ttir::ReturnToOutputMappingAttr::name);
      return;
    }

    // Find the return operation
    func::ReturnOp returnOp;
    for (Block &block : funcOp.getBlocks()) {
      if (auto retOp = dyn_cast<func::ReturnOp>(block.getTerminator())) {
        returnOp = retOp;
        break;
      }
    }

    if (!returnOp) {
      funcOp->emitError() << "Function does not have a return operation";
      failed = true;
      return;
    }

    if (returnOp.getNumOperands() == 0) {
      funcOp->emitWarning()
          << "Function already has no return values, nothing to transform";
      funcOp->removeAttr(ttir::ReturnToOutputMappingAttr::name);
      return;
    }

    Value returnVal = returnOp.getOperands()[0];

    // Check if the return value is a block argument--this indicates the hoisted
    // op has been optimized away, and we must replace it with a copy s.t. the
    // output tensor gets the proper value.
    if (auto blockArg = dyn_cast<BlockArgument>(returnVal)) {
      // This is a NOOP case - we need to copy the input to the output.
      Value outputTensor = funcOp.getArgument(outputArgIdx);

      rewriter.setInsertionPoint(returnOp);

      if (auto inputType = dyn_cast<RankedTensorType>(returnVal.getType())) {
        auto outputType = dyn_cast<RankedTensorType>(outputTensor.getType());

        assert(inputType == outputType);

        rewriter.create<linalg::CopyOp>(returnOp.getLoc(), returnVal,
                                        outputTensor);
      }

      // Remove the mapping attribute since we've applied it
      funcOp->removeAttr(ttir::ReturnToOutputMappingAttr::name);
      return;
    }

    // Find the operation that produces the return value
    Operation *producer = returnVal.getDefiningOp();
    if (!producer) {
      funcOp->emitError() << "Return value is not produced by an operation";
      failed = true;
      return;
    }

    // Handle different types of operations
    bool transformed = false;

    // Helper lambda to replace tensor.empty with output tensor
    auto replaceEmptyWithOutput =
        [&](tensor::EmptyOp emptyOp, Value outputTensor,
            ArrayRef<ReassociationIndices> reassocIndices = {}) {
          auto outputTensorType =
              mlir::cast<RankedTensorType>(outputTensor.getType());
          auto emptyOpType = mlir::cast<RankedTensorType>(emptyOp.getType());

          rewriter.setInsertionPoint(emptyOp);

          if (outputTensorType.getRank() != emptyOpType.getRank() &&
              !reassocIndices.empty()) {
            // Need to collapse the output tensor to match the expected shape
            auto collapsedType = RankedTensorType::get(
                emptyOpType.getShape(), outputTensorType.getElementType());

            Value collapsedTensor = rewriter.create<tensor::CollapseShapeOp>(
                emptyOp.getLoc(), collapsedType, outputTensor, reassocIndices);

            rewriter.replaceOp(emptyOp, collapsedTensor);
          } else {
            // Ranks match or no reassociation provided, replace directly
            rewriter.replaceOp(emptyOp, outputTensor);
          }
          return true;
        };

    // Helper lambda to find and replace empty ops in DPS init operands
    auto processInitOperands = [&](Operation *op) {
      auto dpsInterface = dyn_cast<DestinationStyleOpInterface>(op);
      if (!dpsInterface) {
        return false;
      }

      bool found = false;
      for (OpOperand &initOperand : dpsInterface.getDpsInitsMutable()) {
        Value initValue = initOperand.get();
        if (auto emptyOp = initValue.getDefiningOp<tensor::EmptyOp>()) {
          Value outputTensor = funcOp.getArgument(outputArgIdx);
          replaceEmptyWithOutput(emptyOp, outputTensor);
          found = true;
        }
      }
      return found;
    };

    // Handle tensor.expand_shape operations
    if (auto expandOp = mlir::dyn_cast<tensor::ExpandShapeOp>(producer)) {
      // Get the input to the expand_shape operation
      Value expandInput = expandOp.getSrc();
      Operation *expandInputProducer = expandInput.getDefiningOp();

      // Check if the input is produced by a linalg.reduce operation
      if (auto reduceOp =
              mlir::dyn_cast_or_null<linalg::ReduceOp>(expandInputProducer)) {
        // Find the tensor.empty operation that feeds into the linalg.reduce
        for (OpOperand &output : reduceOp->getOpOperands()) {
          // Check if this is an output operand (init tensor)
          if (reduceOp.isInitTensor(&output)) {
            Value outputBuffer = output.get();

            // Check if it's defined by a linalg.fill operation
            if (auto fillOp = mlir::dyn_cast_or_null<linalg::FillOp>(
                    outputBuffer.getDefiningOp())) {
              // Get the destination of the fill operation
              Value fillDest = fillOp.getDpsInitOperand(0)->get();

              // Check if it's defined by a tensor.empty operation
              if (auto emptyOp = mlir::dyn_cast_or_null<tensor::EmptyOp>(
                      fillDest.getDefiningOp())) {
                Value outputTensor = funcOp.getArgument(outputArgIdx);
                transformed = replaceEmptyWithOutput(
                    emptyOp, outputTensor, expandOp.getReassociationIndices());
              }
            } else if (auto emptyOp = mlir::dyn_cast_or_null<tensor::EmptyOp>(
                           outputBuffer.getDefiningOp())) {
              Value outputTensor = funcOp.getArgument(outputArgIdx);
              transformed = replaceEmptyWithOutput(
                  emptyOp, outputTensor, expandOp.getReassociationIndices());
            }
          }
        }
      }
    }
    // Handle operations implementing DestinationStyleOpInterface
    else if (auto dpsOp =
                 mlir::dyn_cast<DestinationStyleOpInterface>(producer)) {
      transformed = processInitOperands(producer);
    }
    // Handle other operations that might create temporary tensors
    else {
      funcOp->emitWarning()
          << "Unhandled operation type: " << producer->getName().getStringRef()
          << ". Operation must implement DestinationStyleOpInterface or be "
             "tensor.expand_shape.";
    }

    if (!transformed) {
      funcOp->emitWarning()
          << "Could not find any tensor.empty operations to replace";
      return;
    }

    // Remove the mapping attribute since we've applied it
    funcOp->removeAttr(ttir::ReturnToOutputMappingAttr::name);
  });

  return failed ? failure() : success();
}

namespace {
// Pass which re-enables DPS semantics after lowering passes that may have
// replaced DPS outputs with new empty tensors. This is a workaround for
// transformations (TOSA lowering, custom decompositions, etc.) that don't
// preserve the original DPS structure.
//
// The pass looks for functions with the "ttir.return_to_output_mapping"
// attribute, which indicates:
// 1. The function originally used DPS semantics
// 2. The return value should correspond to a specific input parameter
// 3. Intermediate transformations may have broken this correspondence
//
// The pass checks if the designated output parameter is unused. If it is
// unused, this indicates that DPS has been broken (likely by a lowering that
// created new tensor.empty ops). The pass then traces back from the return
// value to find these tensor.empty ops and replaces them with the appropriate
// function argument, effectively re-enabling DPS.
class ReenableLostDPS : public impl::ReenableLostDPSBase<ReenableLostDPS> {
  using impl::ReenableLostDPSBase<ReenableLostDPS>::ReenableLostDPSBase;
  void runOnOperation() override {
    if (failed(reenableDpsFromAttr(getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::tt::transforms
