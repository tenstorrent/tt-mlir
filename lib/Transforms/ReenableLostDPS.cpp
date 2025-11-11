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

static tensor::EmptyOp findEmptyOp(Value value) {
  if (auto emptyOp = value.getDefiningOp<tensor::EmptyOp>()) {
    return emptyOp;
  }

  // Handle cases where empty op is filled via FillOp--seems to be common
  // pattern in linalg.
  if (auto fillOp = value.getDefiningOp<linalg::FillOp>()) {
    Value fillInput = fillOp.getDpsInitOperand(0)->get();
    return fillInput.getDefiningOp<tensor::EmptyOp>();
  }

  return nullptr;
}

// Function to transform defining ops of return values s.t. they use original
// DPS parameter (stashed in return_to_output_mapping attr) instead of empty ops
// created during lowering.
static LogicalResult reenableDpsFromAttr(ModuleOp moduleOp) {
  IRRewriter rewriter(moduleOp.getContext());

  // Find all functions with the return_to_output_mapping attribute.
  auto result = moduleOp.walk([&](func::FuncOp funcOp) {
    // Only process functions that have the mapping attribute.
    if (!funcOp->hasAttr(ttir::ReturnToOutputMappingAttr::name)) {
      return WalkResult::skip();
    }

    // Get the return_to_output_mapping attribute.
    auto mappingAttr = funcOp->getAttrOfType<IntegerAttr>(
        ttir::ReturnToOutputMappingAttr::name);
    if (!mappingAttr) {
      funcOp->emitError() << "Function has ttir.return_to_output_mapping "
                             "attribute but it's not an IntegerAttr";
      return WalkResult::interrupt();
    }

    // Get the output operand index.
    unsigned outputArgIdx = mappingAttr.getInt();
    if (outputArgIdx >= funcOp.getNumArguments()) {
      funcOp->emitError() << "Output argument index " << outputArgIdx
                          << " is out of range (function has "
                          << funcOp.getNumArguments() << " arguments)";
      return WalkResult::interrupt();
    }

    // Check if the output parameter is actually used in the function
    // If it's already being used, then DPS is already enabled and we don't
    // need to do anything. This is a more reliable check than relying on
    // pass-specific attributes.
    BlockArgument outputArg = funcOp.getArgument(outputArgIdx);
    if (!outputArg.use_empty()) {
      // The output argument is already being used, so DPS is already enabled
      // Remove the mapping attribute and skip this function.
      funcOp->removeAttr(ttir::ReturnToOutputMappingAttr::name);
      return WalkResult::skip();
    }

    // Find the return operation.
    func::ReturnOp returnOp;
    for (Block &block : funcOp.getBlocks()) {
      if (auto retOp = llvm::dyn_cast<func::ReturnOp>(block.getTerminator())) {
        returnOp = retOp;
        break;
      }
    }

    if (!returnOp) {
      funcOp->emitError() << "Function does not have a return operation";
      return WalkResult::interrupt();
    }

    if (returnOp.getNumOperands() == 0) {
      funcOp->emitWarning()
          << "Function already has no return values, nothing to transform";
      funcOp->removeAttr(ttir::ReturnToOutputMappingAttr::name);
      return WalkResult::skip();
    }

    Value returnVal = returnOp.getOperands()[0];
    Value outputTensor = funcOp.getArgument(outputArgIdx);

    // Check if the return value is a block argument--this indicates the hoisted
    // op has been optimized away, and we must replace it with a copy s.t. the
    // output tensor gets the proper value.
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(returnVal)) {
      // This is a NOOP case - we need to copy the input to the output.

      rewriter.setInsertionPoint(returnOp);

      if (auto inputType =
              llvm::dyn_cast<RankedTensorType>(returnVal.getType())) {
        auto outputType =
            llvm::dyn_cast<RankedTensorType>(outputTensor.getType());

        assert(inputType == outputType);

        rewriter.create<linalg::CopyOp>(returnOp.getLoc(), returnVal,
                                        outputTensor);
      }

      // Remove the mapping attribute since we've applied it.
      funcOp->removeAttr(ttir::ReturnToOutputMappingAttr::name);
      return WalkResult::skip();
    }

    // Find the operation that produces the return value.
    Operation *producer = returnVal.getDefiningOp();
    if (!producer) {
      funcOp->emitError() << "Return value is not produced by an operation";
      return WalkResult::interrupt();
    }

    // Handle different types of operations.
    bool transformed = false;

    // Helper lambda to replace tensor.empty with output tensor.
    auto replaceEmptyWithOutput =
        [&](tensor::EmptyOp emptyOp, Value outputTensor,
            ArrayRef<ReassociationIndices> reassocIndices = {}) {
          auto outputTensorType =
              mlir::cast<RankedTensorType>(outputTensor.getType());
          auto emptyOpType = mlir::cast<RankedTensorType>(emptyOp.getType());

          rewriter.setInsertionPoint(emptyOp);

          if (outputTensorType.getRank() != emptyOpType.getRank() &&
              !reassocIndices.empty()) {
            // Need to collapse the output tensor to match the expected shape.
            auto collapsedType = RankedTensorType::get(
                emptyOpType.getShape(), outputTensorType.getElementType());

            Value collapsedTensor = rewriter.create<tensor::CollapseShapeOp>(
                emptyOp.getLoc(), collapsedType, outputTensor, reassocIndices);

            rewriter.replaceOp(emptyOp, collapsedTensor);
          } else {
            // Ranks match or no reassociation provided, replace directly.
            rewriter.replaceOp(emptyOp, outputTensor);
          }
          return true;
        };

    // Special handling for tensor.expand_shape.
    if (auto expandOp = mlir::dyn_cast<tensor::ExpandShapeOp>(producer)) {
      // For expand_shape, we need to look at its input producer.
      Value expandInput = expandOp.getSrc();
      if (Operation *inputProducer = expandInput.getDefiningOp()) {
        if (auto dpsOp =
                mlir::dyn_cast<DestinationStyleOpInterface>(inputProducer)) {
          // Process the DPS operation that feeds into expand_shape.
          for (OpOperand &initOperand : dpsOp.getDpsInitsMutable()) {
            if (auto emptyOp = findEmptyOp(initOperand.get())) {
              // For expand_shape, we need reassociation indices.
              transformed = replaceEmptyWithOutput(
                  emptyOp, outputTensor, expandOp.getReassociationIndices());
            }
          }
        }
      }
    }
    // Handle regular DPS operations.
    else if (auto dpsOp =
                 mlir::dyn_cast<DestinationStyleOpInterface>(producer)) {
      // Process all init operands.
      for (OpOperand &initOperand : dpsOp.getDpsInitsMutable()) {
        if (auto emptyOp = findEmptyOp(initOperand.get())) {
          transformed = replaceEmptyWithOutput(emptyOp, outputTensor);
        }
      }
    }
    // Insert linalg.copy if the DPS semantic can't be re-enabled directly.
    if (!transformed) {
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<linalg::CopyOp>(returnOp.getLoc(), returnVal,
                                      outputTensor);
    }

    // Remove the mapping attribute since we've applied it.
    funcOp->removeAttr(ttir::ReturnToOutputMappingAttr::name);

    return WalkResult::advance();
  });

  return success(!result.wasInterrupted());
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
  void runOnOperation() final {
    if (failed(reenableDpsFromAttr(getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace
} // namespace mlir::tt::transforms
