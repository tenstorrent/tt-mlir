// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNTraits.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/AllGatherOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/AllReduceOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ArgMaxOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatOpDecompositionRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatOpReshapeRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ConcatenateHeadsOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv2dRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/CumSumOpDimRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/CumSumOpRankRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/EmbeddingOpSqueezeWeightRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ExplicateOperandBroadcastsRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MultiplyOpDecompositionRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceScatterOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SubtractOpImplicitBroadcastRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/UpsampleOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <tuple>
#include <utility>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNWORKAROUNDS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// If the layout of the output result has changed as a result of applying a
// workaround, this method transforms the layout back to the previous state
// by inserting a ToLayoutOp after the op result output in order to maintain
// the workarounds changes locally.
//
static void revertOutputLayout(wa::TTNNWorkaroundInterface &op,
                               PatternRewriter &rewriter,
                               wa::WorkaroundResults &workaroundResults,
                               mlir::TypedValue<RankedTensorType> newOpResult) {
  // Check if the data type of the output result has changed.
  if (!workaroundResults.isModified()) {
    return;
  }

  // Insert the toLayoutOp after the op output.
  rewriter.setInsertionPointAfter(op);

  // Cast the data type back to the previous data type by inserting ToLayoutOp.
  mlir::Value castLayoutOp = utils::createToLayoutOp(
      op.getOperation(), newOpResult, rewriter,
      workaroundResults.tensorLayoutResult.previousValue,
      workaroundResults.tensorBufferTypeResult.previousValue,
      workaroundResults.tensorMemoryLayoutResult.previousValue,
      workaroundResults.tensorDataTypeResult.previousValue, "_workaround");

  // Replace the new output result with the casted output result.
  rewriter.replaceUsesWithIf(
      newOpResult, castLayoutOp, [&](OpOperand &operand) {
        return operand.getOwner() != castLayoutOp.getDefiningOp();
      });
}

// Helper method to apply workarounds to an input operand. This method inserts a
// ToLayoutOp with the specified tensor layout, buffer type, and memory layout.
// It returns true if the workarounds were successfully applied.
static bool workaroundInputOperand(
    OpOperand &inputOperand, const wa::TTNNOperandWorkarounds &inputWorkaround,
    PatternRewriter &rewriter, wa::TTNNWorkaroundInterface op) {
  // Get the current input tensor layout, buffer type and memory layout from the
  // input operand.
  auto inputValue =
      mlir::cast<mlir::TypedValue<RankedTensorType>>(inputOperand.get());
  TTNNLayoutAttr inputLayoutAttr =
      utils::getLayoutAttrFromTensor(inputValue.getType());

  // Apply the workarounds on the input operand workaround arguments
  wa::WorkaroundResults inputWorkaroundResults =
      applyWorkarounds(inputWorkaround, inputLayoutAttr);

  // If there were no modifications by workarounds, return false.
  if (!inputWorkaroundResults.isModified()) {
    return false;
  }

  // Apply the workarounds on the input operand by inserting the ToLayoutOp with
  // the desired tensor layout, buffer type and memory layout.
  mlir::Value insertedToLayoutOpValue = utils::createToLayoutOp(
      op.getOperation(), inputValue, rewriter,
      inputWorkaroundResults.tensorLayoutResult.targetValue,
      inputWorkaroundResults.tensorBufferTypeResult.targetValue,
      inputWorkaroundResults.tensorMemoryLayoutResult.targetValue,
      inputWorkaroundResults.tensorDataTypeResult.targetValue, "_workaround");

  // Insert to layout op between the current op and the input operand
  // to convert the input operand to the desired tensor layout, buffer type.
  rewriter.modifyOpInPlace(op, [&]() {
    // Update the input operand with the new toLayout op operand.
    op->setOperand(inputOperand.getOperandNumber(), insertedToLayoutOpValue);
  });

  return true;
}

// Helper method to apply workarounds to output results.
// - For DPS results, this method only verifies that the output result matches
// the
//   corresponding DPS destination operand. At this stage, DPS results should
//   already be propagated.
// - For non-DPS operations, this method applies the necessary workarounds to
// the
//   output result and returns true if the workarounds were successfully
//   applied.
static bool
workaroundOutputOperand(mlir::TypedValue<RankedTensorType> opResult,
                        const wa::TTNNOperandWorkarounds &outputWorkaround,
                        PatternRewriter &rewriter,
                        wa::TTNNWorkaroundInterface op) {
  // Get the current output tensor layout, buffer type and memory layout from
  // the input operand.
  TTNNLayoutAttr opResultLayoutAttr =
      utils::getLayoutAttrFromTensor(opResult.getType());

  // Apply the workarounds on the output result workaround arguments
  wa::WorkaroundResults outputWorkaroundResults =
      wa::applyWorkarounds(outputWorkaround, opResultLayoutAttr);

  // If there were no modifications by workarounds, return false.
  if (!outputWorkaroundResults.isModified()) {
    return false;
  }

  // Create the data type attribute.
  Type elementType = utils::getElementType(
      rewriter.getContext(),
      outputWorkaroundResults.tensorLayoutResult.targetValue,
      outputWorkaroundResults.tensorDataTypeResult.targetValue);

  // Get the input operand type.
  RankedTensorType opResultType =
      mlir::cast<RankedTensorType>(opResult.getType());

  // Create the new output layout attribute with the updated tensor layout,
  // buffer type, memory layout and data type.
  TTNNLayoutAttr newOutputLayoutAttr =
      opResultLayoutAttr.withElementType(elementType, opResultType.getShape())
          .withBufferType(
              outputWorkaroundResults.tensorBufferTypeResult.targetValue)
          .withMemoryLayout(
              outputWorkaroundResults.tensorMemoryLayoutResult.targetValue);

  // Create the new output result type with the updated data type and layout.
  RankedTensorType newOutputResultType = utils::RankedTensorTypeFactory::create(
      utils::RankedTensorTypeFactory::create(
          opResultType,
          mlir::tt::ttcore::dataTypeToElementType(
              rewriter.getContext(),
              outputWorkaroundResults.tensorDataTypeResult.targetValue)),
      newOutputLayoutAttr);

  // Update the type of result with applied workarounds.
  rewriter.modifyOpInPlace(op, [&]() {
    opResult.setType(newOutputResultType);

    // Some ops defines attributes with tensor layout, buffer type and memory
    // layout, hence we need to update the attributes as well. For example,
    // the empty op defines layout and memory_config attributes.
    TTNNLayoutOpInterface layoutOp =
        mlir::dyn_cast<TTNNLayoutOpInterface>(op.getOperation());
    if (outputWorkaroundResults.tensorLayoutResult.isModified() && layoutOp) {
      LayoutAttr updatedLayoutAttr = rewriter.getAttr<LayoutAttr>(
          outputWorkaroundResults.tensorLayoutResult.targetValue);
      layoutOp.setLayoutAttr(updatedLayoutAttr);
    }

    TTNNDtypeOpInterface dtypeOp =
        mlir::dyn_cast<TTNNDtypeOpInterface>(op.getOperation());
    if (outputWorkaroundResults.tensorDataTypeResult.isModified() && dtypeOp) {
      ttcore::DataTypeAttr updatedDataTypeAttr =
          rewriter.getAttr<ttcore::DataTypeAttr>(
              outputWorkaroundResults.tensorDataTypeResult.targetValue);
      dtypeOp.setDtypeAttr(updatedDataTypeAttr);
    }

    TTNNMemoryConfigOpInterface memoryConfigOp =
        mlir::dyn_cast<TTNNMemoryConfigOpInterface>(op.getOperation());
    if ((outputWorkaroundResults.tensorBufferTypeResult.isModified() ||
         outputWorkaroundResults.tensorMemoryLayoutResult.isModified()) &&
        memoryConfigOp) {

      MemoryConfigAttr currentMemoryConfig =
          memoryConfigOp.getMemoryConfigAttr();

      MemoryConfigAttr updatedMemoryConfig =
          MemoryConfigAttr::Builder(currentMemoryConfig)
              .setBufferType(
                  outputWorkaroundResults.tensorBufferTypeResult.targetValue)
              .setTensorMemoryLayout(
                  outputWorkaroundResults.tensorMemoryLayoutResult.targetValue);

      // Update the changed memory config attribute.
      memoryConfigOp.setMemoryConfigAttr(updatedMemoryConfig);

      TTNNDeviceOperandInterface deviceOperandOp =
          mlir::dyn_cast<TTNNDeviceOperandInterface>(op.getOperation());

      // If the target value for buffer type is SystemMemory, we need to remove
      // the device operand from the operation.
      if (outputWorkaroundResults.tensorBufferTypeResult.isModified() &&
          outputWorkaroundResults.tensorBufferTypeResult.targetValue ==
              BufferType::SystemMemory &&
          deviceOperandOp) {
        // Remove the device operand from the operation.
        deviceOperandOp.setDevice(nullptr);
      }

      // If the target value for buffer type is not SystemMemory and the
      // operation has a required device operand, we need to set the device
      // operand to a default device if it is not already set.
      if (outputWorkaroundResults.tensorBufferTypeResult.isModified() &&
          outputWorkaroundResults.tensorBufferTypeResult.targetValue !=
              BufferType::SystemMemory &&
          deviceOperandOp && !deviceOperandOp.getDevice()) {
        // Set the device operand to a default device.
        Value device = utils::getOrInsertDevice(rewriter, op);
        deviceOperandOp.setDevice(device);
      }
    }
  });

  revertOutputLayout(op, rewriter, outputWorkaroundResults, opResult);

  return true;
}

// TTNNWorkaroundInterface rewriter applies workarounds to the operands of TTNN
// operations. TTNNWorkaroundInterface is an interface on TTNN_Op, so this
// pattern should match each op in the IR. Each op has a default implementation
// of the interface that returns a default TTNNOperandsWorkarounds object
// without workarounds. For each op that is required, we can override the
// default implementation to return the specific workarounds for the op.
//
// The main goal of the rewriter is to apply workaround changes to the input and
// output operands of TTNN operations. The idea is to insert a ToLayoutOp before
// input operands and after output results to apply the necessary workarounds in
// order to keep workaround changes consistent and local to the affected op. The
// rewriter processes both input and output operands of TTNN operations:
// 1. **Input Operands**: The rewriter iterates through all input tensor
// operands and applies the necessary workarounds.
//    - If the input workarounds makes any changes to the input operand layout,
//    we are inserting a ToLayoutOp before the op to transform the layout to the
//    desired tensor layout, buffer type, and memory layout.
// 2. **Output Operands**: The rewriter iterates through all output tensor
// results and applies the necessary workarounds.
//    - Workarounds are applied by updating the output result type with the new
//    tensor layout, buffer type, and memory layout.
//    - If the output workaround makes any changes to the output layout, we
//    are inserting a ToLayoutOp after the op to transform the layout back to
//    the previous state in order to maintain the workarounds changes locally.
//    - For operations that define attributes with tensor layout, buffer type,
//    and memory layout, these attributes are also updated.
//      For example, the empty op defines layout and memory_config attributes.
class TTNNOperandsWorkaroundsRewriter
    : public OpInterfaceRewritePattern<wa::TTNNWorkaroundInterface> {
public:
  TTNNOperandsWorkaroundsRewriter(MLIRContext *ctx,
                                  const std::set<mlir::StringRef> *enabledOps)
      : OpInterfaceRewritePattern<wa::TTNNWorkaroundInterface>(ctx),
        enabledOps(enabledOps) {}

  LogicalResult matchAndRewrite(wa::TTNNWorkaroundInterface op,
                                PatternRewriter &rewriter) const final {
    if (!enabledOps->count(op.getOperation()->getName().getStringRef())) {
      return failure();
    }

    // To layout op is a special case, we don't want to rewrite it. We use it
    // to apply workarounds to the operands and results of TTNN operations.
    if (mlir::isa<ttnn::ToLayoutOp>(op.getOperation())) {
      return failure();
    }

    bool modified = false;
    // Get the operands workarounds for the current operation.
    wa::TTNNOperandsWorkarounds operandsWorkarounds =
        op.getOperandsWorkarounds();

    // Filter out all the input tensor operands.
    auto inputTensorsOperands =
        llvm::make_filter_range(op->getOpOperands(), [](OpOperand &v) {
          return ttmlir::utils::isRankedTensor(v.get());
        });

    // Apply workarounds to all input tensor operands.
    llvm::for_each(
        llvm::zip_equal(inputTensorsOperands,
                        operandsWorkarounds.getInputOperandWorkarounds()),
        [&](std::tuple<mlir::OpOperand &, const wa::TTNNOperandWorkarounds &>
                pair) {
          modified = std::get<1>(pair).hasAnyWorkaround() &&
                     workaroundInputOperand(std::get<0>(pair),
                                            std::get<1>(pair), rewriter, op);
        });

    // Filter out all the output tensor results.
    auto outputTensorResults =
        llvm::make_filter_range(op->getOpResults(), [](OpResult v) {
          return ttmlir::utils::isRankedTensor(v);
        });

    // Apply workarounds to all output tensor results.
    llvm::for_each(
        llvm::zip_equal(outputTensorResults,
                        operandsWorkarounds.getOutputOperandWorkarounds()),
        [&](std::tuple<mlir::OpResult, const wa::TTNNOperandWorkarounds &>
                pair) {
          modified |= std::get<1>(pair).hasAnyWorkaround() &&
                      workaroundOutputOperand(
                          mlir::cast<mlir::TypedValue<RankedTensorType>>(
                              std::get<0>(pair)),
                          std::get<1>(pair), rewriter, op);
        });

    // Return success if the transformations were applied.
    return modified ? success() : failure();
  }

private:
  // Set of ops that are enabled for workarounds.
  const std::set<mlir::StringRef> *enabledOps;
};

// Pass to apply workarounds to the operands of TTNN operations.
class TTNNWorkarounds : public impl::TTNNWorkaroundsBase<TTNNWorkarounds> {
public:
  using impl::TTNNWorkaroundsBase<TTNNWorkarounds>::TTNNWorkaroundsBase;

  void runOnOperation() final {
    if (decompositionWorkaroundsEnabled) {
      RewritePatternSet patterns(&getContext());
      patterns.add<
          workarounds::decomposition::ConcatOpDecompositionRewritePattern,
          workarounds::decomposition::ConcatOpReshapeRewritePattern,
          workarounds::decomposition::TTNNReduceScatterWorkarounds,
          workarounds::decomposition::TTNNAllGatherWorkarounds,
          workarounds::decomposition::TTNNAllReduceWorkarounds,
          workarounds::decomposition::CumSumOpDimRewritePattern,
          workarounds::decomposition::CumSumOpRankRewritePattern,
          workarounds::decomposition::EmbeddingOpSqueezeWeightRewritePattern,
          workarounds::decomposition::ArgMaxOpRewritePattern,
          workarounds::decomposition::UpsampleOpBilinearPaddingRewritePattern,
          workarounds::decomposition::MultiplyOpDecompositionRewritePattern,
          workarounds::decomposition::SubtractOpImplicitBroadcastRewritePattern,
          workarounds::decomposition::ExplicateOperandBroadcastsRewritePattern,
          workarounds::decomposition::Conv2dRewritePattern<Conv2dOp>,
          workarounds::decomposition::Conv2dRewritePattern<ConvTranspose2dOp>,
          workarounds::decomposition::ConcatenateHeadsOpRewritePattern>(
          &getContext());

      runRewritePatterns(std::move(patterns),
                         GreedyRewriteConfig::kNoLimit /*maxIterations*/);
    }
    if (layoutWorkaroundsEnabled) {
      RewritePatternSet patterns(&getContext());

      std::set<mlir::StringRef> enabledOps;
      if (optimizerEnabled) {
        enabledOps = enabledOpsForWorkaroundWithOptimizer;
      } else {
        enabledOps = utils::getAllTTNNDialectOps(&getContext());
      }

      patterns.add<TTNNOperandsWorkaroundsRewriter>(&getContext(), &enabledOps);

      // All layout workarounds should be applied during the first iteration.
      // If the workarounds are not applied in the first iteration, it
      // indicates a bug in the workarounds implementation. Although the
      // workarounds are applied in the first iteration, the rewriter must
      // iterate through the IR once more to confirm that the fixpoint is
      // reached. If the fixpoint is not reached in the second iteration, it
      // indicates a bug in the workarounds implementation.
      const int64_t maxIterations = 2;
      runRewritePatterns(std::move(patterns), maxIterations);
    }
  }

private:
  // Runs rewrite patterns with specified maximum number of iterations the
  // rewriter will perform on the IR. The rewriter will iterate through the IR
  // until a fixpoint is reached.
  void runRewritePatterns(RewritePatternSet &&patterns, int64_t maxIterations) {
    FrozenRewritePatternSet patternSet(std::move(patterns));
    GreedyRewriteConfig config = GreedyRewriteConfig();
    config.setMaxIterations(maxIterations);
    // This configuration specifies that the rewriter should traverse the IR
    // in a top-down order.
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), patternSet, config))) {
      signalPassFailure();
      return;
    }
  }

  static const std::set<mlir::StringRef> enabledOpsForWorkaroundWithOptimizer;
};

const std::set<mlir::StringRef>
    TTNNWorkarounds::TTNNWorkarounds::enabledOpsForWorkaroundWithOptimizer = {
        ttnn::WhereOp::getOperationName(), ttnn::FullOp::getOperationName()};

} // namespace mlir::tt::ttnn
