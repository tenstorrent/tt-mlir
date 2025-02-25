// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/CumSumOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceOpsRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RepeatOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
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
      workaroundResults.tensorDataTypeResult.previousValue);

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
      inputWorkaroundResults.tensorDataTypeResult.targetValue);

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

  // Create tensor memory layout attribute.
  TensorMemoryLayoutAttr outputMemLayoutAttr =
      outputWorkaroundResults.tensorMemoryLayoutResult.targetValue
          ? ttnn::TensorMemoryLayoutAttr::get(
                rewriter.getContext(),
                *outputWorkaroundResults.tensorMemoryLayoutResult.targetValue)
          : nullptr;

  // Create the new output layout attribute with the updated tensor layout,
  // buffer type, memory layout and data type.
  TTNNLayoutAttr newOutputLayoutAttr =
      opResultLayoutAttr
          .withElementType(rewriter.getContext(), elementType,
                           opResultType.getShape())
          .withBufferType(
              rewriter.getContext(),
              outputWorkaroundResults.tensorBufferTypeResult.targetValue)
          .withMemoryLayout(rewriter.getContext(), outputMemLayoutAttr);

  // Create the new output result type with the updated data type and layout.
  RankedTensorType newOutputResultType =
      ttnn::utils::createRankedTensorTypeWithEncoding(
          ttnn::utils::createRankedTensorTypeWithElementType(
              opResultType,
              mlir::tt::dataTypeToElementType(
                  rewriter.getContext(),
                  outputWorkaroundResults.tensorDataTypeResult.targetValue)),
          newOutputLayoutAttr);

  // Update the type of result with applied workarounds.
  rewriter.modifyOpInPlace(op, [&]() {
    opResult.setType(newOutputResultType);

    // Some ops defines attributes with tensor layout, buffer type and memory
    // layout, hence we need to update the attributes as well. For example,
    // the empty op defines layout and memory_config attributes.
    if (outputWorkaroundResults.tensorLayoutResult.isModified() &&
        op->getAttrDictionary().get("layout")) {
      LayoutAttr updatedLayoutAttr = rewriter.getAttr<LayoutAttr>(
          outputWorkaroundResults.tensorLayoutResult.targetValue);
      op->setAttr("layout", updatedLayoutAttr);
    }

    if (outputWorkaroundResults.tensorDataTypeResult.isModified() &&
        op->getAttrDictionary().get("dtype")) {
      DataTypeAttr updatedDataTypeAttr = rewriter.getAttr<DataTypeAttr>(
          outputWorkaroundResults.tensorDataTypeResult.targetValue);
      op->setAttr("dtype", updatedDataTypeAttr);
    }

    if ((outputWorkaroundResults.tensorBufferTypeResult.isModified() ||
         outputWorkaroundResults.tensorMemoryLayoutResult.isModified()) &&
        op->getAttrDictionary().get("memory_config")) {

      MemoryConfigAttr currentMemoryConfig =
          mlir::cast<MemoryConfigAttr>(op->getAttr("memory_config"));

      // Create the output memory config attribute.
      // Check if the buffer type got updated.
      if (outputWorkaroundResults.tensorBufferTypeResult.isModified()) {
        currentMemoryConfig = currentMemoryConfig.withBufferType(
            rewriter.getContext(),
            outputWorkaroundResults.tensorBufferTypeResult.targetValue);
      }

      // Check if the memory layout got updated.
      if (outputWorkaroundResults.tensorMemoryLayoutResult.isModified()) {
        currentMemoryConfig = currentMemoryConfig.withMemoryLayout(
            rewriter.getContext(),
            outputWorkaroundResults.tensorMemoryLayoutResult.targetValue
                .value());
      }

      // Update the changed memory config attribute.
      op->setAttr("memory_config", currentMemoryConfig);
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
  TTNNOperandsWorkaroundsRewriter(MLIRContext *ctx)
      : OpInterfaceRewritePattern<wa::TTNNWorkaroundInterface>(ctx) {}

  LogicalResult matchAndRewrite(wa::TTNNWorkaroundInterface op,
                                PatternRewriter &rewriter) const final {

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
};

//
// Two workarounds are implemented here to avoid issues in ttnn
//
// 1. all_reduce ops are broken down into reduce_scatter and all_gather ops
// because current support of all_reduce in TTNN is not stable.
// 2. reduce_scatter op in TTNN currently does not support two dimensional
// tensor correctly. As a temporary workaround, we insert reshape ops front
// and back to make the tensor as four dimensional tensor.
class TTNNAllReduceWorkarounds : public OpRewritePattern<ttnn::AllReduceOp> {
public:
  using OpRewritePattern<ttnn::AllReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::AllReduceOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(op.getInput().getType());
    llvm::SmallVector<int64_t> inputTypeShape(inputType.getShape());
    size_t scatter_dim = op.getScatterDim();
    int32_t scatter_num = op.getScatterNum();
    Value deviceValue = op.getDevice();
    Location loc = op.getLoc();
    uint32_t clusterAxis =
        1; // TODO(tapspatel) Hard-code to 1 to prevent any changing behaviour
           // while all_reduce code is updated with new algorithm.

    // TODO(wooseoklee): Once it supports two dimensional tensor
    // (https://github.com/tenstorrent/tt-metal/issues/15010), we can remove
    // this workaround solution.
    if (inputTypeShape.size() < 4) {
      std::vector<int64_t> reshapedInputShape(4, 1);
      for (size_t i = 0; i < inputTypeShape.size(); ++i) {
        reshapedInputShape[i + inputTypeShape.size()] = inputTypeShape[i];
      }

      ArrayAttr reshapedInputShapeAttr =
          rewriter.getI32ArrayAttr(std::vector<int32_t>(
              reshapedInputShape.begin(), reshapedInputShape.end()));

      auto reshapedInputType =
          RankedTensorType::Builder(inputType).setShape(reshapedInputShape);

      ttnn::ReshapeOp preReshapeOp = rewriter.create<ttnn::ReshapeOp>(
          loc, Type(reshapedInputType), op.getInput(), reshapedInputShapeAttr);

      scatter_dim = scatter_dim + (4 - inputTypeShape.size());

      reshapedInputShape[scatter_dim] =
          static_cast<int32_t>(reshapedInputShape[scatter_dim] / scatter_num);

      auto scatteredInputType =
          RankedTensorType::Builder(inputType).setShape(reshapedInputShape);

      ttnn::ReduceScatterOp reduceScatterOp =
          rewriter.create<ttnn::ReduceScatterOp>(
              loc, Type(scatteredInputType), preReshapeOp.getResult(),
              deviceValue, scatter_dim, op.getMathOp());

      RankedTensorType outputType = mlir::cast<RankedTensorType>(op.getType());
      SmallVector<int64_t> outputTypeShape(outputType.getShape());

      std::vector<int64_t> reshapedOutputShape(4, 1);
      for (size_t i = 0; i < outputTypeShape.size(); ++i) {
        reshapedOutputShape[i + outputTypeShape.size()] = outputTypeShape[i];
      }

      auto reshapedOutputType =
          RankedTensorType::Builder(outputType).setShape(reshapedOutputShape);

      ttnn::AllGatherOp allGatherOp = rewriter.create<ttnn::AllGatherOp>(
          loc, Type(reshapedOutputType), reduceScatterOp.getResult(),
          deviceValue, scatter_dim, clusterAxis);

      ArrayAttr reshapedOutputShapeAttr = rewriter.getI32ArrayAttr(
          std::vector<int32_t>(outputTypeShape.begin(), outputTypeShape.end()));

      rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(op, Type(outputType),
                                                   allGatherOp.getResult(),
                                                   reshapedOutputShapeAttr);
    } else {
      // TODO(wooseoklee): Once ttnn supports all_reduce op
      // (https://github.com/tenstorrent/tt-metal/issues/13835), we can convert
      // directly to ttnn.all_reduce.
      inputTypeShape[scatter_dim] = inputTypeShape[scatter_dim] / scatter_num;
      auto scatteredInputType =
          RankedTensorType::Builder(inputType).setShape(inputTypeShape);

      ttnn::ReduceScatterOp reduceScatterOp =
          rewriter.create<ttnn::ReduceScatterOp>(loc, Type(scatteredInputType),
                                                 op.getInput(), deviceValue,
                                                 scatter_dim, op.getMathOp());

      rewriter.replaceOpWithNewOp<ttnn::AllGatherOp>(
          op, op.getType(), reduceScatterOp.getResult(), deviceValue,
          scatter_dim, clusterAxis);
    }
    return success();
  }
};

// Pass to apply workarounds to the operands of TTNN operations.
class TTNNWorkarounds : public impl::TTNNWorkaroundsBase<TTNNWorkarounds> {
public:
  using impl::TTNNWorkaroundsBase<TTNNWorkarounds>::TTNNWorkaroundsBase;

  void runOnOperation() final {
    if (decompositionWorkaroundsEnabled) {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNAllReduceWorkarounds,
                   workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
                       ttnn::SumOp, /*keepDimUnsupported*/ false>,
                   workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
                       ttnn::MaxOp, /*keepDimUnsupported*/ false>,
                   workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
                       ttnn::MeanOp, /*keepDimUnsupported*/ false>,
                   workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
                       ttnn::MinOp, /*keepDimUnsupported*/ false>,
                   workarounds::decomposition::CumSumOpRewritePattern>(
          &getContext());

      runRewritePatterns(std::move(patterns),
                         GreedyRewriteConfig::kNoLimit /*maxIterations*/);
    }
    if (repeatFoldingWorkaroundEnabled) {
      RewritePatternSet patterns(&getContext());
      patterns.add<workarounds::decomposition::TTNNRepeatFoldingWorkaround>(
          &getContext());
      runRewritePatterns(std::move(patterns), GreedyRewriteConfig::kNoLimit);
    }
    if (layoutWorkaroundsEnabled) {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNOperandsWorkaroundsRewriter>(&getContext());

      // All layout workarounds should be applied during the first iteration. If
      // the workarounds are not applied in the first iteration, it indicates a
      // bug in the workarounds implementation. Although the workarounds are
      // applied in the first iteration, the rewriter must iterate through the
      // IR once more to confirm that the fixpoint is reached. If the fixpoint
      // is not reached in the second iteration, it indicates a bug in the
      // workarounds implementation.
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
    config.maxIterations = maxIterations;
    // This configuration specifies that the rewriter should traverse the IR
    // in a top-down order.
    config.useTopDownTraversal = true;
    if (failed(applyPatternsGreedily(getOperation(), patternSet, config))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace mlir::tt::ttnn
