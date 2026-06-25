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
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/AllReduceReshapeOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/AllToAllDispatchMetadataDrainCoreRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv2dEnableKernelStrideFoldingRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/Conv2dRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/DistributedRMSNormWidthShardInputRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/EmbeddingOpSqueezeWeightRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/FillCacheInputPadRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/GatherOpRank1RewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/GroupNormAffineReshapeRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/GroupNormChannelPadRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/IntegerProdOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/LinearOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MoeComputeRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MoeGptLayoutRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/NLPConcatHeadsDecodeInputRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PadHighDimRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PagedUpdateCacheOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PointToPointOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RMSNormConfigRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceScatterConfigRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceScatterOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RotaryEmbeddingOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SamplingOpRank2RewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionDecodeAttentionSinkRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionDecodeBroadcastMaskRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ScaledDotProductAttentionPadTileDimsRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SliceStaticOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/SplitQueryKeyValueAndSplitHeadsOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/TopKRouterGptDecompositionRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/UpsampleOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
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

#include <limits>
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

  // When the operand is a const-eval'able constant whose L1 residency is
  // intended, tag the inserted op so ConstEvalHoist permits hoisting it
  // despite the L1-resident result.
  if (inputWorkaround.allowL1ConstEval) {
    insertedToLayoutOpValue.getDefiningOp()->setAttr(
        utils::g_ConstEvalAllowedAttrName, rewriter.getUnitAttr());
  }

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

  ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(op);

  // Create the new output layout attribute with the updated tensor layout,
  // buffer type, memory layout and data type.
  TTNNLayoutAttr newOutputLayoutAttr =
      TTNNLayoutAttr::Builder(opResultLayoutAttr, opResultType.getShape())
          .setElementType(elementType)
          .setBufferType(
              outputWorkaroundResults.tensorBufferTypeResult.targetValue)
          .setMemoryLayout(
              outputWorkaroundResults.tensorMemoryLayoutResult.targetValue)
          .buildWithCanonicalCorePlacement(deviceAttr);

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

    // The buffer type / memory layout / page layout changes are encoded in the
    // result tensor's TTNNLayoutAttr (set above).
    TTNNDeviceOperandInterface deviceOperandOp =
        mlir::dyn_cast<TTNNDeviceOperandInterface>(op.getOperation());
    if (outputWorkaroundResults.tensorBufferTypeResult.isModified() &&
        deviceOperandOp) {
      BufferType targetBufferType =
          outputWorkaroundResults.tensorBufferTypeResult.targetValue;
      if (targetBufferType == BufferType::SystemMemory) {
        // Moving to host: drop the device operand.
        deviceOperandOp.setDevice(nullptr);
      } else if (!deviceOperandOp.getDevice()) {
        // Moving to a device buffer type and no device operand is set yet:
        // attach a default one.
        deviceOperandOp.setDevice(utils::getOrInsertDevice(rewriter, op));
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

// Two workarounds are implemented here to avoid issues in ttnn
//
// 1. all_reduce ops are broken down into reduce_scatter and all_gather ops
// because current support of all_reduce in TTNN is not stable.
// 2. We prefer using the last tensor dimensions for reduce_scatter.
// In transformers, trailing dimensions are typically larger, which gives better
// utilization.
// 3. The selected tensor dimension must be divisible by the number of devices
// along the cluster axis used for all_reduce. For tiled layout, this
// divisibility is checked in per-dimension tile counts.
class TTNNAllReduceWorkarounds : public OpRewritePattern<ttnn::AllReduceOp> {
public:
  using OpRewritePattern<ttnn::AllReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::AllReduceOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(op.getInput().getType());
    Location loc = op.getLoc();
    uint32_t clusterAxis = op.getClusterAxis();
    auto deviceDesc = ttcore::lookupDevice(op);
    ::llvm::ArrayRef<int64_t> meshShape = deviceDesc.getMeshShape();

    // Algorithm: iterate through all tensor dimension values and select the
    // last tensor dimension which is divisible by number of devices along the
    // cluster axis on which we are performing the all reduce.
    // For tiled layout, this divisibility check is done on per-dim tile counts.
    auto sizeOfDevices = meshShape[clusterAxis];
    auto inputShape = inputType.getShape();
    auto inputLayout = utils::getLayoutAttrFromTensor(inputType);
    llvm::SmallVector<int64_t> shapeInTileCounts(inputShape.begin(),
                                                 inputShape.end());
    llvm::SmallVector<int64_t> tilePaddedShape;
    if (inputLayout.isTiled()) {
      tilePaddedShape = utils::getTilePaddedShape(shapeInTileCounts);
      if (!shapeInTileCounts.empty()) {
        shapeInTileCounts[shapeInTileCounts.size() - 1] =
            tilePaddedShape[shapeInTileCounts.size() - 1] / TILE_WIDTH;
      }
      if (shapeInTileCounts.size() > 1) {
        shapeInTileCounts[shapeInTileCounts.size() - 2] =
            tilePaddedShape[shapeInTileCounts.size() - 2] / TILE_HEIGHT;
      }
    }

    int64_t selectedDim = -1;
    for (int64_t dim = shapeInTileCounts.size() - 1; dim >= 0; --dim) {
      if (shapeInTileCounts[dim] % sizeOfDevices == 0) {
        selectedDim = dim;
        break;
      }
    }

    if (selectedDim < 0) {
      // If all the dimensions are not evenly divisible by the number of
      // devices in the cluster, use the all-gather + local reduce breakdown
      // approach.
      return rewriteAsAllGatherLocalReduce(op, meshShape, rewriter);
    }

    Value reduceScatterInput = op.getInput();
    RankedTensorType reduceScatterInputType = inputType;

    // If the input is tiled and selectedDim is one of the tile-sensitive dims,
    // pad first so the reduce_scatter split produces equal-sized slices on each
    // device.
    if (inputLayout.isTiled() &&
        selectedDim >= static_cast<int64_t>(inputShape.size()) - 2 &&
        inputShape[selectedDim] != tilePaddedShape[selectedDim]) {
      llvm::SmallVector<int32_t> padding(inputShape.size() * 2, 0);
      padding[selectedDim * 2 + 1] =
          tilePaddedShape[selectedDim] - inputShape[selectedDim];

      llvm::SmallVector<int64_t> paddedShape(inputShape.begin(),
                                             inputShape.end());
      paddedShape[selectedDim] = tilePaddedShape[selectedDim];
      auto paddedType =
          ttnn::utils::RankedTensorTypeFactory::create(inputType, paddedShape);

      reduceScatterInput = rewriter.create<ttnn::PadOp>(
          ttmlir::utils::appendLocationSuffix(loc, "_pad_for_reduce_scatter"),
          paddedType, op.getInput(), padding, /*pad_value=*/mlir::APFloat(0.0f),
          /*use_multicore=*/false);
      reduceScatterInputType = paddedType;
    }

    // TODO(wooseoklee): Once ttnn supports all_reduce op
    // (https://github.com/tenstorrent/tt-metal/issues/13835), we can
    // convert directly to ttnn.all_reduce.

    // Build reduce_scatter output type.
    llvm::SmallVector<int64_t> reduceScatterShape(
        reduceScatterInputType.getShape().begin(),
        reduceScatterInputType.getShape().end());
    reduceScatterShape[selectedDim] =
        reduceScatterShape[selectedDim] / meshShape[clusterAxis];
    auto reduceScatterOutputType = ttnn::utils::RankedTensorTypeFactory::create(
        reduceScatterInputType, reduceScatterShape);

    // Create a new reducer scatter op.
    ttnn::ReduceScatterOp reduceScatterOp =
        rewriter.create<ttnn::ReduceScatterOp>(
            ttmlir::utils::appendLocationSuffix(loc, "_reduce_scatter"),
            reduceScatterOutputType, reduceScatterInput, op.getReduceType(),
            selectedDim, clusterAxis, nullptr, nullptr, nullptr, nullptr);

    // all_gather restores the reduce_scatter input shape.
    auto allGatherOutputType = ttnn::utils::RankedTensorTypeFactory::create(
        reduceScatterInputType, reduceScatterInputType.getShape());
    ttnn::AllGatherOp allGatherOp = rewriter.create<ttnn::AllGatherOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_all_gather"),
        allGatherOutputType, reduceScatterOp.getResult(), selectedDim,
        clusterAxis, nullptr /*sub_device_id*/, nullptr /*num_links*/,
        nullptr /*topology*/);

    // If padding was added, crop back to the original shape.
    if (reduceScatterInputType.getShape() != inputType.getShape()) {
      llvm::SmallVector<int32_t> begins(inputShape.size(), 0);
      llvm::SmallVector<int32_t> ends(inputShape.begin(), inputShape.end());
      llvm::SmallVector<int32_t> steps(inputShape.size(), 1);
      auto sliceOp = rewriter.create<ttnn::SliceStaticOp>(
          ttmlir::utils::appendLocationSuffix(loc,
                                              "_slice_for_reduce_scatter_pad"),
          op.getType(), allGatherOp.getResult(),
          rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
          rewriter.getI32ArrayAttr(steps));
      rewriter.replaceOp(op, sliceOp.getResult());
    } else {
      rewriter.replaceOp(op, allGatherOp.getResult());
    }
    return success();
  }

private:
  LogicalResult
  rewriteAsAllGatherLocalReduce(ttnn::AllReduceOp op,
                                ::llvm::ArrayRef<int64_t> meshShape,
                                PatternRewriter &rewriter) const {
    RankedTensorType inputType = op.getInput().getType();
    Location loc = op.getLoc();
    uint32_t clusterAxis = op.getClusterAxis();

    // Use allGather + Reduce breakdown.
    // Increase the rank of the current input shape by 1.
    ArrayRef<int64_t> inputTypeShape = inputType.getShape();
    llvm::SmallVector<int64_t> expandedInputShape = {1};
    expandedInputShape.append(inputTypeShape.begin(), inputTypeShape.end());
    ArrayAttr reshapedInputShapeAttr =
        rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(
            expandedInputShape.begin(), expandedInputShape.end()));
    RankedTensorType reshapedInputType =
        ttnn::utils::RankedTensorTypeFactory::create(inputType,
                                                     expandedInputShape);

    ttnn::ReshapeOp leadingReshapeOp = rewriter.create<ttnn::ReshapeOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_reshape"), reshapedInputType,
        op.getInput(), reshapedInputShapeAttr);

    // Create a new all gather op.
    expandedInputShape[0] = meshShape[clusterAxis];
    RankedTensorType allGatherOutputType =
        ttnn::utils::RankedTensorTypeFactory::create(reshapedInputType,
                                                     expandedInputShape);
    ttnn::AllGatherOp allGatherOp = rewriter.create<ttnn::AllGatherOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_allGather"),
        allGatherOutputType, leadingReshapeOp.getResult(), 0, clusterAxis,
        nullptr /*sub_device_id*/, nullptr /*num_links*/, nullptr /*topology*/);
    // Create a new reduce op.
    ArrayAttr reduceDimAttr =
        rewriter.getI32ArrayAttr(llvm::ArrayRef<int32_t>{0});
    switch (op.getReduceType()) {
    case ttcore::ReduceType::Sum:
      rewriter.replaceOpWithNewOp<ttnn::SumOp>(op, op.getType(), allGatherOp,
                                               false, reduceDimAttr);
      break;
    case ttcore::ReduceType::Mean:
      rewriter.replaceOpWithNewOp<ttnn::MeanOp>(op, op.getType(), allGatherOp,
                                                false, reduceDimAttr);
      break;
    case ttcore::ReduceType::Max:
      rewriter.replaceOpWithNewOp<ttnn::MaxOp>(op, op.getType(), allGatherOp,
                                               false, reduceDimAttr);
      break;
    case ttcore::ReduceType::Min:
      rewriter.replaceOpWithNewOp<ttnn::MinOp>(op, op.getType(), allGatherOp,
                                               false, reduceDimAttr);
      break;
    case ttcore::ReduceType::Std:
      return op.emitOpError() << "std is not supported";
    case ttcore::ReduceType::Var:
      return op.emitOpError() << "var is not supported";
    case ttcore::ReduceType::Prod:
      return op.emitOpError() << "prod is not supported";
    case ttcore::ReduceType::Invalid:
      return op.emitOpError() << "invalid is not supported";
    }
    return success();
  }
};

// This pattern wraps an si32-indexed gather in a fill-style mask, modeled
// on what JAX emits for `jax.lax.gather(..., mode='fill')`:
//
//   mask     = idx < 0
//   safe     = max(idx, 0)
//   safe_u32 = to_layout(safe, dtype = ui32)
//   raw      = ttnn.gather(input, safe_u32, dim)
//   result   = where(mask, NaN, raw)
//
// Lanes whose original index was negative end up as NaN in the output,
// making the failure visible
// https://github.com/tenstorrent/tt-metal/issues/43869
class GatherSi32Workaround : public OpRewritePattern<ttnn::GatherOp> {
public:
  using OpRewritePattern<ttnn::GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::GatherOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType indexType = op.getIndex().getType();
    if (ttcore::elementTypeToDataType(indexType.getElementType()) !=
        ttcore::DataType::Int32) {
      return failure();
    }

    Location loc = op.getLoc();
    Value device = ttnn::utils::getOrInsertDevice(rewriter, op);

    RankedTensorType outputType = op.getResult().getType();

    RankedTensorType maskType = ttnn::utils::RankedTensorTypeFactory::create(
        indexType, ttcore::elementTypeToDataType(outputType.getElementType()));
    TTNNLayoutAttr indexLayout =
        ttnn::utils::getLayoutAttrFromTensor(indexType);

    // %zero = ttnn.full(0 : si32, shape = idx_shape)
    auto zero = rewriter.create<ttnn::FullOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_zero"), indexType,
        rewriter.getI32IntegerAttr(0), device);

    // %mask = ttnn.lt(idx, zero) -> numeric mask tensor
    auto mask = rewriter.create<ttnn::LessThanOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_idx_lt_zero"), maskType,
        op.getIndex(), zero.getResult());

    // %safe = ttnn.maximum(idx, zero) -> si32 (negatives clamped to 0)
    auto safeIdx = rewriter.create<ttnn::MaximumOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_clamp"), indexType,
        op.getIndex(), zero.getResult());

    // %safe_u32 = ttnn.to_layout(%safe) -> ui32
    ttnn::ToLayoutOp safeIdxU32 = ttnn::utils::createToLayoutOp(
        op.getOperation(),
        mlir::cast<mlir::TypedValue<RankedTensorType>>(safeIdx.getResult()),
        rewriter, indexLayout.getLayout(), indexLayout.getBufferType(),
        indexLayout.getMemLayout(), ttcore::DataType::UInt32, "_to_u32");

    // %raw = ttnn.gather(input, %safe_u32, dim)
    auto rawGather = rewriter.create<ttnn::GatherOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_safe_gather"), outputType,
        op.getInput(), safeIdxU32.getResult(), op.getDimAttr());

    //   - float => NaN
    //   - int   => int32_min (for unsigned this makes a large positive number,
    //   and for signed this makes a large negative number)
    mlir::Type outputElemType = outputType.getElementType();
    mlir::Attribute fillValue;
    if (mlir::isa<mlir::FloatType>(outputElemType)) {
      fillValue =
          rewriter.getF32FloatAttr(std::numeric_limits<float>::quiet_NaN());
    } else if (mlir::isa<mlir::IntegerType>(outputElemType)) {
      fillValue =
          rewriter.getI32IntegerAttr(std::numeric_limits<int32_t>::min());
    } else {
      return failure();
    }

    // %fill = ttnn.full(fill_value, shape = output_shape)
    auto fillTensor = rewriter.create<ttnn::FullOp>(
        ttmlir::utils::appendLocationSuffix(loc, "_fill"), outputType,
        fillValue, device);

    // %result = ttnn.where(mask, fill_value, raw)
    rewriter.replaceOpWithNewOp<ttnn::WhereOp>(op, outputType, mask.getResult(),
                                               fillTensor.getResult(),
                                               rawGather.getResult());

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
      patterns.add<
          GatherSi32Workaround,
          workarounds::decomposition::GatherOpRank1RewritePattern,
          workarounds::decomposition::SamplingOpRank2RewritePattern,
          workarounds::decomposition::TTNNAllReduceReshapeWorkarounds,
          workarounds::decomposition::TTNNAllGatherWorkarounds,
          workarounds::decomposition::TTNNReduceScatterWorkarounds,
          workarounds::decomposition::EmbeddingOpSqueezeWeightRewritePattern,
          workarounds::decomposition::GroupNormChannelPadRewritePattern,
          workarounds::decomposition::GroupNormAffineReshapeRewritePattern,
          workarounds::decomposition::IntegerProdOpRewritePattern,
          workarounds::decomposition::UpsampleOpBilinearPaddingRewritePattern,
          workarounds::decomposition::RotaryEmbeddingOpRewritePattern,
          workarounds::decomposition::FillCacheInputPadRewritePattern<
              ttnn::FillCacheOp>,
          workarounds::decomposition::FillCacheInputPadRewritePattern<
              ttnn::PagedFillCacheOp>,
          workarounds::decomposition::Conv2dRewritePattern<Conv2dOp>,
          workarounds::decomposition::Conv2dRewritePattern<ConvTranspose2dOp>,
          workarounds::decomposition::
              Conv2dEnableKernelStrideFoldingRewritePattern<Conv2dOp>,
          workarounds::decomposition::
              Conv2dEnableKernelStrideFoldingRewritePattern<ConvTranspose2dOp>,
          workarounds::decomposition::PadHighDimRewritePattern,
          workarounds::decomposition::NLPConcatHeadsDecodeInputRewritePattern,
          workarounds::decomposition::
              SplitQueryKeyValueAndSplitHeadsOpRewritePattern,
          // PagedUpdateCacheOpRewritePattern added below — conditionally.

          workarounds::decomposition::
              ScaledDotProductAttentionDecodeAttentionSinkRewritePattern,
          workarounds::decomposition::
              ScaledDotProductAttentionDecodeBroadcastMaskRewritePattern,
          workarounds::decomposition::
              ScaledDotProductAttentionPadTileDimsRewritePattern,
          workarounds::decomposition::PointToPointOpRewritePattern,
          workarounds::decomposition::RMSNormConfigRewritePattern,
          workarounds::decomposition::
              DistributedRMSNormWidthShardInputRewritePattern,
          workarounds::decomposition::ReduceScatterConfigRewritePattern,
          workarounds::decomposition::TopKRouterGptDecompositionRewritePattern,
          workarounds::decomposition::
              AllToAllDispatchMetadataDrainCoreRewritePattern,
          workarounds::decomposition::MoeComputeRewritePattern,
          workarounds::decomposition::SliceStaticOpRewritePattern,
          workarounds::decomposition::MoeGptLayoutRewritePattern>(
          &getContext());
      patterns.add<workarounds::decomposition::LinearOpRewritePattern>(
          &getContext(), /*benefit=*/2);

      // The all_reduce decomposition workaround can be disabled via a pass
      // option, e.g. once TTNN provides stable native all_reduce support.
      if (allReduceWorkaroundEnabled) {
        patterns.add<TTNNAllReduceWorkarounds>(&getContext());
      }

      // PagedUpdateCacheOpRewritePattern is only needed below opt-level 2.
      // At level >= 2 the greedy sharding optimizer drives the upstream
      // producer to L1 height-sharded and inserts a proper ToMemoryConfigOp
      // via beam search: metal's own grid TT_FATAL (tt-metal #45016) makes the
      // constraint query reject any other operand-1 layout.
      if (optimizationLevel < 2) {
        patterns
            .add<workarounds::decomposition::PagedUpdateCacheOpRewritePattern>(
                &getContext());
      }

      patterns.add<
          workarounds::decomposition::
              PagedScaledDotProductAttentionDecodeProgramConfigRewritePattern>(
          &getContext(), optimizationLevel);

      runRewritePatterns(std::move(patterns),
                         GreedyRewriteConfig::kNoLimit /*maxIterations*/);
    }
    if (layoutWorkaroundsEnabled) {
      RewritePatternSet patterns(&getContext());

      std::set<mlir::StringRef> enabledOps;
      if (optimizationLevel >= 1) {
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
        ttnn::WhereOp::getOperationName(), ttnn::FullOp::getOperationName(),
        ttnn::EmbeddingOp::getOperationName(),
        ttnn::ScatterOp::getOperationName(),
        // TopK's operands workaround forces input bf16 + indices ui16/ui32;
        // without it, opt_level>=1 dtype propagation picks f32. See #8141.
        ttnn::TopKOp::getOperationName(),
        // FlashMlaPrefill's operands workaround forces Q/K/V/output to a
        // tt-metal SDPA-supported dtype (bf16). Without it, opt_level>=1 leaves
        // f32 operands
        ttnn::FlashMlaPrefillOp::getOperationName(),
        // The moe_compute weight packers and the op itself require
        // layout and data type workarounds.
        ttnn::PrepareMoEComputeW0W1WeightsOp::getOperationName(),
        ttnn::PrepareMoEComputeW2WeightsOp::getOperationName(),
        ttnn::MoeComputeOp::getOperationName(),
        // Conv3d's runtime kernel hard-rejects Tile input
        // (TT_FATAL @ conv3d_device_operation.cpp:49); without the
        // workaround running here, the optimizer's layout propagation
        // picks Tile for the input and downstream OpModel queries
        // (LegalOpConfigAnalysis, OperationValidationAndFallback) see
        // an inconsistent view between in-IR layouts and runtime
        // contract.
        ttnn::Conv3dOp::getOperationName(),
        // Sampling's operands workaround forces ROW_MAJOR layout on
        // index/param tensors, UINT32 dtype on k, and ROW_MAJOR+UINT32 on
        // the result (the kernel hard-rejects anything else and produces
        // UINT32).
        ttnn::SamplingOp::getOperationName()};
} // namespace mlir::tt::ttnn
