// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNWorkaroundsPass.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ArgMaxOpRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/CumSumOpDimRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/CumSumOpRankRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/EmbeddingOpSqueezeWeightRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/ReduceOpsRewritePattern.h"
#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/RepeatOpRewritePattern.h"
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
      opResultLayoutAttr.withElementType(elementType, opResultType.getShape())
          .withBufferType(
              outputWorkaroundResults.tensorBufferTypeResult.targetValue)
          .withMemoryLayout(outputMemLayoutAttr);

  // Create the new output result type with the updated data type and layout.
  RankedTensorType newOutputResultType = utils::RankedTensorTypeFactory::create(
      utils::RankedTensorTypeFactory::create(
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
            outputWorkaroundResults.tensorBufferTypeResult.targetValue);
      }

      // Check if the memory layout got updated.
      if (outputWorkaroundResults.tensorMemoryLayoutResult.isModified()) {
        currentMemoryConfig = currentMemoryConfig.withMemoryLayout(
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
// 3. It doesn't really matter which tensor dimension we do the
// reduce scatter and the all gather on but they must be equal to each other
// and within the constraints of the rank of the tensor.
// 3-1. It turned out that using any dimension other than 3 generates incorrect
// output under the current ttnn implementation. Temporarily use dimension == 3.
// 4. We also need to
// make sure the tensor dimension we select is divisible by the number of
// devices along the cluster axis dimension we want to perform the all
// reduce on.
class TTNNAllReduceWorkarounds : public OpRewritePattern<ttnn::AllReduceOp> {
public:
  using OpRewritePattern<ttnn::AllReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::AllReduceOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(op.getInput().getType());
    llvm::SmallVector<int64_t> inputTypeShape(inputType.getShape());
    Location loc = op.getLoc();
    uint32_t clusterAxis = op.getClusterAxis();
    Value deviceValue = op.getDevice();
    auto deviceDesc = lookupDevice(op);
    ::llvm::ArrayRef<int64_t> meshShape = deviceDesc.getMeshShape();

    // TODO(hongseok): Restore dynamic dimension selection once the issue
    // (https://github.com/tenstorrent/tt-metal/issues/19433) is resolved.
    // Currently, dimension 3 must be used to produce correct outputs.
    int32_t dimension =
        std::min(3, static_cast<int32_t>(inputTypeShape.size() - 1));
    // If the target dimension is not evenly divisible by the number of devices
    // in the cluster, use the all-gather + local reduce breakdown approach.
    if (inputTypeShape[dimension] % meshShape[clusterAxis] != 0) {
      // Estimate memory usage of AllGather + LocalReduce breakdown and check if
      // it exceeds the allowed memory limit. This breakdown requires
      // significantly more memory than ReduceScatter + AllGather due to
      // internal padding and temporary buffers. To avoid potential memory
      // blowup, enforce a size constraint based on DRAM capacity.
      if (exceedsAllGatherReduceMemLimit(getCurrentScopeSystemDesc(op),
                                         inputType, meshShape[clusterAxis],
                                         0.05)) {
        return rewriteAsAllGatherLocalReduce(op, meshShape, rewriter);
      }
    }

    // TODO(wooseoklee): Once it supports two dimensional tensor
    // (https://github.com/tenstorrent/tt-metal/issues/15010), we can remove
    // this workaround solution.
    if (inputTypeShape.size() < 4) {
      // We need to expand the current inputShape size to a tensor with
      // rank=4. We do this by adding leading 1's to the inputShape to create
      // a new shape with rank=4.
      uint32_t requiredOnesInput = 4 - inputTypeShape.size();
      llvm::SmallVector<int64_t> reshapedInputShape(requiredOnesInput, 1);
      reshapedInputShape.append(inputTypeShape);

      ArrayAttr reshapedInputShapeAttr =
          rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(
              reshapedInputShape.begin(), reshapedInputShape.end()));
      auto reshapedInputType =
          RankedTensorType::Builder(inputType).setShape(reshapedInputShape);

      // Create a new reshape op.
      ttnn::ReshapeOp preReshapeOp = rewriter.create<ttnn::ReshapeOp>(
          loc, Type(reshapedInputType), op.getInput(), reshapedInputShapeAttr,
          /* memory_config */ nullptr);

      // Determine new dimension since entire tensor shape got shifted.
      dimension = dimension + requiredOnesInput;

      // Determine the shape of its input tensor. The new tensor
      // shape at the scatter_dim will be tensor_shape[scatter_dim] =
      // original_tensor_shape / num_devices.
      reshapedInputShape[dimension] =
          reshapedInputShape[dimension] / meshShape[clusterAxis];
      auto scatteredInputType =
          RankedTensorType::Builder(inputType).setShape(reshapedInputShape);

      // Create a new reduce scatter op.
      ttnn::ReduceScatterOp reduceScatterOp =
          rewriter.create<ttnn::ReduceScatterOp>(
              loc, Type(scatteredInputType), preReshapeOp.getResult(),
              deviceValue, op.getReduceType(), dimension, clusterAxis);

      // We need to reshape the output to tensor rank=4 as well.
      RankedTensorType outputType = mlir::cast<RankedTensorType>(op.getType());
      llvm::SmallVector<int64_t> outputTypeShape(outputType.getShape());

      uint32_t requiredOnesOutput = 4 - outputTypeShape.size();
      llvm::SmallVector<int64_t> reshapedOutputShape(requiredOnesOutput, 1);
      reshapedOutputShape.append(outputTypeShape);

      auto reshapedOutputType =
          RankedTensorType::Builder(outputType).setShape(reshapedOutputShape);
      ArrayAttr reshapedOutputShapeAttr =
          rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(
              outputTypeShape.begin(), outputTypeShape.end()));

      // Create a new all gather op.
      ttnn::AllGatherOp allGatherOp = rewriter.create<ttnn::AllGatherOp>(
          loc, Type(reshapedOutputType), reduceScatterOp.getResult(),
          deviceValue, dimension, clusterAxis);

      rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
          op, Type(outputType), allGatherOp.getResult(),
          reshapedOutputShapeAttr, /* memory_config */ nullptr);
    } else {
      // TODO(wooseoklee): Once ttnn supports all_reduce op
      // (https://github.com/tenstorrent/tt-metal/issues/13835), we can
      // convert directly to ttnn.all_reduce.

      // Determine the shape of its input tensor. The new tensor
      // shape at the scatter_dim will be tensor_shape[scatter_dim] =
      // original_tensor_shape / num_devices.
      inputTypeShape[dimension] =
          inputTypeShape[dimension] / meshShape[clusterAxis];
      auto scatteredInputType =
          RankedTensorType::Builder(inputType).setShape(inputTypeShape);

      // Create a new reducer scatter op.
      ttnn::ReduceScatterOp reduceScatterOp =
          rewriter.create<ttnn::ReduceScatterOp>(
              loc, Type(scatteredInputType), op.getInput(), deviceValue,
              op.getReduceType(), dimension, clusterAxis);

      // Replace all_reduce op with all_gather op.
      rewriter.replaceOpWithNewOp<ttnn::AllGatherOp>(
          op, op.getType(), reduceScatterOp.getResult(), deviceValue, dimension,
          clusterAxis);
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
    Value deviceValue = op.getDevice();

    // Use allGather + Reduce breakdown.
    // Increase the rank of the current input shape by 1.
    ArrayRef<int64_t> inputTypeShape = inputType.getShape();
    llvm::SmallVector<int64_t> expandedInputShape = {1};
    expandedInputShape.append(inputTypeShape.begin(), inputTypeShape.end());
    ArrayAttr reshapedInputShapeAttr =
        rewriter.getI32ArrayAttr(llvm::SmallVector<int32_t>(
            expandedInputShape.begin(), expandedInputShape.end()));
    RankedTensorType reshapedInputType =
        RankedTensorType::Builder(inputType).setShape(expandedInputShape);

    ttnn::ReshapeOp leadingReshapeOp = rewriter.create<ttnn::ReshapeOp>(
        loc, reshapedInputType, op.getInput(), reshapedInputShapeAttr,
        /* memory_config */ nullptr);

    // Create a new all gather op.
    expandedInputShape[0] = meshShape[clusterAxis];
    RankedTensorType allGatherOutputType =
        RankedTensorType::Builder(reshapedInputType)
            .setShape(expandedInputShape);
    ttnn::AllGatherOp allGatherOp = rewriter.create<ttnn::AllGatherOp>(
        loc, allGatherOutputType, leadingReshapeOp.getResult(), deviceValue, 0,
        clusterAxis);
    // Create a new reduce op.
    ArrayAttr reduceDimAttr =
        rewriter.getI32ArrayAttr(llvm::ArrayRef<int32_t>{0});
    switch (op.getReduceType()) {
    case ReduceType::Sum:
      rewriter.replaceOpWithNewOp<ttnn::SumOp>(op, op.getType(), allGatherOp,
                                               false, reduceDimAttr);
      break;
    case ReduceType::Mean:
      rewriter.replaceOpWithNewOp<ttnn::MeanOp>(op, op.getType(), allGatherOp,
                                                false, reduceDimAttr);
      break;
    case ReduceType::Max:
      rewriter.replaceOpWithNewOp<ttnn::MaxOp>(op, op.getType(), allGatherOp,
                                               false, reduceDimAttr);
      break;
    case ReduceType::Min:
      rewriter.replaceOpWithNewOp<ttnn::MinOp>(op, op.getType(), allGatherOp,
                                               false, reduceDimAttr);
      break;
    case ReduceType::Std:
      return op.emitOpError() << "std is not supported";
    case ReduceType::Var:
      return op.emitOpError() << "var is not supported";
    }
    return success();
  }
  bool exceedsAllGatherReduceMemLimit(tt::SystemDescAttr systemDesc,
                                      RankedTensorType inputType,
                                      int64_t numOfDevicesInCluster,
                                      float memoryLimitFactor = 0.05) const {
    // Estimate additional memory required when using AllGather + LocalReduce,
    // compared to the baseline ReduceScatter + AllGather breakdown.
    //
    // Let:
    //   - a = size of input tensor
    //   - N = number of devices in the cluster
    //
    // Memory usage estimation:
    //   - ReduceScatter + AllGather ≈ (1 + 1/N) * a
    //   - AllGather + LocalReduce ≈ (N + 2 * ceil_to_32_multiple(N)) * a
    //
    // The LocalReduce implementation allocates two extra padded buffers,
    // hence the 2 * ceil_to_32_multiple(N) term.
    //
    // Since we cannot determine the actual available memory at runtime,
    // we apply a conservative heuristic: if the *additional* memory required
    // exceeds a fixed fraction of total DRAM size, we reject this breakdown.
    auto chipDesc = systemDesc.getChipDescs()[0];
    size_t dramCapacity =
        chipDesc.getUsableDramChannelSize() * chipDesc.getNumDramChannels();
    size_t inputTensorSize =
        inputType.getNumElements() * inputType.getElementTypeBitWidth() / 8;

    // Estimated memory usage for AllGather + LocalReduce
    // tt-metal transpose the tensor and pad it to tile size. Refer to the
    // issue: https://github.com/tenstorrent/tt-metal/issues/20540
    int64_t paddedN = ((numOfDevicesInCluster + 31) / 32) * 32;
    size_t memAllgatherLocalReduce =
        (numOfDevicesInCluster + 2 * paddedN) * inputTensorSize;

    // Estimated memory usage for ReduceScatter + AllGather
    double memReduceScatterAllGather =
        (1.0 + 1.0 / static_cast<double>(numOfDevicesInCluster)) *
        inputTensorSize;

    // Additional memory required
    double overhead = static_cast<double>(memAllgatherLocalReduce) -
                      memReduceScatterAllGather;

    // Compare against memory limit threshold
    double threshold = static_cast<double>(dramCapacity) * memoryLimitFactor;

    return overhead <= threshold;
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
          TTNNAllReduceWorkarounds,
          workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
              ttnn::SumOp, /*keepDimUnsupported*/ false>,
          workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
              ttnn::MaxOp, /*keepDimUnsupported*/ false>,
          workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
              ttnn::MeanOp, /*keepDimUnsupported*/ false>,
          workarounds::decomposition::ReduceOpsKeepDimRewritePattern<
              ttnn::MinOp, /*keepDimUnsupported*/ false>,
          workarounds::decomposition::CumSumOpDimRewritePattern,
          workarounds::decomposition::CumSumOpRankRewritePattern,
          workarounds::decomposition::EmbeddingOpSqueezeWeightRewritePattern,
          workarounds::decomposition::ArgMaxOpRewritePattern,
          workarounds::decomposition::ReduceOpsPadInputRewritePattern<
              ttnn::MaxOp>,
          workarounds::decomposition::ReduceOpsPadInputRewritePattern<
              ttnn::MinOp>>(&getContext());

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
    config.setMaxIterations(maxIterations);
    // This configuration specifies that the rewriter should traverse the IR
    // in a top-down order.
    config.setUseTopDownTraversal(true);
    if (failed(applyPatternsGreedily(getOperation(), patternSet, config))) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace mlir::tt::ttnn
