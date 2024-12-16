// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

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
#include "ttmlir/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <tuple>
#include <utility>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNWORKAROUNDS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// Helper method to get the tensor layout attribute from the op operand.
static TTNNLayoutAttr getLayoutAttrFromOpOperand(OpOperand &opOperand) {
  auto tensorType = mlir::cast<RankedTensorType>(opOperand.get().getType());
  return mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());
}

// Helper method to get the tensor layout attribute from the op result.
static TTNNLayoutAttr getLayoutAttrFromOpResult(OpResult &opResult) {
  auto tensorType = mlir::cast<RankedTensorType>(opResult.getType());
  return mlir::cast<TTNNLayoutAttr>(tensorType.getEncoding());
}

// Helper method to get the element type for the given tensor layout and data.
static Type getElementType(MLIRContext *context, Layout tensorLayout,
                           DataType dataType) {
  return tensorLayout == Layout::Tile
             ? TileType::get(context, {ttnn::TILE_HEIGHT, ttnn::TILE_WIDTH},
                             dataType)
             : ttnn::utils::createRowMajorTypeFromDtype(context, dataType);
}

// Helper method to insert a ToLayoutOp to convert the input operand to the
// desired tensor layout, buffer type and memory layout.
static mlir::Value
createToLayoutOp(wa::TTNNWorkaroundInterface &op, OpOperand &inputOperand,
                 PatternRewriter &rewriter, Layout targetTensorLayout,
                 BufferType targetTensorBufferType,
                 std::optional<TensorMemoryLayout> targetTensorMemoryLayout) {
  TTNNLayoutAttr inputLayoutAttr = getLayoutAttrFromOpOperand(inputOperand);

  // Create element type based on tensor layout.
  Type elementType = getElementType(rewriter.getContext(), targetTensorLayout,
                                    inputLayoutAttr.getDataType());

  // Create tensor memory layout attribute.
  ttnn::TensorMemoryLayoutAttr outputMemLayoutAttr =
      targetTensorMemoryLayout.has_value()
          ? ttnn::TensorMemoryLayoutAttr::get(rewriter.getContext(),
                                              targetTensorMemoryLayout.value())
          : nullptr;

  // Create the output memory config attribute.
  ttnn::MemoryConfigAttr outputMemConfigAttr = ttnn::MemoryConfigAttr::get(
      rewriter.getContext(),
      ttnn::BufferTypeAttr::get(rewriter.getContext(), targetTensorBufferType),
      ttnn::ShardSpecAttr::get(
          op.getContext(),
          ttnn::ShapeAttr::get(rewriter.getContext(),
                               inputLayoutAttr.getMemref().getShape())),
      outputMemLayoutAttr);

  // Get the input operand type.
  RankedTensorType inputOperandType =
      mlir::cast<RankedTensorType>(inputOperand.get().getType());

  // Create a ToLayoutOp to convert the input operand to the desired
  // tensor layout, buffer type and memory layout.
  return rewriter
      .create<ttnn::ToLayoutOp>(
          op.getLoc(),
          ttnn::utils::createRankedTensorTypeWithEncoding(
              inputOperandType,
              inputLayoutAttr
                  .withElementType(rewriter.getContext(), elementType)
                  .withBufferType(rewriter.getContext(), targetTensorBufferType)
                  .withMemoryLayout(rewriter.getContext(),
                                    outputMemLayoutAttr)),
          inputOperand.get(),
          LayoutAttr::get(rewriter.getContext(), targetTensorLayout),
          DataTypeAttr::get(rewriter.getContext(),
                            inputLayoutAttr.getDataType()),
          outputMemConfigAttr,
          (targetTensorBufferType == ttnn::BufferType::SystemMemory)
              ? nullptr
              : utils::getOrInsertDevice(rewriter, op))
      ->getResult(0);
}

// Helper method to apply workarounds to an input operand. This method inserts a
// ToLayoutOp with the specified tensor layout, buffer type, and memory layout.
// It returns true if the workarounds were successfully applied.
static bool workaroundInputOperand(
    OpOperand &inputOperand, const wa::TTNNOperandWorkarounds &inputWorkaround,
    PatternRewriter &rewriter, wa::TTNNWorkaroundInterface op) {
  // Get the current input tensor layout, buffer type and memory layout from the
  // input operand.
  TTNNLayoutAttr inputLayoutAttr = getLayoutAttrFromOpOperand(inputOperand);

  // Apply the workarounds on the input operand workaround arguments
  wa::WorkaroundResult inputWorkaroundResult =
      applyWorkarounds(inputWorkaround, inputLayoutAttr);

  // If there were no modifications by workarounds, return false.
  if (!inputWorkaroundResult.modified()) {
    return false;
  }

  // Apply the workarounds on the input operand by inserting the ToLayoutOp with
  // the desired tensor layout, buffer type and memory layout.
  mlir::Value insertedToLayoutOpValue = createToLayoutOp(
      op, inputOperand, rewriter,
      inputWorkaroundResult.targetTensorLayoutResult.first,
      inputWorkaroundResult.targetTensorBufferTypeResult.first,
      inputWorkaroundResult.targetTensorMemoryLayoutResult.first);

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
static bool workaroundOutputOperand(
    OpResult &opResult, const wa::TTNNOperandWorkarounds &outputWorkaround,
    PatternRewriter &rewriter, wa::TTNNWorkaroundInterface op) {
  // Get the current output tensor layout, buffer type and memory layout from
  // the input operand.
  TTNNLayoutAttr opResultLayoutAttr = getLayoutAttrFromOpResult(opResult);

  // Apply the workarounds on the output result workaround arguments
  wa::WorkaroundResult outputWorkaroundResult =
      wa::applyWorkarounds(outputWorkaround, opResultLayoutAttr);

  // At this point, the DPS result should already be propagated, hence we only
  // need to verify that the output workaround is in sync with the current DPS
  // result.
  assert(!(outputWorkaroundResult.modified() &&
           mlir::isa<DestinationStyleOpInterface>(op.getOperation())) &&
         "Output operand workarounds not supported for DPS ops");

  // If there were no modifications by workarounds, return false.
  if (!outputWorkaroundResult.modified()) {
    return false;
  }

  // Create the data type attribute.
  Type elementType =
      getElementType(rewriter.getContext(),
                     outputWorkaroundResult.targetTensorLayoutResult.first,
                     opResultLayoutAttr.getDataType());

  // Get the input operand type.
  RankedTensorType opResultType =
      mlir::cast<RankedTensorType>(opResult.getType());

  // Create tensor memory layout attribute.
  TensorMemoryLayoutAttr outputMemLayoutAttr =
      outputWorkaroundResult.targetTensorMemoryLayoutResult.first.has_value()
          ? ttnn::TensorMemoryLayoutAttr::get(
                rewriter.getContext(),
                outputWorkaroundResult.targetTensorMemoryLayoutResult.first
                    .value())
          : nullptr;

  // Create the new output result type with the updated tensor layout, buffer
  // type and memory layout.
  RankedTensorType newOutputResultType =
      ttnn::utils::createRankedTensorTypeWithEncoding(
          opResultType,
          opResultLayoutAttr.withElementType(rewriter.getContext(), elementType)
              .withBufferType(
                  rewriter.getContext(),
                  outputWorkaroundResult.targetTensorBufferTypeResult.first)
              .withMemoryLayout(rewriter.getContext(), outputMemLayoutAttr));

  // Update the type of result with applied workarounds.
  rewriter.modifyOpInPlace(op, [&]() {
    opResult.setType(newOutputResultType);

    // Some ops defines attributes with tensor layout, buffer type and memory
    // layout, hence we need to update the attributes as well. For example,
    // the empty op defines layout and memory_config attributes.
    if (outputWorkaroundResult.targetTensorLayoutResult.second &&
        op->getAttrDictionary().get("layout")) {
      LayoutAttr updatedLayoutAttr = rewriter.getAttr<LayoutAttr>(
          outputWorkaroundResult.targetTensorLayoutResult.first);
      op->setAttr("layout", updatedLayoutAttr);
    }

    if ((outputWorkaroundResult.targetTensorBufferTypeResult.second ||
         outputWorkaroundResult.targetTensorMemoryLayoutResult.second) &&
        op->getAttrDictionary().get("memory_config")) {

      MemoryConfigAttr currentMemoryConfig =
          mlir::cast<MemoryConfigAttr>(op->getAttr("memory_config"));

      // Create the output memory config attribute.
      // Check if the buffer type got updated.
      if (outputWorkaroundResult.targetTensorBufferTypeResult.second) {
        currentMemoryConfig = currentMemoryConfig.withBufferType(
            rewriter.getContext(),
            outputWorkaroundResult.targetTensorBufferTypeResult.first);
      }

      // Check if the memory layout got updated.
      if (outputWorkaroundResult.targetTensorMemoryLayoutResult.second) {
        currentMemoryConfig = currentMemoryConfig.withMemoryLayout(
            rewriter.getContext(),
            outputWorkaroundResult.targetTensorMemoryLayoutResult.first
                .value());
      }

      // Update the changed memory config attribute.
      op->setAttr("memory_config", currentMemoryConfig);
    }
  });

  return true;
}

// Propagate the workaround changes for DPS input operands if they are applied
// in above graph transforms, either in a pattern for a current op, or in a
// pattern matched for a previous ops.
static bool propagateDpsInitChangesToDpsResults(wa::TTNNWorkaroundInterface &op,
                                                PatternRewriter &rewriter) {
  // Check if the op is a DPS op.
  if (!mlir::isa<DestinationStyleOpInterface>(op.getOperation())) {
    return false;
  }

  bool modified = false;

  auto dpsOp = mlir::cast<DestinationStyleOpInterface>(op.getOperation());
  mlir::OperandRange dpsInits = dpsOp.getDpsInits();

  // Iterate through all dps destination operands and propagate the changes if
  // any.
  for (size_t dpsInitIndex = 0; dpsInitIndex < dpsInits.size();
       dpsInitIndex++) {
    OpOperand *dpsInit = dpsOp.getDpsInitOperand(dpsInitIndex);
    OpResult tiedDpsResult = dpsOp.getTiedOpResult(dpsInit);

    // If the DPS destination is changed, update the DPS result as well.
    if (tiedDpsResult.getType() != dpsInit->get().getType()) {
      modified = true;
      rewriter.modifyOpInPlace(
          op, [&]() { tiedDpsResult.setType(dpsInit->get().getType()); });
    }
  }

  return modified;
}

// TTNNWorkaroundInterface rewriter applies workarounds to the operands of TTNN
// operations. TTNNWorkaroundInterface is an interface on TTNN_Op, so this
// pattern should match each op in the IR.
//
// The rewriter processes both input and output operands of TTNN operations:
// 1. **Input Operands**: The rewriter iterates through all input tensor
// operands and applies the necessary workarounds.
//    - Workarounds are applied by inserting ToLayoutOp with the desired tensor
//    layout, buffer type, and memory layout.
// 2. **DPS result propagation**: The rewriter propagates changes to tied DPS
// destination operands to ensure consistency with previous graph
// transformations, either in the current op match or previous op matches.
// 3. **Output Operands**: Output workarounds are applied only if the operation
// is not a DPS op.
//    - At this stage, all DPS result changes should be propagated. An assertion
//    ensures that the output result workaround matches
//      the corresponding DPS output result.
//    - Workarounds are applied by updating the output result type with the new
//    tensor layout, buffer type, and memory layout.
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

    // To layout op is a special case, we don't want to rewrite it.
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

    // Propagate the workaround changes for DPS input operands to DPS results if
    // they are applied in above graph transforms, either in a pattern for a
    // current op, or in a pattern matched for a previous ops.
    modified |= propagateDpsInitChangesToDpsResults(op, rewriter);

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
                      workaroundOutputOperand(std::get<0>(pair),
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
          deviceValue, scatter_dim);

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
          scatter_dim);
    }
    return success();
  }
};

class TTNNTypecastWorkarounds : public OpRewritePattern<ttnn::ReshapeOp> {
public:
  using OpRewritePattern<ttnn::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType =
        mlir::cast<RankedTensorType>(op.getInput().getType());

    if (inputType.getElementType().isBF16() ||
        inputType.getElementType().isF16() || inputType.isF32() ||
        inputType.isF64()) {
    }

    // The remainder of the types require a typecastOp before and after the
    // actual operation
    DataTypeAttr toDtypeAttr =
        DataTypeAttr::get(op.getContext(), DataType::BFloat16);

    rewriter.setInsertionPoint(op);
    ttnn::TypecastOp toType = rewriter.create<ttnn::TypecastOp>(
        op.getLoc(), op.getResult().getType(), op.getInput(), toDtypeAttr);

    ttnn::ReshapeOp reshape = rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
        op, op.getType(), toType.getResult(), op.getShape());

    DataTypeAttr fromDtypeAttr = DataTypeAttr::get(
        op.getContext(),
        elementTypeToDataType(op.getInput().getType().getElementType()));

    rewriter.setInsertionPointAfter(op);
    ttnn::TypecastOp fromType = rewriter.create<ttnn::TypecastOp>(
        op.getLoc(), reshape.getResult().getType(), reshape.getResult(),
        fromDtypeAttr);

    rewriter.replaceAllOpUsesWith(reshape, fromType);

    return success();
  }
};

// Pass to apply workarounds to the operands of TTNN operations.
class TTNNWorkarounds : public impl::TTNNWorkaroundsBase<TTNNWorkarounds> {
public:
  using impl::TTNNWorkaroundsBase<TTNNWorkarounds>::TTNNWorkaroundsBase;

  void runOnOperation() final {
    if (decompositionWorkaroundsEnabled) {
      // Placeholder for workaround decomposition patterns.
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNAllReduceWorkarounds>(&getContext());
      patterns.add<TTNNTypecastWorkarounds>(&getContext());

      FrozenRewritePatternSet patternSet(std::move(patterns));
      GreedyRewriteConfig config = GreedyRewriteConfig();
      config.useTopDownTraversal = true;
      config.maxIterations = GreedyRewriteConfig::kNoLimit;
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet,
                                              config))) {
        signalPassFailure();
        return;
      }
    }
    if (layouotWorkaroundsEnabled) {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNOperandsWorkaroundsRewriter>(&getContext());

      FrozenRewritePatternSet patternSet(std::move(patterns));
      GreedyRewriteConfig config = GreedyRewriteConfig();
      // This configuration specifies that the rewriter should traverse the IR
      // in a top-down order.
      config.useTopDownTraversal = true;
      // This configuration specifies the maximum number of iterations the
      // rewriter will perform on the IR. The rewriter will iterate through the
      // IR until a fixpoint is reached. All workarounds should be applied
      // during the first iteration. If the workarounds are not applied in the
      // first iteration, it indicates a bug in the workarounds implementation.
      // Although the workarounds are applied in the first iteration, the
      // rewriter must iterate through the IR once more to confirm that the
      // fixpoint is reached. If the fixpoint is not reached in the second
      // iteration, it indicates a bug in the workarounds implementation.
      config.maxIterations = 2;
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet,
                                              config))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace mlir::tt::ttnn
