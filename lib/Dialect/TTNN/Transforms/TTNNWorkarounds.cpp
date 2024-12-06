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

// Pass to apply workarounds to the operands of TTNN operations.
class TTNNWorkarounds : public impl::TTNNWorkaroundsBase<TTNNWorkarounds> {
public:
  using impl::TTNNWorkaroundsBase<TTNNWorkarounds>::TTNNWorkaroundsBase;

  void runOnOperation() final {
    {
      // Placeholder for workaround decomposition patterns.
    }
    {
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
