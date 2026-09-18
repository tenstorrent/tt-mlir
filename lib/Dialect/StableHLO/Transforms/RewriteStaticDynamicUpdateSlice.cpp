// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"

#include <algorithm>
#include <optional>

namespace mlir::tt::stablehlo {
#define GEN_PASS_DEF_REWRITESTATICDYNAMICUPDATESLICEPASS
#include "ttmlir/Dialect/StableHLO/Transforms/Passes.h.inc"

namespace {

static std::optional<int64_t> getScalarIntExpression(Value value,
                                                     int depth = 0) {
  if (depth > 16) {
    return std::nullopt;
  }

  auto constantOp = value.getDefiningOp<mlir::stablehlo::ConstantOp>();
  if (constantOp) {
    auto denseAttr = dyn_cast<DenseIntElementsAttr>(constantOp.getValue());
    if (!denseAttr || denseAttr.getNumElements() != 1) {
      return std::nullopt;
    }

    return denseAttr.getSplatValue<APInt>().getSExtValue();
  }

  if (auto addOp = value.getDefiningOp<mlir::stablehlo::AddOp>()) {
    std::optional<int64_t> lhs =
        getScalarIntExpression(addOp.getLhs(), depth + 1);
    std::optional<int64_t> rhs =
        getScalarIntExpression(addOp.getRhs(), depth + 1);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return *lhs + *rhs;
  }

  if (auto subtractOp = value.getDefiningOp<mlir::stablehlo::SubtractOp>()) {
    std::optional<int64_t> lhs =
        getScalarIntExpression(subtractOp.getLhs(), depth + 1);
    std::optional<int64_t> rhs =
        getScalarIntExpression(subtractOp.getRhs(), depth + 1);
    if (!lhs || !rhs) {
      return std::nullopt;
    }
    return *lhs - *rhs;
  }

  return std::nullopt;
}

static std::optional<int64_t>
getClampedStart(int64_t start, int64_t operandSize, int64_t updateSize) {
  if (ShapedType::isDynamic(operandSize) || ShapedType::isDynamic(updateSize) ||
      updateSize > operandSize) {
    return std::nullopt;
  }

  return std::clamp(start, int64_t(0), operandSize - updateSize);
}

static Value createSlice(Location loc, Value input,
                         llvm::ArrayRef<int64_t> start,
                         llvm::ArrayRef<int64_t> limit, IRRewriter &rewriter) {
  auto inputType = cast<RankedTensorType>(input.getType());
  SmallVector<int64_t> shape;
  shape.reserve(inputType.getRank());
  for (auto [startDim, limitDim] : llvm::zip_equal(start, limit)) {
    shape.push_back(limitDim - startDim);
  }

  auto resultType = RankedTensorType::get(shape, inputType.getElementType(),
                                          inputType.getEncoding());
  SmallVector<int64_t> strides(inputType.getRank(), 1);

  return rewriter
      .create<mlir::stablehlo::SliceOp>(loc, resultType, input,
                                        rewriter.getDenseI64ArrayAttr(start),
                                        rewriter.getDenseI64ArrayAttr(limit),
                                        rewriter.getDenseI64ArrayAttr(strides))
      .getResult();
}

static LogicalResult rewriteIfStaticSlice(mlir::stablehlo::DynamicSliceOp op,
                                          IRRewriter &rewriter) {
  auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!operandType || !resultType ||
      operandType.getRank() != resultType.getRank()) {
    return failure();
  }

  llvm::ArrayRef<int64_t> sliceSizes = op.getSliceSizes();
  if (static_cast<int64_t>(sliceSizes.size()) != operandType.getRank() ||
      resultType.getShape() != sliceSizes) {
    return failure();
  }

  SmallVector<int64_t> starts;
  starts.reserve(operandType.getRank());
  for (auto [dim, startIndex] : llvm::enumerate(op.getStartIndices())) {
    std::optional<int64_t> start = getScalarIntExpression(startIndex);
    if (!start) {
      return failure();
    }

    std::optional<int64_t> clampedStart =
        getClampedStart(*start, operandType.getDimSize(dim), sliceSizes[dim]);
    if (!clampedStart) {
      return failure();
    }
    starts.push_back(*clampedStart);
  }

  if (static_cast<int64_t>(starts.size()) != operandType.getRank()) {
    return failure();
  }

  SmallVector<int64_t> limits;
  limits.reserve(operandType.getRank());
  for (auto [start, size] : llvm::zip_equal(starts, sliceSizes)) {
    limits.push_back(start + size);
  }

  SmallVector<int64_t> strides(operandType.getRank(), 1);
  rewriter.setInsertionPoint(op);
  auto slice = rewriter.create<mlir::stablehlo::SliceOp>(
      op.getLoc(), resultType, op.getOperand(),
      rewriter.getDenseI64ArrayAttr(starts),
      rewriter.getDenseI64ArrayAttr(limits),
      rewriter.getDenseI64ArrayAttr(strides));
  rewriter.replaceOp(op, slice.getResult());
  return success();
}

static LogicalResult
rewriteIfSingleAxisStaticUpdate(mlir::stablehlo::DynamicUpdateSliceOp op,
                                IRRewriter &rewriter) {
  auto operandType = dyn_cast<RankedTensorType>(op.getOperand().getType());
  auto updateType = dyn_cast<RankedTensorType>(op.getUpdate().getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!operandType || !updateType || !resultType) {
    return failure();
  }

  if (operandType.getRank() != updateType.getRank() ||
      operandType.getRank() != resultType.getRank()) {
    return failure();
  }

  if (operandType.getShape() != resultType.getShape()) {
    return failure();
  }

  SmallVector<int64_t> starts;
  starts.reserve(operandType.getRank());
  for (Value startIndex : op.getStartIndices()) {
    std::optional<int64_t> start = getScalarIntExpression(startIndex);
    if (!start) {
      return failure();
    }
    starts.push_back(*start);
  }

  if (static_cast<int64_t>(starts.size()) != operandType.getRank()) {
    return failure();
  }

  int64_t updateDim = -1;
  for (int64_t dim = 0; dim < operandType.getRank(); ++dim) {
    int64_t operandSize = operandType.getDimSize(dim);
    int64_t updateSize = updateType.getDimSize(dim);
    int64_t start = starts[dim];

    std::optional<int64_t> clampedStart =
        getClampedStart(start, operandSize, updateSize);
    if (!clampedStart) {
      return failure();
    }
    starts[dim] = *clampedStart;

    if (updateSize == operandSize) {
      continue;
    }

    if (updateDim != -1) {
      return failure();
    }
    updateDim = dim;
  }

  rewriter.setInsertionPoint(op);
  if (updateDim == -1) {
    rewriter.replaceOp(op, op.getUpdate());
    return success();
  }

  Location loc = op.getLoc();
  SmallVector<Value> pieces;
  SmallVector<int64_t> zeroStart(operandType.getRank(), 0);
  SmallVector<int64_t> operandLimit(operandType.getShape().begin(),
                                    operandType.getShape().end());

  int64_t start = starts[updateDim];
  int64_t updateSize = updateType.getDimSize(updateDim);
  if (start > 0) {
    SmallVector<int64_t> beforeLimit = operandLimit;
    beforeLimit[updateDim] = start;
    pieces.push_back(
        createSlice(loc, op.getOperand(), zeroStart, beforeLimit, rewriter));
  }

  pieces.push_back(op.getUpdate());

  int64_t afterStartValue = start + updateSize;
  if (afterStartValue < operandType.getDimSize(updateDim)) {
    SmallVector<int64_t> afterStart = zeroStart;
    afterStart[updateDim] = afterStartValue;
    pieces.push_back(
        createSlice(loc, op.getOperand(), afterStart, operandLimit, rewriter));
  }

  if (pieces.size() == 1) {
    rewriter.replaceOp(op, pieces.front());
    return success();
  }

  auto concat = rewriter.create<mlir::stablehlo::ConcatenateOp>(
      loc, resultType, pieces, updateDim);
  rewriter.replaceOp(op, concat.getResult());
  return success();
}

class RewriteStaticDynamicUpdateSlicePass
    : public impl::RewriteStaticDynamicUpdateSlicePassBase<
          RewriteStaticDynamicUpdateSlicePass> {
public:
  using impl::RewriteStaticDynamicUpdateSlicePassBase<
      RewriteStaticDynamicUpdateSlicePass>::
      RewriteStaticDynamicUpdateSlicePassBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(module.getContext());

    SmallVector<mlir::stablehlo::DynamicSliceOp> dynamicSliceWorklist;
    module.walk([&](mlir::stablehlo::DynamicSliceOp op) {
      dynamicSliceWorklist.push_back(op);
    });

    for (mlir::stablehlo::DynamicSliceOp op : dynamicSliceWorklist) {
      (void)rewriteIfStaticSlice(op, rewriter);
    }

    SmallVector<mlir::stablehlo::DynamicUpdateSliceOp> worklist;
    module.walk([&](mlir::stablehlo::DynamicUpdateSliceOp op) {
      worklist.push_back(op);
    });

    for (mlir::stablehlo::DynamicUpdateSliceOp op : worklist) {
      (void)rewriteIfSingleAxisStaticUpdate(op, rewriter);
    }
  }
};

} // namespace
} // namespace mlir::tt::stablehlo
