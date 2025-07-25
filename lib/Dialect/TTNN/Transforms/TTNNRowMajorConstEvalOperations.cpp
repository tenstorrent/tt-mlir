// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNROWMAJORCONSTEVALOPERATIONS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class ExplicateBroadcastsRewriter : public RewritePattern {
public:
  ExplicateBroadcastsRewriter(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasTrait<ttnn::BroadcastableTrait>()) {
      return failure();
    }

    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp || !ttmlir::utils::isConstEvalFunc(funcOp)) {
      return failure();
    }

    auto resultShape =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
            .getShape();
    bool hasChanged = false;

    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      mlir::Value operand = op->getOperand(i);
      auto operandShape =
          mlir::cast<mlir::RankedTensorType>(operand.getType()).getShape();
      if (operandShape == resultShape) {
        continue;
      }

      RankedTensorType newOutputType = utils::RankedTensorTypeFactory::create(
          mlir::cast<RankedTensorType>(operand.getType()), resultShape);
      auto broadcastDims = ttmlir::utils::getBroadcastDimensions<int64_t>(
          operandShape, resultShape);
      auto shapeAttr =
          ttnn::ShapeAttr::get(rewriter.getContext(), broadcastDims);
      auto repeatOp = rewriter.create<ttnn::RepeatOp>(
          op->getLoc(), newOutputType, operand, shapeAttr);

      rewriter.modifyOpInPlace(op, [&]() { op->setOperand(i, repeatOp); });
      hasChanged = true;
    }

    return success(hasChanged);
  }
};

struct ElementTypeConverter : public mlir::TypeConverter {
  ElementTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([](mlir::RankedTensorType type)
                      -> std::optional<mlir::RankedTensorType> {
      assert(type.getEncoding() &&
             "RowMajorConstEvalOperations only supports RankedTensorType with "
             "encoding");

      auto encoding = mlir::cast<ttnn::TTNNLayoutAttr>(type.getEncoding());
      if (encoding.getLayout() == Layout::RowMajor) {
        return std::nullopt;
      }

      auto newEncoding = utils::convertTTNNLayoutToRowMajor(
          type.getContext(), encoding, type.getShape());
      return type.cloneWithEncoding(newEncoding);
    });

    addSourceMaterialization(materializeRowMajorTensor);
    addTargetMaterialization(materializeRowMajorTensor);
  }

  static Value materializeRowMajorTensor(mlir::OpBuilder &builder,
                                         mlir::Type resultType,
                                         mlir::ValueRange inputs,
                                         mlir::Location loc) {
    assert(inputs.size() == 1 &&
           "RowMajorConstEvalOperations only supports one input");
    RankedTensorType rankedResultType =
        mlir::cast<RankedTensorType>(resultType);
    auto encoding =
        mlir::cast<ttnn::TTNNLayoutAttr>(rankedResultType.getEncoding());
    Layout layout = encoding.getLayout();
    auto toLayoutOp =
        builder.create<ttnn::ToLayoutOp>(loc, resultType, inputs[0], layout,
                                         /*dtype=*/nullptr,
                                         /*memory_config=*/nullptr);
    utils::moveDeviceOpToTopOfBlock(toLayoutOp);
    return toLayoutOp;
  }
};

class RowMajorRewriter : public ConversionPattern {
public:
  RowMajorRewriter(const TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IRMapping mapping;
    mapping.map(op->getOperands(), operands);
    Operation *newOp = rewriter.clone(*op, mapping);

    Type newType = getTypeConverter()->convertType(op->getResult(0).getType());
    rewriter.modifyOpInPlace(newOp,
                             [&]() { newOp->getResult(0).setType(newType); });
    if (auto layoutUpdatableOp =
            dyn_cast<TTNNLayoutUpdatableInterface>(newOp)) {
      layoutUpdatableOp.updateLayoutAttribute(
          ttnn::LayoutAttr::get(newOp->getContext(), ttnn::Layout::RowMajor));
    }

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

class RowMajorConstEvalOperations
    : public impl::TTNNRowMajorConstEvalOperationsBase<
          RowMajorConstEvalOperations> {
public:
  using impl::TTNNRowMajorConstEvalOperationsBase<
      RowMajorConstEvalOperations>::TTNNRowMajorConstEvalOperationsBase;

  bool checkDynamicallyLegal(Operation *op) const {
    func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp || !ttmlir::utils::isConstEvalFunc(funcOp)) {
      return true;
    }

    if (!isa<TTNNDialect>(op->getDialect())) {
      return true;
    }

    if (op->getNumResults() == 0) {
      return true;
    }

    if (!isa<RankedTensorType>(op->getResult(0).getType())) {
      return true;
    }

    if (isa<PrepareConv2dWeightsOp>(op) || isa<PrepareConv2dBiasOp>(op)) {
      return true;
    }

    assert(op->getNumResults() == 1 &&
           "RowMajorConstEvalOperations only supports operations with a single "
           "result");

    Type resultTypeConverted =
        typeConverter.convertType(op->getResult(0).getType());
    if (resultTypeConverted == op->getResult(0).getType()) {
      return true;
    }

    return false;
  }

  void runOnOperation() final {
    // First apply broadcast explication patterns
    RewritePatternSet broadcastPatterns(&getContext());
    broadcastPatterns.add<ExplicateBroadcastsRewriter>(&getContext());
    FrozenRewritePatternSet frozenBroadcastPatterns(
        std::move(broadcastPatterns));
    if (failed(
            applyPatternsGreedily(getOperation(), frozenBroadcastPatterns))) {
      signalPassFailure();
      return;
    }

    mlir::ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return checkDynamicallyLegal(op); });

    RewritePatternSet patterns(&getContext());
    patterns.add<RowMajorRewriter>(typeConverter, &getContext());
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

private:
  ElementTypeConverter typeConverter;
};
} // namespace
} // namespace mlir::tt::ttnn
