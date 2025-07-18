// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNROWMAJORCONSTEVALOPERATIONS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

struct ElementTypeConverter : public mlir::TypeConverter {
  ElementTypeConverter() {
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
    return builder.create<ttnn::ToLayoutOp>(loc, resultType, inputs[0], layout,
                                            /*dtype=*/nullptr,
                                            /*memory_config=*/nullptr);
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

    SmallVector<Type> convertedTypes;
    Type newType = getTypeConverter()->convertType(op->getResult(0).getType());
    rewriter.modifyOpInPlace(newOp,
                             [&]() { newOp->getResult(0).setType(newType); });

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

    if (isa<PrepareConv2dWeightsOp>(op) || isa<PrepareConv2dBiasOp>(op)) {
      return true;
    }

    assert(op->getNumResults() == 1 &&
           "RowMajorConstEvalOperations only supports operations with a single "
           "result");

    Type resultTypeConverted =
        typeConverter.convertType(op->getResult(0).getType());
    if (!resultTypeConverted) {
      return true;
    }

    return false;
  }

  void runOnOperation() final {
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
