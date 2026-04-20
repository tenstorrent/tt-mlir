// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRRANKNORMALIZATION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

static constexpr int64_t minRank = 2;

/// Expands a shape array to minRank by prepending 1s.
static SmallVector<int32_t> expandShape(ArrayRef<int32_t> shape) {
  if (static_cast<int64_t>(shape.size()) >= minRank) {
    return SmallVector<int32_t>(shape);
  }

  int64_t numOnesToAdd = minRank - shape.size();
  SmallVector<int32_t> newShape(numOnesToAdd, 1);
  newShape.append(shape.begin(), shape.end());
  return newShape;
}

/// Expands a tensor type to minRank by prepending 1s.
/// Example: tensor<32xf32> -> tensor<1x32xf32>.
static RankedTensorType expandRank(RankedTensorType type) {
  if (type.getRank() >= minRank) {
    return type;
  }

  int64_t numOnesToAdd = minRank - type.getRank();
  SmallVector<int64_t> newShape(numOnesToAdd, 1);
  newShape.append(type.getShape().begin(), type.getShape().end());

  return RankedTensorType::get(newShape, type.getElementType(),
                               type.getEncoding());
}

static bool needsRankExpansion(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.getRank() < minRank;
  }
  return false;
}

/// TypeConverter that expands tensor types with rank < minRank.
class RankNormalizationTypeConverter : public TypeConverter {
public:
  RankNormalizationTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](RankedTensorType type) -> RankedTensorType {
      return expandRank(type);
    });
    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }

private:
  static Value materializeCast(OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) {
    assert(inputs.size() == 1 && "Expected single input.");
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs.front())
        .getResult(0);
  }
};

/// Converts operations by expanding low-rank tensor operands and results.
class GenericRankNormalizationPattern : public ConversionPattern {
public:
  GenericRankNormalizationPattern(TypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ModuleOp>(op) || isa<func::FuncOp>(op)) {
      return failure();
    }

    bool needsConversion =
        llvm::any_of(op->getOperandTypes(), needsRankExpansion) ||
        llvm::any_of(op->getResultTypes(), needsRankExpansion);
    if (!needsConversion) {
      return failure();
    }

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newResultTypes))) {
      return failure();
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(newResultTypes);
    state.addAttributes(op->getAttrs());
    state.addSuccessors(op->getSuccessors());
    for (Region &region : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(region, *newRegion, newRegion->end());
    }

    Operation *newOp = rewriter.create(state);

    // Update attributes that encode shape information.
    if (auto reshapeOp = dyn_cast<ttir::ReshapeOp>(newOp)) {
      updateReshapeShapeAttr(reshapeOp);
    } else if (auto constantOp = dyn_cast<ttir::ConstantOp>(newOp)) {
      updateConstantValueAttr(constantOp);
    } else if (auto arangeOp = dyn_cast<ttir::ArangeOp>(newOp)) {
      updateArangeDimension(arangeOp);
    } else if (auto sliceOp = dyn_cast<ttir::SliceStaticOp>(newOp)) {
      updateSliceStaticAttrs(sliceOp);
    } else if (auto fullOp = dyn_cast<ttir::FullOp>(newOp)) {
      updateDenseI32ShapeAttr(fullOp);
    } else if (auto zerosOp = dyn_cast<ttir::ZerosOp>(newOp)) {
      updateDenseI32ShapeAttr(zerosOp);
    } else if (auto onesOp = dyn_cast<ttir::OnesOp>(newOp)) {
      updateDenseI32ShapeAttr(onesOp);
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

private:
  /// Ops with `DenseI32ArrayAttr` `shape` (ttir.full, zeros, ones): keep shape
  /// attr aligned with the promoted result rank.
  template <typename OpTy>
  static void updateDenseI32ShapeAttr(OpTy op) {
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) {
      return;
    }
    ArrayRef<int32_t> currentShape = op.getShape();
    if (static_cast<int64_t>(currentShape.size()) == resultType.getRank()) {
      return;
    }
    OpBuilder builder(op.getContext());
    op.setShapeAttr(builder.getDenseI32ArrayAttr(expandShape(currentShape)));
  }

  static void updateConstantValueAttr(ttir::ConstantOp constantOp) {
    auto valueAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
    if (!valueAttr) {
      return;
    }
    auto resultType =
        dyn_cast<RankedTensorType>(constantOp.getResult().getType());
    if (!resultType) {
      return;
    }
    auto valueType = dyn_cast<RankedTensorType>(valueAttr.getType());
    if (!valueType || valueType.getShape() == resultType.getShape()) {
      return;
    }
    constantOp.setValueAttr(DenseElementsAttr::getFromRawBuffer(
        resultType, valueAttr.getRawData()));
  }

  static void updateArangeDimension(ttir::ArangeOp arangeOp) {
    auto resultType =
        dyn_cast<RankedTensorType>(arangeOp.getResult().getType());
    if (!resultType || resultType.getRank() < minRank) {
      return;
    }
    // When expanding 1D to 2D, dimension 0 shifts to dimension 1.
    if (arangeOp.getArangeDimension() == 0 && resultType.getRank() == minRank) {
      OpBuilder builder(arangeOp.getContext());
      arangeOp.setArangeDimensionAttr(builder.getI64IntegerAttr(1));
    }
  }

  static void updateReshapeShapeAttr(ttir::ReshapeOp reshapeOp) {
    auto resultType =
        dyn_cast<RankedTensorType>(reshapeOp.getResult().getType());
    if (!resultType) {
      return;
    }

    ArrayAttr shapeAttr = reshapeOp.getShapeAttr();
    SmallVector<int32_t> currentShape;
    for (Attribute attr : shapeAttr) {
      currentShape.push_back(cast<IntegerAttr>(attr).getInt());
    }

    if (static_cast<int64_t>(currentShape.size()) == resultType.getRank()) {
      return;
    }

    OpBuilder builder(reshapeOp.getContext());
    reshapeOp.setShapeAttr(builder.getI32ArrayAttr(expandShape(currentShape)));
  }

  static void updateSliceStaticAttrs(ttir::SliceStaticOp sliceOp) {
    auto resultType = dyn_cast<RankedTensorType>(sliceOp.getResult().getType());
    auto inputType = dyn_cast<RankedTensorType>(sliceOp.getInput().getType());
    if (!resultType || !inputType) {
      return;
    }

    ArrayAttr beginsAttr = sliceOp.getBeginsAttr();
    ArrayAttr endsAttr = sliceOp.getEndsAttr();
    ArrayAttr stepAttr = sliceOp.getStepAttr();

    // Check if attributes already match the expanded rank
    if (static_cast<int64_t>(beginsAttr.size()) == inputType.getRank()) {
      return;
    }

    // Extract current attribute values
    SmallVector<int32_t> begins, ends, step;
    for (Attribute attr : beginsAttr) {
      begins.push_back(cast<IntegerAttr>(attr).getInt());
    }
    for (Attribute attr : endsAttr) {
      ends.push_back(cast<IntegerAttr>(attr).getInt());
    }
    for (Attribute attr : stepAttr) {
      step.push_back(cast<IntegerAttr>(attr).getInt());
    }

    // Prepend values for new leading dimensions
    int64_t numDimsToAdd = inputType.getRank() - begins.size();
    SmallVector<int32_t> newBegins(numDimsToAdd, 0);
    SmallVector<int32_t> newEnds(numDimsToAdd, 1);
    SmallVector<int32_t> newStep(numDimsToAdd, 1);

    newBegins.append(begins.begin(), begins.end());
    newEnds.append(ends.begin(), ends.end());
    newStep.append(step.begin(), step.end());

    OpBuilder builder(sliceOp.getContext());
    sliceOp.setBeginsAttr(builder.getI32ArrayAttr(newBegins));
    sliceOp.setEndsAttr(builder.getI32ArrayAttr(newEnds));
    sliceOp.setStepAttr(builder.getI32ArrayAttr(newStep));
  }
};

/// Converts external CPU-hoisted function declarations.
/// Lower benefit ensures this pattern runs after the standard function
/// conversion pattern, which handles non-external functions.
class FuncOpRankNormalizationPattern
    : public OpConversionPattern<func::FuncOp> {
public:
  FuncOpRankNormalizationPattern(TypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<func::FuncOp>(converter, ctx, /*benefit=*/0) {}

  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypeConverter::SignatureConversion signatureConversion(
        funcOp.getNumArguments());
    for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
      Type convertedType =
          getTypeConverter()->convertType(funcOp.getArgumentTypes()[i]);
      if (!convertedType) {
        return failure();
      }
      signatureConversion.addInputs(i, convertedType);
    }

    SmallVector<Type> newResultTypes;
    if (failed(getTypeConverter()->convertTypes(funcOp.getResultTypes(),
                                                newResultTypes))) {
      return failure();
    }

    auto newFuncType = FunctionType::get(
        funcOp.getContext(), signatureConversion.getConvertedTypes(),
        newResultTypes);
    rewriter.modifyOpInPlace(funcOp, [&]() { funcOp.setType(newFuncType); });
    return success();
  }
};

class TTIRRankNormalization
    : public impl::TTIRRankNormalizationBase<TTIRRankNormalization> {
public:
  using TTIRRankNormalizationBase::TTIRRankNormalizationBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    RankNormalizationTypeConverter typeConverter;
    ConversionTarget target(*ctx);
    target.addLegalDialect<ttnn::TTNNDialect>();

    // A func participates in rank normalization iff its body contains at least
    // one TTIR-dialect op. Funcs without any TTIR op (e.g. pure-TTNN
    // const-eval helpers, external decls, or already-lowered subgraphs) are
    // left entirely untouched: signature, block args, body, and return op all
    // keep their original ranks. This avoids inserting
    // `builtin.unrealized_conversion_cast` ops at func boundaries that nothing
    // downstream knows how to clean up, and prevents the verifier mismatch
    // between a (legal, unmodified) func signature and a (rewritten)
    // `func.return` operand.
    DenseSet<func::FuncOp> participatingFuncs;
    module.walk([&](func::FuncOp funcOp) {
      bool hasTTIROp = false;
      funcOp.walk([&](Operation *inner) {
        if (isa<ttir::TTIRDialect>(inner->getDialect())) {
          hasTTIROp = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (hasTTIROp) {
        participatingFuncs.insert(funcOp);
      }
    });

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      // Per-op test: does this op carry any rank<2 type that the patterns
      // would want to rewrite? `func::FuncOp` has no SSA operands or results
      // of its own; its sig types live inside the `FunctionType` attribute,
      // so it needs its own check.
      bool needsRewrite;
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        needsRewrite =
            llvm::any_of(funcOp.getArgumentTypes(), needsRankExpansion) ||
            llvm::any_of(funcOp.getResultTypes(), needsRankExpansion);
      } else {
        needsRewrite =
            llvm::any_of(op->getOperandTypes(), needsRankExpansion) ||
            llvm::any_of(op->getResultTypes(), needsRankExpansion);
      }
      if (!needsRewrite) {
        return true;
      }
      // Op needs a rewrite. Suppress it if the op lives in (or is) a func
      // that has no TTIR ops to normalize -- promoting that func's sig or
      // any of its ops would only insert a builtin.unrealized_conversion_cast
      // at the boundary that nothing downstream knows how to clean up. This
      // prevents both (a) the flatbuffer crash on stray casts and (b) the
      // verifier mismatch between an unmodified func sig and a rewritten
      // func.return operand.
      func::FuncOp parentFunc = isa<func::FuncOp>(op)
                                    ? cast<func::FuncOp>(op)
                                    : op->getParentOfType<func::FuncOp>();
      if (parentFunc && !participatingFuncs.contains(parentFunc)) {
        return true;
      }
      return false;
    });

    RewritePatternSet patterns(ctx);
    patterns.add<GenericRankNormalizationPattern>(typeConverter, ctx);
    patterns.add<FuncOpRankNormalizationPattern>(typeConverter, ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttir
