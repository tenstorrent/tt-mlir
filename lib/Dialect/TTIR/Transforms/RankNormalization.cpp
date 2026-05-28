// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
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
///
/// For tensors carrying a `ttnn::TTNNLayoutAttr` encoding, defers to
/// `ttnn::utils::RankedTensorTypeFactory::create` so the layout's affine map
/// and memref shape are rebuilt for the new rank. A naive
/// `RankedTensorType::get` would propagate the original rank-1 layout onto
/// the rank-2 tensor, breaking downstream passes (e.g. TTIRToD2M) that index
/// into the encoding's shape.
static RankedTensorType expandRank(RankedTensorType type) {
  if (type.getRank() >= minRank) {
    return type;
  }

  int64_t numOnesToAdd = minRank - type.getRank();
  SmallVector<int64_t> newShape(numOnesToAdd, 1);
  newShape.append(type.getShape().begin(), type.getShape().end());

  if (isa_and_nonnull<ttnn::TTNNLayoutAttr>(type.getEncoding())) {
    return ttnn::utils::RankedTensorTypeFactory::create(type, newShape);
  }

  return RankedTensorType::get(newShape, type.getElementType(),
                               type.getEncoding());
}

static bool needsRankExpansion(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.getRank() < minRank;
  }
  return false;
}

/// Collects functions that participate in rank normalization. A non-external
/// function participates if its body contains at least one TTIR-dialect op.
/// An external declaration participates only if it is called from a
/// participating func.
static DenseSet<func::FuncOp> collectParticipatingFuncs(ModuleOp module) {
  DenseSet<func::FuncOp> result;

  module.walk([&](func::FuncOp funcOp) {
    if (funcOp.isExternal()) {
      return;
    }

    bool hasTTIROp = false;
    funcOp.walk([&](Operation *inner) {
      if (isa<ttir::TTIRDialect>(inner->getDialect())) {
        hasTTIROp = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasTTIROp) {
      result.insert(funcOp);
    }
  });

  // include external declarations that are called from
  //  participating functions (their signatures must be promoted to match).
  SmallVector<func::FuncOp> externalCallees;
  for (func::FuncOp participatingFunc : result) {
    participatingFunc.walk([&](func::CallOp callOp) {
      auto callee = module.lookupSymbol<func::FuncOp>(callOp.getCallee());
      if (callee && callee.isExternal() &&
          (llvm::any_of(callee.getArgumentTypes(), needsRankExpansion) ||
           llvm::any_of(callee.getResultTypes(), needsRankExpansion))) {
        externalCallees.push_back(callee);
      }
    });
  }
  result.insert(externalCallees.begin(), externalCallees.end());

  return result;
}

/// TypeConverter that expands tensor types with rank < minRank.
class RankNormalizationTypeConverter : public TypeConverter {
public:
  RankNormalizationTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](RankedTensorType type) -> RankedTensorType {
      return expandRank(type);
    });
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
    } else if (auto broadcastOp = dyn_cast<ttir::BroadcastOp>(newOp)) {
      updateBroadcastDimensionsAttr(broadcastOp);
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

  static void updateBroadcastDimensionsAttr(ttir::BroadcastOp broadcastOp) {
    auto inputType =
        dyn_cast<RankedTensorType>(broadcastOp.getInput().getType());
    if (!inputType) {
      return;
    }

    ArrayRef<int64_t> currentBroadcastDimensions =
        broadcastOp.getBroadcastDimensions();
    if (static_cast<int64_t>(currentBroadcastDimensions.size()) ==
        inputType.getRank()) {
      return;
    }

    if (static_cast<int64_t>(currentBroadcastDimensions.size()) >
        inputType.getRank()) {
      TT_assertv(false,
                 "broadcast_dimensions rank ({}) exceeds input rank ({})",
                 currentBroadcastDimensions.size(), inputType.getRank());
    }

    int64_t numDimsToAdd =
        inputType.getRank() - currentBroadcastDimensions.size();
    SmallVector<int64_t> newBroadcastDimensions(numDimsToAdd, 1);
    newBroadcastDimensions.append(currentBroadcastDimensions.begin(),
                                  currentBroadcastDimensions.end());

    OpBuilder builder(broadcastOp.getContext());
    broadcastOp.setBroadcastDimensionsAttr(
        builder.getDenseI64ArrayAttr(newBroadcastDimensions));
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

/// After `applyFullConversion` promotes the signatures of any `func.func`
/// referenced by a `ttnn.d2m_subgraph` op, the matching call sites keep their
/// original-rank operand/result types because they live in the (legal) TTNN
/// dialect and are never visited by the conversion. Walks every
/// `ttnn.d2m_subgraph` op and rebuilds it so every input, output (DPS
/// buffer), and result type matches the (possibly-promoted) callee signature,
/// inserting `ttnn.reshape` ops on each side as needed. This re-establishes
/// the `D2MSubgraphOp::verify()` invariant and also restores the global
/// rank >= minRank invariant inside TTIR (so downstream passes such as
/// TTIRToD2M never see rank-1 tensors).
static void insertCallsiteReshapesForD2MSubgraphs(ModuleOp module) {
  SmallVector<ttnn::D2MSubgraphOp> ops;
  module.walk([&](ttnn::D2MSubgraphOp op) { ops.push_back(op); });

  for (ttnn::D2MSubgraphOp op : ops) {
    func::FuncOp callee = op.getD2MMainFunc();
    if (!callee) {
      continue;
    }

    auto calleeArgTypes = callee.getArgumentTypes();
    auto calleeResultTypes = callee.getResultTypes();

    bool changed = false;
    for (auto [in, exp] :
         llvm::zip_equal(op.getInputs().getTypes(), calleeArgTypes)) {
      if (in != exp) {
        changed = true;
        break;
      }
    }
    if (!changed) {
      for (auto [out, exp] :
           llvm::zip_equal(op.getOutputs().getTypes(), calleeResultTypes)) {
        if (out != exp) {
          changed = true;
          break;
        }
      }
    }
    if (!changed) {
      for (auto [res, exp] :
           llvm::zip_equal(op.getResults().getTypes(), calleeResultTypes)) {
        if (res != exp) {
          changed = true;
          break;
        }
      }
    }
    if (!changed) {
      continue;
    }

    OpBuilder builder(op);
    auto buildReshape = [&](Value v, Type targetType) -> Value {
      if (v.getType() == targetType) {
        return v;
      }
      auto target = cast<RankedTensorType>(targetType);
      SmallVector<int32_t> shape;
      shape.reserve(target.getRank());
      for (int64_t d : target.getShape()) {
        shape.push_back(static_cast<int32_t>(d));
      }
      return builder.create<ttnn::ReshapeOp>(op.getLoc(), target, v,
                                             builder.getI32ArrayAttr(shape));
    };

    SmallVector<Value> newInputs;
    newInputs.reserve(op.getInputs().size());
    for (auto [input, expected] :
         llvm::zip_equal(op.getInputs(), calleeArgTypes)) {
      newInputs.push_back(buildReshape(input, expected));
    }

    SmallVector<Value> newOutputs;
    newOutputs.reserve(op.getOutputs().size());
    for (auto [output, expected] :
         llvm::zip_equal(op.getOutputs(), calleeResultTypes)) {
      newOutputs.push_back(buildReshape(output, expected));
    }

    SmallVector<Type> newResultTypes(calleeResultTypes.begin(),
                                     calleeResultTypes.end());
    auto newOp = builder.create<ttnn::D2MSubgraphOp>(
        op.getLoc(), newResultTypes, newInputs, newOutputs,
        op.getD2mFuncAttr());

    builder.setInsertionPointAfter(newOp);
    SmallVector<Value> finalResults;
    finalResults.reserve(op.getNumResults());
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op.getResults(), newOp.getResults())) {
      finalResults.push_back(buildReshape(newResult, oldResult.getType()));
    }

    op.replaceAllUsesWith(finalResults);
    op.erase();
  }
}

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
    target.addLegalDialect<ttcore::TTCoreDialect>();

    auto participatingFuncs = collectParticipatingFuncs(module);

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
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
      return;
    }

    insertCallsiteReshapesForD2MSubgraphs(module);
  }
};

} // namespace

} // namespace mlir::tt::ttir
