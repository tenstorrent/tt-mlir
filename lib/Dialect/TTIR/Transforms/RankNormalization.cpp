// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

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

/// Some ops have rank invariants (e.g. require a 1D tensor, or have a verifier
/// that ties result rank/shape to other attributes) that this pass would break
/// by promoting to rank 2. Treat those ops as legal so the rewriter leaves
/// them and their operands/results untouched.
///
/// `ttir.mesh_shard` is included because its folder returns `getInput()` when
/// the shard volume is 1. MLIR's dialect conversion framework runs
/// `legalizeWithFold` on illegal ops before applying patterns; if our pass
/// marks `mesh_shard` illegal while its operand has been transiently promoted
/// (via the public-function entry reshape) but its declared result type has
/// not yet been rewritten, the fold returns a rank-2 value where a rank-0/1
/// is expected and the framework asserts. Keeping `mesh_shard` legal avoids
/// the fold call entirely; we instead wrap each rank-strict op with explicit
/// `ttir.reshape` ops in `insertRankStrictOpBoundaries` so the rest of the
/// body can still operate on rank>=2.
///
/// TTIR reduction ops (Sum, Mean, Max, Min, Prod, ReduceAnd, ReduceOr,
/// ArgMax) are included because their verifier derives the expected output
/// shape from the input rank, `dim_arg`, and `keep_dim`. Naively promoting
/// just the result type to rank>=2 (without also adjusting `dim_arg` indices
/// and/or flipping `keep_dim`) produces a verifier failure of the form
/// "Expected output shape (...), got (...)". Wrapping with boundary reshapes
/// preserves all three quantities together.
///
/// `ttir.dot_general` is included because its result rank is a function of
/// the operand ranks (result rank = lhs_rank + rhs_rank - 2*|contract| -
/// |batch|). Promoting only the operands to rank>=2 without also adjusting
/// the result type and the batch/contract dim attributes produces an op
/// whose declared result type is inconsistent with its operands. The
/// inconsistency is silently carried through to
/// `TTIRToTTIRDecomposition::DotGeneralToMatmulConversionPattern`, which
/// rebuilds the output shape from the post-promotion operand ranks
/// (e.g. (1x40960) x (1x64) -> 1x40960x1x64) and then `replaceOpWithNewOp`s
/// the dot_general with a reshape of that 4D shape. The dialect-conversion
/// framework inserts a `builtin.unrealized_conversion_cast` to bridge back
/// to the dot_general's original 2D result type. That cast escapes
/// `TTIRToTTNNCommon` (whose 1:1 type converter installs no materialization
/// callback) and the pipeline aborts with "failed to legalize unresolved
/// materialization that remained live after conversion". Wrapping with
/// boundary reshapes keeps the op's operand and result types exactly as
/// SHLO->TTIR declared them.
static bool hasRankStrictOperandInvariants(Operation *op) {
  return isa<ttir::PagedUpdateCacheOp, ttir::PagedFillCacheOp,
             ttir::ScaledDotProductAttentionDecodeOp,
             ttir::PagedScaledDotProductAttentionDecodeOp,
             ttir::MeshShardOp, ttir::SumOp, ttir::MeanOp, ttir::MaxOp,
             ttir::MinOp, ttir::ProdOp, ttir::ReduceAndOp, ttir::ReduceOrOp,
             ttir::ArgMaxOp, ttir::DotGeneralOp>(op);
}

/// Public, defined functions are the module's external boundary (e.g. JAX's
/// entry point as seen by the PJRT runtime). Their signature must remain
/// rank-0 to match the contract the runtime collected before this pass ran;
/// otherwise the runtime sees a shape mismatch between the MLIR module and
/// the compiled flatbuffer.
static bool isBoundaryFunc(func::FuncOp op) {
  return op.isPublic() && !op.isDeclaration();
}

/// Marker attribute placed on the reshape ops we insert at function
/// boundaries. The body conversion pattern skips ops carrying this attr so
/// our boundary reshapes are not promoted along with the rest of the body.
static constexpr StringRef kBoundaryReshapeAttr = "ttir.boundary_reshape";

/// For boundary funcs, insert entry-side `ttir.reshape` ops so every rank-0
/// block argument is immediately reshaped to rank>=2 and the rest of the
/// body operates on the promoted value. The block argument itself (and
/// therefore the function signature) stays rank-0.
///
/// Uses of block args that go directly into rank-strict ops (or into the
/// boundary demote reshape we may have already inserted just before such an
/// op in `insertRankStrictOpBoundaries`) are intentionally NOT replaced: the
/// rank-strict op needs to keep seeing its original-rank value, and the
/// demote reshape will itself read the rank-promoted value via its own use.
static void insertEntryBoundaryReshapes(func::FuncOp funcOp) {
  Block &entry = funcOp.getBody().front();
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(&entry);
  for (BlockArgument arg : llvm::to_vector(entry.getArguments())) {
    auto argType = dyn_cast<RankedTensorType>(arg.getType());
    if (!argType || argType.getRank() >= minRank) {
      continue;
    }
    RankedTensorType promoted = expandRank(argType);
    SmallVector<int32_t> shape(promoted.getShape().begin(),
                               promoted.getShape().end());
    auto reshape = builder.create<ttir::ReshapeOp>(
        arg.getLoc(), promoted, arg, builder.getI32ArrayAttr(shape));
    reshape->setAttr(kBoundaryReshapeAttr, builder.getUnitAttr());

    SmallPtrSet<Operation *, 4> excepted;
    excepted.insert(reshape);
    for (Operation *user : arg.getUsers()) {
      if (hasRankStrictOperandInvariants(user)) {
        excepted.insert(user);
      }
    }
    arg.replaceAllUsesExcept(reshape.getResult(), excepted);
  }
}

/// For ops in `hasRankStrictOperandInvariants` that have rank<minRank
/// operands or results, wrap them with `ttir.reshape` ops so the rank-strict
/// op continues to receive (and produce) values of its original rank while
/// the surrounding IR is promoted to rank>=2 by the rest of the pass.
///
/// For each rank<minRank operand we insert a "demote" reshape just before the
/// op (rank>=2 -> original rank) and rewire the op to read from the demote.
/// For each rank<minRank result we insert a "promote" reshape just after the
/// op (original rank -> rank>=2) and replace all other uses of the result
/// with the promoted value.
///
/// At the time we insert these reshapes the producers/consumers may still be
/// rank<minRank; once Phase 2 promotes them, the reshape's input/output ranks
/// naturally diverge and the reshape stops being identity. We mark each
/// inserted reshape with `kBoundaryReshapeAttr` so the body conversion
/// pattern leaves it (and the wrapped op) alone.
static void insertRankStrictOpBoundaries(Operation *op) {
  OpBuilder builder(op->getContext());

  builder.setInsertionPoint(op);
  for (auto en : llvm::enumerate(op->getOperandTypes())) {
    auto declared = dyn_cast<RankedTensorType>(en.value());
    if (!declared || declared.getRank() >= minRank) {
      continue;
    }

    Value operand = op->getOperand(en.index());
    SmallVector<int32_t> shape(declared.getShape().begin(),
                               declared.getShape().end());
    auto demote = builder.create<ttir::ReshapeOp>(
        op->getLoc(), declared, operand, builder.getI32ArrayAttr(shape));
    demote->setAttr(kBoundaryReshapeAttr, builder.getUnitAttr());
    op->setOperand(en.index(), demote.getResult());
  }

  builder.setInsertionPointAfter(op);
  for (Value result : op->getResults()) {
    auto declared = dyn_cast<RankedTensorType>(result.getType());
    if (!declared || declared.getRank() >= minRank) {
      continue;
    }

    RankedTensorType promotedType = expandRank(declared);
    SmallVector<int32_t> shape(promotedType.getShape().begin(),
                               promotedType.getShape().end());
    auto promote = builder.create<ttir::ReshapeOp>(
        op->getLoc(), promotedType, result, builder.getI32ArrayAttr(shape));
    promote->setAttr(kBoundaryReshapeAttr, builder.getUnitAttr());
    result.replaceAllUsesExcept(promote.getResult(), promote);
  }
}

/// For boundary funcs, insert exit-side squeeze reshapes after the body has
/// been promoted to rank>=2. For each return operand whose declared result
/// type is rank<minRank, take the now-promoted producer (peeling through any
/// `unrealized_conversion_cast` the framework inserted as a type bridge) and
/// replace the return operand with a `ttir.reshape` from rank>=2 back to the
/// declared rank, so the return matches the unchanged function signature.
static void insertExitBoundaryReshapes(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  funcOp.walk([&](func::ReturnOp returnOp) {
    builder.setInsertionPoint(returnOp);
    for (auto en : llvm::enumerate(returnOp.getOperands())) {
      Type declared = funcOp.getResultTypes()[en.index()];
      auto declaredType = dyn_cast<RankedTensorType>(declared);
      if (!declaredType || declaredType.getRank() >= minRank) {
        continue;
      }

      Value operand = en.value();
      Value source = operand;
      Operation *castOp = nullptr;
      if (auto cast = operand.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (cast.getInputs().size() == 1) {
          source = cast.getInputs()[0];
          castOp = cast;
        }
      }

      auto sourceType = dyn_cast<RankedTensorType>(source.getType());
      if (!sourceType || sourceType.getRank() < minRank) {
        continue;
      }

      SmallVector<int32_t> shape(declaredType.getShape().begin(),
                                 declaredType.getShape().end());
      auto reshape = builder.create<ttir::ReshapeOp>(
          returnOp.getLoc(), declared, source,
          builder.getI32ArrayAttr(shape));
      reshape->setAttr(kBoundaryReshapeAttr, builder.getUnitAttr());
      returnOp.setOperand(en.index(), reshape.getResult());

      if (castOp && castOp->use_empty()) {
        castOp->erase();
      }
    }
  });
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
    Value input = inputs.front();
    auto inputTensor = dyn_cast<RankedTensorType>(input.getType());
    auto outputTensor = dyn_cast<RankedTensorType>(type);

    // When both sides are tensors with the same element type and element
    // count (i.e. only leading 1-dims differ -- exactly what `expandRank`
    // produces), emit a `ttir.reshape` instead of an
    // `unrealized_conversion_cast`.  Casts inserted by the framework to
    // bridge rank-strict ops and their rank-promoted neighbors can survive
    // past legalization and cause a fatal "failed to legalize operation
    // 'builtin.unrealized_conversion_cast'" error.  A reshape is a legal
    // TTIR op that the rest of the pipeline understands.
    if (inputTensor && outputTensor &&
        inputTensor.getElementType() == outputTensor.getElementType() &&
        inputTensor.getNumElements() == outputTensor.getNumElements()) {
      SmallVector<int32_t> shape(outputTensor.getShape().begin(),
                                 outputTensor.getShape().end());
      auto reshape = builder.create<ttir::ReshapeOp>(
          loc, outputTensor, input, builder.getI32ArrayAttr(shape));
      reshape->setAttr(kBoundaryReshapeAttr, builder.getUnitAttr());
      return reshape.getResult();
    }

    return builder.create<UnrealizedConversionCastOp>(loc, type, input)
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

    if (hasRankStrictOperandInvariants(op)) {
      return failure();
    }

    // Reshapes we inserted at function boundaries must not be promoted; they
    // are intentionally rank-asymmetric (rank-0 ↔ rank-2) to translate
    // between the unchanged function signature and the promoted body.
    if (op->hasAttr(kBoundaryReshapeAttr)) {
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

    // Phase 1a: wrap rank-strict ops (those with operand-rank invariants
    // that this pass would otherwise break, e.g. `ttir.mesh_shard`) with
    // explicit demote/promote reshapes so the surrounding IR can be promoted
    // to rank>=2 without changing the rank-strict op's operand or result
    // types. This must run BEFORE Phase 1b (entry boundary reshapes) so
    // that we observe the operand types as originally declared by the
    // rank-strict op, not the rank-promoted values that the entry reshape
    // would have substituted. It must run before Phase 2 (the conversion
    // framework would otherwise call `legalizeWithFold` on the rank-strict
    // op, which can crash when the operand value rank no longer matches the
    // op's declared result rank).
    SmallVector<Operation *> rankStrictOps;
    module.walk([&](Operation *op) {
      if (hasRankStrictOperandInvariants(op)) {
        rankStrictOps.push_back(op);
      }
    });
    for (Operation *op : rankStrictOps) {
      insertRankStrictOpBoundaries(op);
    }

    // Phase 1b: insert entry-side boundary reshapes on every public function.
    // This lets the body conversion below promote internal ops to rank>=2
    // while the function arguments themselves stay rank-0. Direct uses of
    // public block args by rank-strict ops are skipped (those uses already
    // flow through the demote reshape that Phase 1a inserted).
    module.walk([](func::FuncOp funcOp) {
      if (isBoundaryFunc(funcOp)) {
        insertEntryBoundaryReshapes(funcOp);
      }
    });

    // Phase 2: existing op-level conversion. Boundary funcs and our marker
    // reshapes are kept legal so the conversion framework leaves them alone.
    RankNormalizationTypeConverter typeConverter;
    ConversionTarget target(*ctx);

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (hasRankStrictOperandInvariants(op)) {
        return true;
      }
      if (op->hasAttr(kBoundaryReshapeAttr)) {
        return true;
      }
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (isBoundaryFunc(funcOp)) {
          return true;
        }
        if (llvm::any_of(funcOp.getArgumentTypes(), needsRankExpansion) ||
            llvm::any_of(funcOp.getResultTypes(), needsRankExpansion)) {
          return false;
        }
        return true;
      }
      // The terminator of a boundary function returns the (already-squeezed)
      // rank-0 values that match the unchanged function signature. Don't let
      // the framework rewrite it to take rank-2 operands.
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        if (auto parent = returnOp->getParentOfType<func::FuncOp>()) {
          if (isBoundaryFunc(parent)) {
            return true;
          }
        }
      }
      if (llvm::any_of(op->getOperandTypes(), needsRankExpansion) ||
          llvm::any_of(op->getResultTypes(), needsRankExpansion)) {
        return false;
      }
      return true;
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

    // Phase 3: insert exit-side squeezes now that the body is fully promoted.
    // Doing this post-conversion lets us pull the rank>=2 producer directly
    // and avoid leaving stray `unrealized_conversion_cast` bridges in the IR.
    module.walk([](func::FuncOp funcOp) {
      if (isBoundaryFunc(funcOp)) {
        insertExitBoundaryReshapes(funcOp);
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttir
