// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/MoeWorkaroundsRewritePatterns.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttnn::workarounds::decomposition {

namespace {

static Value createReshape(PatternRewriter &rewriter, Location loc, Value input,
                           llvm::ArrayRef<int64_t> outputShape,
                           StringRef suffix) {
  auto inputType = cast<RankedTensorType>(input.getType());
  RankedTensorType outputType =
      ttnn::utils::RankedTensorTypeFactory::create(inputType, outputShape);
  SmallVector<int32_t> shapeI32(outputShape.begin(), outputShape.end());
  return rewriter
      .create<ttnn::ReshapeOp>(
          ttmlir::utils::appendLocationSuffix(loc, suffix), outputType, input,
          rewriter.getI32ArrayAttr(shapeI32), ttnn::MemoryConfigAttr())
      .getResult();
}

static ttcore::TileSizeAttr
getPreferredTileSize(ttcore::ChipDescAttr chipDesc) {
  llvm::ArrayRef<ttcore::TileSizeAttr> supported =
      chipDesc.getSupportedTileSizes();
  if (supported.empty()) {
    return {};
  }

  auto betterTile = [](ttcore::TileSizeAttr lhs, ttcore::TileSizeAttr rhs) {
    int64_t lhsArea = lhs.getY() * lhs.getX();
    int64_t rhsArea = rhs.getY() * rhs.getX();
    if (lhsArea != rhsArea) {
      return lhsArea < rhsArea;
    }
    if (lhs.getY() != rhs.getY()) {
      return lhs.getY() < rhs.getY();
    }
    return lhs.getX() < rhs.getX();
  };

  ttcore::TileSizeAttr preferred = supported.front();
  for (ttcore::TileSizeAttr tile : supported.drop_front()) {
    if (betterTile(preferred, tile)) {
      preferred = tile;
    }
  }
  return preferred;
}

class SparseMatmulTileDimsRewritePattern
    : public OpRewritePattern<ttnn::SparseMatmulOp> {
public:
  using OpRewritePattern<ttnn::SparseMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::SparseMatmulOp srcOp,
                                PatternRewriter &rewriter) const override;
};

// Generate a default program config for sparse_matmul based on tiled shapes.
static ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr
createProgramConfig(MLIRContext *ctx, RankedTensorType aType,
                    RankedTensorType bType, ttcore::DeviceAttr deviceAttr,
                    int64_t tileH, int64_t tileW) {
  auto gridShape = deviceAttr.getWorkerGrid().getShape();
  int64_t coreY = gridShape[0];
  int64_t coreX = gridShape[1];

  auto aShape = aType.getShape();
  auto bShape = bType.getShape();

  int64_t M = aShape[aShape.size() - 2];
  int64_t N = bShape[bShape.size() - 1];
  int64_t NTiles = (N + tileW - 1) / tileW;

  int64_t usedCoreX = std::min(coreX, NTiles);
  while (usedCoreX > 1) {
    int64_t pcn = (NTiles + usedCoreX - 1) / usedCoreX;
    if ((NTiles + pcn - 1) / pcn == usedCoreX) {
      break;
    }
    --usedCoreX;
  }

  int64_t perCoreM = std::max(static_cast<int64_t>(1), M / tileH);
  int64_t perCoreN =
      std::max(static_cast<int64_t>(1), (NTiles + usedCoreX - 1) / usedCoreX);

  auto gridAttr = ttnn::CoreCoordAttr::get(ctx, usedCoreX, coreY);
  auto hopCoresAttr = ttnn::CoreRangeSetAttr::get(ctx, {});

  return ttnn::MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
      ctx, gridAttr,
      /*in0_block_w=*/1, /*out_subblock_h=*/1, /*out_subblock_w=*/1,
      /*out_block_h=*/1, /*out_block_w=*/1,
      /*per_core_m=*/static_cast<uint64_t>(perCoreM),
      /*per_core_n=*/static_cast<uint64_t>(perCoreN),
      /*fuse_batch=*/false, /*fused_activation=*/nullptr,
      /*mcast_in0=*/true, /*gather_in0=*/false,
      /*hop_cores=*/hopCoresAttr, /*num_global_cb_receivers=*/0,
      /*untilize_out=*/false);
}

template <typename OpTy>
static OpTy getSingleUserOfType(Value value) {
  SmallVector<Operation *> users;
  for (Operation *user : value.getUsers()) {
    if (mlir::isa<ttnn::DeallocateOp>(user)) {
      continue;
    }
    users.push_back(user);
  }

  if (users.size() != 1) {
    return nullptr;
  }
  return mlir::dyn_cast<OpTy>(users.front());
}

static bool isPermutation(ttnn::PermuteOp permuteOp,
                          llvm::ArrayRef<int64_t> expected) {
  if (!permuteOp) {
    return false;
  }
  auto perm = permuteOp.getPermutation();
  if (perm.size() != expected.size()) {
    return false;
  }
  return llvm::equal(perm, expected);
}

static Value cloneLikeWithNewOperands(PatternRewriter &rewriter, Operation *op,
                                      llvm::ArrayRef<Value> operands,
                                      Type resultType) {
  OperationState state(op->getLoc(), op->getName().getStringRef());
  state.addOperands(operands);
  state.addTypes(resultType);
  state.addAttributes(op->getAttrs());
  return rewriter.create(state)->getResult(0);
}

// Rewrite already-tiled gate-up activation chain from flattened layout:
//
//   reshape -> permute -> reshape(flat) -> add -> swiglu -> reshape -> permute
//
// to an M-preserving layout equivalent to legacy factoring:
//
//   add(on 5D) -> permute -> reshape -> permute -> swiglu
//
// This preserves semantics while avoiding the [BD*dimB*M, 1, 1, N] flattening
// for the elementwise activation path.
static LogicalResult
rewriteAlreadyTiledGateUpActivationChain(ttnn::SparseMatmulOp srcOp,
                                         PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  auto gateUpReshape = getSingleUserOfType<ttnn::ReshapeOp>(srcOp.getResult());
  if (!gateUpReshape) {
    return failure();
  }

  auto gateUpShape = gateUpReshape.getType().getShape();
  if (gateUpShape.size() != 5) {
    return failure();
  }

  auto preFlattenPermute =
      getSingleUserOfType<ttnn::PermuteOp>(gateUpReshape.getResult());
  if (!isPermutation(preFlattenPermute, {0, 1, 3, 2, 4})) {
    return failure();
  }

  auto flattened =
      getSingleUserOfType<ttnn::ReshapeOp>(preFlattenPermute.getResult());
  if (!flattened) {
    return failure();
  }

  auto flattenedShape = flattened.getType().getShape();
  if (flattenedShape.size() != 4 || flattenedShape[2] != 1) {
    return failure();
  }

  // Legacy flattened paths may carry either:
  //   [BD*dimB*M, 1, 1, N]    (single-expert chunk), or
  //   [BD*dimB*M, E, 1, N]    (multi-expert chunk, e.g. E=4).
  // Accept both, but require consistency with the gate-up expert dimension.
  int64_t flattenedExpertDim = flattenedShape[1];
  int64_t gateUpExpertDim = gateUpShape[2];
  if (flattenedExpertDim != 1 && flattenedExpertDim != gateUpExpertDim) {
    return failure();
  }

  auto preActAdd = getSingleUserOfType<ttnn::AddOp>(flattened.getResult());
  if (!preActAdd) {
    return failure();
  }

  // Build rewritten activation ops at the original pre-activation add site.
  // Inserting at srcOp can violate SSA dominance because the operands in this
  // subgraph are defined later than srcOp.
  rewriter.setInsertionPoint(preActAdd);

  SmallVector<ttnn::SliceStaticOp> preActSlices;
  for (Operation *user : preActAdd.getResult().getUsers()) {
    if (mlir::isa<ttnn::DeallocateOp>(user)) {
      continue;
    }
    auto slice = mlir::dyn_cast<ttnn::SliceStaticOp>(user);
    if (!slice) {
      return failure();
    }
    preActSlices.push_back(slice);
  }
  if (preActSlices.size() != 2) {
    return failure();
  }

  ttnn::SliceStaticOp gateSlice;
  ttnn::SliceStaticOp valueSlice;
  for (ttnn::SliceStaticOp slice : preActSlices) {
    auto beginsOpt =
        ttmlir::utils::getIntegerVector<int64_t>(slice.getBegins());
    auto stepsOpt = ttmlir::utils::getIntegerVector<int64_t>(slice.getStep());
    if (!beginsOpt || !stepsOpt) {
      return failure();
    }
    auto &begins = *beginsOpt;
    auto &steps = *stepsOpt;
    if (begins.size() != 4 || steps.size() != 4 || steps[3] != 2) {
      return failure();
    }
    if (begins[3] == 1) {
      gateSlice = slice;
    } else if (begins[3] == 0) {
      valueSlice = slice;
    } else {
      return failure();
    }
  }
  if (!gateSlice || !valueSlice) {
    return failure();
  }

  auto gateClamp =
      getSingleUserOfType<ttnn::ClampScalarOp>(gateSlice.getResult());
  auto gateAdd = gateClamp
                     ? getSingleUserOfType<ttnn::AddOp>(gateClamp.getResult())
                     : nullptr;
  if (!gateClamp || !gateAdd) {
    return failure();
  }

  auto valueClamp =
      getSingleUserOfType<ttnn::ClampScalarOp>(valueSlice.getResult());
  if (!valueClamp) {
    return failure();
  }

  SmallVector<ttnn::MultiplyOp> valueClampMulUsers;
  for (Operation *user : valueClamp.getResult().getUsers()) {
    if (mlir::isa<ttnn::DeallocateOp>(user)) {
      continue;
    }
    auto mul = mlir::dyn_cast<ttnn::MultiplyOp>(user);
    if (!mul) {
      return failure();
    }
    valueClampMulUsers.push_back(mul);
  }
  if (valueClampMulUsers.size() != 2) {
    return failure();
  }

  ttnn::MultiplyOp valueMul;
  ttnn::SigmoidOp valueSigmoid;
  for (ttnn::MultiplyOp mul : valueClampMulUsers) {
    if (auto sig = getSingleUserOfType<ttnn::SigmoidOp>(mul.getResult())) {
      if (valueMul) {
        return failure();
      }
      valueMul = mul;
      valueSigmoid = sig;
    }
  }
  if (!valueMul || !valueSigmoid) {
    return failure();
  }

  ttnn::MultiplyOp valueGatedMul;
  for (ttnn::MultiplyOp mul : valueClampMulUsers) {
    if (mul == valueMul) {
      continue;
    }
    bool hasClamp = mul.getLhs() == valueClamp.getResult() ||
                    mul.getRhs() == valueClamp.getResult();
    bool hasSigmoid = mul.getLhs() == valueSigmoid.getResult() ||
                      mul.getRhs() == valueSigmoid.getResult();
    if (hasClamp && hasSigmoid) {
      valueGatedMul = mul;
    }
  }
  if (!valueGatedMul) {
    return failure();
  }

  auto finalMul = getSingleUserOfType<ttnn::MultiplyOp>(gateAdd.getResult());
  if (!finalMul) {
    return failure();
  }
  Value finalOtherOperand =
      finalMul.getLhs() == gateAdd.getResult()   ? finalMul.getRhs()
      : finalMul.getRhs() == gateAdd.getResult() ? finalMul.getLhs()
                                                 : Value();
  if (!finalOtherOperand || finalOtherOperand != valueGatedMul.getResult()) {
    return failure();
  }

  auto finalReshape =
      getSingleUserOfType<ttnn::ReshapeOp>(finalMul.getResult());
  auto finalPermute =
      finalReshape
          ? getSingleUserOfType<ttnn::PermuteOp>(finalReshape.getResult())
          : nullptr;
  if (!finalReshape || !isPermutation(finalPermute, {0, 2, 1, 3})) {
    return failure();
  }

  auto finalType = mlir::cast<RankedTensorType>(finalPermute.getType());
  auto finalShape = finalType.getShape();
  if (finalShape.size() != 4) {
    return failure();
  }

  int64_t dimA = gateUpShape[0];
  int64_t dimB = gateUpShape[1];
  int64_t E = gateUpShape[2];
  int64_t M = gateUpShape[3];
  int64_t expandedN = gateUpShape[4];
  int64_t reducedN = finalShape[3];

  if (finalShape[0] != dimA * dimB || finalShape[1] != E ||
      finalShape[2] != M) {
    return failure();
  }
  if (expandedN != reducedN * 2) {
    return failure();
  }

  auto addBiasType = mlir::cast<RankedTensorType>(preActAdd.getRhs().getType());
  auto addBiasShape = addBiasType.getShape();
  if (addBiasShape.size() != 4 || addBiasShape[3] != expandedN) {
    return failure();
  }
  int64_t biasE = addBiasShape[1];
  if (biasE != 1 && biasE != E) {
    return failure();
  }

  auto loc = srcOp.getLoc();

  // Match main-branch friendly layout through activation:
  // [dimA, dimB, E, M, N] -> [dimA*dimB, E, M, N], then bias add.
  auto preSliceInput = ttir_to_ttnn::utils::generateReshape(
      cast<TypedValue<RankedTensorType>>(gateUpReshape.getResult()),
      {dimA * dimB, E, M, expandedN}, rewriter, loc);

  auto preSliceInputType =
      mlir::cast<RankedTensorType>(preSliceInput.getResult().getType());
  auto newPreActAdd = cloneLikeWithNewOperands(
      rewriter, preActAdd.getOperation(),
      {preSliceInput.getResult(), preActAdd.getRhs()}, preSliceInputType);

  auto sliceOutType = ttnn::utils::RankedTensorTypeFactory::create(
      preSliceInputType,
      {finalShape[0], finalShape[1], finalShape[2], reducedN});

  auto buildHalfSlice = [&](int32_t start, Location sliceLoc) -> Value {
    SmallVector<int32_t> begins = {0, 0, 0, start};
    SmallVector<int32_t> ends = {static_cast<int32_t>(finalShape[0]),
                                 static_cast<int32_t>(finalShape[1]),
                                 static_cast<int32_t>(finalShape[2]),
                                 static_cast<int32_t>(expandedN)};
    SmallVector<int32_t> steps = {1, 1, 1, 2};
    return rewriter
        .create<ttnn::SliceStaticOp>(sliceLoc, sliceOutType, newPreActAdd,
                                     rewriter.getI32ArrayAttr(begins),
                                     rewriter.getI32ArrayAttr(ends),
                                     rewriter.getI32ArrayAttr(steps))
        .getResult();
  };

  Value newGateSlice = buildHalfSlice(/*start=*/1, gateSlice.getLoc());
  Value newValueSlice = buildHalfSlice(/*start=*/0, valueSlice.getLoc());

  Value newGateClamp = cloneLikeWithNewOperands(
      rewriter, gateClamp.getOperation(), {newGateSlice}, sliceOutType);
  Value newGateAdd = cloneLikeWithNewOperands(
      rewriter, gateAdd.getOperation(),
      {gateAdd.getLhs() == gateClamp.getResult() ? newGateClamp
                                                 : gateAdd.getLhs(),
       gateAdd.getRhs() == gateClamp.getResult() ? newGateClamp
                                                 : gateAdd.getRhs()},
      sliceOutType);

  Value newValueClamp = cloneLikeWithNewOperands(
      rewriter, valueClamp.getOperation(), {newValueSlice}, sliceOutType);
  Value newValueMul = cloneLikeWithNewOperands(
      rewriter, valueMul.getOperation(),
      {valueMul.getLhs() == valueClamp.getResult() ? newValueClamp
                                                   : valueMul.getLhs(),
       valueMul.getRhs() == valueClamp.getResult() ? newValueClamp
                                                   : valueMul.getRhs()},
      sliceOutType);
  Value newValueSigmoid = cloneLikeWithNewOperands(
      rewriter, valueSigmoid.getOperation(), {newValueMul}, sliceOutType);
  auto remapValueGatedMulOperand = [&](Value operand) -> Value {
    if (operand == valueClamp.getResult()) {
      return newValueClamp;
    }
    if (operand == valueSigmoid.getResult()) {
      return newValueSigmoid;
    }
    return operand;
  };
  Value newValueGatedMul = cloneLikeWithNewOperands(
      rewriter, valueGatedMul.getOperation(),
      {remapValueGatedMulOperand(valueGatedMul.getLhs()),
       remapValueGatedMulOperand(valueGatedMul.getRhs())},
      sliceOutType);

  Value newFinalMul = cloneLikeWithNewOperands(
      rewriter, finalMul.getOperation(),
      {finalMul.getLhs() == gateAdd.getResult() ? newGateAdd : newValueGatedMul,
       finalMul.getRhs() == gateAdd.getResult() ? newGateAdd
                                                : newValueGatedMul},
      finalType);

  // Rewrite down-projection post-sparse chain to keep tiled M layout through
  // bias add, matching main-branch style:
  //   sparse -> add -> reshape -> permute -> reshape
  // instead of:
  //   sparse -> permute -> reshape -> add -> reshape -> permute.
  if (auto downSparse =
          getSingleUserOfType<ttnn::SparseMatmulOp>(finalPermute.getResult())) {
    if (downSparse.getIsInputASparse() && !downSparse.getIsInputBSparse()) {
      auto downPermute =
          getSingleUserOfType<ttnn::PermuteOp>(downSparse.getResult());
      auto downFlatten =
          downPermute
              ? getSingleUserOfType<ttnn::ReshapeOp>(downPermute.getResult())
              : nullptr;
      auto downAdd =
          downFlatten
              ? getSingleUserOfType<ttnn::AddOp>(downFlatten.getResult())
              : nullptr;
      auto downReshape =
          downAdd ? getSingleUserOfType<ttnn::ReshapeOp>(downAdd.getResult())
                  : nullptr;
      auto downFinalPermute =
          downReshape
              ? getSingleUserOfType<ttnn::PermuteOp>(downReshape.getResult())
              : nullptr;

      if (isPermutation(downPermute, {0, 2, 1, 3}) &&
          isPermutation(downFinalPermute, {2, 0, 1, 3})) {
        auto sparseType =
            cast<RankedTensorType>(downSparse.getResult().getType());
        auto sparseShape = sparseType.getShape();
        auto downReshapeType = cast<RankedTensorType>(downReshape.getType());
        auto downReshapeShape = downReshapeType.getShape();
        if (sparseShape.size() == 4 && downReshapeShape.size() == 4) {
          int64_t T = sparseShape[0];
          int64_t EDown = sparseShape[1];
          int64_t MDown = sparseShape[2];
          int64_t HDown = sparseShape[3];
          int64_t BDown = downReshapeShape[0];
          int64_t SDown = downReshapeShape[1];
          int64_t EReshaped = downReshapeShape[2];
          int64_t HReshaped = downReshapeShape[3];

          Value flattenValue = downFlatten.getResult();
          Value bias = downAdd.getLhs() == flattenValue   ? downAdd.getRhs()
                       : downAdd.getRhs() == flattenValue ? downAdd.getLhs()
                                                          : Value();

          if (MDown > 0 && EDown == EReshaped && HDown == HReshaped &&
              SDown > 0 && SDown % MDown == 0 && bias) {
            int64_t dimBDown = SDown / MDown;
            if (dimBDown > 0 && BDown > 0 && BDown * dimBDown == T) {
              rewriter.setInsertionPoint(downAdd);
              Value newDownAdd = cloneLikeWithNewOperands(
                  rewriter, downAdd.getOperation(),
                  {downAdd.getLhs() == flattenValue ? downSparse.getResult()
                                                    : downAdd.getLhs(),
                   downAdd.getRhs() == flattenValue ? downSparse.getResult()
                                                    : downAdd.getRhs()},
                  sparseType);

              auto downReshape5D = ttir_to_ttnn::utils::generateReshape(
                  cast<TypedValue<RankedTensorType>>(newDownAdd),
                  {BDown, dimBDown, EDown, MDown, HDown}, rewriter,
                  downAdd.getLoc());
              auto downPermute5D = ttir_to_ttnn::utils::generatePermute(
                  cast<TypedValue<RankedTensorType>>(downReshape5D.getResult()),
                  {2, 0, 1, 3, 4}, rewriter, downAdd.getLoc());
              auto downCombineInput = ttir_to_ttnn::utils::generateReshape(
                  cast<TypedValue<RankedTensorType>>(downPermute5D.getResult()),
                  {EDown, BDown, SDown, HDown}, rewriter, downAdd.getLoc());
              rewriter.replaceOp(downFinalPermute,
                                 downCombineInput.getResult());
            }
          }
        }
      }
    }
  }

  rewriter.replaceOp(finalPermute, newFinalMul);
  return success();
}

LogicalResult SparseMatmulTileDimsRewritePattern::matchAndRewrite(
    ttnn::SparseMatmulOp srcOp, PatternRewriter &rewriter) const {

  auto inputA = srcOp.getA();
  auto inputB = srcOp.getB();
  auto sparsityMask = srcOp.getSparsity();

  auto inputAType = mlir::cast<RankedTensorType>(inputA.getType());
  auto inputBType = mlir::cast<RankedTensorType>(inputB.getType());
  auto sparsityType = mlir::cast<RankedTensorType>(sparsityMask.getType());
  auto inputAShape = inputAType.getShape();
  int64_t mDim = inputAShape[inputAShape.size() - 2];

  bool isASparse = srcOp.getIsInputASparse();
  bool isBSparse = srcOp.getIsInputBSparse();
  auto loc = srcOp.getLoc();
  auto chipDesc = ttcore::getOpChipDescAttr(srcOp);
  auto preferredTile = getPreferredTileSize(chipDesc);
  if (!preferredTile) {
    srcOp.emitError("chip description has no supported tile sizes");
    return failure();
  }
  int64_t tileHeight = preferredTile.getY();
  int64_t tileWidth = preferredTile.getX();
  if (tileHeight <= 0 || tileWidth <= 0) {
    srcOp.emitError("preferred tile size must be positive");
    return failure();
  }

  if (!isASparse && isBSparse) {
    if (mDim == tileHeight) {
      // Handle already-tiled gate-up subgraph where frontend reconstruction
      // flattened [dimA, dimB, M] to [dimA*dimB*M, 1, 1] before SwiGLU.
      // Rewrite to keep the M axis through elementwise activation.
      return rewriteAlreadyTiledGateUpActivationChain(srcOp, rewriter);
    }

    if (mDim > tileHeight) {
      return failure();
    }

    // Gate-up: A=[BD, S, 1, H], sparsity=[1, 1, BD*S/M, E]
    auto bShape = inputBType.getShape();
    auto spShape = sparsityType.getShape();

    int64_t BD = inputAShape[0];
    int64_t S = inputAShape[1];
    int64_t H = inputAShape[3];
    int64_t E = spShape[3];
    int64_t N = bShape[3];
    int64_t reducedTokens = spShape[2];
    int64_t M = (BD * S) / reducedTokens;

    bool splitSeq = (S % M == 0) && (S >= M);
    bool splitBd = (BD % M == 0) && (BD >= M);
    if (!splitSeq && !splitBd) {
      return srcOp.emitError("Neither seq_len (")
             << S << ") nor BD (" << BD << ") is divisible by tile size " << M;
    }

    int64_t dimA = splitSeq ? BD : BD / M;
    int64_t dimB = splitSeq ? S / M : S;

    // Tile A: [BD, S, 1, H] -> [dimA, dimB, M, H]
    Value tiledA;
    if (splitSeq) {
      tiledA = ttir_to_ttnn::utils::generateReshape(inputA, {BD, S / M, M, H},
                                                    rewriter, loc)
                   .getResult();
    } else {
      auto reshaped = ttir_to_ttnn::utils::generateReshape(
          inputA, {BD / M, M, S, H}, rewriter, loc);
      tiledA = ttir_to_ttnn::utils::generatePermute(
                   cast<TypedValue<RankedTensorType>>(reshaped.getResult()),
                   {0, 2, 1, 3}, rewriter, loc)
                   .getResult();
    }

    // Tile sparsity: [1, 1, BD*S/M, E] -> [dimA, dimB, 1, E]
    auto tiledSparsity = ttir_to_ttnn::utils::generateReshape(
        sparsityMask, {dimA, dimB, 1, E}, rewriter, loc);

    // Create tiled sparse_matmul
    auto tiledATyped = cast<TypedValue<RankedTensorType>>(tiledA);
    auto deviceAttr = ttcore::lookupDevice(srcOp);
    auto programConfigAttr =
        createProgramConfig(rewriter.getContext(), tiledATyped.getType(),
                            inputBType, deviceAttr, tileHeight, tileWidth);

    llvm::SmallVector<int64_t> tiledOutShape = {dimA, dimB, 1, E, M, N};
    auto tiledOutType = ttnn::utils::RankedTensorTypeFactory::create(
        tiledATyped.getType(), tiledOutShape);

    mlir::IntegerAttr nnzAttr = nullptr;
    if (auto nnz = srcOp.getNnz()) {
      nnzAttr = rewriter.getI64IntegerAttr(*nnz);
    }

    auto tiledResult = rewriter.create<ttnn::SparseMatmulOp>(
        loc, tiledOutType, tiledATyped, inputB,
        cast<TypedValue<RankedTensorType>>(tiledSparsity.getResult()),
        srcOp.getIsInputASparse(), srcOp.getIsInputBSparse(), nnzAttr,
        programConfigAttr, /*memory_config=*/nullptr, /*dtype=*/nullptr,
        /*compute_config=*/nullptr);

    // Untile output while preserving the legacy gate-up factoring.
    //
    // Historically, the frontend reconstructed gate-up outputs via:
    //   squeeze(dim=2) -> permute(0, 1, 3, 2, 4) -> reshape(BD, S, E, N).
    // Rebuild that 4D tensor first, then expand back to the frontend-visible
    // 6D type [BD, S, 1, E, 1, N]. This keeps type compatibility while making
    // the subsequent squeeze path recover the legacy layout-friendly shape.
    auto squeezed = ttir_to_ttnn::utils::generateReshape(
        cast<TypedValue<RankedTensorType>>(tiledResult.getResult()),
        {dimA, dimB, E, M, N}, rewriter, loc);

    auto legacyPermuted = ttir_to_ttnn::utils::generatePermute(
        cast<TypedValue<RankedTensorType>>(squeezed.getResult()),
        {0, 1, 3, 2, 4}, rewriter, loc);

    auto legacyGateUp = ttir_to_ttnn::utils::generateReshape(
        cast<TypedValue<RankedTensorType>>(legacyPermuted.getResult()),
        {BD, S, E, N}, rewriter, loc);

    auto untiled = ttir_to_ttnn::utils::generateReshape(
        cast<TypedValue<RankedTensorType>>(legacyGateUp.getResult()),
        {BD, S, 1, E, 1, N}, rewriter, loc);

    rewriter.replaceOp(srcOp, untiled.getResult());
    return success();

  } else if (isASparse && !isBSparse) {
    if (mDim >= tileHeight) {
      return failure();
    }

    // Down: A=[BD*S, E, 1, inter], sparsity=[1, 1, BD*S/M, E]
    auto bShape = inputBType.getShape();
    auto spShape = sparsityType.getShape();

    int64_t totalTokens = inputAShape[0];
    int64_t E = inputAShape[1];
    int64_t inter = inputAShape[3];
    int64_t H = bShape[3];
    int64_t reducedTokens = spShape[2];
    int64_t M = totalTokens / reducedTokens;

    // Preserve legacy factoring when the graph has:
    //   [BD, S, E, inter] -> reshape([BD*S, E, 1, inter]) -> sparse_matmul
    // and the sparse_matmul output is immediately restored with:
    //   reshape([BD, S, E, H]).
    //
    // In that case, consume [BD, S, E, inter] directly and produce [BD, S, E,
    // H] directly, so the [BD*S, E, 1, *] flatten/unflatten pair disappears.
    TypedValue<RankedTensorType> logicalInputA =
        cast<TypedValue<RankedTensorType>>(inputA);
    ttnn::ReshapeOp preFlatten =
        mlir::dyn_cast_if_present<ttnn::ReshapeOp>(inputA.getDefiningOp());
    ttnn::ReshapeOp postRestore =
        getSingleUserOfType<ttnn::ReshapeOp>(srcOp.getResult());
    bool preserveLegacyFactoring = false;
    int64_t restoredBD = -1;
    int64_t restoredS = -1;
    if (preFlatten && postRestore) {
      auto preFlattenInputType =
          mlir::cast<RankedTensorType>(preFlatten->getOperand(0).getType());
      auto preFlattenOutputType =
          mlir::cast<RankedTensorType>(preFlatten.getType());
      auto postRestoreType =
          mlir::cast<RankedTensorType>(postRestore.getType());

      auto preFlattenInputShape = preFlattenInputType.getShape();
      auto preFlattenOutputShape = preFlattenOutputType.getShape();
      auto postRestoreShape = postRestoreType.getShape();

      if (preFlattenInputShape.size() == 4 &&
          preFlattenOutputShape.size() == 4 && postRestoreShape.size() == 4 &&
          preFlattenOutputShape[0] == totalTokens &&
          preFlattenOutputShape[1] == E && preFlattenOutputShape[2] == 1 &&
          preFlattenOutputShape[3] == inter &&
          postRestoreShape[0] == preFlattenInputShape[0] &&
          postRestoreShape[1] == preFlattenInputShape[1] &&
          postRestoreShape[2] == preFlattenInputShape[2] &&
          postRestoreShape[3] == H && preFlattenInputShape[2] == E &&
          preFlattenInputShape[0] * preFlattenInputShape[1] == totalTokens) {
        logicalInputA =
            cast<TypedValue<RankedTensorType>>(preFlatten->getOperand(0));
        restoredBD = postRestoreShape[0];
        restoredS = postRestoreShape[1];
        preserveLegacyFactoring = true;
      }
    }

    // Tile A: [BD*S, E, 1, inter] (or [BD, S, E, inter]) -> [BD*S/M, E, M,
    // inter]
    auto reshaped = ttir_to_ttnn::utils::generateReshape(
        logicalInputA, {totalTokens / M, M, E, inter}, rewriter, loc);
    auto tiledA = ttir_to_ttnn::utils::generatePermute(
        cast<TypedValue<RankedTensorType>>(reshaped.getResult()), {0, 2, 1, 3},
        rewriter, loc);

    auto tiledATyped = cast<TypedValue<RankedTensorType>>(tiledA.getResult());
    auto deviceAttr = ttcore::lookupDevice(srcOp);
    auto programConfigAttr =
        createProgramConfig(rewriter.getContext(), tiledATyped.getType(),
                            inputBType, deviceAttr, tileHeight, tileWidth);

    llvm::SmallVector<int64_t> tiledOutShape = {totalTokens / M, E, M, H};
    auto tiledOutType = ttnn::utils::RankedTensorTypeFactory::create(
        tiledATyped.getType(), tiledOutShape);

    mlir::IntegerAttr nnzAttr = nullptr;
    if (auto nnz = srcOp.getNnz()) {
      nnzAttr = rewriter.getI64IntegerAttr(*nnz);
    }

    auto tiledResult = rewriter.create<ttnn::SparseMatmulOp>(
        loc, tiledOutType, tiledATyped, inputB, sparsityMask,
        srcOp.getIsInputASparse(), srcOp.getIsInputBSparse(), nnzAttr,
        programConfigAttr,
        /*memory_config=*/nullptr, /*dtype=*/nullptr,
        /*compute_config=*/nullptr);

    auto permuted = ttir_to_ttnn::utils::generatePermute(
        cast<TypedValue<RankedTensorType>>(tiledResult.getResult()),
        {0, 2, 1, 3}, rewriter, loc);

    if (preserveLegacyFactoring) {
      auto restored = ttir_to_ttnn::utils::generateReshape(
          cast<TypedValue<RankedTensorType>>(permuted.getResult()),
          {restoredBD, restoredS, E, H}, rewriter, loc);

      postRestore.getResult().replaceAllUsesWith(restored.getResult());
      rewriter.eraseOp(postRestore);

      SmallVector<Operation *> srcResultUsers;
      for (Operation *user : srcOp.getResult().getUsers()) {
        srcResultUsers.push_back(user);
      }
      for (Operation *user : srcResultUsers) {
        if (mlir::isa<ttnn::DeallocateOp>(user)) {
          rewriter.eraseOp(user);
        }
      }
      rewriter.eraseOp(srcOp);

      if (preFlatten->use_empty()) {
        rewriter.eraseOp(preFlatten);
      }
      return success();
    }

    // Default untile output: [BD*S/M, E, M, H] -> [BD*S, E, 1, H]
    auto untiled = ttir_to_ttnn::utils::generateReshape(
        cast<TypedValue<RankedTensorType>>(permuted.getResult()),
        {totalTokens, E, 1, H}, rewriter, loc);
    rewriter.replaceOp(srcOp, untiled.getResult());
    return success();
  }

  // Not an untiled case we handle — leave as-is.
  return failure();
}

class MoeAllToAllDispatchCanonicalizeInputsRewritePattern
    : public OpRewritePattern<ttnn::AllToAllDispatchOp> {
public:
  using OpRewritePattern<ttnn::AllToAllDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::AllToAllDispatchOp op,
                                PatternRewriter &rewriter) const override {
    Value inputTensor = op.getInputTensor();
    Value expertIndices = op.getExpertIndices();
    auto inputType = cast<RankedTensorType>(inputTensor.getType());
    auto indicesType = cast<RankedTensorType>(expertIndices.getType());

    if (inputType.getRank() == 4 && indicesType.getRank() == 4) {
      return failure();
    }

    bool modified = false;
    if (inputType.getRank() == 3) {
      auto shape = inputType.getShape();
      if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) {
        return failure();
      }
      inputTensor =
          createReshape(rewriter, op.getLoc(), inputTensor,
                        {shape[0], 1, shape[1], shape[2]}, "_input_4d");
      inputType = cast<RankedTensorType>(inputTensor.getType());
      modified = true;
    }

    int64_t B = inputType.getShape()[0];
    int64_t S = inputType.getShape()[2];
    if (indicesType.getRank() == 2) {
      int64_t K = indicesType.getShape()[1];
      if (B <= 0 || S <= 0 || K <= 0) {
        return failure();
      }
      expertIndices = createReshape(rewriter, op.getLoc(), expertIndices,
                                    {B, 1, S, K}, "_expert_indices_4d");
      modified = true;
    } else if (indicesType.getRank() == 3) {
      auto shape = indicesType.getShape();
      if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) {
        return failure();
      }
      expertIndices = createReshape(rewriter, op.getLoc(), expertIndices,
                                    {shape[0], 1, shape[1], shape[2]},
                                    "_expert_indices_4d");
      modified = true;
    }

    if (!modified) {
      return failure();
    }

    auto newOp = rewriter.create<ttnn::AllToAllDispatchOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_moe_workaround"),
        cast<RankedTensorType>(op.getDispatched().getType()),
        cast<RankedTensorType>(op.getMetadata().getType()), inputTensor,
        expertIndices, op.getExpertMapping(), op.getNumDevicesAttr(),
        op.getClusterAxisAttr(), op.getMemoryConfigAttr());

    rewriter.replaceOp(op, {newOp.getDispatched(), newOp.getMetadata()});
    return success();
  }
};

class MoeAllToAllCombineCanonicalizeInputLayoutRewritePattern
    : public OpRewritePattern<ttnn::AllToAllCombineOp> {
public:
  using OpRewritePattern<ttnn::AllToAllCombineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::AllToAllCombineOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = cast<RankedTensorType>(op.getInputTensor().getType());
    auto metadataType =
        cast<RankedTensorType>(op.getExpertMetadata().getType());
    auto mappingType = cast<RankedTensorType>(op.getExpertMapping().getType());
    if (inputType.getRank() != 4 || metadataType.getRank() != 4 ||
        mappingType.getRank() != 4) {
      return failure();
    }

    auto inputShape = inputType.getShape();
    auto metadataShape = metadataType.getShape();
    auto mappingShape = mappingType.getShape();

    int64_t metadataBD = metadataShape[1];
    int64_t metadataS = metadataShape[2];
    int64_t eGlobal = mappingShape[2];
    int64_t totalDevices = mappingShape[3];
    int64_t maybeE = inputShape[2];

    int64_t eLocal = -1;
    if (totalDevices > 0 && eGlobal > 0 && eGlobal % totalDevices == 0) {
      eLocal = eGlobal / totalDevices;
    }
    bool expertDimMatches =
        (maybeE == eGlobal) || (eLocal > 0 && maybeE == eLocal);
    bool looksLikeBdseh = (inputShape[0] == metadataBD) &&
                          (inputShape[1] == metadataS) && expertDimMatches;
    if (!looksLikeBdseh) {
      return failure();
    }

    DenseI64ArrayAttr permutation = rewriter.getDenseI64ArrayAttr({2, 0, 1, 3});
    RankedTensorType permutedType =
        ttnn::utils::RankedTensorTypeFactory::create(
            inputType,
            {inputShape[2], inputShape[0], inputShape[1], inputShape[3]});

    auto permute = rewriter.create<ttnn::PermuteOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_to_ebdsh"),
        permutedType, op.getInputTensor(), permutation,
        ttnn::MemoryConfigAttr(), FloatAttr());

    // Preserve decode behavior while avoiding stale shard-dim values from the
    // pre-canonicalized [BD, S, E, H] layout. Once converted to [E, BD, S, H],
    // output_shard_dim should depend on S (decode: S==1 -> 2, else 1).
    IntegerAttr outputShardDimAttr = op.getOutputShardDimAttr();
    if (metadataS > 0) {
      int64_t canonicalOutputShardDim = (metadataS == 1) ? 2 : 1;
      outputShardDimAttr = rewriter.getI64IntegerAttr(canonicalOutputShardDim);
    }

    auto newOp = rewriter.create<ttnn::AllToAllCombineOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_moe_workaround"),
        cast<RankedTensorType>(op.getResult().getType()), permute.getResult(),
        op.getExpertMetadata(), op.getExpertMapping(), op.getNumDevicesAttr(),
        op.getClusterAxisAttr(), op.getNumExpertsPerTokAttr(),
        outputShardDimAttr, op.getMemoryConfigAttr());

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

class MoeExpertTokenRemapCanonicalizeTopkRewritePattern
    : public OpRewritePattern<ttnn::MoeExpertTokenRemapOp> {
public:
  using OpRewritePattern<ttnn::MoeExpertTokenRemapOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::MoeExpertTokenRemapOp op,
                                PatternRewriter &rewriter) const override {
    Value topkTensor = op.getTopkTensor();
    auto topkType = cast<RankedTensorType>(topkTensor.getType());
    if (topkType.getRank() != 2 && topkType.getRank() != 3) {
      return failure();
    }

    auto metadataType =
        cast<RankedTensorType>(op.getExpertMetadata().getType());
    if (metadataType.getRank() != 4) {
      return failure();
    }
    int64_t BD = metadataType.getShape()[1];
    int64_t S = metadataType.getShape()[2];
    if (BD <= 0 || S <= 0) {
      return failure();
    }

    auto repeatAlongDim0 = [&](Value value, int64_t repeats,
                               StringRef suffix) -> FailureOr<Value> {
      if (repeats <= 1) {
        return value;
      }
      auto valueType = cast<RankedTensorType>(value.getType());
      auto outShape = llvm::SmallVector<int64_t>(valueType.getShape().begin(),
                                                 valueType.getShape().end());
      if (outShape[0] <= 0) {
        return failure();
      }
      outShape[0] *= repeats;
      auto outType =
          ttnn::utils::RankedTensorTypeFactory::create(valueType, outShape);
      SmallVector<Value> repeatedInputs(repeats, value);
      return rewriter
          .create<ttnn::ConcatOp>(
              ttmlir::utils::appendLocationSuffix(op.getLoc(), suffix), outType,
              repeatedInputs, /*dim=*/0, ttnn::MemoryConfigAttr())
          .getResult();
    };

    if (topkType.getRank() == 2) {
      int64_t BS = topkType.getShape()[0];
      int64_t E = topkType.getShape()[1];
      if (BS <= 0 || E <= 0 || BS % S != 0) {
        return failure();
      }
      int64_t B = BS / S;
      if (B <= 0 || BD % B != 0) {
        return failure();
      }
      int64_t repeats = BD / B;

      topkTensor = createReshape(rewriter, op.getLoc(), topkTensor, {B, S, E},
                                 "_topk_3d");
      FailureOr<Value> repeated =
          repeatAlongDim0(topkTensor, repeats, "_topk_repeat_dim0");
      if (failed(repeated)) {
        return failure();
      }
      topkTensor = *repeated;
      topkTensor = createReshape(rewriter, op.getLoc(), topkTensor,
                                 {1, BD, S, E}, "_topk_4d");
    } else {
      int64_t B = topkType.getShape()[0];
      int64_t STopk = topkType.getShape()[1];
      int64_t E = topkType.getShape()[2];
      if (B <= 0 || STopk <= 0 || E <= 0 || BD % B != 0) {
        return failure();
      }
      int64_t repeats = BD / B;
      FailureOr<Value> repeated =
          repeatAlongDim0(topkTensor, repeats, "_topk_repeat_dim0");
      if (failed(repeated)) {
        return failure();
      }
      topkTensor = *repeated;
      topkTensor = createReshape(rewriter, op.getLoc(), topkTensor,
                                 {1, BD, STopk, E}, "_topk_4d");
    }

    auto newOp = rewriter.create<ttnn::MoeExpertTokenRemapOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_moe_workaround"),
        cast<RankedTensorType>(op.getMapping().getType()),
        cast<RankedTensorType>(op.getReduced().getType()), topkTensor,
        op.getExpertMapping(), op.getExpertMetadata(),
        op.getReductionSizeAttr(), op.getMemoryConfigAttr());
    rewriter.replaceOp(op, {newOp.getMapping(), newOp.getReduced()});
    return success();
  }
};

// Runtime workaround for decode-style all_to_all_combine:
// output_shard_dim=2 with result shape [K, B, 1, H] can fail in runtime.
// Rewrite to output_shard_dim=1 with an intermediate [K, 1, B, H] output,
// then reshape back to the original result type.
class MoeAllToAllCombineDecodeShardDimRewritePattern
    : public OpRewritePattern<ttnn::AllToAllCombineOp> {
public:
  using OpRewritePattern<ttnn::AllToAllCombineOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::AllToAllCombineOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType outputType = op.getResult().getType();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    if (op.getOutputShardDim() != 2 || outputType.getRank() != 4 ||
        outputShape[2] != 1) {
      return failure();
    }

    RankedTensorType workaroundType = RankedTensorType::get(
        {outputShape[0], outputShape[2], outputShape[1], outputShape[3]},
        outputType.getElementType(), outputType.getEncoding());

    auto combineOp = rewriter.create<ttnn::AllToAllCombineOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_moe_workaround"),
        workaroundType, op.getInputTensor(), op.getExpertMetadata(),
        op.getExpertMapping(), op.getNumDevicesAttr(), op.getClusterAxisAttr(),
        op.getNumExpertsPerTokAttr(),
        rewriter.getI64IntegerAttr(/*output_shard_dim=*/1),
        op.getMemoryConfigAttr());

    SmallVector<int32_t> reshapeShape;
    reshapeShape.reserve(outputShape.size());
    for (int64_t dim : outputShape) {
      reshapeShape.push_back(static_cast<int32_t>(dim));
    }

    rewriter.replaceOpWithNewOp<ttnn::ReshapeOp>(
        op, outputType, combineOp.getResult(),
        rewriter.getI32ArrayAttr(reshapeShape), op.getMemoryConfigAttr());
    return success();
  }
};

} // namespace

void populateMoeWorkaroundPatterns(RewritePatternSet &patterns) {
  patterns.add<SparseMatmulTileDimsRewritePattern,
               MoeAllToAllDispatchCanonicalizeInputsRewritePattern,
               MoeAllToAllCombineCanonicalizeInputLayoutRewritePattern,
               MoeExpertTokenRemapCanonicalizeTopkRewritePattern,
               MoeAllToAllCombineDecodeShardDimRewritePattern>(
      patterns.getContext());
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
