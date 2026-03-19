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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

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

// Return the unique user of the requested type while tolerating other
// non-Deallocate users. This is useful for already-tiled MoE chains where the
// legacy path may keep extra users alive, but we still want to rewrite the
// matching sparse_matmul->activation->sparse_matmul subgraph.
template <typename OpTy>
static OpTy getUniqueUserOfType(Value value) {
  OpTy found;
  for (Operation *user : value.getUsers()) {
    if (mlir::isa<ttnn::DeallocateOp>(user)) {
      continue;
    }
    auto op = mlir::dyn_cast<OpTy>(user);
    if (!op) {
      continue;
    }
    if (found) {
      return nullptr;
    }
    found = op;
  }
  return found;
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

static bool isPointwiseRankPreservingOp(Operation *op) {
  if (!mlir::isa<ttnn::AddOp, ttnn::MultiplyOp, ttnn::ClampScalarOp,
                 ttnn::SigmoidOp, ttnn::SiluOp>(op)) {
    return false;
  }
  if (op->getNumResults() != 1) {
    return false;
  }
  auto resultType = mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType) {
    return false;
  }
  for (Value operand : op->getOperands()) {
    auto operandType = mlir::dyn_cast<RankedTensorType>(operand.getType());
    if (operandType && operandType.getRank() == resultType.getRank()) {
      return true;
    }
  }
  return false;
}

// Walk forward through pointwise rank-preserving ops, recording the shortest
// distance to each reachable op. The reachable value set is used to determine
// whether a candidate common op directly merges both slice branches.
static DenseMap<Operation *, int64_t>
collectForwardPointwiseDistances(Value start, DenseSet<Value> &reachable) {
  DenseMap<Operation *, int64_t> opDistance;
  DenseMap<Value, int64_t> valueDistance;
  SmallVector<Value> worklist = {start};
  valueDistance[start] = 0;

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    int64_t distance = valueDistance.lookup(v);
    reachable.insert(v);

    for (Operation *user : v.getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(user) ||
          !isPointwiseRankPreservingOp(user)) {
        continue;
      }

      int64_t newDistance = distance + 1;
      auto opIt = opDistance.find(user);
      if (opIt != opDistance.end() && opIt->second <= newDistance) {
        continue;
      }
      opDistance[user] = newDistance;

      for (Value result : user->getResults()) {
        auto valueIt = valueDistance.find(result);
        if (valueIt != valueDistance.end() && valueIt->second <= newDistance) {
          continue;
        }
        valueDistance[result] = newDistance;
        worklist.push_back(result);
      }
    }
  }

  return opDistance;
}

// Find the nearest common pointwise rank-preserving op reachable from both
// slices. Prefer an op that directly consumes one value from each branch when
// available, otherwise fall back to the earliest common pointwise user.
static Operation *
findNearestCommonPointwiseUser(MutableArrayRef<ttnn::SliceStaticOp> slices) {
  if (slices.size() != 2) {
    return nullptr;
  }
  DenseSet<Value> reach0, reach1;
  auto dist0 = collectForwardPointwiseDistances(slices[0].getResult(), reach0);
  auto dist1 = collectForwardPointwiseDistances(slices[1].getResult(), reach1);

  Operation *bestDirectMerge = nullptr;
  int64_t bestDirectMaxDist = 0;
  int64_t bestDirectSumDist = 0;
  Operation *bestCommon = nullptr;
  int64_t bestCommonMaxDist = 0;
  int64_t bestCommonSumDist = 0;

  for (const auto &it : dist0) {
    Operation *op = it.first;
    int64_t distFrom0 = it.second;
    auto dist1It = dist1.find(op);
    if (dist1It == dist1.end()) {
      continue;
    }
    int64_t distFrom1 = dist1It->second;
    int64_t maxDist = std::max(distFrom0, distFrom1);
    int64_t sumDist = distFrom0 + distFrom1;

    bool has0OnlyOperand = false;
    bool has1OnlyOperand = false;
    for (Value operand : op->getOperands()) {
      bool from0 = reach0.count(operand);
      bool from1 = reach1.count(operand);
      has0OnlyOperand |= from0 && !from1;
      has1OnlyOperand |= from1 && !from0;
    }
    bool directlyMergesBranches = has0OnlyOperand && has1OnlyOperand;

    auto isBetterCandidate = [&](Operation *bestOp, int64_t bestMaxDist,
                                 int64_t bestSumDist) {
      return !bestOp || maxDist < bestMaxDist ||
             (maxDist == bestMaxDist && sumDist < bestSumDist);
    };

    if (directlyMergesBranches) {
      if (isBetterCandidate(bestDirectMerge, bestDirectMaxDist,
                            bestDirectSumDist)) {
        bestDirectMerge = op;
        bestDirectMaxDist = maxDist;
        bestDirectSumDist = sumDist;
      }
    } else if (isBetterCandidate(bestCommon, bestCommonMaxDist,
                                 bestCommonSumDist)) {
      bestCommon = op;
      bestCommonMaxDist = maxDist;
      bestCommonSumDist = sumDist;
    }
  }

  return bestDirectMerge ? bestDirectMerge : bestCommon;
}

// Collect all ops between the slices and finalCommonOp (exclusive of both
// slices and finalCommonOp) and return them in topological order.
static SmallVector<Operation *>
collectActivationSubgraph(MutableArrayRef<ttnn::SliceStaticOp> slices,
                          Operation *finalCommonOp) {
  DenseSet<Operation *> subgraphOps;
  SmallVector<Value> worklist;
  for (auto slice : slices) {
    worklist.push_back(slice.getResult());
  }
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (Operation *user : v.getUsers()) {
      if (mlir::isa<ttnn::DeallocateOp>(user)) {
        continue;
      }
      if (user == finalCommonOp) {
        continue;
      }
      if (!isPointwiseRankPreservingOp(user)) {
        continue;
      }
      if (subgraphOps.insert(user).second) {
        for (Value result : user->getResults()) {
          worklist.push_back(result);
        }
      }
    }
  }

  SmallVector<Operation *> topoOrder;
  DenseSet<Operation *> sorted;
  bool progress = true;
  while (progress && sorted.size() < subgraphOps.size()) {
    progress = false;
    for (Operation *op : subgraphOps) {
      if (sorted.count(op)) {
        continue;
      }
      bool ready = true;
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (def && subgraphOps.count(def) && !sorted.count(def)) {
          ready = false;
          break;
        }
      }
      if (ready) {
        topoOrder.push_back(op);
        sorted.insert(op);
        progress = true;
      }
    }
  }
  return topoOrder;
}

// Clone all ops from slices through finalCommonOp with new shapes, returning
// the new finalCommonOp value. The shape mapping replaces the old flattened
// [BD*dimB*M, E, 1, X] layout with an M-preserving [dimA*dimB, E, M, X].
static Value cloneActivationChainWithNewShapes(
    MutableArrayRef<ttnn::SliceStaticOp> slices, Value newPreSliceInput,
    int64_t dimA, int64_t dimB, int64_t E, int64_t M, int64_t expandedN,
    SmallVector<Operation *> &topoOrder, Operation *finalCommonOp,
    PatternRewriter &rewriter) {

  auto preSliceInputType =
      mlir::cast<RankedTensorType>(newPreSliceInput.getType());
  int64_t reducedN = expandedN / 2;
  SmallVector<int64_t> actShape = {dimA * dimB, E, M, reducedN};
  auto sliceOutType =
      ttnn::utils::RankedTensorTypeFactory::create(preSliceInputType, actShape);

  auto makeNewType = [&](RankedTensorType oldType) -> RankedTensorType {
    return ttnn::utils::RankedTensorTypeFactory::create(
        oldType, {dimA * dimB, E, M, oldType.getShape()[3]});
  };

  DenseMap<Value, Value> valueMapping;

  for (auto slice : slices) {
    auto beginsOpt =
        ttmlir::utils::getIntegerVector<int64_t>(slice.getBegins());
    if (!beginsOpt) {
      llvm::consumeError(beginsOpt.takeError());
      continue;
    }
    int32_t startN = static_cast<int32_t>((*beginsOpt)[3]);
    SmallVector<int32_t> begins = {0, 0, 0, startN};
    SmallVector<int32_t> ends = {static_cast<int32_t>(dimA * dimB),
                                 static_cast<int32_t>(E),
                                 static_cast<int32_t>(M),
                                 static_cast<int32_t>(expandedN)};
    SmallVector<int32_t> steps = {1, 1, 1, 2};
    Value newSlice =
        rewriter
            .create<ttnn::SliceStaticOp>(
                slice.getLoc(), sliceOutType, newPreSliceInput,
                rewriter.getI32ArrayAttr(begins),
                rewriter.getI32ArrayAttr(ends),
                rewriter.getI32ArrayAttr(steps))
            .getResult();
    valueMapping[slice.getResult()] = newSlice;
  }

  for (Operation *op : topoOrder) {
    SmallVector<Value> newOperands;
    for (Value operand : op->getOperands()) {
      auto it = valueMapping.find(operand);
      newOperands.push_back(it != valueMapping.end() ? it->second : operand);
    }
    auto oldType = mlir::cast<RankedTensorType>(op->getResult(0).getType());
    Value newResult =
        cloneLikeWithNewOperands(rewriter, op, newOperands, makeNewType(oldType));
    valueMapping[op->getResult(0)] = newResult;
  }

  // Clone finalCommonOp separately (not in topoOrder).
  SmallVector<Value> finalOpOperands;
  for (Value operand : finalCommonOp->getOperands()) {
    auto it = valueMapping.find(operand);
    finalOpOperands.push_back(it != valueMapping.end() ? it->second : operand);
  }
  auto finalOpOldType =
      mlir::cast<RankedTensorType>(finalCommonOp->getResult(0).getType());
  return cloneLikeWithNewOperands(rewriter, finalCommonOp, finalOpOperands,
                                  makeNewType(finalOpOldType));
}

// Build the common M-preserving pre-slice tensor from the tiled gate-up sparse
// output. Accepts either [dimA, dimB, E, M, N] or [dimA, dimB, 1, E, M, N] and
// produces [dimA*dimB, E, M, N] ready for slicing. Keep this chain minimal so
// the resulting graph stays close to the old frontend-modeled path from main.
static Value buildMPreservingPreSliceChain(Value tiledGateUpResult, Value bias,
                                           Operation *origAdd, int64_t dimA,
                                           int64_t dimB, int64_t E, int64_t M,
                                           int64_t expandedN,
                                           PatternRewriter &rewriter,
                                           Location loc) {
  auto preSliceReshape = ttir_to_ttnn::utils::generateReshape(
      cast<TypedValue<RankedTensorType>>(tiledGateUpResult),
      {dimA * dimB, E, M, expandedN},
      rewriter, loc);

  return cloneLikeWithNewOperands(rewriter, origAdd,
                                  {preSliceReshape.getResult(), bias},
                                  preSliceReshape.getType());
}

// Verify that the two slices have step=2 on the last dimension and that their
// begins differ only in the last element (0 vs 1).
static bool verifyHalfSlices(MutableArrayRef<ttnn::SliceStaticOp> slices) {
  if (slices.size() != 2) {
    return false;
  }
  for (auto slice : slices) {
    auto beginsOpt =
        ttmlir::utils::getIntegerVector<int64_t>(slice.getBegins());
    auto stepsOpt = ttmlir::utils::getIntegerVector<int64_t>(slice.getStep());
    if (!beginsOpt) {
      llvm::consumeError(beginsOpt.takeError());
      if (stepsOpt) {
        (void)*stepsOpt;
      } else {
        llvm::consumeError(stepsOpt.takeError());
      }
      return false;
    }
    if (!stepsOpt) {
      llvm::consumeError(stepsOpt.takeError());
      return false;
    }
    auto &begins = *beginsOpt;
    auto &steps = *stepsOpt;
    if (begins.size() != 4 || steps.size() != 4 || steps[3] != 2) {
      return false;
    }
    if (begins[3] != 0 && begins[3] != 1) {
      return false;
    }
  }
  return true;
}

// Rewrite already-tiled gate-up activation chain from flattened layout:
//
//   sparse -> reshape(5D) -> permute -> reshape(flat) -> add -> activation
//   -> reshape -> permute
//
// to an M-preserving layout:
//
//   sparse -> reshape(5D) -> reshape(4D) -> add(4D) -> activation
//
// Handles both clamp-based and standard SwiGLU activation patterns by using
// generic graph traversal to find the nearest common pointwise merge op.
static LogicalResult
rewriteAlreadyTiledGateUpActivationChain(ttnn::SparseMatmulOp srcOp,
                                         PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  auto gateUpReshape = getUniqueUserOfType<ttnn::ReshapeOp>(srcOp.getResult());
  if (!gateUpReshape) {
    return failure();
  }

  auto gateUpShape = gateUpReshape.getType().getShape();
  if (gateUpShape.size() != 5) {
    return failure();
  }

  auto preFlattenPermute =
      getUniqueUserOfType<ttnn::PermuteOp>(gateUpReshape.getResult());
  if (!isPermutation(preFlattenPermute, {0, 1, 3, 2, 4})) {
    return failure();
  }

  auto flattened =
      getUniqueUserOfType<ttnn::ReshapeOp>(preFlattenPermute.getResult());
  if (!flattened) {
    return failure();
  }

  auto flattenedShape = flattened.getType().getShape();
  if (flattenedShape.size() != 4 || flattenedShape[2] != 1) {
    return failure();
  }

  auto preActAdd = getUniqueUserOfType<ttnn::AddOp>(flattened.getResult());
  if (!preActAdd) {
    return failure();
  }

  SmallVector<ttnn::SliceStaticOp> preActSlices;
  for (Operation *user : preActAdd.getResult().getUsers()) {
    if (mlir::isa<ttnn::DeallocateOp>(user)) {
      continue;
    }
    auto slice = mlir::dyn_cast<ttnn::SliceStaticOp>(user);
    if (!slice) {
      continue;
    }
    preActSlices.push_back(slice);
  }
  if (!verifyHalfSlices(preActSlices)) {
    return failure();
  }

  Operation *finalMergeOp = findNearestCommonPointwiseUser(preActSlices);
  if (!finalMergeOp) {
    return failure();
  }

  auto finalReshape =
      getUniqueUserOfType<ttnn::ReshapeOp>(finalMergeOp->getResult(0));
  auto finalPermute =
      finalReshape
          ? getUniqueUserOfType<ttnn::PermuteOp>(finalReshape.getResult())
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
  if (flattenedShape[1] != E) {
    return failure();
  }

  Value addBias = preActAdd.getLhs() == flattened.getResult()
                      ? preActAdd.getRhs()
                      : preActAdd.getLhs();
  auto addBiasType = mlir::cast<RankedTensorType>(addBias.getType());
  auto addBiasShape = addBiasType.getShape();
  if (addBiasShape.size() != 4 || addBiasShape[0] != 1 ||
      addBiasShape[1] != E || addBiasShape[2] != 1 ||
      addBiasShape[3] != expandedN) {
    return failure();
  }

  // Validate activation subgraph BEFORE creating any new ops to avoid
  // dangling ops on failure.
  auto topoOrder = collectActivationSubgraph(preActSlices, finalMergeOp);
  for (Operation *op : topoOrder) {
    auto type = mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!type || type.getShape().size() != 4) {
      return failure();
    }
  }

  rewriter.setInsertionPoint(preActAdd);
  auto loc = srcOp.getLoc();

  Value preSliceInput = buildMPreservingPreSliceChain(
      srcOp.getResult(), addBias, preActAdd.getOperation(), dimA, dimB,
      E, M, expandedN, rewriter, loc);

  Value newFinalMerge = cloneActivationChainWithNewShapes(
      preActSlices, preSliceInput, dimA, dimB, E, M, expandedN, topoOrder,
      finalMergeOp, rewriter);

  rewriter.replaceOp(finalPermute, newFinalMerge);
  return success();
}

// Erase an op and all its DeallocateOp users from the IR.
// Returns false if the op still has non-DeallocateOp users (skip erasure).
static bool eraseOpAndDeallocates(PatternRewriter &rewriter, Operation *op) {
  for (Value result : op->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (!mlir::isa<ttnn::DeallocateOp>(user)) {
        return false;
      }
    }
  }
  for (Value result : op->getResults()) {
    for (Operation *user : llvm::make_early_inc_range(result.getUsers())) {
      rewriter.eraseOp(user);
    }
  }
  rewriter.eraseOp(op);
  return true;
}

// Rewrite the gate-up chain for the untiled case. Instead of creating an
// untile chain and hoping the already-tiled rewriter matches later, this
// directly walks the original 4D chain:
//
//   srcOp -> reshape [BD,S,E,N] -> add(bias) -> slices -> activation
//   -> finalMergeOp -> reshape -> ...
//
// and rewrites it using the tiled sparse_matmul result.
static LogicalResult
rewriteUntiledGateUpChain(ttnn::SparseMatmulOp origSrcOp,
                          Value tiledSparseResult, int64_t dimA, int64_t dimB,
                          int64_t E, int64_t M, int64_t expandedN,
                          PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  auto origReshape =
      getSingleUserOfType<ttnn::ReshapeOp>(origSrcOp.getResult());
  if (!origReshape) {
    return failure();
  }
  auto origReshapeShape = origReshape.getType().getShape();
  if (origReshapeShape.size() != 4) {
    return failure();
  }

  int64_t BD = origReshapeShape[0];
  int64_t S = origReshapeShape[1];
  if (origReshapeShape[2] != E || origReshapeShape[3] != expandedN) {
    return failure();
  }
  if (BD * S != dimA * dimB * M) {
    return failure();
  }

  auto origAdd = getSingleUserOfType<ttnn::AddOp>(origReshape.getResult());
  if (!origAdd) {
    return failure();
  }

  Value origBias = origAdd.getLhs() == origReshape.getResult()
                       ? origAdd.getRhs()
                       : origAdd.getLhs();
  if (origAdd.getLhs() != origReshape.getResult() &&
      origAdd.getRhs() != origReshape.getResult()) {
    return failure();
  }

  auto biasType = mlir::cast<RankedTensorType>(origBias.getType());
  auto biasShape = biasType.getShape();
  if (biasShape.size() != 4) {
    return failure();
  }

  SmallVector<ttnn::SliceStaticOp> origSlices;
  for (Operation *user : origAdd.getResult().getUsers()) {
    if (mlir::isa<ttnn::DeallocateOp>(user)) {
      continue;
    }
    auto slice = mlir::dyn_cast<ttnn::SliceStaticOp>(user);
    if (!slice) {
      return failure();
    }
    origSlices.push_back(slice);
  }
  if (!verifyHalfSlices(origSlices)) {
    return failure();
  }

  Operation *finalMergeOp = findNearestCommonPointwiseUser(origSlices);
  if (!finalMergeOp) {
    return failure();
  }

  int64_t reducedN = expandedN / 2;
  auto finalMergeType =
      mlir::cast<RankedTensorType>(finalMergeOp->getResult(0).getType());
  auto finalMergeShape = finalMergeType.getShape();
  if (finalMergeShape.size() != 4 || finalMergeShape[0] != BD ||
      finalMergeShape[1] != S || finalMergeShape[2] != E ||
      finalMergeShape[3] != reducedN) {
    return failure();
  }

  auto topoOrder = collectActivationSubgraph(origSlices, finalMergeOp);
  for (Operation *op : topoOrder) {
    auto type = mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!type || type.getShape().size() != 4) {
      return failure();
    }
  }

  rewriter.setInsertionPoint(origAdd);
  auto loc = origSrcOp.getLoc();

  Value preSliceInput = buildMPreservingPreSliceChain(
      tiledSparseResult, origBias, origAdd.getOperation(), dimA, dimB, E, M,
      expandedN, rewriter, loc);

  Value newFinalMergeVal = cloneActivationChainWithNewShapes(
      origSlices, preSliceInput, dimA, dimB, E, M, expandedN, topoOrder,
      finalMergeOp, rewriter);

  // Back-permute: [dimA*dimB, E, M, reducedN] -> [BD, S, E, reducedN]
  auto backPermute = ttir_to_ttnn::utils::generatePermute(
      cast<TypedValue<RankedTensorType>>(newFinalMergeVal), {0, 2, 1, 3},
      rewriter, loc);

  rewriter.replaceOp(finalMergeOp, backPermute.getResult());

  // Clean up the dead original chain in reverse topological order.
  for (auto it = topoOrder.rbegin(); it != topoOrder.rend(); ++it) {
    eraseOpAndDeallocates(rewriter, *it);
  }
  for (auto slice : origSlices) {
    eraseOpAndDeallocates(rewriter, slice.getOperation());
  }
  eraseOpAndDeallocates(rewriter, origAdd.getOperation());
  eraseOpAndDeallocates(rewriter, origReshape.getOperation());

  return success();
}

// Rewrite already-tiled down-projection post-sparse chain:
//
//   sparse -> permute -> reshape(flatten) -> add -> reshape -> permute
//
// to:
//
//   sparse -> add -> reshape -> permute -> reshape
//
// This keeps the expert axis resident through the add and removes two
// expensive layout transforms around the down-projection path.
static LogicalResult
rewriteAlreadyTiledDownProjectionPostSparseChain(ttnn::SparseMatmulOp srcOp,
                                                 PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  auto srcType = cast<RankedTensorType>(srcOp.getResult().getType());
  if (srcType.getRank() != 4) {
    return failure();
  }
  auto srcShape = srcType.getShape();
  int64_t TSrc = srcShape[0];
  int64_t EDown = srcShape[1];
  int64_t MDown = srcShape[2];
  int64_t HDown = srcShape[3];
  if (TSrc <= 0 || EDown <= 0 || MDown <= 0 || HDown <= 0) {
    return failure();
  }

  auto downPermute = getSingleUserOfType<ttnn::PermuteOp>(srcOp.getResult());
  if (!isPermutation(downPermute, {0, 2, 1, 3})) {
    return failure();
  }

  auto downFlatten =
      getSingleUserOfType<ttnn::ReshapeOp>(downPermute.getResult());
  if (!downFlatten) {
    return failure();
  }

  auto downAdd = getSingleUserOfType<ttnn::AddOp>(downFlatten.getResult());
  if (!downAdd) {
    return failure();
  }
  Value bias = downAdd.getLhs() == downFlatten.getResult()   ? downAdd.getRhs()
               : downAdd.getRhs() == downFlatten.getResult() ? downAdd.getLhs()
                                                             : Value();
  if (!bias) {
    return failure();
  }

  auto downReshaped = getSingleUserOfType<ttnn::ReshapeOp>(downAdd.getResult());
  auto downPermuteBack =
      downReshaped
          ? getSingleUserOfType<ttnn::PermuteOp>(downReshaped.getResult())
          : nullptr;
  if (!downReshaped || !isPermutation(downPermuteBack, {2, 0, 1, 3})) {
    return failure();
  }

  auto downFlattenType = cast<RankedTensorType>(downFlatten.getType());
  auto downReshapedType = cast<RankedTensorType>(downReshaped.getType());
  auto downPermuteBackType = cast<RankedTensorType>(downPermuteBack.getType());
  if (downFlattenType.getRank() != 4 || downReshapedType.getRank() != 4 ||
      downPermuteBackType.getRank() != 4) {
    return failure();
  }
  auto downFlattenShape = downFlattenType.getShape();
  auto downReshapedShape = downReshapedType.getShape();
  auto downPermuteBackShape = downPermuteBackType.getShape();

  int64_t TDown = downReshapedShape[0];
  int64_t SDown = downReshapedShape[1];
  int64_t EReshaped = downReshapedShape[2];
  int64_t HReshaped = downReshapedShape[3];

  auto hasExpectedBiasShape = [&](Value maybeBias) {
    auto biasType = dyn_cast<RankedTensorType>(maybeBias.getType());
    if (!biasType || biasType.getRank() != 4) {
      return false;
    }
    auto biasShape = biasType.getShape();
    return biasShape[0] == 1 && biasShape[1] == EDown && biasShape[2] == 1 &&
           biasShape[3] == HDown;
  };

  if (TDown <= 0 || MDown <= 0 || EDown != EReshaped || HDown != HReshaped ||
      SDown <= 0 || SDown % MDown != 0 || !hasExpectedBiasShape(bias) ||
      downFlattenShape[0] != TSrc * MDown || downFlattenShape[1] != EDown ||
      downFlattenShape[2] != 1 || downFlattenShape[3] != HDown) {
    return failure();
  }

  int64_t groupedS = SDown / MDown;
  if (TSrc != TDown * groupedS || downPermuteBackShape[0] != EDown ||
      downPermuteBackShape[1] != TDown || downPermuteBackShape[2] != SDown ||
      downPermuteBackShape[3] != HDown) {
    return failure();
  }

  rewriter.setInsertionPoint(downAdd);

  Value newDownAdd = cloneLikeWithNewOperands(
      rewriter, downAdd.getOperation(), {srcOp.getResult(), bias}, srcType);

  auto grouped = ttir_to_ttnn::utils::generateReshape(
      cast<TypedValue<RankedTensorType>>(newDownAdd),
      {TDown, groupedS, EDown, MDown, HDown}, rewriter, srcOp.getLoc());
  auto regroupPermute = ttir_to_ttnn::utils::generatePermute(
      cast<TypedValue<RankedTensorType>>(grouped.getResult()), {2, 0, 1, 3, 4},
      rewriter, srcOp.getLoc());
  auto regroupReshape = ttir_to_ttnn::utils::generateReshape(
      cast<TypedValue<RankedTensorType>>(regroupPermute.getResult()),
      {EDown, TDown, SDown, HDown}, rewriter, srcOp.getLoc());

  rewriter.replaceOp(downPermuteBack, regroupReshape.getResult());
  return success();
}

LogicalResult SparseMatmulTileDimsRewritePattern::matchAndRewrite(
    ttnn::SparseMatmulOp srcOp, PatternRewriter &rewriter) const {

  auto a = srcOp.getA();
  auto b = srcOp.getB();
  auto sparsity = srcOp.getSparsity();

  auto aType = mlir::cast<RankedTensorType>(a.getType());
  auto bType = mlir::cast<RankedTensorType>(b.getType());
  auto spType = mlir::cast<RankedTensorType>(sparsity.getType());
  auto aShape = aType.getShape();
  int64_t mDim = aShape[aShape.size() - 2];

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
      // Handle already-tiled gate-up subgraph where the preceding graph
      // flattened [dimA, dimB, M] to [dimA*dimB*M, E, 1] before SwiGLU.
      // Rewrite to keep the M axis through elementwise activation.
      return rewriteAlreadyTiledGateUpActivationChain(srcOp, rewriter);
    }

    if (mDim > tileHeight) {
      return failure();
    }

    // Gate-up: A=[BD, S, 1, H], sparsity=[1, 1, BD*S/M, E]
    auto bShape = bType.getShape();
    auto spShape = spType.getShape();

    int64_t BD = aShape[0];
    int64_t S = aShape[1];
    int64_t H = aShape[3];
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
      tiledA = ttir_to_ttnn::utils::generateReshape(a, {BD, S / M, M, H},
                                                    rewriter, loc)
                   .getResult();
    } else {
      auto reshaped = ttir_to_ttnn::utils::generateReshape(a, {BD / M, M, S, H},
                                                           rewriter, loc);
      tiledA = ttir_to_ttnn::utils::generatePermute(
                   cast<TypedValue<RankedTensorType>>(reshaped.getResult()),
                   {0, 2, 1, 3}, rewriter, loc)
                   .getResult();
    }

    // Tile sparsity: [1, 1, BD*S/M, E] -> [dimA, dimB, 1, E]
    auto tiledSparsity = ttir_to_ttnn::utils::generateReshape(
        sparsity, {dimA, dimB, 1, E}, rewriter, loc);

    // Create tiled sparse_matmul
    auto tiledATyped = cast<TypedValue<RankedTensorType>>(tiledA);
    auto deviceAttr = ttcore::lookupDevice(srcOp);
    auto programConfigAttr =
        createProgramConfig(rewriter.getContext(), tiledATyped.getType(), bType,
                            deviceAttr, tileHeight, tileWidth);

    llvm::SmallVector<int64_t> tiledOutShape = {dimA, dimB, 1, E, M, N};
    auto tiledOutType = ttnn::utils::RankedTensorTypeFactory::create(
        tiledATyped.getType(), tiledOutShape);

    mlir::IntegerAttr nnzAttr = nullptr;
    if (auto nnz = srcOp.getNnz()) {
      nnzAttr = rewriter.getI64IntegerAttr(*nnz);
    }

    auto tiledResult = rewriter.create<ttnn::SparseMatmulOp>(
        loc, tiledOutType, tiledATyped, b,
        cast<TypedValue<RankedTensorType>>(tiledSparsity.getResult()),
        srcOp.getIsInputASparse(), srcOp.getIsInputBSparse(), nnzAttr,
        programConfigAttr, /*memory_config=*/nullptr, /*dtype=*/nullptr,
        /*compute_config=*/nullptr);

    // Try to rewrite the entire gate-up chain (add + activation) directly,
    // avoiding the intermediate untile chain that the already-tiled rewriter
    // cannot match due to different chain shapes.
    if (succeeded(rewriteUntiledGateUpChain(srcOp, tiledResult.getResult(),
                                            dimA, dimB, E, M, N, rewriter))) {
      eraseOpAndDeallocates(rewriter, srcOp);
      return success();
    }

    // Fallback: untile output while preserving the legacy gate-up factoring.
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
    if (mDim == tileHeight) {
      return rewriteAlreadyTiledDownProjectionPostSparseChain(srcOp, rewriter);
    }

    if (mDim > tileHeight) {
      return failure();
    }

    // Down: A=[BD*S, E, 1, inter], sparsity=[1, 1, BD*S/M, E]
    auto bShape = bType.getShape();
    auto spShape = spType.getShape();

    int64_t totalTokens = aShape[0];
    int64_t E = aShape[1];
    int64_t inter = aShape[3];
    int64_t H = bShape[3];
    int64_t reducedTokens = spShape[2];
    int64_t M = totalTokens / reducedTokens;

    // Tile A: [BD*S, E, 1, inter] -> [BD*S/M, E, M, inter]
    auto reshaped = ttir_to_ttnn::utils::generateReshape(
        a, {totalTokens / M, M, E, inter}, rewriter, loc);
    auto tiledA = ttir_to_ttnn::utils::generatePermute(
        cast<TypedValue<RankedTensorType>>(reshaped.getResult()), {0, 2, 1, 3},
        rewriter, loc);

    auto tiledATyped = cast<TypedValue<RankedTensorType>>(tiledA.getResult());
    auto deviceAttr = ttcore::lookupDevice(srcOp);
    auto programConfigAttr =
        createProgramConfig(rewriter.getContext(), tiledATyped.getType(), bType,
                            deviceAttr, tileHeight, tileWidth);

    llvm::SmallVector<int64_t> tiledOutShape = {totalTokens / M, E, M, H};
    auto tiledOutType = ttnn::utils::RankedTensorTypeFactory::create(
        tiledATyped.getType(), tiledOutShape);

    mlir::IntegerAttr nnzAttr = nullptr;
    if (auto nnz = srcOp.getNnz()) {
      nnzAttr = rewriter.getI64IntegerAttr(*nnz);
    }

    auto tiledResult = rewriter.create<ttnn::SparseMatmulOp>(
        loc, tiledOutType, tiledATyped, b, sparsity, srcOp.getIsInputASparse(),
        srcOp.getIsInputBSparse(), nnzAttr, programConfigAttr,
        /*memory_config=*/nullptr, /*dtype=*/nullptr,
        /*compute_config=*/nullptr);

    // Untile output: [BD*S/M, E, M, H] -> [BD*S, E, 1, H]
    auto permuted = ttir_to_ttnn::utils::generatePermute(
        cast<TypedValue<RankedTensorType>>(tiledResult.getResult()),
        {0, 2, 1, 3}, rewriter, loc);

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

    auto newOp = rewriter.create<ttnn::AllToAllCombineOp>(
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_moe_workaround"),
        cast<RankedTensorType>(op.getResult().getType()), permute.getResult(),
        op.getExpertMetadata(), op.getExpertMapping(), op.getNumDevicesAttr(),
        op.getClusterAxisAttr(), op.getNumExpertsPerTokAttr(),
        op.getOutputShardDimAttr(), op.getMemoryConfigAttr());

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
