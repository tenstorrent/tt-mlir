// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Fusing/SplitQKVFusingPatterns.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::fusing {

namespace {

// ============================================================================
// SplitQueryKeyValueAndSplitHeadsFusing helpers
// ============================================================================

static constexpr unsigned kRoleTraceMaxDepth = 20;

enum OutputDims {
  O_BATCH = 0,
  O_NUM_HEADS = 1,
  O_SEQ_LEN = 2,
  O_HEAD_DIM = 3,
};

enum class QKVRole { Query, Key, Value };

// Pure structural match: the ops forming one slice → reshape → (permute) chain.
struct SliceReshapeMatch {
  SliceStaticOp sliceOp;
  ReshapeOp reshapeOp;
  PermuteOp permuteOp; // nullptr if absent

  Operation *getFinalOp() {
    return permuteOp ? permuteOp.getOperation() : reshapeOp.getOperation();
  }

  RankedTensorType getFinalType() {
    return permuteOp ? permuteOp.getType() : reshapeOp.getType();
  }
};

// Structural match enriched with Q/K/V role after forward tracing to SDPA.
struct QKVHead {
  SliceReshapeMatch match;
  QKVRole role;

  int64_t numHeads() { return match.getFinalType().getShape()[O_NUM_HEADS]; }
  int64_t headDim() { return match.getFinalType().getShape()[O_HEAD_DIM]; }
  int64_t seqLen() { return match.getFinalType().getShape()[O_SEQ_LEN]; }
};

QKVHead *findByRole(SmallVector<QKVHead> &heads, QKVRole role) {
  for (auto &h : heads) {
    if (h.role == role) {
      return &h;
    }
  }
  return nullptr;
}

bool validateQKVDimensions(QKVHead &q, QKVHead &k, QKVHead &v) {
  return q.headDim() == k.headDim() && k.headDim() == v.headDim() &&
         k.numHeads() == v.numHeads();
}

// Find which dimension is being sliced and return its index along with
// bounds. Returns nullopt if multiple dimensions are sliced or no dimension
// is sliced.
std::optional<std::tuple<size_t, int64_t, int64_t>>
findSlicedDimensionWithBounds(SliceStaticOp sliceOp,
                              ArrayRef<int64_t> inputShape) {
  auto begins = sliceOp.getBegins();
  auto ends = sliceOp.getEnds();

  std::optional<std::tuple<size_t, int64_t, int64_t>> result;
  for (size_t dim = 0; dim < begins.size(); ++dim) {
    int64_t start = mlir::cast<mlir::IntegerAttr>(begins[dim]).getInt();
    int64_t end = mlir::cast<mlir::IntegerAttr>(ends[dim]).getInt();

    if (start != 0 || end != inputShape[dim]) {
      if (result.has_value()) {
        return std::nullopt; // Multiple dimensions being sliced
      }
      result = {dim, start, end};
    }
  }
  return result;
}

// Trace forward from a value to find which QKV role it plays in attention.
// Returns the role if the value reaches an SDPA Q/K/V operand. Traces through
// intermediate ops (RoPE, permutes, etc.).
std::optional<QKVRole> findQKVRole(Value start) {
  llvm::SmallVector<std::pair<Value, unsigned>, 32> queue;
  llvm::DenseSet<Value> visited;
  size_t queueIdx = 0;
  queue.push_back({start, 0});
  visited.insert(start);

  std::optional<QKVRole> foundRole;
  while (queueIdx < queue.size()) {
    auto item = queue[queueIdx++];
    Value v = item.first;
    unsigned depth = item.second;
    if (depth >= kRoleTraceMaxDepth) {
      continue;
    }

    for (Operation *user : v.getUsers()) {
      auto role = llvm::TypeSwitch<Operation *, std::optional<QKVRole>>(user)
                      .Case<ScaledDotProductAttentionOp,
                            ScaledDotProductAttentionDecodeOp,
                            PagedScaledDotProductAttentionDecodeOp>(
                          [&](auto op) -> std::optional<QKVRole> {
                            if (op.getQuery() == v) {
                              return QKVRole::Query;
                            }
                            if (op.getKey() == v) {
                              return QKVRole::Key;
                            }
                            if (op.getValue() == v) {
                              return QKVRole::Value;
                            }
                            return std::nullopt;
                          })
                      .Default([](Operation *) { return std::nullopt; });

      if (role) {
        // A single value should not simultaneously feed multiple SDPA operand
        // roles (Q/K/V). If it does, treat it as ambiguous.
        if (!foundRole) {
          foundRole = *role;
        } else if (*foundRole != *role) {
          return std::nullopt;
        }
      }

      // For cache updates, follow the cache tensor to find SDPA consumption.
      // FillCacheOp and PagedFillCacheOp are in-place ops with no results,
      // so we must explicitly follow the cache tensor they write into.
      Value cacheToFollow;
      if (auto op = mlir::dyn_cast<PagedUpdateCacheOp>(user)) {
        cacheToFollow = op.getCache();
      } else if (auto op = mlir::dyn_cast<FillCacheOp>(user)) {
        cacheToFollow = op.getCache();
      } else if (auto op = mlir::dyn_cast<PagedFillCacheOp>(user)) {
        cacheToFollow = op.getCache();
      }
      if (cacheToFollow) {
        unsigned nextDepth = depth + 1;
        if (nextDepth < kRoleTraceMaxDepth &&
            visited.insert(cacheToFollow).second) {
          queue.push_back({cacheToFollow, nextDepth});
        }
      }

      // Propagate through all ops to handle RoPE and other intermediate ops
      for (Value res : user->getResults()) {
        unsigned nextDepth = depth + 1;
        if (nextDepth < kRoleTraceMaxDepth && visited.insert(res).second) {
          queue.push_back({res, nextDepth});
        }
      }
    }
  }

  return foundRole;
}

// Assign Q/K/V roles to each match by tracing forward to SDPA ops.
// Returns a QKVHead per match if all three unique roles are found.
std::optional<SmallVector<QKVHead>>
identifyQKVRoles(SmallVector<SliceReshapeMatch> &matches) {
  if (matches.size() != 3) {
    return std::nullopt;
  }

  uint8_t seenMask = 0;
  SmallVector<QKVHead> heads;
  heads.reserve(3);

  for (auto &m : matches) {
    auto role = findQKVRole(m.reshapeOp.getResult());
    if (!role) {
      return std::nullopt;
    }

    uint8_t bit = 1u << static_cast<uint8_t>(*role);
    if (seenMask & bit) {
      return std::nullopt;
    }
    seenMask |= bit;
    heads.push_back({m, *role});
  }

  // All three roles must be present.
  if (seenMask != 0b111) {
    return std::nullopt;
  }

  return heads;
}

// Infer Q/K/V roles from slice sizes when forward tracing to SDPA fails.
// The largest slice (by number of heads) is Q; the other two are K and V.
// K and V must have the same head count and head dim. The order of K/V
// doesn't matter — the split op and downstream ops handle it.
std::optional<SmallVector<QKVHead>>
inferQKVRolesFromSliceSizes(SmallVector<SliceReshapeMatch> &matches) {
  if (matches.size() != 3) {
    return std::nullopt;
  }

  // Extract num_heads from each match's final shape.
  SmallVector<int64_t> numHeads;
  for (auto &m : matches) {
    numHeads.push_back(m.getFinalType().getShape()[O_NUM_HEADS]);
  }

  // Find the one with the most heads — that's Q.
  size_t qIdx = 0;
  for (size_t i = 1; i < 3; ++i) {
    if (numHeads[i] > numHeads[qIdx]) {
      qIdx = i;
    }
  }

  // The other two are K and V. They must have equal num_heads and head_dim.
  size_t kvIdx0 = (qIdx == 0) ? 1 : 0;
  size_t kvIdx1 = (qIdx == 2) ? 1 : 2;

  if (numHeads[kvIdx0] != numHeads[kvIdx1]) {
    return std::nullopt;
  }

  int64_t kvHeadDim0 = matches[kvIdx0].getFinalType().getShape()[O_HEAD_DIM];
  int64_t kvHeadDim1 = matches[kvIdx1].getFinalType().getShape()[O_HEAD_DIM];
  if (kvHeadDim0 != kvHeadDim1) {
    return std::nullopt;
  }

  SmallVector<QKVHead> heads(3);
  heads[qIdx] = {matches[qIdx], QKVRole::Query};
  heads[kvIdx0] = {matches[kvIdx0], QKVRole::Key};
  heads[kvIdx1] = {matches[kvIdx1], QKVRole::Value};

  return heads;
}

// Check if heads are already in Q, K, V order by slice position.
// Heads are sorted by slice position from validateSliceCoverage.
bool isQKVOrder(const SmallVector<QKVHead> &heads) {
  return heads[0].role == QKVRole::Query && heads[1].role == QKVRole::Key &&
         heads[2].role == QKVRole::Value;
}

// Reorder concat inputs to match Q, K, V order based on head roles.
// Heads are sorted by slice position, so head index == concat input index.
// Reorder concat inputs so the first 3 (QKV) are in Q, K, V order.
// Any additional inputs (e.g. fc1 from shared-LHS fusion) are preserved
// after the QKV inputs.
void reorderConcatInputs(mlir::PatternRewriter &rewriter, ConcatOp concatOp,
                         const SmallVector<QKVHead> &heads) {
  auto inputs = concatOp.getInputs();
  SmallVector<Value> reorderedInputs(3);
  for (size_t i = 0; i < heads.size(); ++i) {
    switch (heads[i].role) {
    case QKVRole::Query:
      reorderedInputs[0] = inputs[i];
      break;
    case QKVRole::Key:
      reorderedInputs[1] = inputs[i];
      break;
    case QKVRole::Value:
      reorderedInputs[2] = inputs[i];
      break;
    }
  }
  for (size_t i = heads.size(); i < inputs.size(); ++i) {
    reorderedInputs.push_back(inputs[i]);
  }
  rewriter.modifyOpInPlace(
      concatOp, [&]() { concatOp.getInputsMutable().assign(reorderedInputs); });
}

// Reorder the matmul RHS weight by slicing Q/K/V portions from the weight
// tensor and concatenating them in Q, K, V order. Used when the RHS is a
// LoadCachedOp (const-eval has folded the original concat). The new slice +
// concat ops are on constant weights, so the second const-eval pass folds
// them away.
template <typename MatMulOpType>
void reorderWeightViaSliceConcat(mlir::PatternRewriter &rewriter,
                                 MatMulOpType matmulOp,
                                 const SmallVector<QKVHead> &heads) {
  Value weight = matmulOp.getB();
  auto weightType = mlir::cast<RankedTensorType>(weight.getType());
  auto weightShape = weightType.getShape();

  // The slice dimension on the weight corresponds to the output feature dim.
  // For transpose_b=false: last dim. For transpose_b=true: first dim.
  bool transB = matmulOp.getTransposeB();
  size_t sliceDim = transB ? 0 : weightShape.size() - 1;

  auto matmulShape = matmulOp.getType().getShape();

  rewriter.setInsertionPoint(matmulOp);

  // Slice each head's portion from the weight, placed in Q, K, V order.
  SmallVector<Value> slices(3);
  for (size_t i = 0; i < heads.size(); ++i) {
    auto dimAndBounds =
        findSlicedDimensionWithBounds(heads[i].match.sliceOp, matmulShape);
    auto [dim, start, end] = *dimAndBounds;

    SmallVector<int32_t> begins(weightShape.size(), 0);
    SmallVector<int32_t> ends(weightShape.begin(), weightShape.end());
    SmallVector<int32_t> step(weightShape.size(), 1);
    begins[sliceDim] = static_cast<int32_t>(start);
    ends[sliceDim] = static_cast<int32_t>(end);

    SmallVector<int64_t> sliceShape(weightShape);
    sliceShape[sliceDim] = end - start;
    auto sliceTy =
        utils::RankedTensorTypeFactory::create(weightType, sliceShape);

    auto sliceOp = rewriter.create<SliceStaticOp>(
        matmulOp.getLoc(), sliceTy, weight, rewriter.getI32ArrayAttr(begins),
        rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(step));

    size_t targetIdx;
    switch (heads[i].role) {
    case QKVRole::Query:
      targetIdx = 0;
      break;
    case QKVRole::Key:
      targetIdx = 1;
      break;
    case QKVRole::Value:
      targetIdx = 2;
      break;
    }
    slices[targetIdx] = sliceOp.getResult();
  }

  SmallVector<int64_t> concatShape(weightShape);
  concatShape[sliceDim] = 0;
  for (auto &s : slices) {
    concatShape[sliceDim] +=
        mlir::cast<RankedTensorType>(s.getType()).getShape()[sliceDim];
  }
  auto concatTy =
      utils::RankedTensorTypeFactory::create(weightType, concatShape);

  auto concatOp = rewriter.create<ConcatOp>(matmulOp.getLoc(), concatTy, slices,
                                            static_cast<int32_t>(sliceDim),
                                            MemoryConfigAttr());

  rewriter.modifyOpInPlace(
      matmulOp, [&]() { matmulOp.getBMutable().assign(concatOp.getResult()); });
}

// ============================================================================
// Step 1: Structural matching
// ============================================================================

// Collect matmul users that match slice → reshape → (optional permute
// [0,2,1,3]) chains. Unrelated users are ignored. If the reshape's sole user is
// a permute with any other permutation, that chain is rejected.
template <typename MatMulOpType>
std::optional<SmallVector<SliceReshapeMatch>>
matchSliceReshapeChains(MatMulOpType matmulOp) {
  SmallVector<SliceReshapeMatch> matches;

  for (Operation *user : matmulOp.getResult().getUsers()) {
    auto sliceOp = dyn_cast<SliceStaticOp>(user);
    if (!sliceOp || !sliceOp.getResult().hasOneUse()) {
      continue;
    }

    auto reshapeOp =
        dyn_cast<ReshapeOp>(*sliceOp.getResult().getUsers().begin());
    if (!reshapeOp || reshapeOp.getType().getShape().size() != 4) {
      continue;
    }

    PermuteOp permuteOp = nullptr;
    if (reshapeOp.getResult().hasOneUse()) {
      if (auto p =
              dyn_cast<PermuteOp>(*reshapeOp.getResult().getUsers().begin())) {
        if (p.getPermutation() == ArrayRef<int64_t>{0, 2, 1, 3}) {
          permuteOp = p; // Prefill: capture and consume.
        }
        // Decode [2,0,1,3] or others: don't capture, don't reject.
        // The permute stays in IR for NLPCreateQKVHeadsDecodeFusing.
      }
    }

    matches.push_back({sliceOp, reshapeOp, permuteOp});
  }

  return matches;
}

// ============================================================================
// Step 2: Slice coverage validation
// ============================================================================

// Validate that the slices are contiguous, cover the full sliced dimension,
// and all slice along the same dimension. Sorts matches by slice position
// so that match index corresponds to concat input index.
template <typename MatMulOpType>
bool validateSliceCoverage(MatMulOpType matmulOp,
                           SmallVector<SliceReshapeMatch> &matches) {
  auto matmulShape = matmulOp.getType().getShape();
  std::optional<size_t> sliceDim;

  // Extract slice bounds and verify consistent slice dimension.
  SmallVector<std::pair<int64_t, int64_t>> bounds;
  for (auto &m : matches) {
    auto dimAndBounds = findSlicedDimensionWithBounds(m.sliceOp, matmulShape);
    if (!dimAndBounds) {
      return false;
    }
    auto [dim, start, end] = *dimAndBounds;

    if (!sliceDim) {
      sliceDim = dim;
    } else if (*sliceDim != dim) {
      return false;
    }
    bounds.push_back({start, end});
  }

  if (!sliceDim) {
    return false;
  }

  // Sort matches by slice start position.
  SmallVector<size_t> indices(matches.size());
  std::iota(indices.begin(), indices.end(), 0);
  llvm::sort(indices, [&](size_t a, size_t b) {
    return bounds[a].first < bounds[b].first;
  });

  // Validate contiguous coverage starting from 0. The slices don't need to
  // cover the full dimension — the matmul may have additional non-QKV slice
  // users (e.g. fc1 fused via shared-LHS).
  int64_t prevEnd = 0;
  for (size_t i : indices) {
    if (bounds[i].first != prevEnd) {
      return false;
    }
    prevEnd = bounds[i].second;
  }

  // Apply the sorted order.
  SmallVector<SliceReshapeMatch> sorted;
  sorted.reserve(matches.size());
  for (size_t i : indices) {
    sorted.push_back(matches[i]);
  }
  matches = std::move(sorted);
  return true;
}

// ============================================================================
// Step 3: Reshape layout validation
// ============================================================================

// Validate that each reshape correctly splits dimensions for the fused op.
//
// The slice output (reshape input) is [..., H*D] where leading dims encode
// B*S (either as a single flattened dim or already factored). We compute
// batchSeq as the product of all dims except the last.
//
// Valid reshape patterns to 4D:
//   BSHD: [B*S, H*D] -> [B, S, H, D]  (requires permute [0,2,1,3] after)
//   BHSD: [B*S, H*D] -> [B, H, S, D]  (only valid without a permute when
//                                       H=1 or S=1, since a dim-of-1 can be
//                                       placed anywhere without reordering
//                                       data)
bool validateReshapeLayouts(SmallVector<SliceReshapeMatch> &matches) {
  for (auto &m : matches) {
    auto sliceShape = m.sliceOp.getType().getShape();
    auto reshapeShape = m.reshapeOp.getType().getShape();
    if (reshapeShape.size() != 4) {
      return false;
    }

    int64_t sliceHidden = sliceShape.back();
    int64_t batchSeq = 1;
    for (size_t i = 0; i < sliceShape.size() - 1; ++i) {
      batchSeq *= sliceShape[i];
    }

    bool isBSHD = (reshapeShape[0] * reshapeShape[1] == batchSeq) &&
                  (reshapeShape[2] * reshapeShape[3] == sliceHidden);

    bool isBHSD = (reshapeShape[0] * reshapeShape[2] == batchSeq) &&
                  (reshapeShape[1] * reshapeShape[3] == sliceHidden);

    if (m.permuteOp) {
      // If a permute is present, the reshape must produce [B,S,H,D] so the
      // supported permute [0,2,1,3] yields [B,H,S,D].
      if (!isBSHD) {
        return false;
      }
    } else {
      // Without a permute, the reshape must already produce [B,H,S,D]. This
      // is only layout-preserving when either H or S is 1.
      if (!isBHSD) {
        return false;
      }
      if (reshapeShape[1] != 1 && reshapeShape[2] != 1) {
        return false;
      }
    }
  }

  return true;
}

// ============================================================================
// Step 7: Create fused op
// ============================================================================

template <typename MatMulOpType>
mlir::LogicalResult createFusedOp(mlir::PatternRewriter &rewriter,
                                  MatMulOpType matmulOp, QKVHead &q, QKVHead &k,
                                  QKVHead &v) {
  auto qFinalShape = q.match.getFinalType().getShape();
  int64_t batchSize = qFinalShape[O_BATCH];
  int64_t seqLen = q.seqLen();
  int64_t matmulHiddenSize = matmulOp.getType().getShape().back();
  int64_t qkvHiddenSize = q.numHeads() * q.headDim() +
                          k.numHeads() * k.headDim() +
                          v.numHeads() * v.headDim();

  bool isGQA = (q.numHeads() != k.numHeads());

  rewriter.setInsertionPointAfter(matmulOp);

  // If the matmul output contains more than just QKV (e.g. fc1 fused via
  // shared-LHS), insert a slice to extract only the QKV portion.
  Value splitInput = matmulOp.getResult();
  if (qkvHiddenSize != matmulHiddenSize) {
    auto matmulShape = matmulOp.getType().getShape();
    SmallVector<int32_t> begins(matmulShape.size(), 0);
    SmallVector<int32_t> ends(matmulShape.begin(), matmulShape.end());
    SmallVector<int32_t> step(matmulShape.size(), 1);
    ends.back() = static_cast<int32_t>(qkvHiddenSize);

    SmallVector<int64_t> sliceShape(matmulShape);
    sliceShape.back() = qkvHiddenSize;
    RankedTensorType sliceTy =
        utils::RankedTensorTypeFactory::create(matmulOp.getType(), sliceShape);

    auto sliceOp = rewriter.create<SliceStaticOp>(
        matmulOp.getLoc(), sliceTy, splitInput,
        rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
        rewriter.getI32ArrayAttr(step));
    splitInput = sliceOp.getResult();
  }

  SmallVector<int64_t> inputReshapeShape = {batchSize, seqLen, qkvHiddenSize};
  SmallVector<int32_t> inputReshapeShapeI32(inputReshapeShape.begin(),
                                            inputReshapeShape.end());

  RankedTensorType reshapeInputTy = utils::RankedTensorTypeFactory::create(
      matmulOp.getType(), inputReshapeShape);

  auto inputReshape = rewriter.create<ReshapeOp>(
      matmulOp.getLoc(), reshapeInputTy, splitInput,
      rewriter.getI32ArrayAttr(inputReshapeShapeI32), MemoryConfigAttr());

  auto splitOp = rewriter.create<SplitQueryKeyValueAndSplitHeadsOp>(
      matmulOp.getLoc(),
      TypeRange{q.match.getFinalType(), k.match.getFinalType(),
                v.match.getFinalType()},
      inputReshape.getResult(),
      Value(), // no separate KV input
      rewriter.getUI32IntegerAttr(q.numHeads()),
      isGQA ? rewriter.getUI32IntegerAttr(k.numHeads()) : IntegerAttr(),
      rewriter.getBoolAttr(false) /*transpose_key*/, MemoryConfigAttr());

  rewriter.replaceOp(q.match.getFinalOp(), splitOp.getQuery());
  rewriter.replaceOp(k.match.getFinalOp(), splitOp.getKey());
  rewriter.replaceOp(v.match.getFinalOp(), splitOp.getValue());

  return mlir::success();
}

// ============================================================================
// NLPCreateQKVHeadsDecodeFusing helpers
// ============================================================================

// Returns the single PermuteOp user of `result` if it has permutation
// [2, 0, 1, 3], or nullptr otherwise.
PermuteOp getDecodePermuteUser(Value result) {
  if (!result.hasOneUse()) {
    return nullptr;
  }
  auto permuteOp = dyn_cast<PermuteOp>(*result.getUsers().begin());
  if (!permuteOp) {
    return nullptr;
  }
  if (permuteOp.getPermutation() != ArrayRef<int64_t>{2, 0, 1, 3}) {
    return nullptr;
  }
  return permuteOp;
}

} // namespace

// ============================================================================
// SplitQueryKeyValueAndSplitHeadsFusing
// ============================================================================

template <typename MatMulOpType>
mlir::LogicalResult
SplitQueryKeyValueAndSplitHeadsFusing<MatMulOpType>::matchAndRewrite(
    MatMulOpType matmulOp, mlir::PatternRewriter &rewriter) const {
  // Structural match: matmul -> 3x (slice -> reshape -> optional permute).
  std::optional<SmallVector<SliceReshapeMatch>> matches =
      matchSliceReshapeChains(matmulOp);
  if (!matches || matches->size() != 3) {
    return mlir::failure();
  }

  // Validate slices are contiguous.
  if (!validateSliceCoverage(matmulOp, *matches)) {
    return mlir::failure();
  }

  // Validate each reshape is a valid dimension split.
  if (!validateReshapeLayouts(*matches)) {
    return mlir::failure();
  }

  // Matmul RHS must be a ConcatOp or LoadCachedOp (const-eval'd weights).
  Value rhs = matmulOp.getB();
  bool isDirectConcat = rhs.getDefiningOp<ConcatOp>() != nullptr;
  bool isLoadCached = rhs.getDefiningOp<ttcore::LoadCachedOp>() != nullptr;
  if (!isDirectConcat && !isLoadCached) {
    return mlir::failure();
  }

  // Identify Q/K/V roles by tracing forward to SDPA ops.
  std::optional<SmallVector<QKVHead>> heads = identifyQKVRoles(*matches);
  if (!heads) {
    // Fallback: infer roles from slice sizes. The largest slice (by hidden dim)
    // is Q, the other two are K and V. K/V ordering doesn't matter — the op
    // splits by num_heads/num_kv_heads regardless, and downstream ops handle
    // the distinction. This handles models with decomposed attention (no fused
    // SDPA ops to trace to).
    heads = inferQKVRolesFromSliceSizes(*matches);
    if (!heads) {
      return mlir::failure();
    }
  }

  // Validate Q/K/V dimension compatibility.
  QKVHead *q = findByRole(*heads, QKVRole::Query);
  QKVHead *k = findByRole(*heads, QKVRole::Key);
  QKVHead *v = findByRole(*heads, QKVRole::Value);
  if (!q || !k || !v || !validateQKVDimensions(*q, *k, *v)) {
    return mlir::failure();
  }

  // Ensure QKV portions are in Q, K, V order on the matmul RHS.
  if (!isQKVOrder(*heads)) {
    if (isDirectConcat) {
      // Fast path: reorder concat inputs in-place.
      reorderConcatInputs(rewriter, rhs.getDefiningOp<ConcatOp>(), *heads);
    } else {
      // LoadCachedOp: slice + reconcat on the weight. The second const-eval
      // pass folds these away.
      reorderWeightViaSliceConcat(rewriter, matmulOp, *heads);
    }
  }

  return createFusedOp(rewriter, matmulOp, *q, *k, *v);
}

// Explicit template instantiations.
template class SplitQueryKeyValueAndSplitHeadsFusing<MatmulOp>;
template class SplitQueryKeyValueAndSplitHeadsFusing<LinearOp>;

// ============================================================================
// NLPCreateQKVHeadsDecodeFusing
// ============================================================================

mlir::LogicalResult NLPCreateQKVHeadsDecodeFusing::matchAndRewrite(
    SplitQueryKeyValueAndSplitHeadsOp splitOp,
    mlir::PatternRewriter &rewriter) const {
  // Must use single fused input (no separate KV tensor).
  if (splitOp.getKvInputTensor()) {
    return mlir::failure();
  }

  // transpose_key must be false (decode variant doesn't support it).
  if (splitOp.getTransposeKey()) {
    return mlir::failure();
  }

  // Input must be 3D with S=1 (decode case).
  auto inputType = splitOp.getInputTensor().getType();
  auto inputShape = inputType.getShape();
  if (inputShape.size() != 3 || inputShape[1] != 1) {
    return mlir::failure();
  }

  // All three outputs must each have exactly one use: a permute [2,0,1,3].
  auto qPermuteOp = getDecodePermuteUser(splitOp.getQuery());
  auto kPermuteOp = getDecodePermuteUser(splitOp.getKey());
  auto vPermuteOp = getDecodePermuteUser(splitOp.getValue());
  if (!qPermuteOp || !kPermuteOp || !vPermuteOp) {
    return mlir::failure();
  }

  // Extract dimensions.
  int64_t batchSize = inputShape[0];
  int64_t hidden = inputShape[2];
  uint32_t numHeads = splitOp.getNumHeads();
  uint32_t numKVHeads =
      splitOp.getNumKvHeads() ? *splitOp.getNumKvHeads() : numHeads;

  rewriter.setInsertionPointAfter(splitOp);

  // Reshape input from [B, 1, hidden] -> [1, 1, B, hidden].
  SmallVector<int64_t> reshapeShape = {1, 1, batchSize, hidden};
  SmallVector<int32_t> reshapeShapeI32(reshapeShape.begin(),
                                       reshapeShape.end());
  RankedTensorType reshapeType =
      utils::RankedTensorTypeFactory::create(inputType, reshapeShape);
  auto reshapeOp = rewriter.create<ReshapeOp>(
      splitOp.getLoc(), reshapeType, splitOp.getInputTensor(),
      rewriter.getI32ArrayAttr(reshapeShapeI32), MemoryConfigAttr());

  bool isGQA = (numHeads != numKVHeads);
  auto decodeOp = rewriter.create<NLPCreateQKVHeadsDecodeOp>(
      splitOp.getLoc(),
      TypeRange{qPermuteOp.getType(), kPermuteOp.getType(),
                vPermuteOp.getType()},
      reshapeOp.getResult(),
      /*batch_offset=*/Value(), rewriter.getUI32IntegerAttr(numHeads),
      isGQA ? rewriter.getUI32IntegerAttr(numKVHeads) : IntegerAttr(),
      /*overlap_qk_coregrid=*/BoolAttr(),
      /*slice_size=*/IntegerAttr(), MemoryConfigAttr());

  // Replace permute ops with decode op outputs.
  rewriter.replaceOp(qPermuteOp, decodeOp.getQuery());
  rewriter.replaceOp(kPermuteOp, decodeOp.getKey());
  rewriter.replaceOp(vPermuteOp, decodeOp.getValue());

  return mlir::success();
}

} // namespace mlir::tt::ttnn::fusing
