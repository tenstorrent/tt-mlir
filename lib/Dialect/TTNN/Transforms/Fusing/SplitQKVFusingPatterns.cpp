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

// Return the single non-DeallocateOp user of a value, or nullptr if there
// are zero or multiple non-deallocate users.
Operation *getSingleNonDeallocUser(Value v) {
  Operation *found = nullptr;
  for (Operation *user : v.getUsers()) {
    if (isa<DeallocateOp>(user)) {
      continue;
    }
    if (found) {
      return nullptr; // Multiple non-deallocate users.
    }
    found = user;
  }
  return found;
}

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
  bool foldedRelabel =
      false; // reshape folds head-split + decode relabel into
             // one reshape to [1,B,H,D]; logical head=[B,H,1,D]

  Operation *getFinalOp() {
    return permuteOp ? permuteOp.getOperation() : reshapeOp.getOperation();
  }

  RankedTensorType getFinalType() {
    if (permuteOp) {
      return permuteOp.getType();
    }
    if (foldedRelabel) {
      // reshapeOp produces [1, B, H, D]; the logical pre-relabel head layout
      // is [B, H, 1, D] (what the split op emits).
      auto sh = reshapeOp.getType().getShape();
      SmallVector<int64_t> headShape = {sh[1], sh[2], 1, sh[3]};
      return utils::RankedTensorTypeFactory::create(reshapeOp.getType(),
                                                    headShape);
    }
    return reshapeOp.getType();
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

// Reorder a tensor by slicing Q/K/V portions and concatenating them in
// Q, K, V order. The slice bounds come from the head roles mapped to the
// matmul output shape. `sliceDim` indicates which dimension of `tensor`
// corresponds to the QKV hidden dimension. Used for both weights and biases.
// The new slice + concat ops are on constant tensors, so the next const-eval
// pass folds them away.
template <typename MatMulOpType>
Value reorderTensorViaSliceConcat(mlir::PatternRewriter &rewriter,
                                  MatMulOpType matmulOp, Value tensor,
                                  size_t sliceDim,
                                  const SmallVector<QKVHead> &heads) {
  auto tensorType = mlir::cast<RankedTensorType>(tensor.getType());
  auto tensorShape = tensorType.getShape();
  auto matmulShape = matmulOp.getType().getShape();

  // Find the end of the last QKV slice to detect shared-LHS remainder.
  int64_t qkvEnd = 0;
  for (size_t i = 0; i < heads.size(); ++i) {
    auto dimAndBounds =
        findSlicedDimensionWithBounds(heads[i].match.sliceOp, matmulShape);
    auto [dim, start, end] = *dimAndBounds;
    qkvEnd = std::max(qkvEnd, end);
  }

  SmallVector<Value> slices(3);
  for (size_t i = 0; i < heads.size(); ++i) {
    auto dimAndBounds =
        findSlicedDimensionWithBounds(heads[i].match.sliceOp, matmulShape);
    auto [dim, start, end] = *dimAndBounds;

    SmallVector<int32_t> begins(tensorShape.size(), 0);
    SmallVector<int32_t> ends(tensorShape.begin(), tensorShape.end());
    SmallVector<int32_t> step(tensorShape.size(), 1);
    begins[sliceDim] = static_cast<int32_t>(start);
    ends[sliceDim] = static_cast<int32_t>(end);

    SmallVector<int64_t> sliceShape(tensorShape);
    sliceShape[sliceDim] = end - start;
    auto sliceTy =
        utils::RankedTensorTypeFactory::create(tensorType, sliceShape);

    auto sliceOp = rewriter.create<SliceStaticOp>(
        matmulOp.getLoc(), sliceTy, tensor, rewriter.getI32ArrayAttr(begins),
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

  // If the tensor is larger than the QKV portion (shared-LHS: e.g. QKV+FC1),
  // slice the remainder and append it so the full tensor is preserved.
  int64_t tensorDimSize = tensorShape[sliceDim];
  if (qkvEnd < tensorDimSize) {
    SmallVector<int32_t> begins(tensorShape.size(), 0);
    SmallVector<int32_t> ends(tensorShape.begin(), tensorShape.end());
    SmallVector<int32_t> step(tensorShape.size(), 1);
    begins[sliceDim] = static_cast<int32_t>(qkvEnd);

    SmallVector<int64_t> sliceShape(tensorShape);
    sliceShape[sliceDim] = tensorDimSize - qkvEnd;
    auto sliceTy =
        utils::RankedTensorTypeFactory::create(tensorType, sliceShape);

    auto sliceOp = rewriter.create<SliceStaticOp>(
        matmulOp.getLoc(), sliceTy, tensor, rewriter.getI32ArrayAttr(begins),
        rewriter.getI32ArrayAttr(ends), rewriter.getI32ArrayAttr(step));
    slices.push_back(sliceOp.getResult());
  }

  SmallVector<int64_t> concatShape(tensorShape);
  concatShape[sliceDim] = 0;
  for (auto &s : slices) {
    concatShape[sliceDim] +=
        mlir::cast<RankedTensorType>(s.getType()).getShape()[sliceDim];
  }
  auto concatTy =
      utils::RankedTensorTypeFactory::create(tensorType, concatShape);

  auto concatOp = rewriter.create<ConcatOp>(matmulOp.getLoc(), concatTy, slices,
                                            static_cast<int32_t>(sliceDim));

  return concatOp.getResult();
}

// ============================================================================
// Step 1: Structural matching
// ============================================================================

// Follow the single non-deallocate user chain through TypecastOps.
// Returns the first non-typecast, non-deallocate user, or nullptr if the
// chain branches or dead-ends.
Operation *lookThroughTypecasts(Value v) {
  Operation *op = getSingleNonDeallocUser(v);
  while (isa_and_nonnull<TypecastOp>(op)) {
    op = getSingleNonDeallocUser(op->getResult(0));
  }
  return op;
}

// Collect matmul users that match slice → (typecast) → reshape → (typecast) →
// (optional permute [0,2,1,3]) chains. Typecasts are looked through
// transparently. Unrelated users are ignored. If the reshape's sole user is a
// permute with any other permutation, that chain is rejected.
template <typename MatMulOpType>
std::optional<SmallVector<SliceReshapeMatch>>
matchSliceReshapeChains(MatMulOpType matmulOp) {
  SmallVector<SliceReshapeMatch> matches;

  for (Operation *user : matmulOp.getResult().getUsers()) {
    auto sliceOp = dyn_cast<SliceStaticOp>(user);
    if (!sliceOp) {
      continue;
    }

    // Look through deallocates and typecasts to find the reshape.
    auto reshapeOp =
        dyn_cast_or_null<ReshapeOp>(lookThroughTypecasts(sliceOp.getResult()));
    if (!reshapeOp || reshapeOp.getType().getShape().size() != 4) {
      continue;
    }

    PermuteOp permuteOp = nullptr;
    if (auto *singleUser = lookThroughTypecasts(reshapeOp.getResult())) {
      if (auto p = dyn_cast<PermuteOp>(singleUser)) {
        if (p.getPermutation() == ArrayRef<int64_t>{0, 2, 1, 3}) {
          permuteOp = p; // Prefill: capture and consume.
        }
        // Decode [2,0,1,3] or others: don't capture, don't reject.
        // The permute stays in IR for NLPCreateQKVHeadsDecodeFusing.
      }
    }

    bool folded = false;
    if (!permuteOp) {
      auto rs = reshapeOp.getType().getShape();
      auto ss = sliceOp.getType().getShape();
      int64_t batchSeq = 1;
      for (size_t i = 0; i + 1 < ss.size(); ++i) {
        batchSeq *= ss[i];
      }
      int64_t hidden = ss.back();
      // Folded decode relabel: a norm-less chain whose reshape collapses the
      // head-split and the [B,H,1,D]->[1,B,H,D] relabel into one reshape to
      // [1, B, H, D]. K/Q keep them separate (q/k-norm sits between).
      if (rs.size() == 4 && rs[0] == 1 && rs[1] == batchSeq &&
          rs[2] * rs[3] == hidden) {
        folded = true;
      }
    }
    matches.push_back({sliceOp, reshapeOp, permuteOp, folded});
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
    if (m.foldedRelabel) {
      // Structurally validated in matchSliceReshapeChains: a folded
      // [1,B,H,D] decode-relabel reshape (logical head layout [B,H,1,D]).
      continue;
    }
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
                                  QKVHead &v,
                                  const OpValidationConfig &validationConfig) {
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
      rewriter.getI32ArrayAttr(inputReshapeShapeI32));

  // Build split op output types in the matmul's element type. If the
  // downstream chain contains typecasts, the final types may differ — we
  // insert typecasts after the split op to bridge the gap.
  auto matmulDataType =
      ttcore::elementTypeToDataType(matmulOp.getType().getElementType());
  auto makeSplitOutputType = [&](RankedTensorType finalType) {
    return utils::RankedTensorTypeFactory::create(finalType, matmulDataType);
  };

  RankedTensorType qSplitTy = makeSplitOutputType(q.match.getFinalType());
  RankedTensorType kSplitTy = makeSplitOutputType(k.match.getFinalType());
  RankedTensorType vSplitTy = makeSplitOutputType(v.match.getFinalType());

  // Validate the fused op before creating it.
  IsolatedIRValidationWrapper validator(rewriter.getContext(),
                                        validationConfig);
  auto numHeadsAttr = rewriter.getUI32IntegerAttr(q.numHeads());
  auto numKVHeadsAttr =
      isGQA ? rewriter.getUI32IntegerAttr(k.numHeads()) : IntegerAttr();
  auto transposeKeyAttr = rewriter.getBoolAttr(false);

  auto validationResult =
      validator.validateOp<SplitQueryKeyValueAndSplitHeadsOp>(
          matmulOp.getOperation(), matmulOp.getLoc(),
          {qSplitTy, kSplitTy, vSplitTy}, inputReshape.getResult(),
          /*kv_input_tensor=*/Value(), numHeadsAttr, numKVHeadsAttr,
          transposeKeyAttr);

  if (!validationResult.isSuccess()) {
    return mlir::failure();
  }

  auto splitOp = rewriter.create<SplitQueryKeyValueAndSplitHeadsOp>(
      matmulOp.getLoc(), TypeRange{qSplitTy, kSplitTy, vSplitTy},
      inputReshape.getResult(),
      Value(), // no separate KV input
      numHeadsAttr, numKVHeadsAttr, transposeKeyAttr);

  // Helper to insert a typecast if the split output dtype differs from what
  // the downstream ops expect.
  auto maybeTypecast = [&](Value splitResult,
                           RankedTensorType finalType) -> Value {
    if (splitResult.getType() == finalType) {
      return splitResult;
    }
    return rewriter
        .create<TypecastOp>(matmulOp.getLoc(), finalType, splitResult)
        .getResult();
  };

  // Wire each split output back to the chain consumers. For folded-relabel
  // chains (norm-less, e.g. V), the original reshape produced [1,B,H,D] while
  // the split emits [B,H,1,D], so re-apply the decode relabel before replacing.
  auto wireOutput = [&](QKVHead &head, Value splitResult) {
    Value val = maybeTypecast(splitResult, head.match.getFinalType());
    if (head.match.foldedRelabel) {
      auto relabelTy = head.match.reshapeOp.getType();
      auto shp = relabelTy.getShape();
      SmallVector<int32_t> shpI32(shp.begin(), shp.end());
      val = rewriter
                .create<ReshapeOp>(matmulOp.getLoc(), relabelTy, val,
                                   rewriter.getI32ArrayAttr(shpI32))
                .getResult();
    }
    rewriter.replaceOp(head.match.getFinalOp(), val);
  };
  wireOutput(q, splitOp.getQuery());
  wireOutput(k, splitOp.getKey());
  wireOutput(v, splitOp.getValue());

  return mlir::success();
}

// ============================================================================
// NLPCreateQKVHeadsDecodeFusing helpers
// ============================================================================

// Returns the single layout-defining user of `result` that performs the decode
// relabel [B, H, 1, D] -> [1, B, H, D], or nullptr otherwise.
//
// This relabel is emitted as a permute [2, 0, 1, 3]. Because only the singleton
// S=1 dim moves, PermuteOp::canonicalize rewrites that permute into an
// equivalent reshape (see the PermuteOp folder / canonicalizer). Canonicalize
// runs between fusing passes, so by the time this pattern sees the IR the
// relabel may be in either form -- accept both. The returned op's single result
// carries the [1, B, H, D] decode-layout type the fused op must produce.
// True if `reshapeOp` performs the decode relabel [B, H, 1, D] -> [1, B, H, D]
// (a pure singleton-S axis move, equivalent to permute [2, 0, 1, 3]).
bool isDecodeRelabelReshape(ReshapeOp reshapeOp) {
  auto inShape =
      mlir::cast<RankedTensorType>(reshapeOp->getOperand(0).getType())
          .getShape();
  auto outShape = reshapeOp.getType().getShape();
  return inShape.size() == 4 && outShape.size() == 4 && inShape[2] == 1 &&
         outShape[0] == 1 && outShape[1] == inShape[0] &&
         outShape[2] == inShape[1] && outShape[3] == inShape[3];
}

Operation *getDecodeLayoutUser(Value result) {
  if (!result.hasOneUse()) {
    return nullptr;
  }
  Operation *user = *result.getUsers().begin();

  if (auto permuteOp = dyn_cast<PermuteOp>(user)) {
    if (permuteOp.getPermutation() == ArrayRef<int64_t>{2, 0, 1, 3}) {
      return permuteOp;
    }
    return nullptr;
  }

  // Reshape form: a pure singleton-S relabel [B, H, 1, D] -> [1, B, H, D],
  // equivalent to permute [2, 0, 1, 3] when the moved S dim is 1.
  if (auto reshapeOp = dyn_cast<ReshapeOp>(user)) {
    return isDecodeRelabelReshape(reshapeOp) ? reshapeOp.getOperation()
                                             : nullptr;
  }

  return nullptr;
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
    } else if (isLoadCached) {
      // LoadCachedOp: slice + reconcat on the weight. The second const-eval
      // pass folds these away.
      bool transB = matmulOp.getTransposeB();
      auto weightShape = mlir::cast<RankedTensorType>(rhs.getType()).getShape();
      size_t weightSliceDim = transB ? 0 : weightShape.size() - 1;
      rewriter.setInsertionPoint(matmulOp);
      Value reorderedWeight = reorderTensorViaSliceConcat(
          rewriter, matmulOp, rhs, weightSliceDim, *heads);
      rewriter.modifyOpInPlace(
          matmulOp, [&]() { matmulOp.getBMutable().assign(reorderedWeight); });
    } else {
      // If it is not const-eval'd, we would add a slice + concat to
      // reorder the inputs, but this would hinder performance.
      return mlir::failure();
    }

    // Reorder bias to match if this is a LinearOp with a bias.
    if constexpr (std::is_same_v<MatMulOpType, LinearOp>) {
      Value bias = matmulOp.getBias();
      if (bias) {
        auto biasConcatOp = bias.getDefiningOp<ConcatOp>();
        auto biasLoadCached = bias.getDefiningOp<ttcore::LoadCachedOp>();
        if (biasConcatOp) {
          reorderConcatInputs(rewriter, biasConcatOp, *heads);
        } else if (biasLoadCached) {
          // Bias is 1D or 2D — the QKV dim is always the last.
          auto biasShape =
              mlir::cast<RankedTensorType>(bias.getType()).getShape();
          size_t biasSliceDim = biasShape.size() - 1;
          rewriter.setInsertionPoint(matmulOp);
          Value reorderedBias = reorderTensorViaSliceConcat(
              rewriter, matmulOp, bias, biasSliceDim, *heads);
          rewriter.modifyOpInPlace(matmulOp, [&]() {
            matmulOp.getBiasMutable().assign(reorderedBias);
          });
        } else {
          return mlir::failure();
        }
      }
    }
  }

  return createFusedOp(rewriter, matmulOp, *q, *k, *v, validationConfig);
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

  // All three outputs must each have exactly one use: the decode relabel
  // [B, H, 1, D] -> [1, B, H, D] (permute [2,0,1,3] or its canonicalized
  // reshape form).
  Operation *qLayoutOp = getDecodeLayoutUser(splitOp.getQuery());
  Operation *kLayoutOp = getDecodeLayoutUser(splitOp.getKey());
  Operation *vLayoutOp = getDecodeLayoutUser(splitOp.getValue());
  if (!qLayoutOp || !kLayoutOp || !vLayoutOp) {
    return mlir::failure();
  }
  auto qOutTy = mlir::cast<RankedTensorType>(qLayoutOp->getResult(0).getType());
  auto kOutTy = mlir::cast<RankedTensorType>(kLayoutOp->getResult(0).getType());
  auto vOutTy = mlir::cast<RankedTensorType>(vLayoutOp->getResult(0).getType());

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
      rewriter.getI32ArrayAttr(reshapeShapeI32));

  bool isGQA = (numHeads != numKVHeads);
  auto numHeadsAttr = rewriter.getUI32IntegerAttr(numHeads);
  auto numKVHeadsAttr =
      isGQA ? rewriter.getUI32IntegerAttr(numKVHeads) : IntegerAttr();
  SmallVector<Type> resultTypes = {qOutTy, kOutTy, vOutTy};

  // Validate the fused op before creating it.
  IsolatedIRValidationWrapper validator(rewriter.getContext(),
                                        validationConfig);
  auto validationResult = validator.validateOp<NLPCreateQKVHeadsDecodeOp>(
      splitOp.getOperation(), splitOp.getLoc(), {qOutTy, kOutTy, vOutTy},
      reshapeOp.getResult(),
      /*batch_offset=*/Value(), numHeadsAttr, numKVHeadsAttr,
      /*overlap_qk_coregrid=*/BoolAttr(),
      /*slice_size=*/IntegerAttr());

  if (!validationResult.isSuccess()) {
    return mlir::failure();
  }

  auto decodeOp = rewriter.create<NLPCreateQKVHeadsDecodeOp>(
      splitOp.getLoc(), resultTypes, reshapeOp.getResult(),
      /*batch_offset=*/Value(), numHeadsAttr, numKVHeadsAttr,
      /*overlap_qk_coregrid=*/BoolAttr(),
      /*slice_size=*/IntegerAttr());

  // Replace permute ops with decode op outputs.
  rewriter.replaceOp(qLayoutOp, decodeOp.getQuery());
  rewriter.replaceOp(kLayoutOp, decodeOp.getKey());
  rewriter.replaceOp(vLayoutOp, decodeOp.getValue());

  return mlir::success();
}

// ============================================================================
// DecodeRelabelThroughRMSNorm
// ============================================================================

// True if `op` is a decode relabel [B, H, 1, D] -> [1, B, H, D], expressed as
// either permute [2, 0, 1, 3] or its canonicalized reshape form.
static bool isDecodeRelabelOp(Operation *op) {
  if (auto permuteOp = dyn_cast<PermuteOp>(op)) {
    return permuteOp.getPermutation() == ArrayRef<int64_t>{2, 0, 1, 3};
  }
  if (auto reshapeOp = dyn_cast<ReshapeOp>(op)) {
    return isDecodeRelabelReshape(reshapeOp);
  }
  return false;
}

mlir::LogicalResult DecodeRelabelThroughRMSNorm::matchAndRewrite(
    RMSNormOp normOp, mlir::PatternRewriter &rewriter) const {
  // Every real consumer of the norm must be a decode relabel
  // [B, H, 1, D] -> [1, B, H, D] (permute [2,0,1,3] or its reshape form). The
  // norm may feed several (e.g. a partial-RoPE rotated half + a passthrough
  // half), so collect them all and sink one shared relabel above the norm.
  SmallVector<Operation *> relabels;
  for (Operation *user : normOp.getResult().getUsers()) {
    if (isa<DeallocateOp>(user)) {
      continue;
    }
    if (!isDecodeRelabelOp(user)) {
      return mlir::failure();
    }
    relabels.push_back(user);
  }
  if (relabels.empty()) {
    return mlir::failure();
  }
  Operation *relabel = relabels.front();

  // Only sink once the head split has been formed: the norm must consume a
  // SplitQueryKeyValueAndSplitHeadsOp output (possibly via a typecast). This
  // enforces split-fusion-first ordering -- otherwise the relabel would merge
  // into the head-splitting reshape and break the split fusion's
  // matmul -> slice -> reshape[B,H,1,D] match.
  Operation *inputDef = normOp.getInput().getDefiningOp();
  while (auto tc = dyn_cast_or_null<TypecastOp>(inputDef)) {
    inputDef = tc.getInput().getDefiningOp();
  }
  if (!isa_and_nonnull<SplitQueryKeyValueAndSplitHeadsOp>(inputDef)) {
    return mlir::failure();
  }

  auto relabelTy =
      mlir::cast<RankedTensorType>(relabel->getResult(0).getType());

  // rms_norm normalizes the last dim, which the relabel leaves untouched, so
  // rms_norm(relabel(x)) == relabel(rms_norm(x)). Sink the relabel above the
  // norm so the norm runs on the [1, B, H, D] decode layout and the relabel
  // moves toward the split output, where NLPCreateQKVHeadsDecodeFusing consumes
  // it. Emit the relabel as a reshape (the canonical S=1 form).
  rewriter.setInsertionPoint(normOp);
  auto outShape = relabelTy.getShape();
  SmallVector<int32_t> shapeI32(outShape.begin(), outShape.end());
  auto newReshape =
      rewriter.create<ReshapeOp>(normOp.getLoc(), relabelTy, normOp.getInput(),
                                 rewriter.getI32ArrayAttr(shapeI32));

  // Clone the norm onto the relabeled input, preserving weight/bias operands,
  // operandSegmentSizes, epsilon and compute_config.
  mlir::OperationState state(normOp.getLoc(), RMSNormOp::getOperationName());
  state.addOperands(newReshape.getResult());
  for (unsigned i = 1; i < normOp->getNumOperands(); ++i) {
    state.addOperands(normOp->getOperand(i));
  }
  state.addAttributes(normOp->getAttrs());
  state.addTypes(relabelTy);
  Operation *newNorm = rewriter.create(state);

  for (Operation *r : relabels) {
    rewriter.replaceOp(r, newNorm->getResult(0));
  }
  return mlir::success();
}

// Returns the single non-dealloc user of `v` if it is a decode relabel
// (permute [2,0,1,3] or its reshape form), else nullptr.
static Operation *getDecodeRelabelConsumer(Value v) {
  Operation *user = getSingleNonDeallocUser(v);
  return (user && isDecodeRelabelOp(user)) ? user : nullptr;
}

// The decode relabel target shape for a [B, H, 1, D'] input: [1, B, H, D'].
static SmallVector<int32_t> decodeRelabelShape(ArrayRef<int64_t> inShape) {
  return {1, static_cast<int32_t>(inShape[0]), static_cast<int32_t>(inShape[1]),
          static_cast<int32_t>(inShape[3])};
}

mlir::LogicalResult DecodeRelabelThroughConcat::matchAndRewrite(
    ConcatOp concatOp, mlir::PatternRewriter &rewriter) const {
  Operation *relabel = getDecodeRelabelConsumer(concatOp.getResult());
  if (!relabel) {
    return mlir::failure();
  }
  // Only commute when concatenating along the last dim (head_dim) of a rank-4
  // [B, H, 1, D] tensor: the relabel moves only leading/singleton dims, so the
  // concat axis is unchanged.
  auto concatTy = concatOp.getType();
  if (concatTy.getRank() != 4 || concatOp.getDim() != 3 ||
      concatTy.getShape()[2] != 1) {
    return mlir::failure();
  }
  auto relabelTy =
      mlir::cast<RankedTensorType>(relabel->getResult(0).getType());

  rewriter.setInsertionPoint(concatOp);
  SmallVector<Value> newInputs;
  for (Value in : concatOp.getInputs()) {
    auto inTy = mlir::cast<RankedTensorType>(in.getType());
    auto inShape = inTy.getShape();
    SmallVector<int64_t> outShape = {1, inShape[0], inShape[1], inShape[3]};
    auto outTy = utils::RankedTensorTypeFactory::create(inTy, outShape);
    newInputs.push_back(rewriter
                            .create<ReshapeOp>(concatOp.getLoc(), outTy, in,
                                               rewriter.getI32ArrayAttr(
                                                   decodeRelabelShape(inShape)))
                            .getResult());
  }
  auto newConcat = rewriter.create<ConcatOp>(concatOp.getLoc(), relabelTy,
                                             newInputs, /*dim=*/3);
  rewriter.replaceOp(relabel, newConcat.getResult());
  return mlir::success();
}

mlir::LogicalResult DecodeRelabelThroughSlice::matchAndRewrite(
    SliceStaticOp sliceOp, mlir::PatternRewriter &rewriter) const {
  Operation *relabel = getDecodeRelabelConsumer(sliceOp.getResult());
  if (!relabel) {
    return mlir::failure();
  }
  // Only commute a pure head_dim (last-dim) slice of a rank-4 [B, H, 1, D]
  // tensor: dims 0,1,2 must be full so the relabel (which only reorders
  // leading/singleton dims) leaves the sliced axis as the last dim.
  auto inTy = mlir::cast<RankedTensorType>(sliceOp->getOperand(0).getType());
  auto inShape = inTy.getShape();
  if (inShape.size() != 4 || inShape[2] != 1) {
    return mlir::failure();
  }
  auto begins = sliceOp.getBegins();
  auto ends = sliceOp.getEnds();
  auto steps = sliceOp.getStep();
  for (int d = 0; d < 3; ++d) {
    if (mlir::cast<mlir::IntegerAttr>(begins[d]).getInt() != 0 ||
        mlir::cast<mlir::IntegerAttr>(ends[d]).getInt() != inShape[d] ||
        mlir::cast<mlir::IntegerAttr>(steps[d]).getInt() != 1) {
      return mlir::failure();
    }
  }
  auto relabelTy =
      mlir::cast<RankedTensorType>(relabel->getResult(0).getType());

  rewriter.setInsertionPoint(sliceOp);
  // Relabel the slice input [B,H,1,D] -> [1,B,H,D].
  SmallVector<int64_t> relInShape = {1, inShape[0], inShape[1], inShape[3]};
  auto relInTy = utils::RankedTensorTypeFactory::create(inTy, relInShape);
  auto relIn = rewriter.create<ReshapeOp>(
      sliceOp.getLoc(), relInTy, sliceOp->getOperand(0),
      rewriter.getI32ArrayAttr(decodeRelabelShape(inShape)));
  // Slice the same head_dim range on the relabeled (now dim 3) layout.
  int64_t d3b = mlir::cast<mlir::IntegerAttr>(begins[3]).getInt();
  int64_t d3e = mlir::cast<mlir::IntegerAttr>(ends[3]).getInt();
  int64_t d3s = mlir::cast<mlir::IntegerAttr>(steps[3]).getInt();
  SmallVector<int32_t> nb = {0, 0, 0, static_cast<int32_t>(d3b)};
  SmallVector<int32_t> ne = {1, static_cast<int32_t>(inShape[0]),
                             static_cast<int32_t>(inShape[1]),
                             static_cast<int32_t>(d3e)};
  SmallVector<int32_t> ns = {1, 1, 1, static_cast<int32_t>(d3s)};
  auto newSlice = rewriter.create<SliceStaticOp>(
      sliceOp.getLoc(), relabelTy, relIn.getResult(),
      rewriter.getI32ArrayAttr(nb), rewriter.getI32ArrayAttr(ne),
      rewriter.getI32ArrayAttr(ns));
  rewriter.replaceOp(relabel, newSlice.getResult());
  return mlir::success();
}

mlir::LogicalResult DecodeRelabelThroughRotary::matchAndRewrite(
    RotaryEmbeddingOp rotaryOp, mlir::PatternRewriter &rewriter) const {
  // The rotary's consumer is the decode relabel [B,H,1,D] -> [1,B,H,D], either
  // directly or through a leading seq-strip slice. At fusing time the rotary
  // output is unpadded [B,H,1,D], so the seq-strip is an identity that
  // canonicalizes away and the relabel consumes the rotary directly; once the
  // seq-len workaround pads the rotary the strip reappears. Accept both.
  // Relabel the rotary input to [1,B,H,D] and re-run the rotary on that layout
  // (heads in dim2) with token_index=0 -- the decode rotary RoPEDecodeFusing
  // produces.
  Operation *next = getSingleNonDeallocUser(rotaryOp.getResult());
  if (!next) {
    return mlir::failure();
  }
  Operation *relabel = nullptr;
  if (isDecodeRelabelOp(next)) {
    relabel = next;
  } else if (auto stripOp = dyn_cast<SliceStaticOp>(next)) {
    auto sTy = mlir::cast<RankedTensorType>(stripOp->getOperand(0).getType());
    auto sShape = sTy.getShape();
    auto geti = [](mlir::ArrayAttr a, int i) {
      return mlir::cast<mlir::IntegerAttr>(a[i]).getInt();
    };
    if (sShape.size() == 4 && geti(stripOp.getBegins(), 0) == 0 &&
        geti(stripOp.getEnds(), 0) == sShape[0] &&
        geti(stripOp.getBegins(), 1) == 0 &&
        geti(stripOp.getEnds(), 1) == sShape[1] &&
        geti(stripOp.getBegins(), 3) == 0 &&
        geti(stripOp.getEnds(), 3) == sShape[3] &&
        geti(stripOp.getBegins(), 2) == 0) {
      relabel = getDecodeRelabelConsumer(stripOp.getResult());
    }
  }
  if (!relabel) {
    return mlir::failure();
  }

  // cos/sin must be single-position (dim -2 == 1): mirrors RoPEDecodeFusing.
  auto cosTy = mlir::cast<RankedTensorType>(rotaryOp.getCosCache().getType());
  if (cosTy.getShape()[cosTy.getRank() - 2] != 1) {
    return mlir::failure();
  }

  Value rotIn = rotaryOp.getInput();
  auto rotInTy = mlir::cast<RankedTensorType>(rotIn.getType());
  auto ris = rotInTy.getShape();
  if (ris.size() != 4 || ris[2] != 1) {
    return mlir::failure();
  }
  auto relabelTy =
      mlir::cast<RankedTensorType>(relabel->getResult(0).getType());

  rewriter.setInsertionPoint(rotaryOp);
  SmallVector<int64_t> relShape = {1, ris[0], ris[1], ris[3]};
  auto relInTy = utils::RankedTensorTypeFactory::create(rotInTy, relShape);
  SmallVector<int32_t> relShapeI32 = {1, static_cast<int32_t>(ris[0]),
                                      static_cast<int32_t>(ris[1]),
                                      static_cast<int32_t>(ris[3])};
  auto relIn = rewriter.create<ReshapeOp>(
      rotaryOp.getLoc(), relInTy, rotIn, rewriter.getI32ArrayAttr(relShapeI32));

  auto tokenIndex = rewriter.getIntegerAttr(
      rewriter.getIntegerType(32, /*isSigned=*/false), 0);
  auto newRotary = rewriter.create<RotaryEmbeddingOp>(
      rotaryOp.getLoc(), relabelTy, relIn.getResult(), rotaryOp.getCosCache(),
      rotaryOp.getSinCache(), tokenIndex, rotaryOp.getComputeConfigAttr());
  rewriter.replaceOp(relabel, newRotary.getResult());
  return mlir::success();
}

} // namespace mlir::tt::ttnn::fusing
