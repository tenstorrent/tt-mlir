// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/SDPAEraseRepeatKV.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::tt::ttnn {

// SDPA Query, Key, Value tensors have shape [B, H, S, D]
// (Batch, NumHeads, SeqLen, HeadDim). The head dim is dim 1.
static constexpr int64_t kNumHeadsDim = 1;

// If `v` is produced by a repeat_interleave on the num-heads dim, return that
// op; otherwise return nullptr.
static RepeatInterleaveOp getHeadRepeat(Value v) {
  auto repeatOp = v.getDefiningOp<RepeatInterleaveOp>();
  if (!repeatOp) {
    return nullptr;
  }
  if (repeatOp.getDim() != kNumHeadsDim) {
    return nullptr;
  }
  return repeatOp;
}

mlir::LogicalResult
SDPAEraseRepeatKV::matchAndRewrite(ScaledDotProductAttentionOp op,
                                   mlir::PatternRewriter &rewriter) const {
  // Both K and V must be expanded by repeat_interleave on the head dim. The
  // SDPA verifier requires key and value to have identical shapes, so we only
  // rewrite when both can be un-expanded by the same factor.
  RepeatInterleaveOp keyRepeat = getHeadRepeat(op.getKey());
  RepeatInterleaveOp valueRepeat = getHeadRepeat(op.getValue());
  if (!keyRepeat || !valueRepeat) {
    return failure();
  }

  uint32_t repeats = keyRepeat.getRepeats();
  if (repeats <= 1 || repeats != valueRepeat.getRepeats()) {
    return failure();
  }

  Value newKey = keyRepeat.getInput();
  Value newValue = valueRepeat.getInput();

  auto queryType = mlir::cast<RankedTensorType>(op.getQuery().getType());
  auto newKeyType = mlir::cast<RankedTensorType>(newKey.getType());
  auto newValueType = mlir::cast<RankedTensorType>(newValue.getType());

  // Query is guaranteed rank 4 by the SDPA verifier; be defensive anyway.
  if (queryType.getRank() != 4 || newKeyType.getRank() != 4 ||
      newValueType.getRank() != 4) {
    return failure();
  }

  int64_t qHeads = queryType.getShape()[kNumHeadsDim];
  int64_t kvHeads = newKeyType.getShape()[kNumHeadsDim];

  // Feeding the un-expanded K/V to SDPA (which broadcasts each KV head across a
  // contiguous group of query heads) is numerically identical to the explicit
  // repeat_interleave expansion iff the expansion produced exactly Hq heads,
  // i.e. Hq == Hkv * repeats. This single equality subsumes the divisibility
  // check (it implies Hq % Hkv == 0, so the resulting GQA op is valid) and pins
  // the broadcast group size to the repeat factor.
  if (qHeads != kvHeads * static_cast<int64_t>(repeats)) {
    return failure();
  }

  // Un-expanded K and V must match head counts (SDPA verifier requirement).
  if (newValueType.getShape()[kNumHeadsDim] != kvHeads) {
    return failure();
  }

  rewriter.modifyOpInPlace(op, [&]() {
    op.getKeyMutable().assign(newKey);
    op.getValueMutable().assign(newValue);
  });

  return success();
}

} // namespace mlir::tt::ttnn
