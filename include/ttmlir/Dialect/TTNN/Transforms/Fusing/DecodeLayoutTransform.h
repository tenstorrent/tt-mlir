// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_DECODELAYOUTTRANSFORM_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_DECODELAYOUTTRANSFORM_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"

#include <optional>

namespace mlir::tt::ttnn::fusing {

// The decode head-split layout transform maps BHSD with seq == 1 ([B,H,1,D])
// to decode layout [1,B,H,D] (i.e. SBHD with S == 1). Because the moved axis
// (seq) has size 1, the transform reorders no data and is therefore
// layout-preserving. It can be expressed either as:
//   - ttnn.permute {permutation = [2,0,1,3]}, or
//   - an equivalent ttnn.reshape producing [1,B,H,D]. Its input is either the
//     4D BHSD form [B,H,1,D] (e.g. a RoPE output) or the folded 2D slice form
//     [B, H*D] (when the canonicalizer merges the to-heads reshape with the
//     permute).
//
// A pre-fusing canonicalizer folds the layout-preserving permute into the
// adjacent reshape, so the decode-fusing patterns must treat both forms as the
// same logical operation. This recognizer is the single place that encodes that
// equivalence.
struct DecodeLayoutMatch {
  RankedTensorType resultType; // The [1, B, H, D] result type.
  int64_t batch;
  int64_t numHeads;
  int64_t headDim;
};

// Returns a match if `op` realizes the decode layout transform [B,H,1,D] ->
// [1,B,H,D], expressed as permute {2,0,1,3} or an equivalent reshape. Returns
// nullopt otherwise.
inline std::optional<DecodeLayoutMatch>
matchDecodeLayoutTransform(mlir::Operation *op) {
  // The result of the transform must be rank-4 decode layout [1, B, H, D].
  auto asDecodeResult =
      [](RankedTensorType resultType) -> std::optional<DecodeLayoutMatch> {
    auto shape = resultType.getShape();
    if (shape.size() != 4 || shape[0] != 1) {
      return std::nullopt;
    }
    return DecodeLayoutMatch{resultType, shape[1], shape[2], shape[3]};
  };

  if (auto permuteOp = mlir::dyn_cast<PermuteOp>(op)) {
    auto inShape =
        mlir::cast<RankedTensorType>(permuteOp.getInput().getType()).getShape();
    // Input must be BHSD with seq (dim 2) == 1.
    if (inShape.size() != 4 || inShape[2] != 1) {
      return std::nullopt;
    }
    if (permuteOp.getPermutation() != llvm::ArrayRef<int64_t>{2, 0, 1, 3}) {
      return std::nullopt;
    }
    return asDecodeResult(permuteOp.getType());
  }

  if (auto reshapeOp = mlir::dyn_cast<ReshapeOp>(op)) {
    std::optional<DecodeLayoutMatch> match =
        asDecodeResult(reshapeOp.getType());
    if (!match) {
      return std::nullopt;
    }
    auto inShape =
        mlir::cast<RankedTensorType>(reshapeOp.getInput().getType()).getShape();
    int64_t b = match->batch;
    int64_t h = match->numHeads;
    int64_t d = match->headDim;
    // 4D BHSD source [B, H, 1, D]: a reshape from this to [1,B,H,D] preserves
    // row-major element order, so it is exactly permute {2,0,1,3}.
    bool from4D = inShape.size() == 4 && inShape[0] == b && inShape[1] == h &&
                  inShape[2] == 1 && inShape[3] == d;
    // Folded 2D slice source [B, H*D] (seq == 1): the canonicalizer merged the
    // to-heads reshape ([B,H*D] -> [B,H,1,D]) and the permute into one reshape.
    bool from2D = inShape.size() == 2 && inShape[0] == b && inShape[1] == h * d;
    if (!from4D && !from2D) {
      return std::nullopt;
    }
    return match;
  }

  return std::nullopt;
}

} // namespace mlir::tt::ttnn::fusing

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_FUSING_DECODELAYOUTTRANSFORM_H
