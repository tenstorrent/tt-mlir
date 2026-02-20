// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Workarounds/Decomposition/PadHighDimRewritePattern.h"

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

namespace mlir::tt::ttnn::workarounds::decomposition {

// Input layout from conv3d: [N=0, D=1, H=2, W=3, C=4]
static constexpr int64_t NDHWC_RANK = 5;
static constexpr int64_t N_DIM = 0;
static constexpr int64_t D_DIM = 1;
static constexpr int64_t H_DIM = 2;
static constexpr int64_t W_DIM = 3;
static constexpr int64_t C_DIM = 4;

LogicalResult
PadHighDimRewritePattern::matchAndRewrite(PadOp srcOp,
                                          PatternRewriter &rewriter) const {

  RankedTensorType inputType = srcOp.getInput().getType();
  int64_t rank = inputType.getRank();

  if (rank != NDHWC_RANK) {
    return failure();
  }

  ArrayRef<int32_t> padding = srcOp.getPadding();

  bool nHasPad = padding[N_DIM * 2] != 0 || padding[N_DIM * 2 + 1] != 0;
  bool cHasPad = padding[C_DIM * 2] != 0 || padding[C_DIM * 2 + 1] != 0;

  if (nHasPad || cHasPad) {
    return failure();
  }

  auto dimHasPad = [&](int64_t d) {
    return padding[d * 2] != 0 || padding[d * 2 + 1] != 0;
  };
  if (!llvm::any_of(std::initializer_list<int64_t>{D_DIM, H_DIM, W_DIM},
                    dimHasPad)) {
    return failure();
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t N = inputShape[N_DIM];
  int64_t D = inputShape[D_DIM];
  int64_t H = inputShape[H_DIM];
  int64_t W = inputShape[W_DIM];
  int64_t C = inputShape[C_DIM];

  // Permute [N,D,H,W,C] -> [C,N,D,H,W] via [4,0,1,2,3].
  SmallVector<int64_t> fwdPerm = {C_DIM, N_DIM, D_DIM, H_DIM, W_DIM};
  auto permuteToFront = ttir_to_ttnn::utils::generatePermute(
      srcOp.getInput(), fwdPerm, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_perm_to_cndhw"));

  // Reshape [C,N,D,H,W] -> [C*N, D, H, W].
  SmallVector<int64_t> squeezedShape = {C * N, D, H, W};
  auto reshapeTo4D = ttir_to_ttnn::utils::generateReshape(
      permuteToFront.getResult(), squeezedShape, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_squeeze_to_4d"));

  // Pad the 4D tensor. Dims: [C*N, D, H, W] — pad dims 1,2,3.
  SmallVector<int32_t> padding4D = {
      0,
      0, // dim 0 (C*N): no padding
      padding[D_DIM * 2],
      padding[D_DIM * 2 + 1], // dim 1 (D)
      padding[H_DIM * 2],
      padding[H_DIM * 2 + 1], // dim 2 (H)
      padding[W_DIM * 2],
      padding[W_DIM * 2 + 1] // dim 3 (W)
  };

  auto pad4D = ttir_to_ttnn::utils::generatePad(
      reshapeTo4D.getResult(), padding4D, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_pad_4d"));

  // Reshape [C*N, D', H', W'] -> [C, N, D', H', W'].
  ArrayRef<int64_t> padded4DShape = pad4D.getResult().getType().getShape();
  int64_t Dp = padded4DShape[1];
  int64_t Hp = padded4DShape[2];
  int64_t Wp = padded4DShape[3];

  SmallVector<int64_t> unsqueezedShape = {C, N, Dp, Hp, Wp};
  auto reshapeTo5D = ttir_to_ttnn::utils::generateReshape(
      pad4D.getResult(), unsqueezedShape, rewriter,
      ttmlir::utils::appendLocationSuffix(srcOp.getLoc(), "_unsqueeze_to_5d"));

  // Step 5: Permute [C,N,D',H',W'] -> [N,D',H',W',C] via [1,2,3,4,0].
  SmallVector<int64_t> invPerm = {1, 2, 3, 4, 0};
  rewriter.replaceOp(srcOp, ttir_to_ttnn::utils::generatePermute(
                                reshapeTo5D.getResult(), invPerm, rewriter,
                                ttmlir::utils::appendLocationSuffix(
                                    srcOp.getLoc(), "_perm_to_ndhwc")));

  return success();
}

} // namespace mlir::tt::ttnn::workarounds::decomposition
