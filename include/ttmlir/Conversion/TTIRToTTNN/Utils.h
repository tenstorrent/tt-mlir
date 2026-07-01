// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_TTIRTOTTNN_UTILS_H
#define TTMLIR_CONVERSION_TTIRTOTTNN_UTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

namespace mlir::tt::ttir_to_ttnn::utils {
// Generates a reshape operation for the given input tensor with the new shape.
mlir::tt::ttnn::ReshapeOp
generateReshape(mlir::TypedValue<mlir::RankedTensorType> input,
                llvm::ArrayRef<int64_t> newShape,
                mlir::PatternRewriter &rewriter, mlir::Location newLoc);

// Generates a reshape operation for the given input tensor that returns 4D
// tensor. Assumes that the input tensor is 4D. First 3 dimensions are flattened
// into 3rd dimension and 4th dimension is kept as is.
mlir::tt::ttnn::ReshapeOp
generateNHWFlatten(mlir::TypedValue<mlir::RankedTensorType> input,
                   mlir::PatternRewriter &rewriter, mlir::Location newLoc);

// Generates a permute operation for the given input tensor with the specified
// permutation pattern.
mlir::tt::ttnn::PermuteOp
generatePermute(mlir::TypedValue<mlir::RankedTensorType> input,
                llvm::ArrayRef<int64_t> permutation,
                mlir::PatternRewriter &rewriter, mlir::Location newLoc);

// Generates a pad operation for the given input tensor with the specified
// padding values.
mlir::tt::ttnn::PadOp
generatePad(mlir::TypedValue<mlir::RankedTensorType> input,
            llvm::ArrayRef<int32_t> padding, mlir::PatternRewriter &rewriter,
            mlir::Location newLoc);

// Emits the primitive-op decomposition of a GroupNorm whose input is in the
// canonical ttnn [N, 1, H*W, C] form, returning the normalized (and optionally
// affine-scaled) result of type `resultType`. This is the single source of
// truth for group-norm decomposition, shared by:
//   * the TTIRToTTNN conversion, for shapes the fused ttnn.group_norm kernel
//     cannot represent (e.g. non-tile-aligned flattened height N*H*W), and
//   * GroupNormDecompositionRewritePattern.
// `weight`/`bias` are optional 1-D [C] tensors (pass null to skip). `device
// AnchorOp` is used only to locate/insert the device for the epsilon constant.
mlir::Value decomposeGroupNorm(
    mlir::TypedValue<mlir::RankedTensorType> input, mlir::Value weight,
    mlir::Value bias, uint32_t numGroups, llvm::APFloat epsilon,
    mlir::RankedTensorType resultType, mlir::Operation *deviceAnchorOp,
    mlir::PatternRewriter &rewriter, mlir::Location loc);
} // namespace mlir::tt::ttir_to_ttnn::utils

#endif
