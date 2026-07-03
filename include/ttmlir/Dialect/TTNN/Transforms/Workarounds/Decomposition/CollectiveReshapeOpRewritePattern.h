// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_COLLECTIVERESHAPEOPREWRITEPATTERN_H
#define TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_COLLECTIVERESHAPEOPREWRITEPATTERN_H

#include "ttmlir/Conversion/TTIRToTTNN/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Utils.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <type_traits>

namespace mlir::tt::ttnn::workarounds::decomposition {

// AllReduceOp and ReduceScatterOp in TTNN produce incorrect results for tensors
// with rank < 4 (see tt-metal issue
// https://github.com/tenstorrent/tt-metal/issues/39953). As a temporary
// workaround, we insert reshape ops front and back to make the tensor four
// dimensional. ReduceScatterOp additionally requires its scatter_dim to be
// shifted by the number of prepended leading dimensions.
template <typename OpTy>
class TTNNCollectiveReshapeWorkaround : public OpRewritePattern<OpTy> {
  static_assert(std::is_same_v<OpTy, ttnn::AllReduceOp> ||
                    std::is_same_v<OpTy, ttnn::ReduceScatterOp>,
                "TTNNCollectiveReshapeWorkaround only supports AllReduceOp and "
                "ReduceScatterOp.");

public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType inputType = op.getInput().getType();
    int64_t rank = inputType.getRank();

    // Only apply workaround for tensors with rank < 4.
    if (rank >= 4) {
      return failure();
    }

    RankedTensorType outputType = op.getResult().getType();
    ArrayRef<int64_t> outputShape = outputType.getShape();

    int64_t numLeadingDims = 4 - rank;

    // Create padded shapes by prepending leading dimensions of size 1.
    SmallVector<int64_t> paddedInputShape(numLeadingDims, 1);
    paddedInputShape.append(inputType.getShape().begin(),
                            inputType.getShape().end());

    SmallVector<int64_t> paddedOutputShape(numLeadingDims, 1);
    paddedOutputShape.append(outputShape.begin(), outputShape.end());

    // Reshape input up to 4D.
    auto reshapeInput = ttir_to_ttnn::utils::generateReshape(
        op.getInput(), paddedInputShape, rewriter,
        ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_to_4d"));

    // Create 4D output tensor type.
    RankedTensorType paddedOutputType =
        ttnn::utils::RankedTensorTypeFactory::create(outputType,
                                                     paddedOutputShape);

    // Create the collective operation on 4D tensors.
    Value collective4D;
    if constexpr (std::is_same_v<OpTy, ttnn::AllReduceOp>) {
      collective4D = rewriter
                         .create<ttnn::AllReduceOp>(
                             ttmlir::utils::appendLocationSuffix(
                                 op.getLoc(), "_all_reduce_4d"),
                             paddedOutputType, reshapeInput.getResult(),
                             op.getReduceType(), op.getClusterAxis(),
                             op.getSubDeviceIdAttr(), op.getNumLinksAttr(),
                             op.getTopologyAttr())
                         .getResult();
    } else {
      // Shift the scatter dimension to account for the prepended dimensions.
      int32_t adjustedScatterDim = op.getScatterDim() + numLeadingDims;
      collective4D =
          rewriter
              .create<ttnn::ReduceScatterOp>(
                  ttmlir::utils::appendLocationSuffix(op.getLoc(),
                                                      "_reduce_scatter_4d"),
                  paddedOutputType, reshapeInput.getResult(),
                  op.getReduceType(), adjustedScatterDim, op.getClusterAxis(),
                  op.getSubDeviceIdAttr(), op.getNumLinksAttr(),
                  op.getTopologyAttr(), op.getComputeConfigAttr())
              .getResult();
    }

    // Reshape back to the original shape.
    rewriter.replaceOp(
        op,
        ttir_to_ttnn::utils::generateReshape(
            mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(collective4D),
            outputShape, rewriter,
            ttmlir::utils::appendLocationSuffix(op.getLoc(), "_reshape_back")));

    return success();
  }
};

} // namespace mlir::tt::ttnn::workarounds::decomposition

#endif // TTMLIR_DIALECT_TTNN_TRANSFORMS_WORKAROUNDS_DECOMPOSITION_COLLECTIVERESHAPEOPREWRITEPATTERN_H
