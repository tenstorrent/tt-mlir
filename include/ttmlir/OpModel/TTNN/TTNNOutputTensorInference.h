// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOUTPUTTENSORINFERENCE_H
#define TTMLIR_OPMODEL_TTNN_TTNNOUTPUTTENSORINFERENCE_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::op_model {

// TODO(jserbedzija): This logic should be moved to tt-metal side.
// Metal issue: https://github.com/tenstorrent/tt-metal/issues/21061

// Calculate the output tensor type of the prepared weights for a conv2d op.
// Conv2dConfigAttr is used to determine the output tensor type.
mlir::RankedTensorType
getPreparedConv2dWeightsOutputTensor(Conv2dOp *op,
                                     Conv2dConfigAttr conv2dConfig);

// Calculate the output tensor type of the prepared weights for a
// conv_transpose2d op. Conv2dConfigAttr is used to determine the output tensor
// type.
mlir::RankedTensorType
getPreparedConvTranspose2dWeightsOutputTensor(ConvTranspose2dOp *op,
                                              Conv2dConfigAttr conv2dConfig);

// Calculate the prepared-weight output tensor types for the
// prepare_moe_compute_w0_w1_weights / _w2_weights ops.
mlir::RankedTensorType
getPreparedMoEComputeW0W1WeightsOutputType(PrepareMoEComputeW0W1WeightsOp *op);

mlir::RankedTensorType
getPreparedMoEComputeW2WeightsOutputType(PrepareMoEComputeW2WeightsOp *op);

// Calculate the device-derived output tensor types for a compute_only
// moe_compute op (the first five tensors tt-metal returns). The op's sixth
// result, combine_output, aliases matmul_output and is not inferred here.
llvm::SmallVector<mlir::RankedTensorType>
getMoeComputeOutputTypes(MoeComputeOp *op);

} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_OPMODEL_TTNN_TTNNOUTPUTTENSORINFERENCE_H
