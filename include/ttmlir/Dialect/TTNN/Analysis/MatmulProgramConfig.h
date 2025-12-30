// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"

#include <optional>

namespace mlir::tt::ttnn {

// Generate matmul program config for an op with given output layout.
// Returns nullopt if output is not sharded or config cannot be generated.
//
// This function generates MatmulMultiCoreReuseMultiCast1DProgramConfig for
// width/height sharded outputs and MatmulMultiCoreReuseMultiCastProgramConfig
// for block sharded outputs.
// Issue that tracks compiler side matmul program configs
// https://github.com/tenstorrent/tt-mlir/issues/6473
std::optional<mlir::Attribute>
generateMatmulProgramConfig(Operation *op, TTNNLayoutAttr outputLayout);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H