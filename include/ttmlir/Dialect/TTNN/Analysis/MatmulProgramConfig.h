// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/IR/Operation.h"

#include <optional>

namespace mlir::tt::ttnn {

/// Compute max subblock size from compute kernel config.
int64_t getMaxSubblockSize(DeviceComputeKernelConfigAttr computeConfig);

/// Generate 1D mcast config for WidthSharded output (mcast_in0=true).
std::optional<mlir::Attribute>
generateMatmul1DWidthConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                            int64_t Kt, TTNNLayoutAttr outputLayout,
                            UnaryWithParamAttr fusedActivation,
                            int64_t maxSubblockSize, bool fuseBatch);

/// Generate 1D mcast config for HeightSharded output (mcast_in0=false).
std::optional<mlir::Attribute>
generateMatmul1DHeightConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                             int64_t Kt, TTNNLayoutAttr outputLayout,
                             UnaryWithParamAttr fusedActivation,
                             int64_t maxSubblockSize, bool fuseBatch);

/// Generate 2D mcast config for BlockSharded output.
std::optional<mlir::Attribute>
generateMatmul2DConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt, int64_t Kt,
                       TTNNLayoutAttr outputLayout,
                       UnaryWithParamAttr fusedActivation,
                       int64_t maxSubblockSize, bool fuseBatch);

/// Generate DRAM-sharded config for WidthSharded output with Mt==1 (decode).
std::optional<mlir::Attribute>
generateMatmulDRAMShardedConfig(MLIRContext *ctx, int64_t Mt, int64_t Nt,
                                int64_t Kt, TTNNLayoutAttr outputLayout,
                                UnaryWithParamAttr fusedActivation,
                                int64_t maxSubblockSize, bool fuseBatch);

/// Legacy wrapper: generate matmul program config for an op with given output
/// layout. Returns nullopt if output is not sharded or config cannot be
/// generated.
///
/// This function generates MatmulMultiCoreReuseMultiCast1DProgramConfig for
/// width/height sharded outputs and MatmulMultiCoreReuseMultiCastProgramConfig
/// for block sharded outputs.
/// Issue that tracks compiler side matmul program configs
/// https://github.com/tenstorrent/tt-mlir/issues/6473
std::optional<mlir::Attribute>
generateMatmulProgramConfig(Operation *op, TTNNLayoutAttr outputLayout);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H
