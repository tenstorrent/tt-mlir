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

// ============================================================================
// DRAM-sharded matmul config generation
// ============================================================================
//
// The weight (in1) is width-sharded across DRAM banks, the activation (in0)
// across L1 cores. MatmulRules.cpp decides eligibility; these own the geometry,
// layouts and configs.

// perCoreNCompute: N tiles each DRAM bank's core computes. perCoreNStorage: N
// tiles each output storage core holds. Both as tt-metal names them.
struct DRAMShardParams {
  int64_t numBanks;
  int64_t kTiles;
  int64_t nTiles;
  int64_t perCoreNCompute;
  int64_t in0BlockW;
  int64_t perCoreM;
  int64_t perCoreNStorage;
  ttcore::DataType weightDataType;
};

// Shard geometry and a CB-fitting in0_block_w for an M×K×N matmul, or nullopt
// when none fits `l1Available`. K and N must be tile-aligned and K in tiles
// divisible by numIn0Cores; the eligibility gate enforces both.
std::optional<DRAMShardParams>
computeShardParams(int64_t M, int64_t K, int64_t N, int64_t numBanks,
                   int64_t numIn0Cores, int64_t numOutCores,
                   ttcore::DataType weightDataType, int64_t l1Available);

// Width-sharded layout over a 1×numCores grid in `bufferType` with canonical
// placement: DRAM over the banks for the weight, L1 over the in0 cores.
TTNNLayoutAttr buildWidthShardedLayout(MLIRContext *ctx,
                                       TTNNLayoutAttr origLayout,
                                       llvm::ArrayRef<int64_t> tensorShape,
                                       BufferType bufferType, int64_t numCores,
                                       ttcore::DeviceAttr deviceAttr);

MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
buildDRAMShardedProgramConfig(MLIRContext *ctx, const DRAMShardParams &p,
                              UnaryWithParamAttr fusedAct);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H
