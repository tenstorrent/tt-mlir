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
// Primitives that build a DRAM-sharded matmul
// (MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig): the weight (in1) is
// width-sharded across DRAM banks and the activation (in0) is width-sharded
// across L1 cores. The optimizer's DRAM-shard rule book (MatmulRules.cpp)
// decides *whether* a matmul is eligible and orchestrates the reshards; these
// functions own the *how* — the shard geometry, layouts, and configs.

// Geometry describing how an M×K×N matmul is sharded for DRAM-sharded
// execution. Produced by computeShardParams; consumed by the builders below
// and by the rule book (which reads e.g. perCoreN to size the output grid).
struct DRAMShardParams {
  int64_t K;
  int64_t N;
  int64_t M;
  int64_t numBanks;
  int64_t numIn0Cores;
  int64_t numOutCores;
  int64_t nPadded;
  int64_t shardH;
  int64_t shardW;
  int64_t kTiles;
  int64_t shardWTiles;
  int64_t in0BlockW;
  int64_t perCoreM;
  int64_t perCoreN;
  int64_t in0ShardW;
  ttcore::DataType weightDataType;
};

// Compute the shard geometry and a circular-buffer-fitting in0_block_w for an
// M×K×N matmul whose weight is BFP and DRAM-interleaved. The weight is sharded
// across `numBanks` DRAM banks and the activation across `numIn0Cores` L1
// cores; `numOutCores` bounds the output storage grid. `l1Available` is the L1
// budget the circular buffers must fit. Returns nullopt when no in0_block_w
// fits. K/kTileSize must be divisible by numIn0Cores (the caller's eligibility
// gate enforces this).
std::optional<DRAMShardParams>
computeShardParams(int64_t M, int64_t K, int64_t N, int64_t numBanks,
                   int64_t numIn0Cores, int64_t numOutCores,
                   ttcore::DataType weightDataType, int64_t l1Available);

// Build the DRAM width-sharded layout for the weight (in1), sharded across
// `p.numBanks` DRAM banks.
TTNNLayoutAttr buildDRAMShardedWeightLayout(MLIRContext *ctx,
                                            TTNNLayoutAttr origLayout,
                                            const DRAMShardParams &p);

// Build an L1 width-sharded layout for `tensorShape` over `numCores`, using
// canonical core placement that wraps across the worker grid (a single-row
// placement would be invalid once numCores exceeds the grid width).
TTNNLayoutAttr buildL1ShardedLayout(MLIRContext *ctx, TTNNLayoutAttr origLayout,
                                    llvm::ArrayRef<int64_t> tensorShape,
                                    int64_t numCores,
                                    ttcore::DeviceAttr deviceAttr);

// Build the DRAM-sharded program config from computed shard params.
MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr
buildDRAMShardedProgramConfig(MLIRContext *ctx, const DRAMShardParams &p,
                              UnaryWithParamAttr fusedAct);

// Build the compute-kernel config for a DRAM-sharded matmul (math fidelity
// follows the weight dtype; fp32 dest-accumulate and packer-L1-accumulate
// enabled).
DeviceComputeKernelConfigAttr
buildComputeConfig(MLIRContext *ctx, ttcore::DataType weightDataType);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_MATMULPROGRAMCONFIG_H
