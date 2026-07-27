// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_ROOFLINE_H
#define TTMLIR_DIALECT_TTNN_UTILS_ROOFLINE_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/Value.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn::roofline {

// Theoretical single-chip hardware constants used by the analytical roofline.
// Shared source of truth for both perf-target estimation
// (TTNNCollectPerfMetrics) and the analytical-time layout cost model so the
// two never diverge.
struct HwSpec {
  uint64_t dramBandwidthBytesPerSec = 0; // aggregate, shared across all cores
  uint64_t aiclkHz = 0;
  uint64_t cyclesPerTileMatmul = 32; // HiFi2: 32 cycles / 32x32x32 tile-mul

  // Per-core L1 SRAM bandwidth. Public specs put L1 at ~1 TB/s per core at
  // ~1 GHz, scaling linearly with clock; 1e12 B/s / 1e9 Hz = 1000 B per Hz.
  uint64_t perCoreL1BandwidthBytesPerSec() const { return aiclkHz * 1000ULL; }
};

// HW spec for a given arch (Wormhole B0, Blackhole). Returns std::nullopt for
// archs the roofline is not calibrated for.
std::optional<HwSpec> getHwSpec(ttcore::Arch arch);

// Number of 32x32x32 tile-multiplies for a matmul with the given lhs and result
// values, accounting for a transposed lhs. Returns 0 if shapes are < rank 2.
uint64_t getNumTileMatmuls(Value lhs, Value result, bool transposeA);

} // namespace mlir::tt::ttnn::roofline

#endif // TTMLIR_DIALECT_TTNN_UTILS_ROOFLINE_H
