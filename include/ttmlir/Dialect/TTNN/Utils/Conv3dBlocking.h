// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_CONV3DBLOCKING_H
#define TTMLIR_DIALECT_TTNN_UTILS_CONV3DBLOCKING_H

#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn::utils {

// Chosen Conv3d block sizes. Unset fields mean "use the device op's default
// (full dimension)". The chosen values obey tt-metal's constraints:
//   - cInBlock is a multiple of hal::get_l1_alignment() (16 on Wormhole) and
//     evenly divides cIn,
//   - cOutBlock is a multiple of TILE_WIDTH (32) and evenly divides the
//     tile-padded C_out.
// If both are nullopt, the Conv3d is expected to fit in L1 with the runtime
// defaults.
struct Conv3dBlocking {
  std::optional<uint32_t> cInBlock;
  std::optional<uint32_t> cOutBlock;
};

// Heuristic that picks Conv3d block sizes to keep the per-core circular-buffer
// footprint (dominated by vol2col_tiled ≈ M_t*K_t tiles and weight_tiled ≈
// K_t*N_t tiles) within a safe fraction of L1. Returns both nullopt when the
// default configuration already fits.
//
// The caller MUST use the returned cInBlock for both the runtime Conv3dConfig
// and the const-eval weight reshape — the weight is pre-blocked according to
// cInBlock, so the two must agree or the kernel will scramble weight reads.
Conv3dBlocking chooseConv3dBlocking(int32_t cIn, int32_t cOut, int32_t kD,
                                    int32_t kH, int32_t kW);

} // namespace mlir::tt::ttnn::utils

#endif // TTMLIR_DIALECT_TTNN_UTILS_CONV3DBLOCKING_H
