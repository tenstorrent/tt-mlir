// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGHEURISTIC_H
#define TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGHEURISTIC_H

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn {

// Conv3d blocking parameters chosen by the heuristic. Fields use the TTNN
// Conv3dConfig naming/order, NOT the (C_in, C_out, T, H, W) tuple order of
// tt-metal's table.
struct Conv3dBlocking {
  uint32_t tOutBlock;
  uint32_t wOutBlock;
  uint32_t hOutBlock;
  uint32_t cOutBlock;
  uint32_t cInBlock;
};

// Look up tt-metal's default conv3d blockings (the `_DEFAULT_BLOCKINGS` table
// in models/tt_dit/utils/conv3d.py) keyed by (in_channels, out_channels,
// kernel). Returns std::nullopt when no entry matches, in which case the caller
// should keep the existing (conversion) default config.
//
// `kernel` is [K_D, K_H, K_W] (TTNN Conv3dOp kernel_size order).
std::optional<Conv3dBlocking>
lookupConv3dBlocking(int64_t inChannels, int64_t outChannels,
                     llvm::ArrayRef<int64_t> kernel);

} // namespace mlir::tt::ttnn

#endif // TTMLIR_DIALECT_TTNN_ANALYSIS_CONV3DCONFIGHEURISTIC_H
