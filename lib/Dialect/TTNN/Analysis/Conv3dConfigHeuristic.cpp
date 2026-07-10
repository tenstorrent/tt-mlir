// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/Conv3dConfigHeuristic.h"

#include "llvm/ADT/ArrayRef.h"

#include <array>
#include <cstdint>
#include <optional>

namespace mlir::tt::ttnn {

namespace {
// One row of tt-metal's `_DEFAULT_BLOCKINGS` table. The block fields are stored
// in tt-metal's value-tuple order, (C_in_block, C_out_block, T_out_block,
// H_out_block, W_out_block), so the entries below can be transcribed verbatim
// from the Python source; the remap to TTNN Conv3dConfig field order happens in
// lookupConv3dBlocking.
struct Conv3dBlockingEntry {
  int64_t inChannels;
  int64_t outChannels;
  std::array<int64_t, 3> kernel; // [K_D, K_H, K_W]
  uint32_t cInBlock;
  uint32_t cOutBlock;
  uint32_t tOutBlock;
  uint32_t hOutBlock;
  uint32_t wOutBlock;
};

// Verbatim mirror of `_DEFAULT_BLOCKINGS` from
// tt-metal/models/tt_dit/utils/conv3d.py. Keep this a faithful, auditable copy
// of the upstream table: {in, out, {kd, kh, kw}, C_in_block, C_out_block,
// T_out_block, H_out_block, W_out_block}.
constexpr Conv3dBlockingEntry kDefaultBlockings[] = {
    {96, 3, {3, 3, 3}, 96, 32, 1, 16, 8},
    {96, 32, {3, 3, 3}, 96, 32, 1, 16, 8},
    {192, 96, {1, 3, 3}, 192, 96, 1, 4, 8},
    {96, 96, {3, 3, 3}, 96, 96, 1, 8, 8},
    {384, 192, {1, 3, 3}, 192, 96, 1, 32, 4},
    {192, 192, {3, 3, 3}, 96, 96, 1, 8, 4},
    {32, 384, {3, 3, 3}, 32, 96, 1, 2, 32},
    {192, 384, {3, 3, 3}, 64, 128, 1, 8, 4},
    {384, 384, {3, 3, 3}, 96, 96, 1, 8, 4},
    {384, 768, {3, 3, 3}, 96, 96, 1, 8, 4},
    {1024, 4096, {3, 3, 3}, 256, 32, 1, 1, 1},
    {512, 4096, {3, 3, 3}, 256, 32, 1, 1, 1},
    {256, 512, {3, 3, 3}, 256, 32, 1, 4, 4},
    {128, 128, {3, 3, 3}, 128, 32, 1, 8, 8},
    {128, 48, {3, 3, 3}, 128, 32, 1, 8, 8},
    {1024, 4096, {1, 3, 3}, 256, 32, 1, 1, 1},
    {1024, 128, {3, 3, 3}, 256, 32, 1, 1, 1},
};
} // namespace

std::optional<Conv3dBlocking>
lookupConv3dBlocking(int64_t inChannels, int64_t outChannels,
                     llvm::ArrayRef<int64_t> kernel) {
  if (kernel.size() != 3) {
    return std::nullopt;
  }
  for (const Conv3dBlockingEntry &entry : kDefaultBlockings) {
    if (entry.inChannels == inChannels && entry.outChannels == outChannels &&
        entry.kernel[0] == kernel[0] && entry.kernel[1] == kernel[1] &&
        entry.kernel[2] == kernel[2]) {
      // Remap from tt-metal's tuple order to TTNN Conv3dConfig field order.
      return Conv3dBlocking{entry.tOutBlock, entry.wOutBlock, entry.hOutBlock,
                            entry.cOutBlock, entry.cInBlock};
    }
  }
  return std::nullopt;
}

} // namespace mlir::tt::ttnn
