// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Utils/Conv3dBlocking.h"

#include <algorithm>
#include <cstdint>

namespace mlir::tt::ttnn::utils {

namespace {

// tt-metal constraints mirrored from:
//   ttnn/cpp/ttnn/operations/experimental/conv3d/device/conv3d_device_operation.cpp:148-206
constexpr uint32_t kTileWidth =
    32; // C_out_block must be a multiple of TILE_WIDTH.
//
// NOTE: the device-op only asserts C_in_block % hal::get_l1_alignment() == 0
// (16 on Wormhole), but in practice the kernel also requires the tiled weight
// layout's inner dim to be tile-aligned. When C_in_block < TILE_WIDTH, the
// intermediate 6D reshape used by reshapeWeightForConv3d produces a memref
// whose inner dim (cb) has size < 32, and the tile-storage padding leaves
// gaps that the subsequent permute/reshape doesn't compact — the kernel then
// reads those padding rows and the output is scrambled (observed PCC ≈ 0.25
// on Qwen3.5-VL patch embed with C_in=32, C_in_block=16). Evidence that this
// is a known problem: tt-metal's own Qwen3-VL demo sidesteps ttnn.conv3d for
// the patch-embed and runs the CPU reference instead
// (models/demos/qwen3_vl/tests/test_model.py: "Use ref model for conv3d for
// now"). Every working tt_dit Conv3d config uses C_in_block ≥ 128.
//
// Conservative rule applied here: require C_in_block to be a multiple of
// TILE_WIDTH. If no such divisor of C_in strictly smaller than C_in exists,
// we leave C_in_block unset — blocking is not possible with correctness.
constexpr uint32_t kCInAlignment = kTileWidth;

// Budgets, in tiles. A bf16 tile is 2 KB. 256 tiles ≈ 512 KB. The two biggest
// CBs (vol2col_tiled and weight_tiled) are both sized by matmul_K_t, so we cap
// matmul_K_t per block here and additionally cap K_t * N_t by the weight
// budget.
constexpr uint32_t kKTilesPerBlockMax = 256;
constexpr uint32_t kWeightTileBudget = 256;

// Largest multiple of kTileWidth at most `cap` that evenly divides
// `paddedCOut`. `paddedCOut` is assumed to be tile-aligned.
uint32_t pickCOutBlock(uint32_t paddedCOut, uint32_t cap) {
  uint32_t candidate = (cap / kTileWidth) * kTileWidth;
  if (candidate == 0) {
    candidate = kTileWidth;
  }
  while (candidate > kTileWidth && paddedCOut % candidate != 0) {
    candidate -= kTileWidth;
  }
  return candidate;
}

} // namespace

Conv3dBlocking chooseConv3dBlocking(int32_t cInSigned, int32_t cOutSigned,
                                    int32_t kDSigned, int32_t kHSigned,
                                    int32_t kWSigned) {
  Conv3dBlocking result{};
  if (cInSigned <= 0 || cOutSigned <= 0 || kDSigned <= 0 || kHSigned <= 0 ||
      kWSigned <= 0) {
    return result;
  }

  uint32_t cIn = static_cast<uint32_t>(cInSigned);
  uint32_t cOut = static_cast<uint32_t>(cOutSigned);
  uint32_t kernelVolume = static_cast<uint32_t>(kDSigned) *
                          static_cast<uint32_t>(kHSigned) *
                          static_cast<uint32_t>(kWSigned);
  uint32_t paddedCOut = ((cOut + kTileWidth - 1) / kTileWidth) * kTileWidth;

  // Bail out if defaults already fit. matmul_K_t * N_t tiles of weight_tiled
  // is the dominant term that scales with both K and N.
  uint64_t defaultKTiles =
      (static_cast<uint64_t>(kernelVolume) * cIn + kTileWidth - 1) / kTileWidth;
  uint64_t defaultNTiles = paddedCOut / kTileWidth;
  if (defaultKTiles * defaultNTiles <= kWeightTileBudget) {
    return result;
  }

  // 1) Pick C_in_block: largest mult-of-kCInAlignment divisor of cIn that is
  //    ≤ targetPatchSize / kernelVolume.
  uint32_t targetPatchSize = kKTilesPerBlockMax * kTileWidth;
  uint32_t maxCInBlock =
      std::max(kCInAlignment, targetPatchSize / std::max(kernelVolume, 1u));
  maxCInBlock = (maxCInBlock / kCInAlignment) * kCInAlignment;

  uint32_t effectiveCIn = cIn;
  if (cIn >= kCInAlignment && cIn % kCInAlignment == 0) {
    uint32_t start = std::min(maxCInBlock, cIn);
    start = (start / kCInAlignment) * kCInAlignment;
    for (uint32_t d = start; d >= kCInAlignment; d -= kCInAlignment) {
      if (cIn % d == 0 && d < cIn) {
        result.cInBlock = d;
        effectiveCIn = d;
        break;
      }
    }
  }

  // 2) Pick C_out_block so weight_tiled = K_t × N_t stays within budget.
  uint32_t effectivePatchSize = kernelVolume * effectiveCIn;
  uint32_t effectiveKTiles = (effectivePatchSize + kTileWidth - 1) / kTileWidth;
  uint32_t maxNTiles =
      std::max(1u, kWeightTileBudget / std::max(effectiveKTiles, 1u));
  uint32_t maxCOutBlock = maxNTiles * kTileWidth;
  if (maxCOutBlock > paddedCOut) {
    maxCOutBlock = paddedCOut;
  }
  result.cOutBlock = pickCOutBlock(paddedCOut, maxCOutBlock);

  return result;
}

} // namespace mlir::tt::ttnn::utils
