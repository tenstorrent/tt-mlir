// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_TYPES_TYPES_H
#define TTMLIR_DIALECT_TTNN_TYPES_TYPES_H
#include "llvm/ADT/StringRef.h"
#include <array>
#include <cstdint>

namespace mlir::tt::ttnn {
inline constexpr uint32_t TILE_HEIGHT = 32;
inline constexpr uint32_t TILE_WIDTH = 32;
// Width of the partial-statistics tensor produced by
// layer_norm_pre_all_gather: two tiles (sum_x2 | sum_x) per device.
inline constexpr int64_t LAYER_NORM_STATS_WIDTH = 2 * TILE_WIDTH;
// Offset of sum(x) within each device's stats block.
inline constexpr int64_t LAYER_NORM_SUM_X_OFFSET = TILE_WIDTH;
inline constexpr std::array<uint32_t, 2> VALID_CQ_IDS = {0, 1};
inline constexpr llvm::StringLiteral g_TTNNHoistGenericViaD2MAttrName =
    "ttnn.hoist_generic_via_d2m";
inline constexpr llvm::StringLiteral g_TTNNCaptureTracePrefix =
    "run_and_capture_";
inline constexpr llvm::StringLiteral g_TTNNExecuteTracePrefix = "execute_";
} // namespace mlir::tt::ttnn

#endif
