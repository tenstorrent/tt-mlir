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
inline constexpr std::array<uint32_t, 2> VALID_CQ_IDS = {0, 1};
inline constexpr llvm::StringLiteral g_TTNNTraceAttrName = "ttnn.trace";
inline constexpr llvm::StringLiteral g_TTNNHoistGenericViaD2MAttrName =
    "ttnn.hoist_generic_via_d2m";
inline constexpr llvm::StringLiteral g_TTNNCaptureTracePrefix =
    "run_and_capture_";
inline constexpr llvm::StringLiteral g_TTNNExecuteTracePrefix = "execute_";
} // namespace mlir::tt::ttnn

#endif
