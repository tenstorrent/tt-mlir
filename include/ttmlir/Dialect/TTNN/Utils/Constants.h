// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_CONSTANTS_H
#define TTMLIR_DIALECT_TTNN_UTILS_CONSTANTS_H

#include <cstddef>

namespace mlir::tt::ttnn::constants {

// Default L1 small size to use for the ttnn runtime (32kb).
// This reserves a region of L1 memory for L1_SMALL buffers used by convs.
constexpr static std::size_t L1_SMALL_SIZE = 1 << 15;

} // namespace mlir::tt::ttnn::constants

#endif // TTMLIR_DIALECT_TTNN_UTILS_CONSTANTS_H
