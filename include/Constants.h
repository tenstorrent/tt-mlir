// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_UTILS_CONSTANTS_H
#define TTMLIR_DIALECT_TTNN_UTILS_CONSTANTS_H

#include <cstddef>

namespace tt::constants {

// Default L1 small size to use for the ttnn runtime (64kb).
// This reserves a region of L1 memory for L1_SMALL buffers used by convs.
constexpr static std::size_t L1_SMALL_SIZE = 1 << 16;

// Used only in unittests:
// todo(arminaleTT): look into dynamically adjusting this
// getOpRuntime() uses trace capture to run and measure the runtime of an op.
// This requires the device to be opened with sufficient trace region size.
// This number is currently set based on manual testing of supported ops to
// accommodate the highest required trace buffer size (2004992B)
static constexpr size_t opModelDefaultTraceRegionSize = 6000000;

} // namespace tt::constants

#endif // TTMLIR_DIALECT_TTNN_UTILS_CONSTANTS_H
