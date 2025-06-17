// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TEST_TTNN_UTILS_H
#define TT_RUNTIME_DETAIL_TEST_TTNN_UTILS_H

#include "tt/runtime/types.h"

// Utility functions for testing TTNN runtime
namespace tt::runtime::test::ttnn {
Layout getDramInterleavedTileLayout(::tt::target::DataType dataType);
Layout getDramInterleavedRowMajorLayout(::tt::target::DataType dataType);
Layout getHostRowMajorLayout(::tt::target::DataType dataType);
std::optional<size_t> getTraceCacheDebugStat(::tt::runtime::Device device,
                                             const std::string &statName);
} // namespace tt::runtime::test::ttnn

#endif // TT_RUNTIME_DETAIL_TEST_TTNN_UTILS_H
