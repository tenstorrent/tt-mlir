// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TEST_UTILS_H
#define TT_RUNTIME_TEST_UTILS_H

#include "tt/runtime/types.h"

// Utility functions for testing TTNN runtime
namespace tt::runtime::ttnn::test {
Layout getDramInterleavedTileLayout(::tt::target::DataType dataType);
Layout getDramInterleavedRowMajorLayout(::tt::target::DataType dataType);
Layout getHostRowMajorLayout(::tt::target::DataType dataType);
} // namespace tt::runtime::ttnn::test

#endif // TT_RUNTIME_TEST_UTILS_H
