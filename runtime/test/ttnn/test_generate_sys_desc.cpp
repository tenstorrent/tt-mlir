// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#ifndef TT_RUNTIME_ENABLE_TTNN
#error "TT_RUNTIME_ENABLE_TTNN must be defined"
#endif
#include "tt/runtime/runtime.h"
#include <gtest/gtest.h>

TEST(TTNNSysDesc, Sanity) {
  auto sysDesc = ::tt::runtime::getCurrentSystemDesc();
}
