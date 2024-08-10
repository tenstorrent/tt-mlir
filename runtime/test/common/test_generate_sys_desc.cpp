// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/runtime.h"
#include <gtest/gtest.h>

TEST(GenerateSysDesc, Sanity) {
  auto sysDesc = ::tt::runtime::getCurrentSystemDesc();
}
