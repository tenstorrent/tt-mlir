// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TEST_DYLIB_H
#define TT_RUNTIME_TEST_DYLIB_H

#include "tt/runtime/types.h"

namespace tt::runtime::ttnn::test {

void *openSo(std::string path);
std::vector<Tensor> runSoProgram(void *so, std::string func_name,
                                 std::vector<Tensor> inputs, Device device);
bool compareOuts(std::vector<Tensor> &lhs, std::vector<Tensor> &rhs);
} // namespace tt::runtime::ttnn::test

#endif // TT_RUNTIME_TEST_DYLIB_H
