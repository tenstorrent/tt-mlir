// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TEST_TTNN_DYLIB_H
#define TT_RUNTIME_TEST_TTNN_DYLIB_H

#include "tt/runtime/types.h"

namespace tt::runtime::test::ttnn {

void *openSo(std::string path);
void closeSo(void *handle);
std::vector<Tensor> runSoProgram(void *so, std::string func_name,
                                 std::vector<Tensor> inputs, Device device);
bool compareOuts(std::vector<Tensor> &lhs, std::vector<Tensor> &rhs);
} // namespace tt::runtime::test::ttnn

#endif // TT_RUNTIME_TEST_TTNN_DYLIB_H
