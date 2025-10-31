// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TEST_TTNN_DYLIB_H
#define TT_RUNTIME_TEST_TTNN_DYLIB_H

#include "tt/runtime/types.h"

namespace tt::runtime::test::ttnn {

void *openSo(const std::string &path);
void closeSo(void *handle);
std::vector<std::string> getSoPrograms(void *so, std::string path);
std::vector<::tt::runtime::Tensor>
createInputs(void *so, std::string funcName, Device device, std::string path);
std::vector<Tensor> runSoProgram(void *so, const std::string &funcName,
                                 std::vector<Tensor> inputs, Device device);
bool compareOuts(std::vector<Tensor> &lhs, std::vector<Tensor> &rhs);
} // namespace tt::runtime::test::ttnn

#endif // TT_RUNTIME_TEST_TTNN_DYLIB_H
