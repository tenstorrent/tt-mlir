// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TOOLS_TTNN_STANDALONE_COMPILE_SO_HPP
#define TTMLIR_TOOLS_TTNN_STANDALONE_COMPILE_SO_HPP

#include <string>

std::string compileCppToSo(const std::string &cppSource,
                           const std::string &tmpPathDir,
                           const std::string &metalSrcDir,
                           const std::string &metalLibDir,
                           const std::string &standaloneDir);

#endif // TTMLIR_TOOLS_TTNN_STANDALONE_COMPILE_SO_HPP
