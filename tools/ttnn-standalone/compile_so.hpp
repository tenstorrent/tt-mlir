// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TOOLS_TTNN_STANDALONE_COMPILE_SO_HPP
#define TTMLIR_TOOLS_TTNN_STANDALONE_COMPILE_SO_HPP

#include <string>

std::string compile_cpp_to_so(const std::string &cpp_source,
                              const std::string &tmp_path_dir);

#endif // TTMLIR_TOOLS_TTNN_STANDALONE_COMPILE_SO_HPP
