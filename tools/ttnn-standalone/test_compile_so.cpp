// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compile_so.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;

std::string loadFileToString(const std::string &filename) {
  std::ifstream file(filename);
  if (!file) {
    exit(1);
  }
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

int main() {
  fs::path ttmlir_home = fs::path(std::getenv("TT_MLIR_HOME"));

  fs::path tmp_path = ttmlir_home;
  fs::path dylib_cpp = ttmlir_home / "tools/ttnn-standalone/ttnn-dylib.cpp";

  std::string cpp_source = loadFileToString(dylib_cpp.string());

  fs::path metal_src_dir = ttmlir_home / "third_party/tt-metal/src/tt-metal";
  fs::path metal_lib_dir =
      ttmlir_home / "third_party/tt-metal/src/tt-metal-build/lib";
  fs::path ttnn_standalone_dir = ttmlir_home / "tools/ttnn-standalone";

  compile_cpp_to_so(cpp_source, tmp_path.string(), metal_src_dir.string(),
                    metal_lib_dir.string(), ttnn_standalone_dir.string());
}
