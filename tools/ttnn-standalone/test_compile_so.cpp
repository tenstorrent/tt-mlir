// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "compile_so.hpp"

#include <fstream>
#include <iterator>
#include <string>

std::string loadFileToString(const std::string &filename) {
  std::ifstream file(filename);
  if (!file) {
    exit(1);
  }
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

int main() {
  std::string ttmlir_home = std::getenv("TT_MLIR_HOME");

  std::string cpp_source =
      loadFileToString(ttmlir_home + "/tools/ttnn-standalone/ttnn-dylib.cpp");
  std::string tmp_path_dir = ttmlir_home;

  compile_cpp_to_so(cpp_source, tmp_path_dir);
}
