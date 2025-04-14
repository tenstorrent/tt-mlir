// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

std::string compile_cpp_to_so(const std::string &cpp_source,
                              const std::string &tmp_path_dir) {

  fs::path directoryPath = fs::path(tmp_path_dir);
  fs::path filePath = directoryPath / "emitted.cpp";

  std::ofstream outFile(filePath);
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << filePath
              << std::endl;
    exit(1); // TODO: how do we handle errors in this situation?
  }

  outFile << cpp_source;
  outFile.close();

  std::cout << "Successfully wrote C++ code to: " << filePath << std::endl;

  // Compile the C++ code to a shared object.
  //
  std::string pythonScriptPath = "../tools/ttnn-standalone/ci_compile_dylib.py";

  // Check if the script exists
  if (!fs::exists(pythonScriptPath)) {
    std::cerr << "Error: Python script not found: " << pythonScriptPath
              << std::endl;
    exit(1);
  }

  std::string command =
      "python " + pythonScriptPath + " --file " + filePath.string();

  int result = std::system(command.c_str());

  if (result == 0) {
    std::cout << "Python script executed successfully." << std::endl;
  } else {
    std::cerr << "Error: Python script execution failed with code: " << result
              << std::endl;
    exit(1);
  }

  fs::path soPath = filePath;
  soPath.replace_extension(".so");

  return soPath.string();
}
