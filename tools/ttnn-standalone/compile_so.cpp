// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

std::string compile_cpp_to_so(const std::string &cpp_source,
                              const std::string &tmp_path_dir,
                              const std::string &metal_src_dir,
                              const std::string &metal_lib_dir,
                              const std::string &standalone_dir) {
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
  fs::path currDir = fs::path(standalone_dir);
  fs::path pythonScriptPath = currDir / "ci_compile_dylib.py";

  // Check if the script exists
  if (!fs::exists(pythonScriptPath)) {
    std::cerr << "Error: Python script not found: " << pythonScriptPath
              << std::endl;
    exit(1);
  }

  std::string command = "python " + pythonScriptPath.string() + " --file " +
                        filePath.string() + " --metal-src-dir " +
                        metal_src_dir + " --metal-lib-dir " + metal_lib_dir;

  int result = std::system(command.c_str());

  if (result == 0) {
    std::cout << "Python script executed successfully." << std::endl;
  } else {
    std::cerr << "Error: Python script execution failed with code: " << result
              << std::endl;
    exit(1);
  }

  // Return path to .so
  return filePath.replace_extension(".so");
}
