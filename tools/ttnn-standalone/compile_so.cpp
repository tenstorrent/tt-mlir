// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

std::string compileCppToSo(const std::string &cppSource,
                           const std::string &tmpPathDir,
                           const std::string &metalSrcDir,
                           const std::string &metalLibDir,
                           const std::string &standaloneDir) {
  fs::path directoryPath = fs::path(tmpPathDir);
  fs::path filePath = directoryPath / "emitted.cpp";

  std::ofstream outFile(filePath);
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open file for writing: " << filePath
              << std::endl;
    exit(1);
  }

  outFile << cppSource;
  outFile.close();

  std::cout << "Successfully wrote C++ code to: " << filePath << std::endl;

  // Compile the C++ code to a shared object.
  //
  fs::path currDir = fs::path(standaloneDir);
  fs::path pythonScriptPath = currDir / "ci_compile_dylib.py";

  // Check if the script exists
  if (!fs::exists(pythonScriptPath)) {
    std::cerr << "Error: Python script not found: " << pythonScriptPath
              << std::endl;
    exit(1);
  }

  std::string command = "python3 " + pythonScriptPath.string() + " --file " +
                        filePath.string() + " --metal-src-dir " + metalSrcDir +
                        " --metal-lib-dir " + metalLibDir;

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
