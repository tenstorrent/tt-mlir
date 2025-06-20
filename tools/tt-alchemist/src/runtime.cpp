// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "include/runtime.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace tt_alchemist {

Runtime::Runtime() {}

bool Runtime::buildSolution(const std::string &model_dir, BuildFlavor flavor,
                            HardwareTarget target) {
  // Get the build directory
  std::string build_dir = getBuildDir(model_dir, flavor, target);

  // Create the build directory if it doesn't exist
  try {
    fs::create_directories(build_dir);
  } catch (const std::exception &e) {
    last_error_ = "Failed to create build directory: " + std::string(e.what());
    return false;
  }

  // Run CMake to configure the solution
  if (!runCMake(model_dir, build_dir, flavor, target)) {
    return false;
  }

  // Run Make to build the solution
  if (!runMake(build_dir)) {
    return false;
  }

  return true;
}

bool Runtime::runSolution(const std::string &model_dir,
                          const std::string &input_file,
                          const std::string &output_file) {
  // Get the executable path
  std::string executable = getExecutablePath(model_dir, BuildFlavor::RELEASE,
                                             HardwareTarget::GRAYSKULL);

  // Check if the executable exists
  if (!fs::exists(executable)) {
    last_error_ = "Executable not found: " + executable;
    return false;
  }

  // Build the command
  std::stringstream cmd;
  cmd << executable;

  // Add input and output files if provided
  if (!input_file.empty()) {
    cmd << " --input=" << input_file;
  }

  if (!output_file.empty()) {
    cmd << " --output=" << output_file;
  }

  // Execute the command
  int result = std::system(cmd.str().c_str());
  if (result != 0) {
    last_error_ = "Failed to run solution: " + cmd.str();
    return false;
  }

  return true;
}

bool Runtime::profileSolution(const std::string &model_dir,
                              const std::string &input_file,
                              const std::string &report_file) {
  // Get the executable path
  std::string executable = getExecutablePath(model_dir, BuildFlavor::PROFILE,
                                             HardwareTarget::GRAYSKULL);

  // Check if the executable exists
  if (!fs::exists(executable)) {
    last_error_ = "Profiling executable not found: " + executable;
    return false;
  }

  // Build the command
  std::stringstream cmd;
  cmd << executable;

  // Add input file if provided
  if (!input_file.empty()) {
    cmd << " --input=" << input_file;
  }

  // Add report file
  cmd << " --profile-output=" << report_file;

  // Execute the command
  int result = std::system(cmd.str().c_str());
  if (result != 0) {
    last_error_ = "Failed to profile solution: " + cmd.str();
    return false;
  }

  return true;
}

bool Runtime::runCMake(const std::string &model_dir,
                       const std::string &build_dir, BuildFlavor flavor,
                       HardwareTarget target) {
  // Build the command
  std::stringstream cmd;
  cmd << "cd " << build_dir << " && cmake";

  // Add build type
  cmd << " -DCMAKE_BUILD_TYPE=" << buildFlavorToString(flavor);

  // Add hardware target
  cmd << " -DHARDWARE_TARGET=" << hardwareTargetToString(target);

  // Add path to model directory
  cmd << " " << model_dir;

  // Execute the command
  int result = std::system(cmd.str().c_str());
  if (result != 0) {
    last_error_ = "Failed to run CMake: " + cmd.str();
    return false;
  }

  return true;
}

bool Runtime::runMake(const std::string &build_dir) {
  // Build the command
  std::stringstream cmd;
  cmd << "cd " << build_dir << " && make";

  // Execute the command
  int result = std::system(cmd.str().c_str());
  if (result != 0) {
    last_error_ = "Failed to run Make: " + cmd.str();
    return false;
  }

  return true;
}

std::string Runtime::getBuildDir(const std::string &model_dir,
                                 BuildFlavor flavor,
                                 HardwareTarget target) const {
  return model_dir + "/build_" + buildFlavorToString(flavor) + "_" +
         hardwareTargetToString(target);
}

std::string Runtime::getExecutablePath(const std::string &model_dir,
                                       BuildFlavor flavor,
                                       HardwareTarget target) const {
  std::string build_dir = getBuildDir(model_dir, flavor, target);
  return build_dir + "/" + fs::path(model_dir).filename().string();
}

std::string Runtime::buildFlavorToString(BuildFlavor flavor) const {
  switch (flavor) {
  case BuildFlavor::RELEASE:
    return "Release";
  case BuildFlavor::DEBUG:
    return "Debug";
  case BuildFlavor::PROFILE:
    return "Profile";
  default:
    return "Release";
  }
}

std::string Runtime::hardwareTargetToString(HardwareTarget target) const {
  switch (target) {
  case HardwareTarget::GRAYSKULL:
    return "grayskull";
  case HardwareTarget::WORMHOLE:
    return "wormhole";
  case HardwareTarget::BLACKHOLE:
    return "blackhole";
  default:
    return "grayskull";
  }
}

std::string Runtime::getLastError() const { return last_error_; }

} // namespace tt_alchemist
