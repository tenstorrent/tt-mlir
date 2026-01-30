// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <thread>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <sys/wait.h>
#include <stdlib.h>
#include <vector>

#include "profiler_impl.h"

ProfilerManager& ProfilerManager::instance() {
  static ProfilerManager instance;
  return instance;
}

std::string ProfilerManager::getOutputDirectory() {
  return m_outputDirectory;
}

std::string ProfilerManager::getAddress() {
  return m_address;
}

int ProfilerManager::getPort() {
  return m_port;
}

std::string ProfilerManager::getCaptureCommand() {
  std::string outputPath = m_outputDirectory + "/" + tracyOutputFileName;
  return "/code/jan-2/tt-mlir/build/python_packages/tt_profiler/capture-release -o " + outputPath + " -a " + m_address + " -p " + std::to_string(m_port) + " -f";
}

std::string ProfilerManager::getCSVExportTimesCommand() {
  std::string tracyFilePath = m_outputDirectory + "/" + tracyOutputFileName;
  std::string tracyOpsTimesFilePath = m_outputDirectory + "/" + tracyOpsTimeFileName;

  std::string childCalls = "CompileProgram,HWCommandQueue_write_buffer";
  return "/code/jan-2/tt-mlir/build/python_packages/tt_profiler/csvexport-release -u -p TT_DNN -x " + childCalls + " " + tracyFilePath;
}

std::string ProfilerManager::getCSVExportDataCommand() {
  std::string tracyFilePath = m_outputDirectory + "/" + tracyOutputFileName;
  std::string tracyOpsDataFilePath = m_outputDirectory + "/" + tracyOpsDataFileName;

  return "/code/jan-2/tt-mlir/build/python_packages/tt_profiler/csvexport-release -m -s \";\" " + tracyFilePath;
}

std::string ProfilerManager::setOutputDirectory(const std::string& outputDirectory) {
  m_outputDirectory = outputDirectory;
  return m_outputDirectory;
}


std::string ProfilerManager::setAddress(const std::string& address) {
  m_address = address;
  return m_address;
}

int ProfilerManager::setPort(int port) {
  m_port = port;
  return m_port;
}
  
ProcessManager& ProcessManager::instance() {
  static ProcessManager instance;
  return instance;
}

void ProcessManager::start(const std::string& command) {
  if (m_pid > 0) {
    throw std::runtime_error("Process already running with PID: " + std::to_string(m_pid));
  }

  std::cout << "Starting process with command: " << command << std::endl;

  pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error("Failed to fork process for command: " + command);
  }

  if (pid == 0) {
    setsid();
    execl("/bin/sh", "sh", "-c", command.c_str(), nullptr);
    perror("execl failed");
    _exit(1);
  }

  m_pid = pid;
  std::this_thread::sleep_for(std::chrono::seconds(2));
}

void ProcessManager::stop() {
  if (m_pid <= 0) {
    return;
  }

  killpg(m_pid, SIGINT);

  int status;
  waitpid(m_pid, &status, 0);

  m_pid = -1;
}

void ProcessManager::execute(const std::string& command, const std::string& output_file) {
  std::string cmd = command + " > " + output_file;
  int ret = system(cmd.c_str());
  if (ret != 0) {
      throw std::runtime_error("Command failed with return code: " + std::to_string(ret));
  }
}

pid_t ProcessManager::pid() const {
  return m_pid;
}

void post_process_op_times() {
  std::string command = ProfilerManager::instance().getCSVExportTimesCommand();
  std::string opTimesFilePath = ProfilerManager::instance().getOutputDirectory() + "/" + ProfilerManager::instance().tracyOpsTimeFileName;

  try {
    ProcessManager::instance().execute(command, opTimesFilePath);
  } catch (const std::runtime_error& e) {
    throw std::runtime_error("Failed to export ops time data with command: " + command + ". Error: " + e.what());
  }

  std::cout << "Op times exported." << std::endl;
}

void post_process_op_data() {
  std::string command = ProfilerManager::instance().getCSVExportDataCommand();
  std::string opDataFilePath = ProfilerManager::instance().getOutputDirectory() + "/" + ProfilerManager::instance().tracyOpsDataFileName;

  try {
    ProcessManager::instance().execute(command, opDataFilePath);
  } catch (const std::runtime_error& e) {
    throw std::runtime_error("Failed to export ops data with command: " + command + ". Error: " + e.what());
  }

  std::cout << "Op data exported." << std::endl;
}

void start_profiler(std::string outputDirectory, std::string address, int port) {
  std::vector<std::string> profilerEnvVars = {
    "TT_METAL_CLEAR_L1=1",
    "TT_METAL_DEVICE_PROFILER=1",
    "TTNN_OP_PROFILER=1",
    "TT_METAL_DEVICE_PROFILER_DISPATCH=0",
    "TT_METAL_PROFILER_CPP_POST_PROCESS=1",
    "TT_METAL_PROFILER_MID_RUN_DUMP=1",
  };

  for (const auto &envVar : profilerEnvVars) {
    auto pos = envVar.find('=');
    if (pos == std::string::npos) {
        throw std::runtime_error("Invalid env var: " + envVar);
    }

    std::string key = envVar.substr(0, pos);
    std::string value = envVar.substr(pos + 1);

    if (setenv(key.c_str(), value.c_str(), 1) != 0) {
        throw std::runtime_error("Failed to set environment variable: " + envVar);
    }
  }

  ProfilerManager::instance().setOutputDirectory(outputDirectory);
  ProfilerManager::instance().setAddress(address);
  ProfilerManager::instance().setPort(port);
  std::string command = ProfilerManager::instance().getCaptureCommand();

  try {
    ProcessManager::instance().start(command);
  } catch (const std::runtime_error& e) {
    throw std::runtime_error("Failed to start profiler with command: " + command + ". Error: " + e.what());
  }

  std::cout << "Profiler started." << std::endl;
}

void stop_profiler() {
  ProcessManager::instance().stop();
  std::cout << "Profiler stopped." << std::endl;

  post_process_op_times();
  post_process_op_data();
}
