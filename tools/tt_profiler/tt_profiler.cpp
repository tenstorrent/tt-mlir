// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_profiler.hpp"

#include "nanobind_headers.h"

#include <thread>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <sys/wait.h>

namespace nb = nanobind;

class ProfilerManager {
public: 
    static ProfilerManager& instance();

    std::string getOutputDirectory();
    std::string getAddress();
    int getPort();
    std::string getCaptureReleaseCommand();
    std::string getCSVExportTimesCommand();
    std::string getCSVExportDataCommand();

    std::string setOutputDirectory(const std::string& outputDirectory);
    std::string setAddress(const std::string& address);
    int setPort(int port);

    std::string tracyOutputFileName = "output.tracy";
    std::string tracyOpsTimeFileName = "tracy_ops_times.csv";
    std::string tracyOpsDataFileName = "tracy_ops_data.csv";

private:
    ProfilerManager() = default;

    ProfilerManager(const ProfilerManager&) = delete;
    ProfilerManager& operator=(const ProfilerManager&) = delete;

    std::string m_outputDirectory;
    std::string m_address;
    int m_port;
};

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

std::string ProfilerManager::getCaptureReleaseCommand() {
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

class ProcessManager {
public:
    static ProcessManager& instance();

    void start(const std::string& command);
    void stop();
    void execute(const std::string& command, const std::string& output_file);
    pid_t pid() const;

private:
    ProcessManager() = default;

    ProcessManager(const ProcessManager&) = delete;
    ProcessManager& operator=(const ProcessManager&) = delete;

    pid_t m_pid = -1;
};

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
  ProfilerManager::instance().setOutputDirectory(outputDirectory);
  ProfilerManager::instance().setAddress(address);
  ProfilerManager::instance().setPort(port);
  std::string command = ProfilerManager::instance().getCaptureReleaseCommand();

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

namespace tt::profiler::python {
void registerProfilerBindings(nb::module_ &m) {
  m.def("check", []() {
    std::cout << "Profiler started" << std::endl;
  });

  m.def("start_profiler", &start_profiler,
        nb::arg("outputDirectory"),
        nb::arg("address") = "localhost",
        nb::arg("port") = 8086,
        "Start the profiler with given parameters");
  
  m.def("stop_profiler", &stop_profiler,
        "Stop the profiler");
}
} // namespace tt::profiler::python
