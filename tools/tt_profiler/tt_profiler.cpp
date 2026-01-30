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

    std::string setOutputDirectory(const std::string& outputDirectory);
    std::string setAddress(const std::string& address);
    int setPort(int port);

private:
    ProfilerManager() = default;

    ProfilerManager(const ProfilerManager&) = delete;
    ProfilerManager& operator=(const ProfilerManager&) = delete;

    std::string m_outputDirectory;
    std::string m_address;
    int m_port;

    std::string tracyOutputFileName = "output.tracy";
    std::string tracyOpsTimeFileName = "tracy_ops_times.csv";
    std::string tracyOpsDataFileName = "tracy_ops_data.csv";
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

    pid_t start(const std::string& command);
    void stop();
    pid_t pid() const;

private:
    ProcessManager() = default;
    ~ProcessManager();

    ProcessManager(const ProcessManager&) = delete;
    ProcessManager& operator=(const ProcessManager&) = delete;

    pid_t m_pid = -1;
};

ProcessManager& ProcessManager::instance() {
  static ProcessManager instance;
  return instance;
}

pid_t ProcessManager::start(const std::string& command) {
  pid_t pid = fork();
  if (pid < 0) {
    throw std::runtime_error("Failed to fork process for command: " + command);
  }

  if (pid == 0) {
    execl("/bin/sh", "sh", "-c", command.c_str(), nullptr);
    perror("execl failed");
    _exit(127);
  }

  std::this_thread::sleep_for(std::chrono::seconds(2));

  m_pid = pid;
  return m_pid;
}

void ProcessManager::stop() {
  if (m_pid <= 0) {
    return;
  }

  kill(m_pid, SIGTERM);
  int status;
  waitpid(m_pid, &status, 0);
  m_pid = -1;
}

pid_t ProcessManager::pid() const {
  return m_pid;
}

ProcessManager::~ProcessManager() {
  stop();
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
}

void post_process_ops_time_data() {

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
