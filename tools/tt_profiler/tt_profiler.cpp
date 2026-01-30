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
  std::string outputFileName = "output.tracy";
  std::string outputPath = outputDirectory + "/" + outputFileName;
  std::string command = "/code/jan-2/tt-mlir/build/python_packages/tt_profiler/capture-release -o " + outputPath + " -a " + address + " -p " + std::to_string(port) + " -f";

  try {
    ProcessManager::instance().start(command);
  } catch (const std::runtime_error& e) {
    throw std::runtime_error("Failed to start profiler with command: " + command + ". Error: " + e.what());
  }

  std::cout << "Profiler started with output file: " << outputPath << std::endl;
}

void stop_profiler() {
  ProcessManager::instance().stop();
  std::cout << "Profiler stopped." << std::endl;
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
