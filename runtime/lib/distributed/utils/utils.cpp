// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/utils/utils.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"

namespace tt::runtime::distributed::utils {

std::string
getWorkerExecutableCommand(std::uint16_t port,
                           const std::optional<std::string> &workerPathOpt) {
  std::string workerPath = workerPathOpt.value_or(
      std::filesystem::path(RuntimeContext::instance().getMlirHome()) /
      "build/runtime/bin/distributed/worker");

  LOG_ASSERT(std::filesystem::exists(workerPath),
             "Distributed worker path does not exist: ", workerPath);

  std::string portString = std::to_string(port);

  return workerPath + " --port " + portString;
}

uint32_t getNumProcesses(const std::string &rankBindingPath) {
  std::ifstream rankBindingFile(rankBindingPath.c_str());
  std::string line;
  uint32_t rankCount = 0;

  while (std::getline(rankBindingFile, line)) {
    if (line.find("  - rank:") != std::string::npos) {
      rankCount++;
    }
  }

  LOG_ASSERT(rankCount > 0, "Unexpected rank count 0 in rank binding file");

  return rankCount;
}

std::string
getTTRunCommand(uint16_t port,
                const ::tt::runtime::MultiProcessArgs &multiProcessArgs,
                const std::optional<std::string> &workerPathOpt) {
  std::ostringstream oss;

  oss << "cd " << RuntimeContext::instance().getMetalHome() << " && ";

  oss << "./ttnn/ttnn/distributed/ttrun.py " << multiProcessArgs.toArgString()
      << " "
      << "bash -c "
      << "\"" << getWorkerExecutableCommand(port, workerPathOpt) << "\"";

  return oss.str();
}
} // namespace tt::runtime::distributed::utils
