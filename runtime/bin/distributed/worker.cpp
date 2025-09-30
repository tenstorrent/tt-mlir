// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/distributed/worker/command_executor.h"

namespace {

class InputParser {
public:
  InputParser(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
      tokens_.push_back(std::string(argv[i]));
    }
  }

  bool cmdOptionExists(const std::string &option) const {
    return std::find(tokens_.begin(), tokens_.end(), option) != tokens_.end();
  }

  std::string getCmdOption(const std::string &option) const {
    auto iter = std::find(tokens_.begin(), tokens_.end(), option);
    if ((iter != tokens_.end()) && (++iter != tokens_.end())) {
      return *iter;
    }
    return "";
  }

  std::string getOption(const std::string &shortOpt,
                        const std::string &longOpt) const {
    if (cmdOptionExists(shortOpt)) {
      return getCmdOption(shortOpt);
    }
    if (cmdOptionExists(longOpt)) {
      return getCmdOption(longOpt);
    }
    return "";
  }

private:
  std::vector<std::string> tokens_;
};

uint16_t parsePort(const std::string &portStr) {
  LOG_ASSERT(!portStr.empty(), "Unexpected empty port number");

  int portNum = std::stoi(portStr);
  LOG_ASSERT(portNum >= 1 && portNum <= 65535,
             "Port must be between 1 and 65535");

  return static_cast<uint16_t>(portNum);
}

} // namespace

int main(int argc, char **argv) {
  std::string host = "localhost";
  uint16_t port = 8080;
  InputParser inputParser(argc, argv);

  if (inputParser.cmdOptionExists("-h") ||
      inputParser.cmdOptionExists("--host")) {
    host = inputParser.getOption("-h", "--host");
    LOG_ASSERT(!host.empty(), "Unexpected empty host address");
  }

  if (inputParser.cmdOptionExists("-p") ||
      inputParser.cmdOptionExists("--port")) {
    port = parsePort(inputParser.getOption("-p", "--port"));
  }

  LOG_INFO("Connecting to runtime controller ", host, ":", port);

  tt::runtime::distributed::worker::CommandExecutor commandExecutor;
  commandExecutor.connect(host, port);
  commandExecutor.run();

  return 0;
}
