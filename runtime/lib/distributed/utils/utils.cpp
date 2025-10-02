// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/utils/utils.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/utils.h"

namespace tt::runtime::distributed::utils {

std::string getWorkerExecutableCommand(std::uint16_t port) {
  std::string mlirHome = RuntimeContext::instance().getMlirHome();
  std::string portString = std::to_string(port);
  return "/" + mlirHome + "/build/runtime/bin/distributed/worker --port " +
         portString;
}

} // namespace tt::runtime::distributed::utils
