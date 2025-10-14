// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_UTILS_UTILS_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_UTILS_UTILS_H

#include "tt/runtime/types.h"
#include <string>

namespace tt::runtime::distributed::utils {

std::string getWorkerExecutableCommand(std::uint16_t port);

uint32_t getNumProcesses(const std::string &rankBindingPath);

std::string
getTTRunCommand(uint16_t port,
                const ::tt::runtime::MultiProcessArgs &multiProcessArgs);

} // namespace tt::runtime::distributed::utils

#endif
