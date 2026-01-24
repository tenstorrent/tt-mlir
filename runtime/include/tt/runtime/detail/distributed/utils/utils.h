// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_UTILS_UTILS_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_UTILS_UTILS_H

#include "tt/runtime/types.h"

#include <cstdint>
#include <string>

namespace tt::runtime::distributed::utils {

std::string getWorkerExecutableCommand(
    std::uint16_t port,
    const std::optional<std::string> &workerPathOpt = std::nullopt,
    const std::optional<std::string> &hostnameOpt = std::nullopt);

uint32_t getNumProcesses(const std::string &rankBindingPath);

std::string
getTTRunCommand(uint16_t port,
                const ::tt::runtime::MultiProcessArgs &multiProcessArgs,
                const std::optional<std::string> &workerPathOpt = std::nullopt);

} // namespace tt::runtime::distributed::utils

#endif
