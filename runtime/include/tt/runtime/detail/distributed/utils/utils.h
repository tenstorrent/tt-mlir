// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_UTILS_UTILS_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_UTILS_UTILS_H

#include <string>

namespace tt::runtime::distributed::utils {

std::string getWorkerExecutableCommand(std::uint16_t port);

} // namespace tt::runtime::distributed::utils

#endif
