// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_SERVER_UTILS_UTILS_H
#define TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_SERVER_UTILS_UTILS_H

#include <cstdint>

namespace tt::runtime::ttnn::distributed::server {

uint64_t nextCommandId();

} // namespace tt::runtime::ttnn::distributed::server
#endif // TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_SERVER_UTILS_UTILS_H
