// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_UTILS_UTILS_H
#define TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_UTILS_UTILS_H

#include <cstdint>

namespace tt::runtime::ttnn::distributed::client {

uint64_t nextResponseId();

} // namespace tt::runtime::ttnn::distributed::client
#endif // TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_UTILS_UTILS_H
