// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_UTILS_H
#define TT_RUNTIME_TTNN_OPERATIONS_UTILS_H

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "types_generated.h"
#include <cstdint>

namespace tt::runtime::ttnn::operations::utils {

bool isTilized(const ::tt::target::TensorRef *tensorRef);

bool inSystemMemory(const ::tt::target::TensorRef *tensorRef);

::tt::target::MemorySpace
getMemorySpace(const ::tt::target::TensorRef *tensorRef);

::ttnn::DataType getDataType(const ::tt::target::TensorRef *tensorRef);

std::optional<::ttnn::MemoryConfig>
createMemoryConfig(const ::tt::target::MemoryConfigDesc *memcfg,
                   const ::tt::target::TensorRef *tensorRef);

::tt::tt_metal::DistributedTensorConfig distributedTensorConfigFromFlatbuffer(
    const ::tt::target::DistributionStrategy *strategy);

} // namespace tt::runtime::ttnn::operations::utils
#endif
