// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef RUNTIME_LIB_TTMETAL_MESHSCHARD_UTILS_H
#define RUNTIME_LIB_TTMETAL_MESHSCHARD_UTILS_H

#include "ttmlir/Target/TTMetal/Target.h"
#include "ttnn/tensor/xtensor/partition.hpp"

#include "tt/runtime/detail/ttmetal/ttmetal.h"

namespace tt::runtime::ttmetal::meshshard_utils {

std::shared_ptr<tt_metal::DistributedHostBuffer> tensorFullToShard(
    const Tensor &input, const tt_metal::distributed::MeshShape &meshShape,
    const target::DataType dataType, const std::vector<size_t> &tensorShape,
    const target::MeshShardType meshShardType,
    const std::vector<int64_t> &meshShardDims);

std::shared_ptr<tt_metal::HostBuffer> tensorShardToFull(
    const Tensor &input, const tt_metal::distributed::MeshShape &meshShape,
    const target::DataType dataType, const std::vector<size_t> &tensorShape,
    const target::MeshShardType meshShardType,
    const std::vector<int64_t> &meshShardDims);

} // namespace tt::runtime::ttmetal::meshshard_utils

#endif // RUNTIME_LIB_TTMETAL_MESHSCHARD_UTILS_H
