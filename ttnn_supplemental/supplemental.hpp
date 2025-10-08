// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <cstdint>
#include <optional>
#include <vector>

namespace ttnn::supplemental {

enum class MeshShardDirection { FullToShard = 0, ShardToFull = 1 };

enum class MeshShardType {
  Identity = 0,
  Replicate = 1,
  Maximal = 2,
  Devices = 3
};

// Mesh shard operation
ttnn::Tensor mesh_shard(ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice,
                        MeshShardDirection meshShardDirection,
                        MeshShardType meshShardType,
                        std::vector<int> shardShape,
                        std::vector<int64_t> shardDims);

// All gather operation
ttnn::Tensor all_gather(
    ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice, int32_t dim,
    uint32_t cluster_axis, uint32_t num_links,
    std::optional<::tt::tt_metal::MemoryConfig> memory_config = std::nullopt);

// Reduce scatter operation
ttnn::Tensor reduce_scatter(
    ttnn::Tensor &input, ::ttnn::MeshDevice &meshDevice, int32_t scatter_dim,
    uint32_t cluster_axis, uint32_t num_links,
    std::optional<::tt::tt_metal::MemoryConfig> memory_config = std::nullopt);

// Collective permute operation
ttnn::Tensor collective_permute(ttnn::Tensor &input,
                                std::vector<int64_t> source_target_pairs);

// Point to point operation
ttnn::Tensor
point_to_point(ttnn::Tensor &input, std::vector<uint32_t> send_coord,
               std::vector<uint32_t> receive_coord,
               std::optional<ttnn::Tensor> accum_tensor = std::nullopt);

} // namespace ttnn::supplemental
