// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

namespace ttnn::distributed {

enum class MeshShardDirection {
  FullToShard = 0,
  ShardToFull = 1
};

enum class MeshShardType {
  Identity = 0,
  Replicate = 1,
  Maximal = 2,
  Devices = 3
};

ttnn::Tensor mesh_shard(ttnn::Tensor &input,
                       ::ttnn::MeshDevice &meshDevice,
                       MeshShardDirection meshShardDirection,
                       MeshShardType meshShardType,
                       std::vector<int> shardShape,
                       std::vector<int64_t> shardDims);

} // namespace ttnn::distributed
