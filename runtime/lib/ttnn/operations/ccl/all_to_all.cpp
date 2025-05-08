// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"

/*
Currently, TTNN does not support collective permute as a first class API.
Nor do they have send/recv point to point communication support.
Therefore, this algorithm uses the host as a fallback to do the mapping.
The collective permute operation takes a list of source_target_pairs that define
how tensor shards currently living in a src device should move to the dest
device. For example, for a 1x2 mesh system, you could have [0, 1], [1, 0]
source_target_pairs list. This indicates that the device shard living in device
0 should move to device 1, and the device shard living in device 1 should move
to device 0. In the situation where you have incomplete devices as the 'dest',
those devices will acquire a device shard with all values set to 0
*/
namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());
  ::ttnn::MeshDevice *meshDevice = input.mesh_device();
  LOG_ASSERT(meshDevice != nullptr, "Tensor must belong to a mesh device");
  int32_t splitDim = op->split_dim();
  int32_t concatDim = op->concat_dim();
  uint32_t clusterAxis = op->cluster_axis();
  uint32_t splitCount = meshDevice->shape()[clusterAxis];
  auto inputShape = input.get_logical_shape();
  uint32_t splitSize = inputShape[splitDim] / splitCount;

  std::vector<::ttnn::Tensor> slicedTensorsDevice;
  for (uint32_t idx = 0; idx < splitCount; idx++) {
    auto begins = ::ttnn::SmallVector<int32_t>(inputShape.rank(), 0);
    auto ends =
        ::ttnn::SmallVector<int32_t>(inputShape.cbegin(), inputShape.cend());
    auto steps = ::ttnn::SmallVector<int32_t>(inputShape.rank(), 1);
    begins[splitDim] = idx * splitSize;
    ends[splitDim] = (idx + 1) * splitSize;
    ::ttnn::Tensor sliced =
        ::ttnn::slice(input, begins, ends, steps); // device storage
    slicedTensorsDevice.push_back(sliced);
  }
  std::vector<std::vector<::ttnn::Tensor>> scatteredTensors(splitCount);
  for (uint32_t idx = 0; idx < splitCount; idx++) {
    std::vector<::ttnn::Tensor> slicedTensorsHost = // Host
        ::ttnn::distributed::get_device_tensors(
            ::ttnn::from_device(slicedTensorsDevice[idx]));
    for (uint32_t i = 0; i < splitCount; i++) {
      scatteredTensors[i].push_back(slicedTensorsHost[i]);
    }
  }
  std::vector<::ttnn::Tensor> scatteredTensorsMulti(splitCount,
                                                    ::ttnn::Tensor());
  for (uint32_t idx = 0; idx < splitCount; idx++) {
    ::ttnn::Tensor scattered = ::ttnn::distributed::aggregate_as_tensor(
        scatteredTensors[idx], input.get_distributed_tensor_config());

    scatteredTensorsMulti[idx] =
        ::ttnn::to_device(scattered, meshDevice, input.memory_config());
  }
  auto out = ::ttnn::concat(scatteredTensorsMulti, concatDim);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
