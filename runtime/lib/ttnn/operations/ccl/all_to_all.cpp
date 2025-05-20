// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/all_to_all.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"

/*
TTNN does not yet expose an All-to-All collective as a first-class API,
and there is no point-to-point send/recv either.  To achieve the same
effect we fall back to the host:
1. Each device slices its local tensor shard.
2. The slices are copied to the host.
3. The host rearranges the slices so every device gets the piece it needs
   (a manual All-to-All shuffle).
4. The rearranged slices are copied back to the devices and concatenated.
*/
namespace tt::runtime::ttnn::operations::ccl {
void run(const ::tt::target::ttnn::AllToAllOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttnn::MeshDevice *meshDevice = inputTensor.mesh_device();
  LOG_ASSERT(meshDevice != nullptr, "Tensor must belong to a mesh device");

  const int32_t splitDim = op->split_dim();
  const int32_t concatDim = op->concat_dim();
  const uint32_t splitCount = op->split_count();
  const uint32_t clusterAxis = op->cluster_axis();

  const auto meshShape = meshDevice->shape();
  const auto inputShape = inputTensor.logical_shape();
  LOG_ASSERT(inputShape[splitDim] % splitCount == 0,
             "Input dimension along splitDim must be divisible by splitCount");

  const uint32_t splitSize = inputShape[splitDim] / splitCount;

  // 1. Slice the input tensor on-device and materialize per-device host copies.
  //    slicedTensorsMulti[sliceIdx][device_idx]
  std::vector<std::vector<::ttnn::Tensor>> slicedTensorsMulti(splitCount);

  for (uint32_t sliceIdx = 0; sliceIdx < splitCount; ++sliceIdx) {
    auto begins = ::ttnn::SmallVector<int32_t>(inputShape.rank(), 0);
    auto ends =
        ::ttnn::SmallVector<int32_t>(inputShape.cbegin(), inputShape.cend());
    auto steps = ::ttnn::SmallVector<int32_t>(inputShape.rank(), 1);

    begins[splitDim] = sliceIdx * splitSize;
    ends[splitDim] = (sliceIdx + 1) * splitSize;

    ::ttnn::Tensor slice_device =
        ::ttnn::slice(inputTensor, begins, ends, steps); // device
    slicedTensorsMulti[sliceIdx] = ::ttnn::distributed::get_device_tensors(
        ::ttnn::from_device(slice_device)); // host view
  }

  // 2. Transpose shards across devices inside each cluster.
  //    gatheredTensorsSharded[sliceIdx][device_idx]
  std::vector<std::vector<::ttnn::Tensor>> gatheredTensorsSharded(
      splitCount, std::vector<::ttnn::Tensor>(meshDevice->num_devices()));

  // (clusterId, DevId in the cluster) -> flat device index
  const auto clusterToDevIdx = [&](uint32_t clusterId,
                                   uint32_t innerDevId) -> uint32_t {
    uint32_t index;
    if (clusterAxis == 0) {
      index = innerDevId * meshShape[1] + clusterId;
    } else {
      index = clusterId * meshShape[1] + innerDevId;
    }
    return index;
  };

  for (uint32_t clusterId = 0; clusterId < meshShape[1 - clusterAxis];
       ++clusterId) {
    for (uint32_t srcRank = 0; srcRank < splitCount; ++srcRank) {
      const uint32_t srcDev = clusterToDevIdx(clusterId, srcRank);
      for (uint32_t sliceIdx = 0; sliceIdx < splitCount; ++sliceIdx) {
        const uint32_t dstDev = clusterToDevIdx(clusterId, sliceIdx);
        gatheredTensorsSharded[srcRank][dstDev] =
            slicedTensorsMulti[sliceIdx][srcDev];
      }
    }
  }

  // 3. Aggregate per-device shards back into device tensors and move them onto
  // the mesh.
  std::vector<::ttnn::Tensor> gatheredTensorsMulti(splitCount);
  for (uint32_t sliceIdx = 0; sliceIdx < splitCount; ++sliceIdx) {
    ::ttnn::Tensor shardedTensor = ::ttnn::distributed::aggregate_as_tensor(
        gatheredTensorsSharded[sliceIdx],
        inputTensor.distributed_tensor_config());

    gatheredTensorsMulti[sliceIdx] = ::ttnn::to_device(
        shardedTensor, meshDevice, inputTensor.memory_config());
  }

  // 4. Concatenate along the requested dimension and register output.
  ::ttnn::Tensor output = ::ttnn::concat(gatheredTensorsMulti, concatDim);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::ccl
