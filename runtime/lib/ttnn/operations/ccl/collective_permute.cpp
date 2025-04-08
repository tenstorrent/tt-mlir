// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/collective_permute.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"

#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

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
void run(const ::tt::target::ttnn::CollectivePermuteOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  const auto *fbSourceTargetPairs = op->source_target_pairs();
  std::vector<int64_t> sourceTargetPairs(fbSourceTargetPairs->begin(),
                                         fbSourceTargetPairs->end());

  LOG_ASSERT(input.storage_type() == ::ttnn::StorageType::MULTI_DEVICE,
             "Input of collective_permute must be multidevice storage. id:",
             op->in()->global_id());

  // Get list of individual per device tensors.
  std::vector<::ttnn::Tensor> originalDeviceTensors =
      ::ttnn::distributed::get_tensors_from_multi_device_storage(input);

  // Iterate through originalDeviceTensors and create mapping of device_id :
  // owned_storage. Also store device id to IDevice mapping.
  std::unordered_map<int, ::ttnn::Tensor> mappedOwnedStorageTensors;
  std::unordered_map<int, ::ttnn::IDevice *> mappedDeviceIds;

  for (const auto &tensor : originalDeviceTensors) {
    auto *tensorDevice = tensor.device();
    auto deviceId = tensorDevice->id();
    ::ttnn::Tensor hostTensor = ::ttnn::from_device(tensor);
    mappedOwnedStorageTensors[deviceId] = hostTensor;
    mappedDeviceIds[deviceId] = tensorDevice;
  }

  // Iterate through sourceTargetPairs and for each pair, get the source tensor
  // from the map and convert to device storage with dest device.
  std::vector<bool> foundDestDevices(originalDeviceTensors.size(), false);
  std::vector<::ttnn::Tensor> newDeviceTensors(originalDeviceTensors.size(),
                                               ::ttnn::Tensor());

  for (size_t i = 0; i < sourceTargetPairs.size(); i += 2) {
    int64_t src = sourceTargetPairs[i];
    int64_t dest = sourceTargetPairs[i + 1];

    auto srcHostTensorIt = mappedOwnedStorageTensors.find(src);
    LOG_ASSERT(srcHostTensorIt != mappedOwnedStorageTensors.end(),
               "Could not find device id in owned storage tensor map!");
    auto srcHostTensor = srcHostTensorIt->second;

    auto deviceIt = mappedDeviceIds.find(dest);
    LOG_ASSERT(deviceIt != mappedDeviceIds.end(),
               "Could not find device id in device map!");
    auto *device = deviceIt->second;

    std::optional<::ttnn::MemoryConfig> memoryConfig =
        srcHostTensor.memory_config();
    newDeviceTensors[dest] =
        ::ttnn::to_device(srcHostTensor, device, memoryConfig);
    foundDestDevices[dest] = true;
  }

  // Loop through all the devices that did not participate in the swaping and
  // set their tensor device shard values to 0.
  for (size_t i = 0; i < foundDestDevices.size(); i++) {
    if (foundDestDevices[i]) {
      continue;
    }

    auto srcHostTensorIt = mappedOwnedStorageTensors.find(i);
    LOG_ASSERT(srcHostTensorIt != mappedOwnedStorageTensors.end(),
               "Could not find device id in owned storage tensor map!");
    auto srcHostTensor = srcHostTensorIt->second;

    auto deviceIt = mappedDeviceIds.find(i);
    LOG_ASSERT(deviceIt != mappedDeviceIds.end(),
               "Could not find device id in device map!");
    auto *device = deviceIt->second;

    // We need to memset this tensor value to 0 based on collective permute
    // operation semantics
    void *dstPtr = ::tt::tt_metal::get_raw_host_data_ptr(srcHostTensor);
    size_t size = srcHostTensor.volume() * srcHostTensor.element_size();
    std::memset(dstPtr, 0, size);

    std::optional<::ttnn::MemoryConfig> memoryConfig =
        srcHostTensor.memory_config();
    newDeviceTensors[i] =
        ::ttnn::to_device(srcHostTensor, device, memoryConfig);
    foundDestDevices[i] = true;
  }

  // Combine all device tensor shards into a single multi device tensor with
  // multi device storage type.
  ::ttnn::Tensor out = ::ttnn::distributed::create_multi_device_tensor(
      newDeviceTensors, ::ttnn::StorageType::MULTI_DEVICE,
      ::ttnn::distributed::get_distributed_tensor_config_from_tensor(input));

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::ccl
