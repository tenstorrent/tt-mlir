// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/runtime.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/utils.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"
#include "ttnn/tensor/shape/small_vector.hpp"
#include "ttnn/tensor/types.hpp"

namespace tt::runtime::ttnn {

using ::tt::runtime::DeviceRuntime;
using ::tt::tt_metal::BorrowedStorage;
using ::tt::tt_metal::DistributedTensorConfig;
using ::tt::tt_metal::OwnedStorage;
using ::tt::tt_metal::raise_unsupported_storage;

template <typename StorageType, typename ElementType>
static StorageType createStorage(ElementType *ptr, std::uint32_t numElements) {
  if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
    return BorrowedStorage(
        ::tt::tt_metal::borrowed_buffer::Buffer<ElementType>(ptr, numElements),
        [] {}, [] {});
  } else if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
    auto data = std::vector<ElementType>(ptr, ptr + numElements);
    auto buffer = ::tt::tt_metal::owned_buffer::create(std::move(data));
    return OwnedStorage(std::move(buffer));
  } else {
    raise_unsupported_storage<StorageType>();
  }
}

template <typename StorageType>
static StorageType createStorage(void *ptr, std::uint32_t numElements,
                                 ::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return createStorage<StorageType>(static_cast<float *>(ptr), numElements);
  case ::tt::target::DataType::BFloat16:
    return createStorage<StorageType>(static_cast<bfloat16 *>(ptr),
                                      numElements);
  case ::tt::target::DataType::UInt32:
    return createStorage<StorageType>(static_cast<uint32_t *>(ptr),
                                      numElements);
  case ::tt::target::DataType::UInt16:
    return createStorage<StorageType>(static_cast<uint16_t *>(ptr),
                                      numElements);
  default:
    LOG_FATAL("Unsupported data type");
  }
}

static ::ttnn::Tensor
createOwnedTensor(std::shared_ptr<void> data,
                  std::vector<std::uint32_t> const &shape,
                  std::vector<std::uint32_t> const &stride,
                  std::uint32_t itemsize, ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];

  ::tt::tt_metal::SmallVector<uint32_t> small_vector_shape(shape.begin(),
                                                           shape.end());

  return ::ttnn::Tensor(
      createStorage<OwnedStorage>(data.get(), numElements, dataType),
      ::ttnn::Shape(small_vector_shape), utils::toTTNNDataType(dataType),
      ::ttnn::Layout::ROW_MAJOR);
}

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];

  ::tt::tt_metal::SmallVector<uint32_t> small_vector_shape(shape.begin(),
                                                           shape.end());

  auto tensor = std::make_shared<::ttnn::Tensor>(
      createStorage<BorrowedStorage>(data.get(), numElements, dataType),
      ::ttnn::Shape(small_vector_shape), utils::toTTNNDataType(dataType),
      ::ttnn::Layout::ROW_MAJOR);
  return Tensor(std::static_pointer_cast<void>(tensor), data,
                DeviceRuntime::TTNN);
}

Tensor
createTensor(std::vector<std::shared_ptr<void>> &data,
             std::vector<std::uint32_t> const &shape,
             std::vector<std::uint32_t> const &stride, std::uint32_t itemsize,
             ::tt::target::DataType dataType,
             std::unordered_map<std::string, std::string> const &strategy) {
  std::vector<::ttnn::Tensor> tensorShards;
  tensorShards.resize(data.size());
  std::transform(data.begin(), data.end(), tensorShards.begin(),
                 [&](std::shared_ptr<void> &dataShard) -> ::ttnn::Tensor {
                   return createOwnedTensor(dataShard, shape, stride, itemsize,
                                            dataType);
                 });
  DistributedTensorConfig distributionStrategy =
      ::tt::tt_metal::get_distributed_tensor_config(strategy);
  std::shared_ptr<::ttnn::Tensor> tensor = std::make_shared<::ttnn::Tensor>(
      ::ttnn::distributed::api::create_multi_device_tensor(
          tensorShards, ::tt::tt_metal::StorageType::MULTI_DEVICE_HOST,
          distributionStrategy));
  std::shared_ptr<std::vector<std::shared_ptr<void>>> borrowedData =
      std::make_shared<std::vector<std::shared_ptr<void>>>(data);
  return Tensor(std::static_pointer_cast<void>(tensor),
                std::static_pointer_cast<void>(borrowedData),
                DeviceRuntime::TTNN);
}

tt::target::DataType getTensorDataType(Tensor tensor) {
  const ::ttnn::Tensor &nnTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  return utils::fromTTNNDataType(nnTensor.get_dtype());
}

size_t getNumAvailableDevices() {
  return ::tt::tt_metal::GetNumAvailableDevices();
}

Device openDevice(DeviceIds const &deviceIds, size_t numHWCQs) {
  LOG_ASSERT(deviceIds.size(), "No devices specified");
  ::tt::tt_metal::distributed::MeshShape grid =
      std::make_pair(1, deviceIds.size());
  std::shared_ptr<::ttnn::MeshDevice> meshDevice = ::ttnn::MeshDevice::create(
      grid, kL1SmallSize, DEFAULT_TRACE_REGION_SIZE, numHWCQs,
      ::tt::tt_metal::DispatchCoreType::WORKER);

  bool enableAsync = debug::Env::get().enableAsyncTTNN;
  for (::ttnn::Device *device : meshDevice->get_devices()) {
    device->enable_async(enableAsync);
  }

  return Device(std::static_pointer_cast<void>(meshDevice),
                DeviceRuntime::TTNN);
}

void closeDevice(Device device) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE)
  for (::ttnn::Device *ttnnDevice : ttnnMeshDevice.get_devices()) {
    ::tt::tt_metal::detail::DumpDeviceProfileResults(ttnnDevice);
  }
#endif

  ttnnMeshDevice.close_devices();
}

void deallocateBuffers(Device deviceHandle) {
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  for (::ttnn::Device *device : meshDevice.get_devices()) {
    device->deallocate_buffers();
  }
}

static ::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  LOG_ASSERT(isTTNN, "Unsupported binary format");
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             std::vector<Tensor> const &inputHandles,
             std::vector<Tensor> const &outputHandles) {
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  ::tt::target::ttnn::TTNNBinary const &fbb = *getBinary(executableHandle);
  std::vector<::ttnn::Tensor *> inputs;
  inputs.reserve(inputHandles.size());
  for (auto &input : inputHandles) {
    LOG_ASSERT(input.matchesRuntime(DeviceRuntime::TTNN));
    inputs.push_back(static_cast<::ttnn::Tensor *>(input.handle.get()));
  }
  std::vector<::ttnn::Tensor *> outputs;
  outputs.reserve(outputHandles.size());
  for (auto &output : outputHandles) {
    LOG_ASSERT(output.matchesRuntime(DeviceRuntime::TTNN));
    outputs.push_back(static_cast<::ttnn::Tensor *>(output.handle.get()));
  }
  tt::runtime::ttnn::runProgram(meshDevice, fbb.programs()->Get(programIndex),
                                inputs, outputs);
  return Event(nullptr, DeviceRuntime::TTNN);
}

void wait(Event event) {
  // Not implemented
  LOG_ASSERT(event.matchesRuntime(DeviceRuntime::TTNN));
}

} // namespace tt::runtime::ttnn
