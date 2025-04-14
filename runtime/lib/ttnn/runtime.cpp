// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "Constants.h"

#include "tt-metalium/small_vector.hpp"
#include "tt/runtime/detail/common.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/utils.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"
#include "ttnn/tensor/types.hpp"

#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

namespace tt::runtime::ttnn {

using ::tt::runtime::DeviceRuntime;
using ::tt::tt_metal::BorrowedStorage;
using ::tt::tt_metal::DistributedTensorConfig;
using ::tt::tt_metal::OwnedStorage;
using ::tt::tt_metal::raise_unsupported_storage;

template <typename ElementType>
static OwnedStorage createOwnedStorage(ElementType const *ptr,
                                       std::uint32_t numElements) {
  ::tt::tt_metal::owned_buffer::Buffer<ElementType> buffer;
  if (ptr != nullptr) {
    auto data = std::vector<ElementType>(ptr, ptr + numElements);
    buffer = ::tt::tt_metal::owned_buffer::create<ElementType>(std::move(data));
  } else {
    buffer = ::tt::tt_metal::owned_buffer::create<ElementType>(numElements);
  }
  return OwnedStorage(std::move(buffer));
}

template <typename ElementType>
static BorrowedStorage createBorrowedStorage(ElementType *ptr,
                                             std::uint32_t numElements) {
  LOG_ASSERT(ptr != nullptr, "Cannot create borrowed storage from nullptr");
  return BorrowedStorage(
      ::tt::tt_metal::borrowed_buffer::Buffer<ElementType>(ptr, numElements),
      [] {}, [] {});
}

static OwnedStorage createOwnedStorage(void const *ptr,
                                       std::uint32_t numElements,
                                       ::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return createOwnedStorage(static_cast<float const *>(ptr), numElements);
  case ::tt::target::DataType::BFloat16:
    return createOwnedStorage(static_cast<bfloat16 const *>(ptr), numElements);
  case ::tt::target::DataType::UInt32:
    return createOwnedStorage(static_cast<uint32_t const *>(ptr), numElements);
  case ::tt::target::DataType::UInt16:
    return createOwnedStorage(static_cast<uint16_t const *>(ptr), numElements);
  case ::tt::target::DataType::UInt8:
    return createOwnedStorage(static_cast<uint8_t const *>(ptr), numElements);
  case ::tt::target::DataType::Int32:
    return createOwnedStorage(static_cast<int32_t const *>(ptr), numElements);
  default:
    LOG_FATAL("Unsupported data type");
  }
}

static BorrowedStorage createBorrowedStorage(void *ptr,
                                             std::uint32_t numElements,
                                             ::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return createBorrowedStorage(static_cast<float *>(ptr), numElements);
  case ::tt::target::DataType::BFloat16:
    return createBorrowedStorage(static_cast<bfloat16 *>(ptr), numElements);
  case ::tt::target::DataType::UInt32:
    return createBorrowedStorage(static_cast<uint32_t *>(ptr), numElements);
  case ::tt::target::DataType::UInt16:
    return createBorrowedStorage(static_cast<uint16_t *>(ptr), numElements);
  case ::tt::target::DataType::UInt8:
    return createBorrowedStorage(static_cast<uint8_t *>(ptr), numElements);
  case ::tt::target::DataType::Int32:
    return createBorrowedStorage(static_cast<int32_t *>(ptr), numElements);
  default:
    LOG_FATAL("Unsupported data type");
  }
}

static ::ttnn::Tensor
createOwnedTTNNTensor(void const *data, std::vector<std::uint32_t> const &shape,
                      std::vector<std::uint32_t> const &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];

  return ::ttnn::Tensor(createOwnedStorage(data, numElements, dataType),
                        ::ttnn::Shape(shape), utils::toTTNNDataType(dataType),
                        ::ttnn::Layout::ROW_MAJOR);
}

static ::tt::runtime::Tensor createNullTensor() {
  return ::tt::runtime::Tensor(nullptr, nullptr, DeviceRuntime::TTNN);
}

static ::tt::runtime::Tensor toHostSingleTensor(::tt::runtime::Tensor tensor,
                                                bool untilize) {
  const ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);

  const ::ttnn::Tensor &deviceTensor = tensorWrapper.getTensor();
  bool shouldRetain = tensorWrapper.shouldRetain();

  ::ttnn::Tensor hostTensor = ::ttnn::from_device(deviceTensor);

  if (untilize) {
    hostTensor = ::ttnn::to_layout(hostTensor, ::ttnn::Layout::ROW_MAJOR,
                                   std::nullopt, std::nullopt,
                                   static_cast<::ttnn::IDevice *>(nullptr));
  }

  return utils::createRuntimeTensorFromTTNN(hostTensor, shouldRetain);
}

static DeviceVariant getTargetDevice(::ttnn::MeshDevice &meshDevice) {
  if (meshDevice.num_devices() == 1) {
    return std::ref(*(meshDevice.get_device(::ttnn::MeshCoordinate(0, 0))));
  }
  return std::ref(meshDevice);
}

static tt::runtime::MemoryView
createMemoryView(tt::tt_metal::detail::MemoryView const &memoryView) {
  return tt::runtime::MemoryView{
      .numBanks = memoryView.num_banks,
      .totalBytesPerBank = memoryView.total_bytes_per_bank,
      .totalBytesAllocatedPerBank = memoryView.total_bytes_allocated_per_bank,
      .totalBytesFreePerBank = memoryView.total_bytes_free_per_bank,
      .largestContiguousBytesFreePerBank =
          memoryView.largest_contiguous_bytes_free_per_bank,
      .blockTable = memoryView.block_table,
  };
}

static ::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  LOG_ASSERT(isTTNN, "Unsupported binary format");
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

::tt::runtime::Tensor
createOwnedHostTensor(void const *data, std::vector<std::uint32_t> const &shape,
                      std::vector<std::uint32_t> const &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {

  return utils::createRuntimeTensorFromTTNN(
      createOwnedTTNNTensor(data, shape, stride, itemsize, dataType));
}

::tt::runtime::Tensor
createBorrowedHostTensor(void *data, std::vector<std::uint32_t> const &shape,
                         std::vector<std::uint32_t> const &stride,
                         std::uint32_t itemsize,
                         ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];

  ::ttnn::Tensor tensor(createBorrowedStorage(data, numElements, dataType),
                        ::ttnn::Shape(shape), utils::toTTNNDataType(dataType),
                        ::ttnn::Layout::ROW_MAJOR);

  return utils::createRuntimeTensorFromTTNN(tensor);
}

static ::ttnn::Tensor
createOwnedFromBorrowedTTNNTensor(const ::ttnn::Tensor &borrowedTensor) {
  BorrowedStorage borrowedStorage =
      std::get<BorrowedStorage>(borrowedTensor.get_storage());
  OwnedStorage ownedStorage = std::visit(
      [](auto &&bufferVariant) {
        return createOwnedStorage(bufferVariant.data(), bufferVariant.size());
      },
      borrowedStorage.buffer);
  return ::ttnn::Tensor(ownedStorage, borrowedTensor.get_tensor_spec());
}

static ::tt::runtime::Tensor createMultiDeviceHostTensor(
    std::vector<::ttnn::Tensor> const &tensorShards,
    std::unordered_map<std::string, std::string> const &strategy) {
  // Currently metal distributed API allows creating multi-device tensors only
  // from owned tensors, so we have to convert all borrowed tensors into owned.
  // https://github.com/tenstorrent/tt-metal/issues/19177#issuecomment-2779877793
  std::vector<::ttnn::Tensor> ownedTensorShards;
  ownedTensorShards.reserve(tensorShards.size());
  std::transform(
      tensorShards.begin(), tensorShards.end(),
      std::back_inserter(ownedTensorShards),
      [&](::ttnn::Tensor tensorShard) -> ::ttnn::Tensor {
        LOG_ASSERT(
            tensorShard.storage_type() == ::ttnn::StorageType::OWNED ||
                tensorShard.storage_type() == ::ttnn::StorageType::BORROWED,
            "Multi-device host tensor can be created only from host tensors "
            "with owned or borrowed storage");
        return tensorShard.storage_type() == ::ttnn::StorageType::OWNED
                   ? tensorShard
                   : createOwnedFromBorrowedTTNNTensor(tensorShard);
      });

  DistributedTensorConfig distributionStrategy =
      ::tt::tt_metal::get_distributed_tensor_config(strategy);
  ::ttnn::Tensor tensor = ::ttnn::distributed::create_multi_device_tensor(
      ownedTensorShards, ::ttnn::StorageType::MULTI_DEVICE_HOST,
      distributionStrategy);

  return utils::createRuntimeTensorFromTTNN(tensor);
}

::tt::runtime::Tensor createOwnedMultiDeviceHostTensor(
    std::vector<void const *> const &data,
    std::vector<std::uint32_t> const &shape,
    std::vector<std::uint32_t> const &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    std::unordered_map<std::string, std::string> const &strategy) {
  std::vector<::ttnn::Tensor> ttnnTensorShards;
  ttnnTensorShards.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(ttnnTensorShards),
                 [&](void const *dataShard) -> ::ttnn::Tensor {
                   return createOwnedTTNNTensor(dataShard, shape, stride,
                                                itemsize, dataType);
                 });
  return createMultiDeviceHostTensor(ttnnTensorShards, strategy);
}

::tt::runtime::Tensor createMultiDeviceHostTensor(
    std::vector<::tt::runtime::Tensor> const &tensorShards,
    std::unordered_map<std::string, std::string> const &strategy) {
  std::vector<::ttnn::Tensor> ttnnTensorShards;
  ttnnTensorShards.reserve(tensorShards.size());
  std::transform(tensorShards.begin(), tensorShards.end(),
                 std::back_inserter(ttnnTensorShards),
                 [&](::tt::runtime::Tensor tensorShard) -> ::ttnn::Tensor {
                   return tensorShard.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
                 });
  return createMultiDeviceHostTensor(ttnnTensorShards, strategy);
}

::tt::runtime::Tensor createEmptyTensor(
    Device device, Layout layout, std::vector<std::uint32_t> const &shape,
    std::vector<std::uint32_t> const &stride, std::uint32_t itemsize) {
  const LayoutDesc &layoutDesc = layout.as<LayoutDesc>(DeviceRuntime::TTNN);
  if (layoutDesc.isOnHost()) {
    ::ttnn::Tensor tensor =
        createOwnedTTNNTensor(nullptr, shape, stride, itemsize,
                              utils::fromTTNNDataType(layoutDesc.dataType));
    ::tt::runtime::Tensor out = utils::createRuntimeTensorFromTTNN(tensor);
    return ::tt::runtime::ttnn::toLayout(out, device, layout);
  }
  DeviceVariant targetDevice =
      getTargetDevice(device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN));
  ::ttnn::Tensor tensor = std::visit(
      [&](auto &&device) -> ::ttnn::Tensor {
        return ::ttnn::operations::core::allocate_tensor_on_device(
            ::ttnn::Shape(shape), layoutDesc.dataType, layoutDesc.layout,
            &(device.get()), layoutDesc.memoryConfig);
      },
      targetDevice);
  return utils::createRuntimeTensorFromTTNN(tensor);
}

bool isTensorAllocated(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  return ttnnTensor.is_allocated();
}

tt::target::DataType getTensorDataType(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &nnTensor =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  return utils::fromTTNNDataType(nnTensor.get_dtype());
}

std::vector<std::byte> getTensorDataBuffer(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  void *dataPtr = nullptr;
  std::vector<std::byte> dataVec(getTensorElementSize(tensor) *
                                 getTensorVolume(tensor));

  // Need to `memcpy` in each case because the vector will go out of scope if we
  // wait until after the switch case
  switch (getTensorDataType(tensor)) {
  case target::DataType::BFP_BFloat4: {
    dataVec.resize(sizeof(float) * getTensorVolume(tensor));
    auto vec = ttnnTensor.to_vector<float>();
    dataPtr = vec.data();
    LOG_ASSERT(dataPtr != nullptr);
    std::memcpy(dataVec.data(), dataPtr, dataVec.size());
    return dataVec;
  }
  case target::DataType::BFP_BFloat8: {
    dataVec.resize(sizeof(float) * getTensorVolume(tensor));
    auto vec = ttnnTensor.to_vector<float>();
    dataPtr = vec.data();
    LOG_ASSERT(dataPtr != nullptr);
    std::memcpy(dataVec.data(), dataPtr, dataVec.size());
    return dataVec;
  }
  case target::DataType::Float32: {
    auto vec = ttnnTensor.to_vector<float>();
    dataPtr = vec.data();
    LOG_ASSERT(dataPtr != nullptr);
    std::memcpy(dataVec.data(), dataPtr, dataVec.size());
    return dataVec;
  }
  case target::DataType::BFloat16: {
    auto vec = ttnnTensor.to_vector<bfloat16>();
    dataPtr = vec.data();
    LOG_ASSERT(dataPtr != nullptr);
    std::memcpy(dataVec.data(), dataPtr, dataVec.size());
    return dataVec;
  }
  case target::DataType::Int32: {
    auto vec = ttnnTensor.to_vector<std::int32_t>();
    dataPtr = vec.data();
    LOG_ASSERT(dataPtr != nullptr);
    std::memcpy(dataVec.data(), dataPtr, dataVec.size());
    return dataVec;
  }
  case target::DataType::UInt32: {
    auto vec = ttnnTensor.to_vector<std::uint32_t>();
    dataPtr = vec.data();
    LOG_ASSERT(dataPtr != nullptr);
    std::memcpy(dataVec.data(), dataPtr, dataVec.size());
    return dataVec;
  }
  case target::DataType::UInt16: {
    auto vec = ttnnTensor.to_vector<std::uint16_t>();
    dataPtr = vec.data();
    LOG_ASSERT(dataPtr != nullptr);
    std::memcpy(dataVec.data(), dataPtr, dataVec.size());
    return dataVec;
  }
  case target::DataType::UInt8: {
    auto vec = ttnnTensor.to_vector<std::uint8_t>();
    dataPtr = vec.data();
    LOG_ASSERT(dataPtr != nullptr);
    std::memcpy(dataVec.data(), dataPtr, dataVec.size());
    return dataVec;
  }
  default:
    LOG_ERROR("Unsupported datatype for underlying TTNN tensor, returning "
              "empty data vector");
    return {};
  }
}

std::vector<std::uint32_t> getTensorShape(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  std::vector<std::uint32_t> shape;
  for (size_t i = 0; i < ttnnTensor.logical_shape().size(); ++i) {
    shape.push_back(ttnnTensor.logical_shape()[i]);
  }
  return shape;
}

std::vector<std::uint32_t> getTensorStride(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  std::vector<std::uint32_t> stride;
  for (size_t i = 0; i < ttnnTensor.strides().size(); ++i) {
    stride.push_back(ttnnTensor.strides()[i]);
  }
  return stride;
}

std::uint32_t getTensorElementSize(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  return ttnnTensor.element_size();
}

std::uint32_t getTensorVolume(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  return ttnnTensor.volume();
}

TensorDesc getTensorDesc(::tt::runtime::Tensor tensor) {
  TensorDesc desc;
  desc.dataType = getTensorDataType(tensor);
  desc.itemsize = getTensorElementSize(tensor);
  desc.stride = getTensorStride(tensor);
  desc.shape = getTensorShape(tensor);
  return desc;
}

bool getTensorRetain(::tt::runtime::Tensor tensor) {
  const ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
  return tensorWrapper.shouldRetain();
}

void setTensorRetain(::tt::runtime::Tensor tensor, bool retain) {
  ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
  return tensorWrapper.setRetain(retain);
}

size_t getNumAvailableDevices() {
  return ::tt::tt_metal::GetNumAvailableDevices();
}

Device openMeshDevice(const std::vector<uint32_t> &meshShape,
                      const MeshDeviceOptions &options) {
  LOG_ASSERT(meshShape.size() == 2, "Mesh shape must be 2D for now");
  ::ttnn::MeshShape shape(meshShape);

  LOG_ASSERT(options.meshOffset.size() == 2, "Mesh offset must be 2D for now");
  ::ttnn::MeshCoordinate offset(options.meshOffset);

  size_t l1SmallSize =
      options.l1SmallSize.value_or(::tt::constants::L1_SMALL_SIZE);
  ::tt::tt_metal::DispatchCoreType dispatchCoreTypeValue =
      tt::runtime::common::getDispatchCoreType(options.dispatchCoreType);

  ::ttnn::MeshDeviceConfig meshConfig(shape, offset, options.deviceIds);

  std::shared_ptr<::ttnn::MeshDevice> meshDevice = ::ttnn::MeshDevice::create(
      meshConfig, l1SmallSize, DEFAULT_TRACE_REGION_SIZE, options.numHWCQs,
      dispatchCoreTypeValue);

  meshDevice->enable_async(options.enableAsyncTTNN);
  if (options.enableProgramCache) {
    meshDevice->enable_program_cache();
  }

  LOG_DEBUG("Device grid size = { ",
            meshDevice->compute_with_storage_grid_size().x, ", ",
            meshDevice->compute_with_storage_grid_size().y, " }");

  return Device(std::static_pointer_cast<void>(meshDevice),
                DeviceRuntime::TTNN);
}

void closeMeshDevice(Device parentMesh) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      parentMesh.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  LOG_ASSERT(ttnnMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  if (uint32_t numSubMeshes = ttnnMeshDevice.get_submeshes().size()) {
    LOG_WARNING("Calling close on parent mesh device ", ttnnMeshDevice,
                " that has ", numSubMeshes, " unreleased submeshes.");
  }
#endif

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE)
  for (::ttnn::IDevice *ttnnDevice : ttnnMeshDevice.get_devices()) {
    ::tt::tt_metal::detail::DumpDeviceProfileResults(ttnnDevice);
  }
#endif
  ttnnMeshDevice.close();
}

Device createSubMeshDevice(
    Device parentMesh, const std::pair<uint32_t, uint32_t> &meshShape,
    const std::optional<const std::pair<uint32_t, uint32_t>> &meshOffset) {
  ::ttnn::MeshDevice &parentMeshDevice =
      parentMesh.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  LOG_ASSERT(parentMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");

  ::ttnn::MeshShape shape{meshShape.first, meshShape.second};

  std::optional<::ttnn::MeshCoordinate> offset = std::nullopt;
  if (meshOffset.has_value()) {
    offset = ::ttnn::MeshCoordinate{meshOffset.value().first,
                                    meshOffset.value().second};
  }

  std::shared_ptr<::ttnn::MeshDevice> subMeshDevice =
      parentMeshDevice.create_submesh(shape, offset);

  return Device(std::static_pointer_cast<void>(subMeshDevice),
                DeviceRuntime::TTNN);
}

void releaseSubMeshDevice(Device subMesh) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      subMesh.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  LOG_ASSERT(!ttnnMeshDevice.is_parent_mesh(), "Mesh device must be a submesh");

  ttnnMeshDevice.close();
}

void deallocateBuffers(Device deviceHandle) {
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  for (::ttnn::IDevice *device : meshDevice.get_devices()) {
    device->allocator()->deallocate_buffers();
  }
}

void dumpMemoryReport(Device deviceHandle) {
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  for (::ttnn::IDevice *device : meshDevice.get_devices()) {
    ::tt::tt_metal::detail::DumpDeviceMemoryState(device);
  }
}

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device deviceHandle, int deviceID) {
  std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
      memoryMap;
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  auto *device = meshDevice.get_device(deviceID);

  auto dramMemoryView =
      ::tt::tt_metal::detail::GetMemoryView(device, ::ttnn::BufferType::DRAM);
  auto l1MemoryView =
      ::tt::tt_metal::detail::GetMemoryView(device, ::ttnn::BufferType::L1);
  auto l1SmallMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      device, ::ttnn::BufferType::L1_SMALL);
  auto traceMemoryView =
      ::tt::tt_metal::detail::GetMemoryView(device, ::ttnn::BufferType::TRACE);

  memoryMap[tt::runtime::MemoryBufferType::DRAM] =
      createMemoryView(dramMemoryView);
  memoryMap[tt::runtime::MemoryBufferType::L1] = createMemoryView(l1MemoryView);
  memoryMap[tt::runtime::MemoryBufferType::L1_SMALL] =
      createMemoryView(l1SmallMemoryView);
  memoryMap[tt::runtime::MemoryBufferType::TRACE] =
      createMemoryView(traceMemoryView);

  return memoryMap;
}

void wait(Event event) {
  // Nothing to do for ttnn runtime
  LOG_ASSERT(event.matchesRuntime(DeviceRuntime::TTNN));
}

void wait(::tt::runtime::Tensor tensor) {
  LOG_ASSERT(tensor.matchesRuntime(DeviceRuntime::TTNN),
             "Expected ttnn tensor");
  ::tt::runtime::ttnn::wait(tensor.event);
}

void wait(std::vector<::tt::runtime::Tensor> const &tensors) {
  for (const ::tt::runtime::Tensor &tensor : tensors) {
    ::tt::runtime::ttnn::wait(tensor);
  }
}

std::vector<::tt::runtime::Tensor> toHost(::tt::runtime::Tensor tensor,
                                          bool untilize) {
  const ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);

  const ::ttnn::Tensor &multiDeviceTensor = tensorWrapper.getTensor();
  bool shouldRetain = tensorWrapper.shouldRetain();

  std::vector<::tt::runtime::Tensor> hostTensors;
  if (multiDeviceTensor.storage_type() ==
          ::ttnn::StorageType::MULTI_DEVICE_HOST ||
      multiDeviceTensor.storage_type() == ::ttnn::StorageType::MULTI_DEVICE) {
    std::vector<::ttnn::Tensor> singleTensors =
        ::ttnn::distributed::get_device_tensors(multiDeviceTensor);
    for (auto &tensor : singleTensors) {
      hostTensors.push_back(::tt::runtime::ttnn::toHostSingleTensor(
          utils::createRuntimeTensorFromTTNN(tensor, shouldRetain), untilize));
    }
  } else {
    hostTensors.push_back(
        ::tt::runtime::ttnn::toHostSingleTensor(tensor, untilize));
  }
  return hostTensors;
}

::tt::runtime::Tensor toLayout(::tt::runtime::Tensor tensor, Device device,
                               Layout layout, std::optional<bool> retain) {
  const LayoutDesc tensorLayoutDesc = LayoutDesc::fromTensor(tensor);

  const LayoutDesc &desiredLayoutDesc =
      layout.as<LayoutDesc>(DeviceRuntime::TTNN);

  ::ttnn::MeshDevice &meshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  const ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);

  const ::ttnn::Tensor &ttnnTensor = tensorWrapper.getTensor();
  bool shouldRetain = retain.value_or(tensorWrapper.shouldRetain());

  DeviceVariant targetDevice = getTargetDevice(meshDevice);
  if (workaround::Env::get().toLayoutAPIAssumeSingleChip) {
    targetDevice =
        std::ref(*meshDevice.get_device(::ttnn::MeshCoordinate(0, 0)));
  }

  LayoutConverter converter(tensorLayoutDesc, desiredLayoutDesc);
  ::ttnn::Tensor out = converter.convertTensorLayout(ttnnTensor, targetDevice);

  ::tt::runtime::Tensor result =
      utils::createRuntimeTensorFromTTNN(out, shouldRetain);
  static std::atomic<uint64_t> tensorVersion{0};
  result.version.store(tensorVersion++);

  if (!shouldRetain) {
    ::tt::runtime::ttnn::deallocateTensor(tensor);
  }
  return result;
}

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex) {
  const ::tt::target::ttnn::TTNNBinary &fbb = *getBinary(executableHandle);
  LOG_ASSERT(programIndex < fbb.programs()->size(), "Invalid program index");

  const ::tt::target::ttnn::Program *program =
      fbb.programs()->Get(programIndex);
  LOG_ASSERT(inputIndex < program->inputs()->size(), "Invalid input index");
  const ::tt::target::ttnn::TensorRef *input =
      program->inputs()->Get(inputIndex);
  const ::tt::target::ttnn::MemoryConfig *memcfg =
      input->desc()->layout()->memory_desc()->memory_config();

  ::ttnn::Layout inputLayout = utils::inferLayoutFromTileShape(input);
  ::ttnn::DataType inputDataType = utils::toTTNNDataType(
      input->desc()->layout()->memory_desc()->data_type());
  ::ttnn::StorageType inputStorageType = utils::toTTNNStorageType(
      input->desc()->layout()->memory_desc()->storage_type());

  std::optional<::ttnn::MemoryConfig> inputMemoryConfig =
      utils::createMemoryConfigIfNeeded(memcfg);
  LOG_ASSERT(utils::isOnHost(inputStorageType) || inputMemoryConfig.has_value(),
             "Device tensors must have memory config");

  std::shared_ptr<LayoutDesc> layoutDesc = std::make_shared<LayoutDesc>(
      inputStorageType, inputLayout, inputDataType, inputMemoryConfig);

  return Layout(std::static_pointer_cast<void>(layoutDesc),
                DeviceRuntime::TTNN);
}

void memcpy(void *dst, ::tt::runtime::Tensor src) {
  const ::ttnn::Tensor &srcTensor =
      src.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  if (utils::isOnHost(srcTensor.storage_type())) {
    const void *srcPtr = ::tt::tt_metal::get_raw_host_data_ptr(srcTensor);
    size_t size = srcTensor.volume() * srcTensor.element_size();
    std::memcpy(dst, srcPtr, size);
  } else {
    ::tt::tt_metal::memcpy(dst, srcTensor);
  }
}

void memcpy(::tt::runtime::Tensor dst, ::tt::runtime::Tensor src) {
  ::ttnn::Tensor &dstTensor =
      dst.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  const ::ttnn::Tensor &srcTensor =
      src.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  LOG_ASSERT(srcTensor.volume() * srcTensor.element_size() ==
                 dstTensor.volume() * dstTensor.element_size(),
             "Input output tensor size mismatch in memcpy: ",
             srcTensor.volume(), " * ", srcTensor.element_size(),
             " != ", dstTensor.volume(), " * ", dstTensor.element_size());
  if (utils::isOnHost(srcTensor.storage_type()) and
      utils::isOnHost(dstTensor.storage_type())) {
    void *dstPtr = ::tt::tt_metal::get_raw_host_data_ptr(dstTensor);
    const void *srcPtr = ::tt::tt_metal::get_raw_host_data_ptr(srcTensor);
    size_t size = srcTensor.volume() * srcTensor.element_size();
    std::memcpy(dstPtr, srcPtr, size);
  } else {
    ::tt::tt_metal::memcpy(dstTensor, srcTensor);
  }
}

void deallocateTensor(::tt::runtime::Tensor &tensor, bool force) {
  ::ttnn::Tensor &ttnnTensor =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  if (ttnnTensor.storage_type() == ::ttnn::StorageType::BORROWED) {
    return;
  }
  ::ttnn::deallocate(ttnnTensor, force);
}

std::string getOpDebugString(OpContext opContextHandle) {
  auto const &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  return std::string(opContext.debug_info()->c_str());
}

std::string getOpLocInfo(OpContext opContextHandle) {
  auto const &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  return std::string(opContext.loc_info()->c_str());
}

::tt::runtime::Tensor getOpOutputTensor(OpContext opContextHandle,
                                        CallbackContext programContextHandle) {
  auto const &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  auto const &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  const ttnn::ProgramTensorPool &tensorPool = programContext.getTensorPool();
  std::optional<const ::tt::target::ttnn::TensorRef *> tensorRef = std::nullopt;
  const ::ttnn::Tensor *outPtr = nullptr;

  switch (opContext.type_type()) {
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    tensorRef = opContext.type_as_ToMemoryConfigOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    tensorRef = opContext.type_as_ToLayoutOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ToDTypeOp: {
    tensorRef = opContext.type_as_ToDTypeOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    tensorRef = opContext.type_as_TypecastOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    tensorRef = opContext.type_as_ToDeviceOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    tensorRef = opContext.type_as_FromDeviceOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    tensorRef = opContext.type_as_EmptyOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ConstructTensorOp: {
    tensorRef = opContext.type_as_ConstructTensorOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::NamedFullOp: {
    tensorRef = opContext.type_as_NamedFullOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    tensorRef = opContext.type_as_FullOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryOp: {
    tensorRef = opContext.type_as_EltwiseBinaryOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeOp: {
    tensorRef = opContext.type_as_EltwiseBinaryCompositeOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseTernaryWhereOp: {
    tensorRef = opContext.type_as_EltwiseTernaryWhereOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseQuantizationOp: {
    tensorRef = opContext.type_as_EltwiseQuantizationOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryOp: {
    tensorRef = opContext.type_as_EltwiseUnaryOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryCompositeOp: {
    tensorRef = opContext.type_as_EltwiseUnaryCompositeOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::LinearOp: {
    tensorRef = opContext.type_as_LinearOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    tensorRef = opContext.type_as_MatmulOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::MorehCumSumOp: {
    tensorRef = opContext.type_as_MorehCumSumOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionArgMaxOp: {
    tensorRef = opContext.type_as_ReductionArgMaxOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionProdOp: {
    tensorRef = opContext.type_as_ReductionProdOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    tensorRef = opContext.type_as_ReductionOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    tensorRef = opContext.type_as_EmbeddingOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingBackwardOp: {
    tensorRef = opContext.type_as_EmbeddingBackwardOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    tensorRef = opContext.type_as_SoftmaxOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    tensorRef = opContext.type_as_TransposeOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::PadOp: {
    tensorRef = opContext.type_as_PadOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    tensorRef = opContext.type_as_ConcatOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::PermuteOp: {
    tensorRef = opContext.type_as_PermuteOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    tensorRef = opContext.type_as_ReshapeOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    tensorRef = opContext.type_as_SliceOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatOp: {
    tensorRef = opContext.type_as_RepeatOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatInterleaveOp: {
    tensorRef = opContext.type_as_RepeatInterleaveOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    tensorRef = opContext.type_as_Conv2dOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ConvTranspose2dOp: {
    tensorRef = opContext.type_as_ConvTranspose2dOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    tensorRef = opContext.type_as_MaxPool2dOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    tensorRef = opContext.type_as_AllGatherOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    tensorRef = opContext.type_as_ReduceScatterOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::CollectivePermuteOp: {
    tensorRef = opContext.type_as_CollectivePermuteOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    tensorRef = opContext.type_as_MeshShardOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ArangeOp: {
    tensorRef = opContext.type_as_ArangeOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::UpsampleOp: {
    tensorRef = opContext.type_as_UpsampleOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::CpuOp: {
    tensorRef = opContext.type_as_CpuOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ConstantOp: {
    tensorRef = opContext.type_as_ConstantOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    tensorRef = opContext.type_as_FillCacheOp()->cache();
    break;
  }
  case ::tt::target::ttnn::OpType::UpdateCacheOp: {
    tensorRef = opContext.type_as_UpdateCacheOp()->cache();
    break;
  }
  case ::tt::target::ttnn::OpType::GetDeviceOp:
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    LOG_WARNING("getting output tensor is not supported for ",
                ::tt::target::ttnn::EnumNamesOpType()[static_cast<size_t>(
                    opContext.type_type())]);
    return createNullTensor();
  }
  default: {
    LOG_FATAL("Unsupported operation type");
  }
  }

  if (tensorRef.has_value() && tensorPool.contains(tensorRef.value())) {
    outPtr = &tensorPool.getTTNNTensorAndValidate(tensorRef.value());
  } else {
    LOG_WARNING("Output tensor not found in tensor pool");
    return createNullTensor();
  }

  ::ttnn::Tensor hostTensor = ::ttnn::to_layout(
      ::ttnn::from_device(*outPtr), ::ttnn::Layout::ROW_MAJOR, std::nullopt,
      std::nullopt, static_cast<::ttnn::IDevice *>(nullptr));

  return utils::createRuntimeTensorFromTTNN(hostTensor);
}

std::vector<::tt::runtime::Tensor>
submit(Device deviceHandle, Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs) {

  std::shared_ptr<TensorCache> cache = deviceHandle.cache;
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  std::vector<::tt::runtime::Tensor> outputs = ::tt::runtime::ttnn::runProgram(
      meshDevice, executableHandle, programIndex, inputs, cache);

  return outputs;
}

} // namespace tt::runtime::ttnn
