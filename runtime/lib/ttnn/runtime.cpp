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
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"
#include "ttnn/tensor/types.hpp"

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

static std::vector<::tt::runtime::Tensor>
convertInputLayouts(Device deviceHandle, Binary executableHandle,
                    std::uint32_t programIndex,
                    std::vector<::tt::runtime::Tensor> &inputs) {
  // Convert input tensors to the layout expected by the program
  std::vector<::tt::runtime::Tensor> inputsWithLayout;
  inputsWithLayout.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); i++) {
    ::tt::runtime::Tensor &input = inputs[i];
    Layout desiredLayout =
        ::tt::runtime::ttnn::getLayout(executableHandle, programIndex, i);

    const LayoutDesc tensorLayoutDesc = LayoutDesc::fromTensor(input);

    const LayoutDesc &desiredlayoutDesc =
        desiredLayout.as<LayoutDesc>(DeviceRuntime::TTNN);

    // If the input tensor already has the correct layout
    // reuse it and continue
    if (tensorLayoutDesc == desiredlayoutDesc) {
      inputsWithLayout.push_back(input);
      continue;
    }

    // Convert the input tensor to the correct layout
    // Deallocating the original tensor if it is not retained
    ::tt::runtime::Tensor inputWithLayout = ::tt::runtime::ttnn::toLayout(
        input, deviceHandle, desiredLayout, /*retain=*/false);
    inputsWithLayout.push_back(inputWithLayout);

    if (!::tt::runtime::ttnn::getTensorRetain(input)) {
      ::tt::runtime::ttnn::deallocateTensor(input);
    }
  }

  return inputsWithLayout;
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

  return utils::createRuntimeTensorFromTTNN(out, shouldRetain);
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

std::vector<::tt::runtime::Tensor>
getOutputTensors(CallbackContext programContextHandle) {
  auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  return programContext.getTensorPool().gatherOutputTensors();
}

std::vector<std::uint32_t>
getIntermediateInputTensorIds(OpContext opContextHandle) {
  auto const &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  std::vector<std::uint32_t> ids;

  switch (opContext.type_type()) {
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    auto op = opContext.type_as_ToMemoryConfigOp();
    ids.push_back(op->in0()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    auto op = opContext.type_as_ToLayoutOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ToDTypeOp: {
    auto op = opContext.type_as_ToDTypeOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    auto op = opContext.type_as_TypecastOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    auto op = opContext.type_as_ToDeviceOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    auto op = opContext.type_as_FromDeviceOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    auto op = opContext.type_as_FillCacheOp();
    ids.push_back(op->input()->global_id());
    ids.push_back(op->cache()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::UpdateCacheOp: {
    auto op = opContext.type_as_UpdateCacheOp();
    ids.push_back(op->input()->global_id());
    ids.push_back(op->cache()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    auto op = opContext.type_as_DeallocateOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::PadOp: {
    auto op = opContext.type_as_PadOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    auto op = opContext.type_as_ConcatOp();
    for (const auto *input : *op->inputs()) {
      ids.push_back(input->global_id());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::PermuteOp: {
    auto op = opContext.type_as_PermuteOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    auto op = opContext.type_as_ReshapeOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    auto op = opContext.type_as_SliceOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatOp: {
    auto op = opContext.type_as_RepeatOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatInterleaveOp: {
    auto op = opContext.type_as_RepeatInterleaveOp();
    ids.push_back(op->input()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    auto op = opContext.type_as_Conv2dOp();
    ids.push_back(op->input()->global_id());
    ids.push_back(op->weight()->global_id());
    ids.push_back(op->bias()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ConvTranspose2dOp: {
    auto op = opContext.type_as_ConvTranspose2dOp();
    ids.push_back(op->input()->global_id());
    ids.push_back(op->weight()->global_id());
    ids.push_back(op->bias()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    auto op = opContext.type_as_MaxPool2dOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    auto op = opContext.type_as_AllGatherOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    auto op = opContext.type_as_ReduceScatterOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::CollectivePermuteOp: {
    auto op = opContext.type_as_CollectivePermuteOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    auto op = opContext.type_as_MeshShardOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryOp: {
    auto op = opContext.type_as_EltwiseBinaryOp();
    ids.push_back(op->lhs()->global_id());
    ids.push_back(op->rhs()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeOp: {
    auto op = opContext.type_as_EltwiseBinaryCompositeOp();
    ids.push_back(op->lhs()->global_id());
    ids.push_back(op->rhs()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseTernaryWhereOp: {
    auto op = opContext.type_as_EltwiseTernaryWhereOp();
    ids.push_back(op->first()->global_id());
    ids.push_back(op->second()->global_id());
    ids.push_back(op->third()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseQuantizationOp: {
    auto op = opContext.type_as_EltwiseQuantizationOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryOp: {
    auto op = opContext.type_as_EltwiseUnaryOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryCompositeOp: {
    auto op = opContext.type_as_EltwiseUnaryCompositeOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::LinearOp: {
    auto op = opContext.type_as_LinearOp();
    ids.push_back(op->a()->global_id());
    ids.push_back(op->b()->global_id());
    if (op->bias())
      ids.push_back(op->bias()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    auto op = opContext.type_as_MatmulOp();
    ids.push_back(op->a()->global_id());
    ids.push_back(op->b()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::MorehCumSumOp: {
    auto op = opContext.type_as_MorehCumSumOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionArgMaxOp: {
    auto op = opContext.type_as_ReductionArgMaxOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionProdOp: {
    auto op = opContext.type_as_ReductionProdOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    auto op = opContext.type_as_ReductionOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    auto op = opContext.type_as_EmbeddingOp();
    ids.push_back(op->input()->global_id());
    ids.push_back(op->weight()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingBackwardOp: {
    auto op = opContext.type_as_EmbeddingBackwardOp();
    ids.push_back(op->input()->global_id());
    ids.push_back(op->weight()->global_id());
    ids.push_back(op->in_grad()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    auto op = opContext.type_as_SoftmaxOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    auto op = opContext.type_as_TransposeOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::UpsampleOp: {
    auto op = opContext.type_as_UpsampleOp();
    ids.push_back(op->in()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::CpuOp: {
    auto op = opContext.type_as_CpuOp();
    for (const auto *input : *op->ins()) {
      ids.push_back(input->global_id());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp:
  case ::tt::target::ttnn::OpType::ConstructTensorOp:
  case ::tt::target::ttnn::OpType::NamedFullOp:
  case ::tt::target::ttnn::OpType::FullOp:
  case ::tt::target::ttnn::OpType::ArangeOp:
  case ::tt::target::ttnn::OpType::ConstantOp:
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    LOG_DEBUG("Op type has no inputs.");
    break;
  }
  default:
    LOG_WARNING("unhandled op type in getIntermediateInputTensor");
    break;
  }
  return ids;
}

std::uint32_t getIntermediateOutputTensorId(OpContext opContextHandle) {
  auto const &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  // std::optional<const ::tt::target::ttnn::TensorRef *> tensorRef =
  // std::nullopt;
  std::vector<std::uint32_t> ids;

  switch (opContext.type_type()) {
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    ids.push_back(opContext.type_as_ToMemoryConfigOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    ids.push_back(opContext.type_as_ToLayoutOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ToDTypeOp: {
    ids.push_back(opContext.type_as_ToDTypeOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    ids.push_back(opContext.type_as_TypecastOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    ids.push_back(opContext.type_as_ToDeviceOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    ids.push_back(opContext.type_as_FromDeviceOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    ids.push_back(opContext.type_as_EmptyOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ConstructTensorOp: {
    ids.push_back(opContext.type_as_ConstructTensorOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::NamedFullOp: {
    ids.push_back(opContext.type_as_NamedFullOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    ids.push_back(opContext.type_as_FullOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryOp: {
    ids.push_back(opContext.type_as_EltwiseBinaryOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeOp: {
    ids.push_back(
        opContext.type_as_EltwiseBinaryCompositeOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseTernaryWhereOp: {
    ids.push_back(
        opContext.type_as_EltwiseTernaryWhereOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseQuantizationOp: {
    ids.push_back(
        opContext.type_as_EltwiseQuantizationOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryOp: {
    ids.push_back(opContext.type_as_EltwiseUnaryOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryCompositeOp: {
    ids.push_back(
        opContext.type_as_EltwiseUnaryCompositeOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::LinearOp: {
    ids.push_back(opContext.type_as_LinearOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    ids.push_back(opContext.type_as_MatmulOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::MorehCumSumOp: {
    ids.push_back(opContext.type_as_MorehCumSumOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionArgMaxOp: {
    ids.push_back(opContext.type_as_ReductionArgMaxOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionProdOp: {
    ids.push_back(opContext.type_as_ReductionProdOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    ids.push_back(opContext.type_as_ReductionOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    ids.push_back(opContext.type_as_EmbeddingOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingBackwardOp: {
    ids.push_back(opContext.type_as_EmbeddingBackwardOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    ids.push_back(opContext.type_as_SoftmaxOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    ids.push_back(opContext.type_as_TransposeOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::PadOp: {
    ids.push_back(opContext.type_as_PadOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    ids.push_back(opContext.type_as_ConcatOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::PermuteOp: {
    ids.push_back(opContext.type_as_PermuteOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    ids.push_back(opContext.type_as_ReshapeOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    ids.push_back(opContext.type_as_SliceOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatOp: {
    ids.push_back(opContext.type_as_RepeatOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatInterleaveOp: {
    ids.push_back(opContext.type_as_RepeatInterleaveOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    ids.push_back(opContext.type_as_Conv2dOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ConvTranspose2dOp: {
    ids.push_back(opContext.type_as_ConvTranspose2dOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    ids.push_back(opContext.type_as_MaxPool2dOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    ids.push_back(opContext.type_as_AllGatherOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    ids.push_back(opContext.type_as_ReduceScatterOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::CollectivePermuteOp: {
    ids.push_back(opContext.type_as_CollectivePermuteOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    ids.push_back(opContext.type_as_MeshShardOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ArangeOp: {
    ids.push_back(opContext.type_as_ArangeOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::UpsampleOp: {
    ids.push_back(opContext.type_as_UpsampleOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::CpuOp: {
    ids.push_back(opContext.type_as_CpuOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ConstantOp: {
    ids.push_back(opContext.type_as_ConstantOp()->out()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    ids.push_back(opContext.type_as_FillCacheOp()->cache()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::UpdateCacheOp: {
    ids.push_back(opContext.type_as_UpdateCacheOp()->cache()->global_id());
    break;
  }
  case ::tt::target::ttnn::OpType::GetDeviceOp:
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    LOG_WARNING("getting output tensor is not supported for ",
                ::tt::target::ttnn::EnumNamesOpType()[static_cast<size_t>(
                    opContext.type_type())]);
    return ids[0];
  }
  default: {
    LOG_FATAL("Unsupported operation type");
  }
  }
  return ids[0];
}

std::vector<::tt::runtime::Tensor>
getIntermediateInputTensors(OpContext opContextHandle,
                            CallbackContext programContextHandle) {
  auto const &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  std::vector<std::uint32_t> ids =
      getIntermediateInputTensorIds(opContextHandle);
  std::vector<::tt::runtime::Tensor> results;
  results.reserve(ids.size());
  for (auto id : ids) {
    results.push_back(programContext.getTensorPool().getRuntimeTensor(id));
  }
  return results;
}

::tt::runtime::Tensor
getIntermediateOutputTensor(OpContext opContextHandle,
                            CallbackContext programContextHandle) {
  auto const &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);

  std::uint32_t id = getIntermediateOutputTensorId(opContextHandle);
  ::tt::runtime::Tensor result =
      programContext.getTensorPool().getRuntimeTensor(id);

  return result;
}

// Okay you still have to go through and make sure all the names and return
// types are right and then you can test this

std::vector<const ::tt::target::ttnn::TensorRef *>
getIntermediateInputTensorRefs(OpContext opContextHandle) {
  auto const &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  std::vector<const ::tt::target::ttnn::TensorRef *> tensorRefs;

  switch (opContext.type_type()) {
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    auto op = opContext.type_as_ToMemoryConfigOp();
    tensorRefs.push_back(op->in0());
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    auto op = opContext.type_as_ToLayoutOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::ToDTypeOp: {
    auto op = opContext.type_as_ToDTypeOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    auto op = opContext.type_as_TypecastOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    auto op = opContext.type_as_ToDeviceOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    auto op = opContext.type_as_FromDeviceOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    auto op = opContext.type_as_FillCacheOp();
    tensorRefs.push_back(op->input());
    tensorRefs.push_back(op->cache());
    break;
  }
  case ::tt::target::ttnn::OpType::UpdateCacheOp: {
    auto op = opContext.type_as_UpdateCacheOp();
    tensorRefs.push_back(op->input());
    tensorRefs.push_back(op->cache());
    break;
  }
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    auto op = opContext.type_as_DeallocateOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::PadOp: {
    auto op = opContext.type_as_PadOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    auto op = opContext.type_as_ConcatOp();
    for (const auto *input : *op->inputs()) {
      tensorRefs.push_back(input);
    }
    break;
  }
  case ::tt::target::ttnn::OpType::PermuteOp: {
    auto op = opContext.type_as_PermuteOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    auto op = opContext.type_as_ReshapeOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    auto op = opContext.type_as_SliceOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatOp: {
    auto op = opContext.type_as_RepeatOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatInterleaveOp: {
    auto op = opContext.type_as_RepeatInterleaveOp();
    tensorRefs.push_back(op->input());
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    auto op = opContext.type_as_Conv2dOp();
    tensorRefs.push_back(op->input());
    tensorRefs.push_back(op->weight());
    tensorRefs.push_back(op->bias());
    break;
  }
  case ::tt::target::ttnn::OpType::ConvTranspose2dOp: {
    auto op = opContext.type_as_ConvTranspose2dOp();
    tensorRefs.push_back(op->input());
    tensorRefs.push_back(op->weight());
    tensorRefs.push_back(op->bias());
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    auto op = opContext.type_as_MaxPool2dOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    auto op = opContext.type_as_AllGatherOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    auto op = opContext.type_as_ReduceScatterOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::CollectivePermuteOp: {
    auto op = opContext.type_as_CollectivePermuteOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    auto op = opContext.type_as_MeshShardOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryOp: {
    auto op = opContext.type_as_EltwiseBinaryOp();
    tensorRefs.push_back(op->lhs());
    tensorRefs.push_back(op->rhs());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeOp: {
    auto op = opContext.type_as_EltwiseBinaryCompositeOp();
    tensorRefs.push_back(op->lhs());
    tensorRefs.push_back(op->rhs());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseTernaryWhereOp: {
    auto op = opContext.type_as_EltwiseTernaryWhereOp();
    tensorRefs.push_back(op->first());
    tensorRefs.push_back(op->second());
    tensorRefs.push_back(op->third());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseQuantizationOp: {
    auto op = opContext.type_as_EltwiseQuantizationOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryOp: {
    auto op = opContext.type_as_EltwiseUnaryOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryCompositeOp: {
    auto op = opContext.type_as_EltwiseUnaryCompositeOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::LinearOp: {
    auto op = opContext.type_as_LinearOp();
    tensorRefs.push_back(op->a());
    tensorRefs.push_back(op->b());
    if (op->bias())
      tensorRefs.push_back(op->bias());
    break;
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    auto op = opContext.type_as_MatmulOp();
    tensorRefs.push_back(op->a());
    tensorRefs.push_back(op->b());
    break;
  }
  case ::tt::target::ttnn::OpType::MorehCumSumOp: {
    auto op = opContext.type_as_MorehCumSumOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionArgMaxOp: {
    auto op = opContext.type_as_ReductionArgMaxOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionProdOp: {
    auto op = opContext.type_as_ReductionProdOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    auto op = opContext.type_as_ReductionOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    auto op = opContext.type_as_EmbeddingOp();
    tensorRefs.push_back(op->input());
    tensorRefs.push_back(op->weight());
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingBackwardOp: {
    auto op = opContext.type_as_EmbeddingBackwardOp();
    tensorRefs.push_back(op->input());
    tensorRefs.push_back(op->weight());
    tensorRefs.push_back(op->in_grad());
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    auto op = opContext.type_as_SoftmaxOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    auto op = opContext.type_as_TransposeOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::UpsampleOp: {
    auto op = opContext.type_as_UpsampleOp();
    tensorRefs.push_back(op->in());
    break;
  }
  case ::tt::target::ttnn::OpType::CpuOp: {
    auto op = opContext.type_as_CpuOp();
    for (const auto *input : *op->ins()) {
      tensorRefs.push_back(input);
    }
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp:
  case ::tt::target::ttnn::OpType::ConstructTensorOp:
  case ::tt::target::ttnn::OpType::NamedFullOp:
  case ::tt::target::ttnn::OpType::FullOp:
  case ::tt::target::ttnn::OpType::ArangeOp:
  case ::tt::target::ttnn::OpType::ConstantOp:
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    LOG_DEBUG("Op type has no inputs.");
    break;
  }
  default:
    LOG_WARNING("unhandled op type in getIntermediateInputTensor");
    break;
  }
  return tensorRefs;
}

std::vector<const ::tt::target::ttnn::TensorRef *>
getIntermediateOutputTensorRefs(OpContext opContextHandle) {
  auto const &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  // std::optional<const ::tt::target::ttnn::TensorRef *> tensorRef =
  // std::nullopt;
  std::vector<const ::tt::target::ttnn::TensorRef *> tensorRefs;

  switch (opContext.type_type()) {
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    tensorRefs.push_back(opContext.type_as_ToMemoryConfigOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    tensorRefs.push_back(opContext.type_as_ToLayoutOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ToDTypeOp: {
    tensorRefs.push_back(opContext.type_as_ToDTypeOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    tensorRefs.push_back(opContext.type_as_TypecastOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    tensorRefs.push_back(opContext.type_as_ToDeviceOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    tensorRefs.push_back(opContext.type_as_FromDeviceOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    tensorRefs.push_back(opContext.type_as_EmptyOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ConstructTensorOp: {
    tensorRefs.push_back(opContext.type_as_ConstructTensorOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::NamedFullOp: {
    tensorRefs.push_back(opContext.type_as_NamedFullOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    tensorRefs.push_back(opContext.type_as_FullOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryOp: {
    tensorRefs.push_back(opContext.type_as_EltwiseBinaryOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeOp: {
    tensorRefs.push_back(opContext.type_as_EltwiseBinaryCompositeOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseTernaryWhereOp: {
    tensorRefs.push_back(opContext.type_as_EltwiseTernaryWhereOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseQuantizationOp: {
    tensorRefs.push_back(opContext.type_as_EltwiseQuantizationOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryOp: {
    tensorRefs.push_back(opContext.type_as_EltwiseUnaryOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryCompositeOp: {
    tensorRefs.push_back(opContext.type_as_EltwiseUnaryCompositeOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::LinearOp: {
    tensorRefs.push_back(opContext.type_as_LinearOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    tensorRefs.push_back(opContext.type_as_MatmulOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::MorehCumSumOp: {
    tensorRefs.push_back(opContext.type_as_MorehCumSumOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionArgMaxOp: {
    tensorRefs.push_back(opContext.type_as_ReductionArgMaxOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionProdOp: {
    tensorRefs.push_back(opContext.type_as_ReductionProdOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    tensorRefs.push_back(opContext.type_as_ReductionOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    tensorRefs.push_back(opContext.type_as_EmbeddingOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingBackwardOp: {
    tensorRefs.push_back(opContext.type_as_EmbeddingBackwardOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    tensorRefs.push_back(opContext.type_as_SoftmaxOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    tensorRefs.push_back(opContext.type_as_TransposeOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::PadOp: {
    tensorRefs.push_back(opContext.type_as_PadOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    tensorRefs.push_back(opContext.type_as_ConcatOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::PermuteOp: {
    tensorRefs.push_back(opContext.type_as_PermuteOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    tensorRefs.push_back(opContext.type_as_ReshapeOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    tensorRefs.push_back(opContext.type_as_SliceOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatOp: {
    tensorRefs.push_back(opContext.type_as_RepeatOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatInterleaveOp: {
    tensorRefs.push_back(opContext.type_as_RepeatInterleaveOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    tensorRefs.push_back(opContext.type_as_Conv2dOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ConvTranspose2dOp: {
    tensorRefs.push_back(opContext.type_as_ConvTranspose2dOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    tensorRefs.push_back(opContext.type_as_MaxPool2dOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    tensorRefs.push_back(opContext.type_as_AllGatherOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    tensorRefs.push_back(opContext.type_as_ReduceScatterOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::CollectivePermuteOp: {
    tensorRefs.push_back(opContext.type_as_CollectivePermuteOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    tensorRefs.push_back(opContext.type_as_MeshShardOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ArangeOp: {
    tensorRefs.push_back(opContext.type_as_ArangeOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::UpsampleOp: {
    tensorRefs.push_back(opContext.type_as_UpsampleOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::CpuOp: {
    tensorRefs.push_back(opContext.type_as_CpuOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::ConstantOp: {
    tensorRefs.push_back(opContext.type_as_ConstantOp()->out());
    break;
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    tensorRefs.push_back(opContext.type_as_FillCacheOp()->cache());
    break;
  }
  case ::tt::target::ttnn::OpType::UpdateCacheOp: {
    tensorRefs.push_back(opContext.type_as_UpdateCacheOp()->cache());
    break;
  }
  case ::tt::target::ttnn::OpType::GetDeviceOp:
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    LOG_WARNING("getting output tensor is not supported for ",
                ::tt::target::ttnn::EnumNamesOpType()[static_cast<size_t>(
                    opContext.type_type())]);
    return tensorRefs;
  }
  default: {
    LOG_FATAL("Unsupported operation type");
  }
  }
  return tensorRefs;
}
/*
std::vector<::tt::runtime::Tensor>
getIntermediateInputTensors(OpContext opContextHandle,
                            CallbackContext programContextHandle) {
  auto const &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  const ttnn::ProgramTensorPool &tensorPool = programContext.getTensorPool();
  std::vector<const ::tt::target::ttnn::TensorRef *> tensorRefs =
      getIntermediateInputTensorRefs(opContextHandle);
  std::vector<::tt::runtime::Tensor> results;
  results.reserve(tensorRefs.size());

  for (const auto *ref : tensorRefs) {
    const ::ttnn::Tensor *inPtr = nullptr;
    if (tensorPool.contains(ref)) {
      inPtr = &tensorPool.getTTNNTensorAndValidate(ref);
    } else {
      LOG_WARNING("Intermediate input tensor not found in tensor pool");
      return results;
    }

    ::ttnn::Tensor hostTensor = ::ttnn::to_layout(
        ::ttnn::from_device(*inPtr), ::ttnn::Layout::ROW_MAJOR, std::nullopt,
        std::nullopt, static_cast<::ttnn::IDevice *>(nullptr));

    results.push_back(utils::createRuntimeTensorFromTTNN(hostTensor));
  }
  return results;
}

std::vector<std::uint32_t>
getIntermediateInputTensorIds(OpContext opContextHandle) {
  std::vector<const ::tt::target::ttnn::TensorRef *> tensorRefs =
      getIntermediateInputTensorRefs(opContextHandle);
  std::vector<std::uint32_t> result;
  result.reserve(tensorRefs.size());

  for (const auto *ref : tensorRefs) {
    result.push_back(ref->global_id());
  }
  return result;
}

std::vector<::tt::runtime::Tensor>
getIntermediateOutputTensors(OpContext opContextHandle,
                             CallbackContext programContextHandle) {
  auto const &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  const ttnn::ProgramTensorPool &tensorPool = programContext.getTensorPool();

  std::vector<const ::tt::target::ttnn::TensorRef *> tensorRefs =
      getIntermediateOutputTensorRefs(opContextHandle);
  std::vector<::tt::runtime::Tensor> results;
  results.reserve(tensorRefs.size());

  for (const auto *ref : tensorRefs) {
    const ::ttnn::Tensor *outPtr = nullptr;
    if (tensorPool.contains(ref)) {
      outPtr = &tensorPool.getTTNNTensorAndValidate(ref);
    } else {
      LOG_WARNING("Intermediate input tensor not found in tensor pool");
      return results;
    }

    ::ttnn::Tensor hostTensor = ::ttnn::to_layout(
        ::ttnn::from_device(*outPtr), ::ttnn::Layout::ROW_MAJOR, std::nullopt,
        std::nullopt, static_cast<::ttnn::IDevice *>(nullptr));

    results.push_back(utils::createRuntimeTensorFromTTNN(hostTensor));
  }
  return results;
}

std::vector<std::uint32_t>
getIntermediateOutputTensorIds(OpContext opContextHandle) {
  std::vector<const ::tt::target::ttnn::TensorRef *> tensorRefs =
      getIntermediateOutputTensorRefs(opContextHandle);
  std::vector<std::uint32_t> result;
  result.reserve(tensorRefs.size());

  for (const auto *ref : tensorRefs) {
    result.push_back(ref->global_id());
  }
  return result;
}
*/
std::vector<std::uint32_t>
getInputTensorIds(CallbackContext programContextHandle) {
  auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  return programContext.getTensorPool().getProgramInputIds();
}

std::vector<std::uint32_t>
getOutputTensorIds(CallbackContext programContextHandle) {
  auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  return programContext.getTensorPool().getProgramOutputIds();
}

bool isTensorLive(CallbackContext programContextHandle,
                  std::uint32_t global_id) {
  auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  return programContext.getTensorPool().containsId(global_id);
}

::tt::runtime::Tensor getTensor(CallbackContext programContextHandle,
                                std::uint32_t global_id) {
  auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  return programContext.getTensorPool().getRuntimeTensor(global_id);
}

std::vector<::tt::runtime::Tensor>
submit(Device deviceHandle, Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs) {
  // Convert input tensors to the layout expected by the program
  std::vector<::tt::runtime::Tensor> inputsWithLayout =
      ::tt::runtime::ttnn::convertInputLayouts(deviceHandle, executableHandle,
                                               programIndex, inputs);

  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  std::vector<::tt::runtime::Tensor> outputs = ::tt::runtime::ttnn::runProgram(
      meshDevice, executableHandle, programIndex, inputsWithLayout);

  return outputs;
}

} // namespace tt::runtime::ttnn
