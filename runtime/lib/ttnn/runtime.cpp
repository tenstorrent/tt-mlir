// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "Constants.h"

#include "tt/runtime/detail/common.h"
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/dylib.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/debug_apis.h"
#include "tt/runtime/ttnn/program_executor.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/utils.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttmlir/Version.h"
#include "ttnn/tensor/types.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace tt::runtime::ttnn {

using ::tt::runtime::DeviceRuntime;
using ::tt::tt_metal::BorrowedStorage;
using ::tt::tt_metal::DistributedTensorConfig;
using ::tt::tt_metal::OwnedStorage;
using ::tt::tt_metal::raise_unsupported_storage;

template <typename ElementType>
static OwnedStorage createOwnedStorage(const ElementType *ptr,
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

static OwnedStorage createOwnedStorage(const void *ptr,
                                       std::uint32_t numElements,
                                       ::tt::target::DataType dataType) {
  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return createOwnedStorage(static_cast<const float *>(ptr), numElements);
  case ::tt::target::DataType::BFloat16:
    return createOwnedStorage(static_cast<const bfloat16 *>(ptr), numElements);
  case ::tt::target::DataType::UInt32:
    return createOwnedStorage(static_cast<const uint32_t *>(ptr), numElements);
  case ::tt::target::DataType::UInt16:
    return createOwnedStorage(static_cast<const uint16_t *>(ptr), numElements);
  case ::tt::target::DataType::UInt8:
    return createOwnedStorage(static_cast<const uint8_t *>(ptr), numElements);
  case ::tt::target::DataType::Int32:
    return createOwnedStorage(static_cast<const int32_t *>(ptr), numElements);
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
createOwnedTTNNTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];

  return ::ttnn::Tensor(createOwnedStorage(data, numElements, dataType),
                        ::ttnn::Shape(shape), utils::toTTNNDataType(dataType),
                        ::ttnn::Layout::ROW_MAJOR);
}

// static ::tt::runtime::Tensor createNullTensor() {
//   return ::tt::runtime::Tensor(nullptr, nullptr, DeviceRuntime::TTNN);
// }

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
                                   static_cast<::ttnn::MeshDevice *>(nullptr));
  }

  return utils::createRuntimeTensorFromTTNN(hostTensor, shouldRetain);
}

static tt::runtime::MemoryView
createMemoryView(const tt::tt_metal::detail::MemoryView &memoryView) {
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

::tt::runtime::Tensor
createOwnedHostTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {

  return utils::createRuntimeTensorFromTTNN(
      createOwnedTTNNTensor(data, shape, stride, itemsize, dataType));
}

::tt::runtime::Tensor
createBorrowedHostTensor(void *data, const std::vector<std::uint32_t> &shape,
                         const std::vector<std::uint32_t> &stride,
                         std::uint32_t itemsize,
                         ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];

  ::ttnn::Tensor tensor(createBorrowedStorage(data, numElements, dataType),
                        ::ttnn::Shape(shape), utils::toTTNNDataType(dataType),
                        ::ttnn::Layout::ROW_MAJOR);

  return utils::createRuntimeTensorFromTTNN(tensor);
}

::tt::runtime::Tensor createMultiDeviceHostTensor(
    const std::vector<::tt::runtime::Tensor> &tensorShards,
    const std::unordered_map<std::string, std::string> &strategy) {
  std::vector<::ttnn::Tensor> ttnnTensorShards;
  ttnnTensorShards.reserve(tensorShards.size());
  std::transform(tensorShards.begin(), tensorShards.end(),
                 std::back_inserter(ttnnTensorShards),
                 [&](::tt::runtime::Tensor tensorShard) -> ::ttnn::Tensor {
                   return tensorShard
                       .as<::tt::runtime::ttnn::TTNNTensorWrapper>(
                           DeviceRuntime::TTNN)
                       .getTensor();
                 });

  DistributedTensorConfig distributionStrategy =
      ::tt::tt_metal::get_distributed_tensor_config(strategy);
  ::ttnn::Tensor multiDeviceHostTensor =
      ::ttnn::distributed::aggregate_as_tensor(ttnnTensorShards,
                                               distributionStrategy);

  return utils::createRuntimeTensorFromTTNN(multiDeviceHostTensor);
}

::tt::runtime::Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    const std::unordered_map<std::string, std::string> &strategy) {
  std::vector<::tt::runtime::Tensor> tensorShards;
  tensorShards.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(tensorShards),
                 [&](const void *dataShard) -> ::tt::runtime::Tensor {
                   return createOwnedHostTensor(dataShard, shape, stride,
                                                itemsize, dataType);
                 });
  return createMultiDeviceHostTensor(tensorShards, strategy);
}

::tt::runtime::Tensor createEmptyTensor(
    Device device, Layout layout, const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize) {
  const LayoutDesc &layoutDesc = layout.as<LayoutDesc>(DeviceRuntime::TTNN);
  if (layoutDesc.isOnHost()) {
    ::ttnn::Tensor tensor =
        createOwnedTTNNTensor(nullptr, shape, stride, itemsize,
                              utils::fromTTNNDataType(layoutDesc.dataType));
    ::tt::runtime::Tensor out = utils::createRuntimeTensorFromTTNN(tensor);
    return ::tt::runtime::ttnn::toLayout(out, device, layout);
  }
  ::ttnn::MeshDevice &meshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  ::ttnn::Tensor tensor = ::ttnn::operations::core::allocate_tensor_on_device(
      ::ttnn::Shape(shape), layoutDesc.dataType, layoutDesc.layout, &meshDevice,
      layoutDesc.memoryConfig);

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
    Device parentMesh, const std::vector<uint32_t> &meshShape,
    const std::optional<const std::vector<uint32_t>> &meshOffset) {
  ::ttnn::MeshDevice &parentMeshDevice =
      parentMesh.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  LOG_ASSERT(parentMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");

  LOG_ASSERT(meshShape.size() == 2, "Mesh shape must be 2D for now");
  ::ttnn::MeshShape shape{meshShape[0], meshShape[1]};

  std::optional<::ttnn::MeshCoordinate> offset = std::nullopt;
  if (meshOffset.has_value()) {
    LOG_ASSERT(meshOffset.value().size() == 2,
               "Mesh offset must be 2D for now");
    offset =
        ::ttnn::MeshCoordinate{meshOffset.value()[0], meshOffset.value()[1]};
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

void reshapeMeshDevice(Device meshDevice,
                       const std::vector<uint32_t> &meshShape) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  ttnnMeshDevice.reshape(::ttnn::MeshShape(meshShape[0], meshShape[1]));
}

void deallocateBuffers(Device deviceHandle) {
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  ::ttnn::deallocate_buffers(&meshDevice);
}

void dumpMemoryReport(Device deviceHandle) {
  const ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  ::tt::tt_metal::detail::DumpDeviceMemoryState(&meshDevice);
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

void wait(const std::vector<::tt::runtime::Tensor> &tensors) {
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
  std::vector<::ttnn::Tensor> singleTensors =
      ::ttnn::distributed::get_device_tensors(multiDeviceTensor);
  for (auto &tensor : singleTensors) {
    hostTensors.push_back(::tt::runtime::ttnn::toHostSingleTensor(
        utils::createRuntimeTensorFromTTNN(tensor, shouldRetain), untilize));
  }
  return hostTensors;
}

::tt::runtime::Tensor toLayout(::tt::runtime::Tensor tensor, Device device,
                               Layout layout, std::optional<bool> retain) {
  const LayoutDesc tensorLayoutDesc = LayoutDesc::fromTensor(tensor);

  const LayoutDesc &desiredLayoutDesc =
      layout.as<LayoutDesc>(DeviceRuntime::TTNN);

  OptionalMeshDeviceRef meshDevice =
      std::ref(device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN));

  ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);

  const ::ttnn::Tensor &ttnnTensor = tensorWrapper.getTensor();
  bool shouldRetain = retain.value_or(tensorWrapper.shouldRetain());

  LayoutConverter converter(tensorLayoutDesc, desiredLayoutDesc);
  ::ttnn::Tensor out = converter.convertTensorLayout(ttnnTensor, meshDevice);

  ::tt::runtime::Tensor result =
      utils::createRuntimeTensorFromTTNN(out, shouldRetain);

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
    const void *srcPtr = utils::getRawHostDataPtr(srcTensor);
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
  if (utils::isOnHost(srcTensor.storage_type()) &&
      utils::isOnHost(dstTensor.storage_type())) {
    void *dstPtr = utils::getRawHostDataPtr(dstTensor);
    const void *srcPtr = utils::getRawHostDataPtr(srcTensor);
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
  const auto &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  return std::string(opContext.debug_info()->c_str());
}

std::string getOpLocInfo(OpContext opContextHandle) {
  const auto &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  return std::string(opContext.loc_info()->c_str());
}

std::optional<const ::tt::target::ttnn::TensorRef *>
getOpOutputTensorRef(OpContext opContextHandle,
                     CallbackContext programContextHandle) {
  const auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  const auto &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  const ttnn::ProgramTensorPool &tensorPool = programContext.getTensorPool();
  std::optional<const ::tt::target::ttnn::TensorRef *> tensorRef = std::nullopt;

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
  case ::tt::target::ttnn::OpType::Pool2dOp: {
    tensorRef = opContext.type_as_Pool2dOp()->out();
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
    return std::nullopt;
  }
  default: {
    LOG_FATAL("Unsupported operation type");
  }
  }

  if (tensorRef.has_value() && tensorPool.contains(tensorRef.value())) {
    return tensorRef;
  } else {
    LOG_WARNING("Output tensor not found in tensor pool");
    return std::nullopt;
  }
}

std::unique_ptr<CallbackTensorBase>
getOpOutputTensor(OpContext opContextHandle,
                  CallbackContext programContextHandle) {
  const auto &tensorRefPtr =
      getOpOutputTensorRef(opContextHandle, programContextHandle);

  if (!tensorRefPtr.has_value()) {
    LOG_WARNING("Output tensor not found in tensor pool");
    return std::make_unique<CallbackTensor>();
  }

  const auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  const ttnn::ProgramTensorPool &tensorPool = programContext.getTensorPool();

  if (!tensorPool.contains(tensorRefPtr.value())) {
    LOG_WARNING("Output tensor not found in tensor pool");
    return std::make_unique<CallbackTensor>();
  }

  const auto &outPtr =
      &tensorPool.getTTNNTensorAndValidate(tensorRefPtr.value());

  std::shared_ptr<::ttnn::Tensor> hostTensor =
      std::make_shared<::ttnn::Tensor>(::ttnn::to_layout(
          ::ttnn::from_device(*outPtr), ::ttnn::Layout::ROW_MAJOR, std::nullopt,
          std::nullopt, static_cast<::ttnn::IDevice *>(nullptr)));

  return std::make_unique<CallbackTensor>(
      ::tt::runtime::Tensor(std::static_pointer_cast<void>(hostTensor), nullptr,
                            DeviceRuntime::TTNN),
      tensorRefPtr.value(), nullptr, DeviceRuntime::TTNN);
}

std::vector<const tt::target::ttnn::TensorRef *> convertFbTensorRefsToVector(
    const flatbuffers::Vector<flatbuffers::Offset<tt::target::ttnn::TensorRef>>
        *fbVector) {
  std::vector<const tt::target::ttnn::TensorRef *> stdVector;
  if (!fbVector) {
    return stdVector;
  }
  stdVector.reserve(fbVector->size());
  for (size_t i = 0; i < fbVector->size(); i++) {
    stdVector.push_back(fbVector->Get(i));
  }
  return stdVector;
}

std::vector<const ::tt::target::ttnn::TensorRef *>
getOpInputTensorRefs(OpContext opContextHandle,
                     CallbackContext programContextHandle) {

  const auto &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);

  std::vector<const ::tt::target::ttnn::TensorRef *> tensorRefs;

  switch (opContext.type_type()) {
  case ::tt::target::ttnn::OpType::ArangeOp: {
    tensorRefs = {};
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    tensorRefs = {};
    break;
  }
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    tensorRefs = {};
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    tensorRefs = {};
    break;
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    tensorRefs = {opContext.type_as_ToMemoryConfigOp()->in0()};
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    tensorRefs = {opContext.type_as_ToLayoutOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::ToDTypeOp: {
    tensorRefs = {opContext.type_as_ToDTypeOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    tensorRefs = {opContext.type_as_TypecastOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    tensorRefs = {opContext.type_as_ToDeviceOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    tensorRefs = {opContext.type_as_FromDeviceOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    tensorRefs =
        convertFbTensorRefsToVector(opContext.type_as_EltwiseOp()->ins());
    break;
  }
  case ::tt::target::ttnn::OpType::LinearOp: {
    tensorRefs = {opContext.type_as_LinearOp()->a(),
                  opContext.type_as_LinearOp()->b(),
                  opContext.type_as_LinearOp()->bias()};
    break;
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    tensorRefs = {opContext.type_as_MatmulOp()->a(),
                  opContext.type_as_MatmulOp()->b()};
    break;
  }
  case ::tt::target::ttnn::OpType::MorehCumSumOp: {
    tensorRefs = {opContext.type_as_MorehCumSumOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionArgMaxOp: {
    tensorRefs = {opContext.type_as_ReductionArgMaxOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionProdOp: {
    tensorRefs = {opContext.type_as_ReductionProdOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    tensorRefs = {opContext.type_as_ReductionOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    tensorRefs = {opContext.type_as_EmbeddingOp()->input()};
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingBackwardOp: {
    tensorRefs = {opContext.type_as_EmbeddingBackwardOp()->input()};
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    tensorRefs = {opContext.type_as_SoftmaxOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    tensorRefs = {opContext.type_as_TransposeOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::PadOp: {
    tensorRefs = {opContext.type_as_PadOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    tensorRefs =
        convertFbTensorRefsToVector(opContext.type_as_ConcatOp()->inputs());
    break;
  }
  case ::tt::target::ttnn::OpType::PermuteOp: {
    tensorRefs = {opContext.type_as_PermuteOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    tensorRefs = {opContext.type_as_ReshapeOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    tensorRefs = {opContext.type_as_SliceOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatOp: {
    tensorRefs = {opContext.type_as_RepeatOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatInterleaveOp: {
    tensorRefs = {opContext.type_as_RepeatInterleaveOp()->input()};
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    tensorRefs = {opContext.type_as_Conv2dOp()->input()};
    break;
  }
  case ::tt::target::ttnn::OpType::ConvTranspose2dOp: {
    tensorRefs = {opContext.type_as_ConvTranspose2dOp()->input()};
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    tensorRefs = {opContext.type_as_MaxPool2dOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    tensorRefs = {opContext.type_as_AllGatherOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    tensorRefs = {opContext.type_as_ReduceScatterOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::CollectivePermuteOp: {
    tensorRefs = {opContext.type_as_CollectivePermuteOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    tensorRefs = {opContext.type_as_MeshShardOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::UpsampleOp: {
    tensorRefs = {opContext.type_as_UpsampleOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::CpuOp: {
    tensorRefs = convertFbTensorRefsToVector(opContext.type_as_CpuOp()->ins());
    break;
  }
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    tensorRefs = {opContext.type_as_DeallocateOp()->in()};
    break;
  }
  default: {
    LOG_FATAL("Unsupported operation type");
  }
  }
  return tensorRefs;
}

std::vector<std::unique_ptr<CallbackTensorBase>>
getOpInputTensors(OpContext opContextHandle,
                  CallbackContext programContextHandle) {
  const auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  const ttnn::ProgramTensorPool &tensorPool = programContext.getTensorPool();
  std::vector<const ::tt::target::ttnn::TensorRef *> tensorRefs =
      getOpInputTensorRefs(opContextHandle, programContextHandle);
  std::vector<std::unique_ptr<CallbackTensorBase>> tensors;

  for (const auto &tensorRef : tensorRefs) {
    if (!tensorPool.contains(tensorRef)) {
      LOG_WARNING("Output tensor not found in tensor pool");
      tensors.push_back(std::make_unique<CallbackTensor>());
      continue;
    }
    const auto &inPtr = &tensorPool.getTTNNTensorAndValidate(tensorRef);

    std::shared_ptr<::ttnn::Tensor> hostTensor =
        std::make_shared<::ttnn::Tensor>(::ttnn::to_layout(
            ::ttnn::from_device(*inPtr), ::ttnn::Layout::ROW_MAJOR,
            std::nullopt, std::nullopt,
            static_cast<::ttnn::IDevice *>(nullptr)));

    const void *tensorRefPtr = reinterpret_cast<const void *>(tensorRef);
    tensors.push_back(std::make_unique<CallbackTensor>(
        ::tt::runtime::Tensor(std::move(hostTensor), nullptr,
                              DeviceRuntime::TTNN),
        tensorRefPtr, nullptr, DeviceRuntime::TTNN));
  }
  return tensors;
}

std::vector<::tt::runtime::Tensor>
submit(Device deviceHandle, Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs) {

  std::shared_ptr<::ttnn::MeshDevice> meshDevice =
      deviceHandle.asSharedPtr<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  std::vector<::tt::runtime::Tensor> outputs = ::tt::runtime::ttnn::runProgram(
      std::move(meshDevice), executableHandle, programIndex, inputs);

  return outputs;
}

std::vector<Tensor> runProgram(std::shared_ptr<::ttnn::MeshDevice> meshDevice,
                               Binary executableHandle,
                               std::uint32_t programIndex,
                               std::vector<::tt::runtime::Tensor> &inputs) {
  const ::tt::target::ttnn::TTNNBinary &fbb = *getBinary(executableHandle);
  const ::tt::target::ttnn::Program *program =
      fbb.programs()->Get(programIndex);
  ProgramExecutor executor(program, executableHandle, inputs,
                           std::move(meshDevice), programIndex);
  executor.execute();
  std::vector<::tt::runtime::Tensor> outputTensors =
      executor.gatherOutputTensors();
  return outputTensors;
}

} // namespace tt::runtime::ttnn
