// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "Constants.h"

#include "tt-metalium/fabric.hpp"
#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/common.h"
#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/detail/ttnn/debug_apis.h"
#include "tt/runtime/detail/ttnn/layout_converter.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/trace_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttmlir/Target/TTNN/types_generated.h"
#include "ttmlir/Version.h"
#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "types_generated.h"
#include <numeric>

#include <memory>
#include <optional>
#include <vector>

namespace tt::runtime::ttnn {

using ::tt::runtime::DeviceRuntime;

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

template <typename T>
static ::ttnn::Tensor createBorrowedTTNNTensor(void *rawData,
                                               const ::ttnn::Shape &shape) {
  std::uint64_t numElements = shape.volume();
  T *typedData = static_cast<T *>(rawData);
  ::ttsl::Span<T> data(typedData, typedData + numElements);
  ::ttnn::Tensor tensor =
      ::ttnn::Tensor::from_borrowed_data(data, shape, []() {}, []() {});
  return tensor;
}

static ::ttnn::Tensor
createOwnedTTNNTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {
  const void *dataToUse = data;
  ::tt::target::DataType dataTypeToUse = dataType;
  std::vector<std::byte> castedData;
  if (!::tt::runtime::utils::isSupportedDataType(dataType)) {
    dataTypeToUse = ::tt::runtime::utils::getUnsupportedDataTypeAlias(dataType);

    LOG_WARNING("User provided a tensor of data type: ",
                ::tt::target::EnumNameDataType(dataType),
                " which is not supported by runtime/ttnn. Casting to: ",
                ::tt::target::EnumNameDataType(dataTypeToUse),
                ", this may impact throughput and the integrity of the data.");

    uint64_t numElements = std::accumulate(shape.begin(), shape.end(),
                                           static_cast<std::uint64_t>(1),
                                           std::multiplies<std::uint64_t>());

    std::uint32_t itemSizeToUse =
        ::tt::runtime::utils::dataTypeElementSize(dataTypeToUse);

    castedData.resize(itemSizeToUse * numElements);

    if (data != nullptr) {
      ::tt::runtime::utils::handleBufferCast(data, castedData.data(), dataType,
                                             dataTypeToUse, numElements);
    }
    dataToUse = castedData.data();
  }

  ::ttnn::Shape ttnnShape(shape);
  ::ttnn::DataType ttnnDataType = utils::toTTNNDataType(dataTypeToUse);

  switch (ttnnDataType) {
  case ::ttnn::DataType::FLOAT32:
    return utils::createTTNNTensor<float>(dataToUse, ttnnShape, ttnnDataType);
  case ::ttnn::DataType::BFLOAT16:
    return utils::createTTNNTensor<bfloat16>(dataToUse, ttnnShape,
                                             ttnnDataType);
  case ::ttnn::DataType::UINT32:
    return utils::createTTNNTensor<uint32_t>(dataToUse, ttnnShape,
                                             ttnnDataType);
  case ::ttnn::DataType::UINT16:
    return utils::createTTNNTensor<uint16_t>(dataToUse, ttnnShape,
                                             ttnnDataType);
  case ::ttnn::DataType::UINT8:
    return utils::createTTNNTensor<uint8_t>(dataToUse, ttnnShape, ttnnDataType);
  case ::ttnn::DataType::INT32:
    return utils::createTTNNTensor<int32_t>(dataToUse, ttnnShape, ttnnDataType);
  default:
    LOG_FATAL("Unsupported data type");
  }
}

static ::tt::runtime::Tensor
toHostSingleTensor(const ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper,
                   bool untilize, bool blocking) {
  const ::ttnn::Tensor &inputTensor = tensorWrapper.getTensor();
  bool shouldRetain = tensorWrapper.shouldRetain();

  // If the tensor is on host, no event recording needed
  if (utils::isOnHost(inputTensor.storage_type())) {
    ::ttnn::Tensor hostTensor = inputTensor;
    if (untilize) {
      hostTensor = ::ttnn::to_layout(hostTensor, ::ttnn::Layout::ROW_MAJOR,
                                     std::nullopt, std::nullopt);
    }
    return utils::createRuntimeTensorFromTTNN(
        hostTensor, /*meshEvent=*/std::nullopt, shouldRetain);
  }

  ::ttnn::MeshDevice *meshDevice = inputTensor.device();
  LOG_ASSERT(meshDevice, "Device tensor must live on a mesh device");

  // If untilize is true and the data type can be untilized on device
  bool untilizeOnDevice =
      untilize && utils::canUntilizeDataTypeOnDevice(inputTensor.dtype());
  // If blackhole workarounds are enabled, only untilize on device if the
  // architecture is not blackhole
  if (::tt::runtime::workaround::Env::get().blackholeWorkarounds) {
    untilizeOnDevice &= getArch() != ::tt::runtime::Arch::BLACKHOLE;
  }
  if (untilizeOnDevice) {
    ::ttnn::Tensor hostTensor = ::ttnn::from_device(
        ::ttnn::to_layout(inputTensor, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                          std::nullopt),
        blocking);

    std::optional<::ttnn::MeshEvent> meshEvent = std::nullopt;
    if (!blocking) {
      meshEvent = ::ttnn::events::record_mesh_event(
          meshDevice,
          ::ttnn::QueueId(tt::tt_metal::GetCurrentCommandQueueIdForThread()));
    }

    return utils::createRuntimeTensorFromTTNN(hostTensor, meshEvent,
                                              shouldRetain);
  }

  // Host untilization requires data to be fully transferred first
  // Therefore we need to block on from_device if untilize is true
  if (untilize && !blocking) {
    LOG_WARNING("Overriding blocking parameter to true because tensor cannot "
                "be untilized on device.");
    blocking = true;
  }

  ::ttnn::Tensor hostTensor =
      ::ttnn::from_device(inputTensor, /*blocking=*/blocking);

  if (untilize) {
    hostTensor = ::ttnn::to_layout(hostTensor, ::ttnn::Layout::ROW_MAJOR,
                                   std::nullopt, std::nullopt);
  }

  std::optional<::ttnn::MeshEvent> meshEvent = std::nullopt;
  // if we don't need to untilize, then from_device can execute asynchronously
  // in this case we need to populate the event
  if (!untilize && !blocking) {
    meshEvent = ::ttnn::events::record_mesh_event(
        meshDevice,
        ::ttnn::QueueId(tt::tt_metal::GetCurrentCommandQueueIdForThread()));
  }

  return utils::createRuntimeTensorFromTTNN(hostTensor, /*meshEvent=*/meshEvent,
                                            shouldRetain);
}

::tt::runtime::Tensor
createBorrowedHostTensor(void *data, const std::vector<std::uint32_t> &shape,
                         const std::vector<std::uint32_t> &stride,
                         std::uint32_t itemsize,
                         ::tt::target::DataType dataType) {
  LOG_ASSERT(
      data != nullptr ||
          (shape.size() == 0 ||
           std::accumulate(shape.begin(), shape.end(), 1,
                           std::multiplies<std::uint32_t>()) == 0),
      "Cannot create borrowed tensor with null data unless the volume is 0.");
  LOG_ASSERT(::tt::runtime::utils::isSupportedDataType(dataType),
             "Cannot create borrowed tensor with unsupported data type");
  ::ttnn::Shape ttnnShape(shape);

  switch (dataType) {
  case ::tt::target::DataType::Float32:
    return utils::createRuntimeTensorFromTTNN(
        createBorrowedTTNNTensor<float>(data, ttnnShape));
  case ::tt::target::DataType::BFloat16:
    return utils::createRuntimeTensorFromTTNN(
        createBorrowedTTNNTensor<bfloat16>(data, ttnnShape));
  case ::tt::target::DataType::UInt32:
    return utils::createRuntimeTensorFromTTNN(
        createBorrowedTTNNTensor<uint32_t>(data, ttnnShape));
  case ::tt::target::DataType::UInt16:
    return utils::createRuntimeTensorFromTTNN(
        createBorrowedTTNNTensor<uint16_t>(data, ttnnShape));
  case ::tt::target::DataType::UInt8:
    return utils::createRuntimeTensorFromTTNN(
        createBorrowedTTNNTensor<uint8_t>(data, ttnnShape));
  case ::tt::target::DataType::Int32:
    return utils::createRuntimeTensorFromTTNN(
        createBorrowedTTNNTensor<int32_t>(data, ttnnShape));
  default:
    LOG_FATAL("Unsupported data type");
  }
}

::tt::runtime::Tensor
createOwnedHostTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {

  ::tt::runtime::Tensor tensor = utils::createRuntimeTensorFromTTNN(
      createOwnedTTNNTensor(data, shape, stride, itemsize, dataType));
  return tensor;
}

::tt::runtime::Tensor createMultiDeviceHostTensor(
    const std::vector<::tt::runtime::Tensor> &tensorShards,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape) {
  std::vector<::ttnn::Tensor> ttnnTensorShards;
  ttnnTensorShards.reserve(tensorShards.size());
  std::transform(tensorShards.begin(), tensorShards.end(),
                 std::back_inserter(ttnnTensorShards),
                 [&](::tt::runtime::Tensor tensorShard) -> ::ttnn::Tensor {
                   return utils::getTTNNTensorFromRuntimeTensor(tensorShard);
                 });

  LOG_ASSERT(meshShape.size() == 2, "Only 2D mesh shape supported for now.");
  ::ttnn::MeshShape ttnnMeshShape(meshShape[0], meshShape[1]);

  ::ttnn::Tensor multiDeviceHostTensor =
      ::ttnn::distributed::from_host_shards(ttnnTensorShards, ttnnMeshShape);

  return utils::createRuntimeTensorFromTTNN(multiDeviceHostTensor);
}

::tt::runtime::Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape) {
  std::vector<::tt::runtime::Tensor> tensorShards;
  tensorShards.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(tensorShards),
                 [&](const void *dataShard) -> ::tt::runtime::Tensor {
                   return createOwnedHostTensor(dataShard, shape, stride,
                                                itemsize, dataType);
                 });
  return createMultiDeviceHostTensor(tensorShards, strategy, meshShape);
}

::tt::runtime::Tensor createEmptyTensor(
    Device device, Layout layout, const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize) {
  const LayoutDesc &layoutDesc = layout.as<LayoutDesc>(DeviceRuntime::TTNN);
  LOG_ASSERT(::tt::runtime::utils::isSupportedDataType(
                 utils::fromTTNNDataType(layoutDesc.dataType)),
             "Data type must be supported");
  if (layoutDesc.isOnHost()) {
    ::ttnn::Tensor tensor =
        createOwnedTTNNTensor(nullptr, shape, stride, itemsize,
                              utils::fromTTNNDataType(layoutDesc.dataType));
    ::tt::runtime::Tensor out = utils::createRuntimeTensorFromTTNN(tensor);
    return ::tt::runtime::ttnn::toLayout(out, device, layout);
  }
  ::ttnn::MeshDevice &meshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  ::ttnn::TensorSpec tensorSpec(
      ::ttnn::Shape(shape),
      ::ttnn::TensorLayout(
          layoutDesc.dataType, ::ttnn::PageConfig(layoutDesc.layout),
          layoutDesc.memoryConfig.value_or(::ttnn::MemoryConfig{})));
  ::ttnn::Tensor tensor =
      ::tt::tt_metal::allocate_tensor_on_device(tensorSpec, &meshDevice);

  return utils::createRuntimeTensorFromTTNN(tensor);
}

bool isTensorAllocated(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      utils::getTTNNTensorFromRuntimeTensor(tensor);
  return ttnnTensor.is_allocated();
}

tt::target::DataType getTensorDataType(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      utils::getTTNNTensorFromRuntimeTensor(tensor);
  return utils::fromTTNNDataType(ttnnTensor.dtype());
}

std::vector<std::byte> getTensorDataBuffer(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      utils::getTTNNTensorFromRuntimeTensor(tensor);
  void *dataPtr = nullptr;
  std::uint32_t tensorVolume = getTensorVolume(tensor);

  if (tensorVolume == 0) {
    LOG_WARNING("getTensorDataBuffer: Tensor has zero volume; returning an "
                "empty data vector.");
    return {};
  }

  std::vector<std::byte> dataVec(getTensorElementSize(tensor) * tensorVolume);

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
      utils::getTTNNTensorFromRuntimeTensor(tensor);
  std::vector<std::uint32_t> shape;
  for (size_t i = 0; i < ttnnTensor.logical_shape().size(); ++i) {
    shape.push_back(ttnnTensor.logical_shape()[i]);
  }
  return shape;
}

std::vector<std::uint32_t> getTensorStride(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      utils::getTTNNTensorFromRuntimeTensor(tensor);
  std::vector<std::uint32_t> stride;
  for (size_t i = 0; i < ttnnTensor.strides().size(); ++i) {
    stride.push_back(ttnnTensor.strides()[i]);
  }
  return stride;
}

std::uint32_t getTensorElementSize(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      utils::getTTNNTensorFromRuntimeTensor(tensor);
  return ttnnTensor.element_size();
}

std::uint32_t getTensorVolume(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      utils::getTTNNTensorFromRuntimeTensor(tensor);
  return ttnnTensor.physical_volume();
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

Arch getArch() {
  return ::tt::runtime::common::toRuntimeArch(::tt::tt_metal::hal::get_arch());
}

void enablePersistentKernelCache() {
  ::tt::tt_metal::detail::EnablePersistentKernelCache();
}

void disablePersistentKernelCache() {
  ::tt::tt_metal::detail::DisablePersistentKernelCache();
}

size_t getNumAvailableDevices() {
  return ::tt::tt_metal::GetNumAvailableDevices();
}

Device openMeshDevice(const MeshDeviceOptions &options) {
  std::optional<::ttnn::MeshShape> meshShape = std::nullopt;
  if (options.meshShape.has_value()) {
    LOG_ASSERT(options.meshShape.value().size() == 2,
               "Mesh shape must be 2D for now");
    meshShape = ::ttnn::MeshShape(options.meshShape.value());
  }

  LOG_ASSERT(options.meshOffset.size() == 2, "Mesh offset must be 2D for now");
  ::ttnn::MeshCoordinate offset(options.meshOffset);

  size_t l1SmallSize =
      options.l1SmallSize.value_or(::tt::constants::L1_SMALL_SIZE);
  size_t traceRegionSize =
      options.traceRegionSize.value_or(DEFAULT_TRACE_REGION_SIZE);
  ::tt::tt_metal::DispatchCoreType dispatchCoreTypeValue =
      tt::runtime::common::getDispatchCoreType(options.dispatchCoreType);

  ::ttnn::MeshDeviceConfig meshConfig(meshShape, offset, options.deviceIds);

  std::shared_ptr<::ttnn::MeshDevice> meshDevice =
      ::ttnn::MeshDevice::create(meshConfig, l1SmallSize, traceRegionSize,
                                 options.numHWCQs, dispatchCoreTypeValue);

  if (options.enableProgramCache) {
    meshDevice->enable_program_cache();
  } else {
    meshDevice->disable_and_clear_program_cache();
  }

  LOG_DEBUG("Device grid size = { ",
            meshDevice->compute_with_storage_grid_size().x, ", ",
            meshDevice->compute_with_storage_grid_size().y, " }");

  auto ttnnTraceCache =
      std::make_shared<::tt::runtime::ttnn::TraceCache>(meshDevice);
  auto traceCache = std::make_shared<::tt::runtime::TraceCache>(
      std::static_pointer_cast<void>(ttnnTraceCache), DeviceRuntime::TTNN);

  return Device(std::static_pointer_cast<void>(meshDevice), traceCache,
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

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE) && TT_RUNTIME_ENABLE_PERF_TRACE == 1
  ::tt::tt_metal::ReadMeshDeviceProfilerResults(ttnnMeshDevice);
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

  auto ttnnTraceCache =
      std::make_shared<::tt::runtime::ttnn::TraceCache>(subMeshDevice);
  auto traceCache = std::make_shared<::tt::runtime::TraceCache>(
      std::static_pointer_cast<void>(ttnnTraceCache), DeviceRuntime::TTNN);
  return Device(std::static_pointer_cast<void>(subMeshDevice), traceCache,
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

std::vector<uint32_t> getMeshShape(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  std::vector<uint32_t> shape(ttnnMeshDevice.shape().view().begin(),
                              ttnnMeshDevice.shape().view().end());
  return shape;
}

std::vector<int> getDeviceIds(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return ttnnMeshDevice.get_device_ids();
}

size_t getNumHwCqs(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return static_cast<size_t>(ttnnMeshDevice.num_hw_cqs());
}

bool isProgramCacheEnabled(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return ttnnMeshDevice.get_program_cache().is_enabled();
}

size_t getL1SmallSize(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return ttnnMeshDevice.allocator()->get_config().l1_small_size;
}

size_t getTraceRegionSize(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return ttnnMeshDevice.allocator()->get_config().trace_region_size;
}

size_t getNumDramChannels(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return ttnnMeshDevice.num_dram_channels();
}

size_t getDramSizePerChannel(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return ttnnMeshDevice.dram_size_per_channel();
}

size_t getL1SizePerCore(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return ttnnMeshDevice.l1_size_per_core();
}

void releaseTrace(Device meshDevice, std::uint64_t binaryId,
                  size_t mainProgramId) {
  ::tt::runtime::ttnn::TraceCache &traceCache =
      meshDevice.getTraceCache()->as<TraceCache>(DeviceRuntime::TTNN);

  MainProgramKey mainProgramKey(binaryId, mainProgramId);
  traceCache.erase(mainProgramKey);
}

void deallocateBuffers(Device deviceHandle) {
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  ::ttnn::deallocate_buffers(&meshDevice);
}

void dumpMemoryReport(Device deviceHandle) {
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  ::tt::tt_metal::detail::DumpDeviceMemoryState(&meshDevice);
}

void readDeviceProfilerResults(Device deviceHandle) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  LOG_ASSERT(ttnnMeshDevice.is_parent_mesh(),
             "Mesh device must be a parent mesh");

#if defined(TT_RUNTIME_ENABLE_PERF_TRACE)
  ::tt::tt_metal::ReadMeshDeviceProfilerResults(ttnnMeshDevice);
#endif
}

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device deviceHandle) {
  std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
      memoryMap;
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  auto dramMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      &meshDevice, ::ttnn::BufferType::DRAM);
  auto l1MemoryView = ::tt::tt_metal::detail::GetMemoryView(
      &meshDevice, ::ttnn::BufferType::L1);
  auto l1SmallMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      &meshDevice, ::ttnn::BufferType::L1_SMALL);
  auto traceMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      &meshDevice, ::ttnn::BufferType::TRACE);

  memoryMap[tt::runtime::MemoryBufferType::DRAM] =
      createMemoryView(dramMemoryView);
  memoryMap[tt::runtime::MemoryBufferType::L1] = createMemoryView(l1MemoryView);
  memoryMap[tt::runtime::MemoryBufferType::L1_SMALL] =
      createMemoryView(l1SmallMemoryView);
  memoryMap[tt::runtime::MemoryBufferType::TRACE] =
      createMemoryView(traceMemoryView);

  return memoryMap;
}

void setFabricConfig(FabricConfig config) {
  ::tt::tt_fabric::SetFabricConfig(common::toTTFabricConfig(config));
  RuntimeContext::instance().setCurrentFabricConfig(config);
}

void wait(Event event) {
  LOG_FATAL("Waiting on events is not supported for ttnn runtime. Please use "
            "wait on tensors instead.");
}

void wait(::tt::runtime::Tensor tensor, std::optional<uint8_t> cqId) {
  LOG_ASSERT(tensor.matchesRuntime(DeviceRuntime::TTNN),
             "Expected ttnn tensor");

  const ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
  const std::optional<::ttnn::MeshEvent> &meshEvent =
      tensorWrapper.getMeshEvent();

  if (!meshEvent.has_value()) {
    return;
  }

  // If no cqId provided, block and wait until the event is recorded
  if (!cqId.has_value()) {
    ::ttnn::events::event_synchronize(meshEvent.value());
    return;
  }

  // tell cqId to wait until the event is recorded
  ::ttnn::QueueId cqIdValue(cqId.value());
  ::ttnn::events::wait_for_mesh_event(cqIdValue, meshEvent.value());
}

void wait(const std::vector<::tt::runtime::Tensor> &tensors,
          std::optional<uint8_t> cqId) {
  for (const ::tt::runtime::Tensor &tensor : tensors) {
    ::tt::runtime::ttnn::wait(tensor, cqId);
  }
}

std::vector<::tt::runtime::Tensor> toHost(::tt::runtime::Tensor tensor,
                                          bool untilize, bool blocking) {
  const ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);

  ::tt::runtime::Tensor multiDeviceHostTensor =
      ::tt::runtime::ttnn::toHostSingleTensor(tensorWrapper, untilize,
                                              blocking);

  std::vector<::ttnn::Tensor> singleTensors =
      ::ttnn::distributed::get_device_tensors(
          utils::getTTNNTensorFromRuntimeTensor(multiDeviceHostTensor));

  const std::optional<::ttnn::MeshEvent> &meshEvent =
      tensorWrapper.getMeshEvent();

  std::vector<::tt::runtime::Tensor> hostTensors;
  for (const ::ttnn::Tensor &tensor : singleTensors) {
    hostTensors.push_back(utils::createRuntimeTensorFromTTNN(
        tensor, meshEvent, tensorWrapper.shouldRetain()));
  }

  return hostTensors;
}

::tt::runtime::Tensor toLayout(::tt::runtime::Tensor tensor, Device device,
                               Layout layout, std::optional<bool> retain) {
  const std::shared_ptr<LayoutDesc> tensorLayoutDesc =
      LayoutDesc::fromTensor(tensor);

  const LayoutDesc &desiredLayoutDesc =
      layout.as<LayoutDesc>(DeviceRuntime::TTNN);

  OptionalMeshDeviceRef meshDevice =
      std::ref(device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN));

  ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);

  const ::ttnn::Tensor &ttnnTensor = tensorWrapper.getTensor();
  bool shouldRetain = retain.value_or(tensorWrapper.shouldRetain());

  LayoutConverter converter(*tensorLayoutDesc, desiredLayoutDesc);
  ::ttnn::Tensor out = converter.convertTensorLayout(ttnnTensor, meshDevice);

  ::tt::runtime::Tensor result = utils::createRuntimeTensorFromTTNN(
      out, /*meshEvent=*/std::nullopt, shouldRetain);

  if (!shouldRetain) {
    ::tt::runtime::ttnn::deallocateTensor(tensor);
  }
  return result;
}

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex) {
  const ::tt::target::ttnn::TTNNBinary &fbb =
      *utils::getBinary(executableHandle);
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

void memcpy(void *dst, ::tt::runtime::Tensor src,
            std::optional<::tt::target::DataType> dstDataType) {

  if (dstDataType.has_value()) {
    LOG_ASSERT(
        dstDataType.value() == getTensorDataType(src) ||
            !::tt::runtime::utils::isSupportedDataType(dstDataType.value()),
        "If destination data type is specified, it must match the "
        "source data type or be an unsupported data type.");
  }

  const ::ttnn::Tensor &srcTensor = utils::getTTNNTensorFromRuntimeTensor(src);

  // Handle cast and copy
  if (dstDataType.has_value() &&
      !::tt::runtime::utils::isSupportedDataType(dstDataType.value())) {
    LOG_ASSERT(utils::isOnHost(srcTensor.storage_type()),
               "Tensor must be on host");
    const void *srcPtr = utils::getRawHostDataPtr(srcTensor);

    ::tt::target::DataType srcDataType = getTensorDataType(src);
    ::tt::target::DataType unsupportedDataTypeAlias =
        tt::runtime::utils::getUnsupportedDataTypeAlias(dstDataType.value());

    LOG_ASSERT(
        srcDataType == unsupportedDataTypeAlias,
        "Tensor data type must be the alias of the unsupported data type: " +
            std::string(target::EnumNameDataType(unsupportedDataTypeAlias)));

    LOG_WARNING(
        "User is requesting to copy the data from a runtime tensor with "
        "data type: ",
        ::tt::target::EnumNameDataType(srcDataType),
        " into buffer with expected data type: ",
        ::tt::target::EnumNameDataType(*dstDataType),
        ", the values will be casted, this may impact the throughput and the "
        "integrity of the data.");

    // Cast to dstDataType, mempy into dst, and return
    return ::tt::runtime::utils::handleBufferCast(srcPtr, dst, srcDataType,
                                                  dstDataType.value(),
                                                  srcTensor.physical_volume());
  }

  // Handle direct copy without cast
  if (utils::isOnHost(srcTensor.storage_type())) {
    const void *srcPtr = utils::getRawHostDataPtr(srcTensor);
    size_t size = srcTensor.physical_volume() * srcTensor.element_size();
    std::memcpy(dst, srcPtr, size);
  } else {
    ::tt::tt_metal::memcpy(dst, srcTensor);
  }
}

void memcpy(::tt::runtime::Tensor dst, ::tt::runtime::Tensor src) {
  ::ttnn::Tensor &dstTensor = utils::getTTNNTensorFromRuntimeTensor(dst);
  const ::ttnn::Tensor &srcTensor = utils::getTTNNTensorFromRuntimeTensor(src);
  LOG_ASSERT(srcTensor.physical_volume() * srcTensor.element_size() ==
                 dstTensor.physical_volume() * dstTensor.element_size(),
             "Input output tensor size mismatch in memcpy: ",
             srcTensor.physical_volume(), " * ", srcTensor.element_size(),
             " != ", dstTensor.physical_volume(), " * ",
             dstTensor.element_size());
  if (utils::isOnHost(srcTensor.storage_type()) &&
      utils::isOnHost(dstTensor.storage_type())) {
    void *dstPtr = utils::getRawHostDataPtr(dstTensor);
    const void *srcPtr = utils::getRawHostDataPtr(srcTensor);
    size_t size = srcTensor.physical_volume() * srcTensor.element_size();
    std::memcpy(dstPtr, srcPtr, size);
  } else {
    ::tt::tt_metal::memcpy(dstTensor, srcTensor);
  }
}

void deallocateTensor(::tt::runtime::Tensor &tensor, bool force) {
  // If the tensor is retained, do not deallocate
  if (getTensorRetain(tensor)) {
    LOG_DEBUG("Tensor is retained thus not deallocating. To deallocate, set "
              "retain to false first");
    return;
  }
  ::ttnn::Tensor &ttnnTensor = utils::getTTNNTensorFromRuntimeTensor(tensor);
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

std::unordered_map<std::uint32_t, Tensor>
getOpOutputTensor(OpContext opContextHandle,
                  CallbackContext programContextHandle) {
  std::unordered_map<std::uint32_t, Tensor> perDeviceOutputTensors;
  std::optional<tt::runtime::TensorRef> tensorRef =
      getOpOutputRef(opContextHandle, programContextHandle);
  if (!tensorRef) {
    return perDeviceOutputTensors;
  }

  const auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  const ttnn::ProgramTensorPool &tensorPool = programContext.getTensorPool();

  const auto *tensorRefPtr =
      &tensorRef->as<tt::target::ttnn::TensorRef>(DeviceRuntime::TTNN);

  if (!tensorRefPtr) {
    LOG_WARNING("Tensor ref pointer is null when retrieving tensor");
    return perDeviceOutputTensors;
  }

  if (!tensorPool.contains(tensorRefPtr)) {
    LOG_WARNING("Tensor not found in tensor pool when retrieving tensor");
    return perDeviceOutputTensors;
  }

  // Assumption: get_device_tensors returns tensors in row major order so each
  // index of the output list is the logical device id. If you print out the
  // physical device ids of the TTNN::tensor object, they will be different from
  // the logical device ids.
  ::tt::runtime::Tensor outTensor = utils::createRuntimeTensorFromTTNN(
      tensorPool.getTTNNTensorAndValidate(tensorRefPtr));
  std::vector<tt::runtime::Tensor> hostTensors =
      ::tt::runtime::ttnn::toHost(outTensor, true);

  for (size_t i = 0; i < hostTensors.size(); ++i) {
    perDeviceOutputTensors[i] = hostTensors[i];
  }

  return perDeviceOutputTensors;
}

std::optional<tt::runtime::TensorRef>
getOpOutputRef(OpContext opContextHandle,
               CallbackContext programContextHandle) {
  const auto &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);

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
  case ::tt::target::ttnn::OpType::RandOp: {
    tensorRef = opContext.type_as_RandOp()->out();
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
  case ::tt::target::ttnn::OpType::PrepareConv2dWeightsOp: {
    tensorRef = opContext.type_as_PrepareConv2dWeightsOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::PrepareConv2dBiasOp: {
    tensorRef = opContext.type_as_PrepareConv2dBiasOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::BatchNormOp: {
    tensorRef = opContext.type_as_BatchNormOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::RMSNormOp: {
    tensorRef = opContext.type_as_RMSNormOp()->out();
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
  case ::tt::target::ttnn::OpType::PointToPointOp: {
    tensorRef = opContext.type_as_PointToPointOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::BeginTraceCaptureOp: {
    tensorRef = opContext.type_as_BeginTraceCaptureOp()->trace_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatenateHeadsOp: {
    tensorRef = opContext.type_as_ConcatenateHeadsOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::RotaryEmbeddingLlamaOp: {
    tensorRef = opContext.type_as_RotaryEmbeddingLlamaOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::NLPConcatHeadsOp: {
    tensorRef = opContext.type_as_NLPConcatHeadsOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::LoadTensorOp: {
    tensorRef = opContext.type_as_LoadTensorOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::NLPConcatHeadsDecodeOp: {
    tensorRef = opContext.type_as_NLPConcatHeadsDecodeOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::SortOp:
  case ::tt::target::ttnn::OpType::LoadCachedOp:
  case ::tt::target::ttnn::OpType::GetDeviceOp:
  case ::tt::target::ttnn::OpType::DeallocateOp:
  case ::tt::target::ttnn::OpType::FuncCallOp:
  case ::tt::target::ttnn::OpType::WriteTensorOp:
  case ::tt::target::ttnn::OpType::EndTraceCaptureOp:
  case ::tt::target::ttnn::OpType::ExecuteTraceOp:
  case ::tt::target::ttnn::OpType::CaptureOrExecuteTraceOp:
  case ::tt::target::ttnn::OpType::DumpTensorOp: {
    LOG_WARNING("getting output tensor is not supported for ",
                ::tt::target::ttnn::EnumNamesOpType()[static_cast<size_t>(
                    opContext.type_type())]);
    return std::nullopt;
  }
  case ::tt::target::ttnn::OpType::GenericOp: {
    auto size = opContext.type_as_GenericOp()->io_tensors()->size();
    tensorRef = opContext.type_as_GenericOp()->io_tensors()->Get(size - 1);
    break;
  }
  case ::tt::target::ttnn::OpType::NONE: {
    LOG_FATAL("Invalid op type");
    break;
  }
  }

  if (!tensorRef.has_value()) {
    return std::nullopt;
  }

  return utils::createRuntimeTensorRefFromTTNN(tensorRef.value());
}

std::vector<tt::runtime::TensorRef>
getOpInputRefs(OpContext opContextHandle,
               CallbackContext programContextHandle) {

  const auto &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);

  std::vector<const ::tt::target::ttnn::TensorRef *> tensorRefs;

  switch (opContext.type_type()) {
  case ::tt::target::ttnn::OpType::ArangeOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::ConstantOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::RandOp: {
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
  case ::tt::target::ttnn::OpType::EltwiseBinaryOp: {
    tensorRefs = {opContext.type_as_EltwiseBinaryOp()->lhs(),
                  opContext.type_as_EltwiseBinaryOp()->rhs()};
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeOp: {
    tensorRefs = {opContext.type_as_EltwiseBinaryCompositeOp()->lhs(),
                  opContext.type_as_EltwiseBinaryCompositeOp()->rhs()};
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseTernaryWhereOp: {
    tensorRefs = {opContext.type_as_EltwiseTernaryWhereOp()->first(),
                  opContext.type_as_EltwiseTernaryWhereOp()->second(),
                  opContext.type_as_EltwiseTernaryWhereOp()->third()};
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseQuantizationOp: {
    tensorRefs = {opContext.type_as_EltwiseQuantizationOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryOp: {
    tensorRefs = {opContext.type_as_EltwiseUnaryOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryCompositeOp: {
    tensorRefs = {opContext.type_as_EltwiseUnaryCompositeOp()->in()};
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
    tensorRefs = {opContext.type_as_EmbeddingOp()->input(),
                  opContext.type_as_EmbeddingOp()->weight()};
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
    tensorRefs = utils::convertFbTensorRefsToVector(
        opContext.type_as_ConcatOp()->inputs());
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
  case ::tt::target::ttnn::OpType::Pool2dOp: {
    tensorRefs = {opContext.type_as_Pool2dOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::PrepareConv2dWeightsOp: {
    tensorRefs = {opContext.type_as_PrepareConv2dWeightsOp()->weight_tensor()};
    break;
  }
  case ::tt::target::ttnn::OpType::PrepareConv2dBiasOp: {
    tensorRefs = {opContext.type_as_PrepareConv2dBiasOp()->bias_tensor()};
    break;
  }
  case ::tt::target::ttnn::OpType::BatchNormOp: {
    tensorRefs = {opContext.type_as_BatchNormOp()->input(),
                  opContext.type_as_BatchNormOp()->running_mean(),
                  opContext.type_as_BatchNormOp()->running_var(),
                  opContext.type_as_BatchNormOp()->weight(),
                  opContext.type_as_BatchNormOp()->bias()};
    break;
  }
  case ::tt::target::ttnn::OpType::RMSNormOp: {
    tensorRefs = {opContext.type_as_RMSNormOp()->input()};
    if (opContext.type_as_RMSNormOp()->weight()) {
      tensorRefs.push_back(opContext.type_as_RMSNormOp()->weight());
    }
    if (opContext.type_as_RMSNormOp()->bias()) {
      tensorRefs.push_back(opContext.type_as_RMSNormOp()->bias());
    }
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
    tensorRefs =
        utils::convertFbTensorRefsToVector(opContext.type_as_CpuOp()->ins());
    break;
  }
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    tensorRefs = {opContext.type_as_DeallocateOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::UpdateCacheOp: {
    tensorRefs = {opContext.type_as_UpdateCacheOp()->cache(),
                  opContext.type_as_UpdateCacheOp()->input(),
                  opContext.type_as_UpdateCacheOp()->update_index()};
    break;
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    tensorRefs = {opContext.type_as_FillCacheOp()->cache(),
                  opContext.type_as_FillCacheOp()->input()};
    break;
  }
  case ::tt::target::ttnn::OpType::LoadCachedOp: {
    tensorRefs = utils::convertFbTensorRefsToVector(
        opContext.type_as_LoadCachedOp()->inputs());
    break;
  }
  case ::tt::target::ttnn::OpType::SortOp: {
    tensorRefs = {opContext.type_as_SortOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::PointToPointOp: {
    tensorRefs = {opContext.type_as_PointToPointOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::NamedFullOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::FuncCallOp: {
    for (const auto *input : *opContext.type_as_FuncCallOp()->inputs()) {
      tensorRefs.push_back(input);
    }
    break;
  }
  case ::tt::target::ttnn::OpType::WriteTensorOp: {
    tensorRefs = {opContext.type_as_WriteTensorOp()->host_tensor(),
                  opContext.type_as_WriteTensorOp()->device_tensor()};
    break;
  }
  case ::tt::target::ttnn::OpType::BeginTraceCaptureOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::EndTraceCaptureOp: {
    tensorRefs = {opContext.type_as_EndTraceCaptureOp()->trace_id()};
    break;
  }
  case ::tt::target::ttnn::OpType::ExecuteTraceOp: {
    tensorRefs = {opContext.type_as_ExecuteTraceOp()->trace_id()};
    break;
  }
  case ::tt::target::ttnn::OpType::CaptureOrExecuteTraceOp: {
    for (const auto *input :
         *opContext.type_as_CaptureOrExecuteTraceOp()->inputs()) {
      tensorRefs.push_back(input);
    }
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatenateHeadsOp: {
    tensorRefs = {opContext.type_as_ConcatenateHeadsOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::NLPConcatHeadsOp: {
    tensorRefs = {opContext.type_as_NLPConcatHeadsOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::NLPConcatHeadsDecodeOp: {
    tensorRefs = {opContext.type_as_NLPConcatHeadsDecodeOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::GenericOp: {
    for (const auto *input : *opContext.type_as_GenericOp()->io_tensors()) {
      tensorRefs.push_back(input);
    }
    break;
  }
  case ::tt::target::ttnn::OpType::RotaryEmbeddingLlamaOp: {
    tensorRefs = {opContext.type_as_RotaryEmbeddingLlamaOp()->input(),
                  opContext.type_as_RotaryEmbeddingLlamaOp()->cos_cache(),
                  opContext.type_as_RotaryEmbeddingLlamaOp()->sin_cache(),
                  opContext.type_as_RotaryEmbeddingLlamaOp()->trans_mat()};
    break;
  }
  case ::tt::target::ttnn::OpType::DumpTensorOp: {
    tensorRefs = {opContext.type_as_DumpTensorOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::LoadTensorOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::NONE: {
    LOG_FATAL("Invalid op type");
    break;
  }
  }
  std::vector<tt::runtime::TensorRef> rtTensorRefs;
  rtTensorRefs.reserve(tensorRefs.size());

  for (const auto *ref : tensorRefs) {
    rtTensorRefs.emplace_back(utils::createRuntimeTensorRefFromTTNN(ref));
  }

  return rtTensorRefs;
}

std::vector<::tt::runtime::Tensor>
submit(Device deviceHandle, Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs) {

  ProgramExecutor executor(deviceHandle, executableHandle, programIndex,
                           inputs);
  executor.execute();
  std::vector<::tt::runtime::Tensor> outputTensors =
      executor.gatherOutputTensors();

  return outputTensors;
}

std::optional<Tensor>
retrieveTensorFromPool(CallbackContext programContextHandle,
                       tt::runtime::TensorRef tensorRef, bool untilize) {
  const auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  const ttnn::ProgramTensorPool &tensorPool = programContext.getTensorPool();

  const auto *tensorRefPtr =
      &tensorRef.as<tt::target::ttnn::TensorRef>(DeviceRuntime::TTNN);

  if (!tensorRefPtr) {
    LOG_WARNING("Tensor ref pointer is null when retrieving tensor");
    return std::nullopt;
  }

  if (!tensorPool.contains(tensorRefPtr)) {
    LOG_WARNING("Tensor not found in tensor pool when retrieving tensor");
    return std::nullopt;
  }

  ::tt::runtime::Tensor outTensor = utils::createRuntimeTensorFromTTNN(
      tensorPool.getTTNNTensorAndValidate(tensorRefPtr));

  std::vector<tt::runtime::Tensor> hostTensors =
      ::tt::runtime::ttnn::toHost(outTensor, untilize);

  if (hostTensors.empty()) {
    LOG_WARNING("Failed to get host tensor when retrieving tensor");
    return std::nullopt;
  }

  if (hostTensors.size() != 1) {
    LOG_FATAL("Multi device tensor not supported when retrieving tensor");
  }

  return hostTensors[0];
}

void updateTensorInPool(CallbackContext programContextHandle,
                        TensorRef tensorRef, Tensor tensor) {
  auto &programContext =
      programContextHandle.as<tt::runtime::ttnn::ProgramContext>(
          DeviceRuntime::TTNN);
  ttnn::ProgramTensorPool &tensorPool = programContext.getTensorPool();
  const auto *tensorRefPtr =
      &tensorRef.as<tt::target::ttnn::TensorRef>(DeviceRuntime::TTNN);

  if (!tensorRefPtr) {
    LOG_WARNING("Tensor ref pointer is null when updating tensor");
    return;
  }
  if (!tensorPool.contains(tensorRefPtr)) {
    LOG_WARNING("Tensor not found in tensor pool when updating tensor");
    return;
  }

  ::ttnn::Tensor &srcTensor = utils::getTTNNTensorFromRuntimeTensor(tensor);
  ::ttnn::Tensor &dstTensor = tensorPool.getTTNNTensorAndValidate(tensorRefPtr);
  srcTensor = ::ttnn::to_layout(srcTensor, dstTensor.layout());
  if (utils::isOnDevice(dstTensor.storage_type())) {
    srcTensor = ::ttnn::to_device(srcTensor, dstTensor.device(),
                                  dstTensor.memory_config());
  }
  tensorPool.insertTTNNTensorAndValidate(tensorRefPtr, srcTensor);
}

void dumpTensor(::tt::runtime::Tensor tensor, const std::string &filePath) {
  ::ttnn::Tensor ttnnTensor = utils::getTTNNTensorFromRuntimeTensor(tensor);
  ::tt::tt_metal::dump_tensor_flatbuffer(filePath, ttnnTensor);
}

::tt::runtime::Tensor loadTensor(const std::string &filePath,
                                 std::optional<Device> device) {

  ::ttnn::MeshDevice *devicePtr = nullptr;
  if (device.has_value()) {
    devicePtr = &device->as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  }

  ::ttnn::Tensor metalTensor =
      ::tt::tt_metal::load_tensor_flatbuffer(filePath, devicePtr);

  auto tensor = utils::createRuntimeTensorFromTTNN(metalTensor);

  return tensor;
}
} // namespace tt::runtime::ttnn
