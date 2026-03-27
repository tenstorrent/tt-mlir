// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "Constants.h"

#include "tt-metalium/experimental/fabric/fabric.hpp"
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
#include <Python.h>
#include <numeric>

#include <memory>
#include <optional>
#include <vector>

namespace tt::runtime::ttnn {

using ::tt::runtime::DeviceRuntime;

static ::ttnn::Tensor
createOwnedTTNNTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {
  const void *dataToUse = data;
  ::tt::target::DataType dataTypeToUse = dataType;
  std::vector<std::byte> castedData;
  if (!::tt::runtime::utils::isSupportedDataType(dataType)) {
    dataTypeToUse = ::tt::runtime::utils::getUnsupportedDataTypeAlias(dataType);

    LOG_DEBUG("User provided a tensor of data type: ",
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
      untilize && utils::canUntilizeOnDevice(inputTensor.dtype(),
                                             inputTensor.memory_config());
  // If blackhole workarounds are enabled, only untilize on device if the
  // architecture is not blackhole
  if (::tt::runtime::workaround::Env::get().blackholeWorkarounds) {
    untilizeOnDevice &= getArch() != ::tt::target::Arch::Blackhole;
  }
  if (untilizeOnDevice) {
    ::ttnn::Tensor hostTensor = ::ttnn::from_device(
        ::ttnn::to_layout(inputTensor, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                          std::nullopt),
        blocking);

    std::optional<::ttnn::MeshEvent> meshEvent = std::nullopt;
    if (!blocking) {
      meshEvent =
          ::ttnn::events::record_mesh_event(meshDevice, ::ttnn::QueueId{0});
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
    meshEvent =
        ::ttnn::events::record_mesh_event(meshDevice, ::ttnn::QueueId{0});
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
        utils::createBorrowedTTNNTensor<float>(data, ttnnShape));
  case ::tt::target::DataType::BFloat16:
    return utils::createRuntimeTensorFromTTNN(
        utils::createBorrowedTTNNTensor<bfloat16>(data, ttnnShape));
  case ::tt::target::DataType::UInt32:
    return utils::createRuntimeTensorFromTTNN(
        utils::createBorrowedTTNNTensor<uint32_t>(data, ttnnShape));
  case ::tt::target::DataType::UInt16:
    return utils::createRuntimeTensorFromTTNN(
        utils::createBorrowedTTNNTensor<uint16_t>(data, ttnnShape));
  case ::tt::target::DataType::UInt8:
    return utils::createRuntimeTensorFromTTNN(
        utils::createBorrowedTTNNTensor<uint8_t>(data, ttnnShape));
  case ::tt::target::DataType::Int32:
    return utils::createRuntimeTensorFromTTNN(
        utils::createBorrowedTTNNTensor<int32_t>(data, ttnnShape));
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
  return ::tt::runtime::ttnn::createMultiDeviceHostTensor(tensorShards,
                                                          strategy, meshShape);
}

Tensor createMultiDeviceBorrowedHostTensor(
    std::vector<void *> &data, const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape) {
  std::vector<::tt::runtime::Tensor> tensorShards;
  tensorShards.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(tensorShards),
                 [&](void *dataShard) -> ::tt::runtime::Tensor {
                   return createBorrowedHostTensor(dataShard, shape, stride,
                                                   itemsize, dataType);
                 });
  return ::tt::runtime::ttnn::createMultiDeviceHostTensor(tensorShards,
                                                          strategy, meshShape);
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
      ::tt::tt_metal::create_device_tensor(tensorSpec, &meshDevice);

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
  std::uint32_t tensorVolume = ::tt::runtime::ttnn::getTensorVolume(tensor);

  if (tensorVolume == 0) {
    LOG_WARNING("getTensorDataBuffer: Tensor has zero volume; returning an "
                "empty data vector.");
    return {};
  }

  std::vector<std::byte> dataVec(
      ::tt::runtime::ttnn::getTensorElementSize(tensor) * tensorVolume);

  // Need to `memcpy` in each case because the vector will go out of scope if we
  // wait until after the switch case
  switch (::tt::runtime::ttnn::getTensorDataType(tensor)) {
  case target::DataType::BFP_BFloat4: {
    dataVec.resize(sizeof(float) *
                   ::tt::runtime::ttnn::getTensorVolume(tensor));
    auto vec = ttnnTensor.to_vector<float>();
    dataPtr = vec.data();
    LOG_ASSERT(dataPtr != nullptr);
    std::memcpy(dataVec.data(), dataPtr, dataVec.size());
    return dataVec;
  }
  case target::DataType::BFP_BFloat8: {
    dataVec.resize(sizeof(float) *
                   ::tt::runtime::ttnn::getTensorVolume(tensor));
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

std::uint32_t getTensorLogicalVolume(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      utils::getTTNNTensorFromRuntimeTensor(tensor);
  return ttnnTensor.logical_volume();
}

TensorDesc getTensorDesc(::tt::runtime::Tensor tensor) {
  TensorDesc desc;
  desc.dataType = ::tt::runtime::ttnn::getTensorDataType(tensor);
  desc.itemsize = ::tt::runtime::ttnn::getTensorElementSize(tensor);
  desc.stride = ::tt::runtime::ttnn::getTensorStride(tensor);
  desc.shape = ::tt::runtime::ttnn::getTensorShape(tensor);
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

tt::target::Arch getArch() {
  return ::tt::runtime::common::toTargetArch(::tt::tt_metal::hal::get_arch());
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

  uint32_t numUnreleasedSubMeshes = 0;
  for (const auto &subMesh : ttnnMeshDevice.get_submeshes()) {
    if (subMesh->is_initialized()) {
      numUnreleasedSubMeshes++;
    }
  }
  if (numUnreleasedSubMeshes > 0) {
    LOG_WARNING("Calling close on parent mesh device ", ttnnMeshDevice,
                " that has ", numUnreleasedSubMeshes,
                " unreleased submeshes."
                "These submeshes will keep the parent mesh device alive. "
                "To fully close the parent mesh device, please release all of "
                "its submeshes.");
  }

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

void clearProgramCache(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return ttnnMeshDevice.clear_program_cache();
}

size_t getL1SmallSize(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return ttnnMeshDevice.allocator()
      ->get_statistics(::ttnn::BufferType::L1_SMALL)
      .total_allocatable_size_bytes;
}

size_t getTraceRegionSize(Device meshDevice) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      meshDevice.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  return ttnnMeshDevice.allocator()
      ->get_statistics(::ttnn::BufferType::TRACE)
      .total_allocatable_size_bytes;
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
  return utils::getMemoryView(deviceHandle);
}

void setFabricConfig(tt::runtime::FabricConfig config) {
  ::tt::tt_fabric::SetFabricConfig(common::toMetalFabricConfig(config));
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

uint32_t getNumShards(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      utils::getTTNNTensorFromRuntimeTensor(tensor);
  return ::ttnn::distributed::get_device_tensors(ttnnTensor).size();
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

std::vector<::tt::runtime::Tensor>
getDeviceTensors(::tt::runtime::Tensor tensor) {
  const ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      tensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);

  std::vector<::ttnn::Tensor> ttnnTensors =
      ::ttnn::distributed::get_device_tensors(tensorWrapper.getTensor());

  std::vector<Tensor> runtime_tensors;
  runtime_tensors.reserve(ttnnTensors.size());

  for (const ::ttnn::Tensor &ttnnTensor : ttnnTensors) {
    runtime_tensors.emplace_back(utils::createRuntimeTensorFromTTNN(
        ttnnTensor, tensorWrapper.getMeshEvent(),
        tensorWrapper.shouldRetain()));
  }

  return runtime_tensors;
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

bool hasLayout(::tt::runtime::Tensor tensor, Layout layout) {
  const std::shared_ptr<LayoutDesc> tensorLayoutDesc =
      LayoutDesc::fromTensor(tensor);

  const LayoutDesc &desiredLayoutDesc =
      layout.as<LayoutDesc>(DeviceRuntime::TTNN);

  return *tensorLayoutDesc == desiredLayoutDesc;
}

// Return the layout of a given runtime Tensor as a Layout handle
Layout getTensorLayout(::tt::runtime::Tensor tensor) {
  const std::shared_ptr<LayoutDesc> tensorLayoutDesc =
      LayoutDesc::fromTensor(tensor);
  return Layout(std::static_pointer_cast<void>(tensorLayoutDesc),
                DeviceRuntime::TTNN);
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
        dstDataType.value() == ::tt::runtime::ttnn::getTensorDataType(src) ||
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

    ::tt::target::DataType srcDataType =
        ::tt::runtime::ttnn::getTensorDataType(src);
    ::tt::target::DataType unsupportedDataTypeAlias =
        tt::runtime::utils::getUnsupportedDataTypeAlias(dstDataType.value());

    LOG_ASSERT(
        srcDataType == unsupportedDataTypeAlias,
        "Tensor data type must be the alias of the unsupported data type: " +
            std::string(target::EnumNameDataType(unsupportedDataTypeAlias)));

    LOG_DEBUG(
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
    ::tt::tt_metal::copy_to_host(srcTensor.device()->mesh_command_queue(),
                                 srcTensor, reinterpret_cast<std::byte *>(dst));
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
  } else if (utils::isOnHost(srcTensor.storage_type())) {
    ::tt::tt_metal::copy_to_device(srcTensor, dstTensor);
  } else {
    void *dstPtr = utils::getRawHostDataPtr(dstTensor);
    ::tt::tt_metal::copy_to_host(srcTensor.device()->mesh_command_queue(),
                                 srcTensor,
                                 reinterpret_cast<std::byte *>(dstPtr));
  }
}

void deallocateTensor(::tt::runtime::Tensor &tensor, bool force) {
  // If the tensor is retained, do not deallocate
  if (::tt::runtime::ttnn::getTensorRetain(tensor)) {
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
      ::tt::runtime::ttnn::getOpOutputRef(opContextHandle,
                                          programContextHandle);
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
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeScalarOp: {
    tensorRef = opContext.type_as_EltwiseBinaryCompositeScalarOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ExperimentalEltwiseBinaryBackwardOp: {
    tensorRef = opContext.type_as_ExperimentalEltwiseBinaryBackwardOp()->out();
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
  case ::tt::target::ttnn::OpType::SparseMatmulOp: {
    tensorRef = opContext.type_as_SparseMatmulOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::CumSumOp: {
    tensorRef = opContext.type_as_CumSumOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::RandOp: {
    tensorRef = opContext.type_as_RandOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::DropoutOp: {
    tensorRef = opContext.type_as_DropoutOp()->out();
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
  case ::tt::target::ttnn::OpType::AssignOp: {
    tensorRef = opContext.type_as_AssignOp()->output();
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    tensorRef = opContext.type_as_ConcatOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ScatterOp: {
    tensorRef = opContext.type_as_ScatterOp()->out();
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
  case ::tt::target::ttnn::OpType::Conv3dOp: {
    tensorRef = opContext.type_as_Conv3dOp()->out();
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
  case ::tt::target::ttnn::OpType::GlobalAvgPool2dOp: {
    tensorRef = opContext.type_as_GlobalAvgPool2dOp()->out();
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
  case ::tt::target::ttnn::OpType::PrepareConvTranspose2dWeightsOp: {
    tensorRef = opContext.type_as_PrepareConvTranspose2dWeightsOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::PrepareConvTranspose2dBiasOp: {
    tensorRef = opContext.type_as_PrepareConvTranspose2dBiasOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::BatchNormInferenceOp: {
    tensorRef = opContext.type_as_BatchNormInferenceOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::RMSNormOp: {
    tensorRef = opContext.type_as_RMSNormOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::DistributedRMSNormOp: {
    tensorRef = opContext.type_as_DistributedRMSNormOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::LayerNormOp: {
    tensorRef = opContext.type_as_LayerNormOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::GroupNormOp: {
    tensorRef = opContext.type_as_GroupNormOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    tensorRef = opContext.type_as_AllGatherOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::AllReduceOp: {
    tensorRef = opContext.type_as_AllReduceOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    tensorRef = opContext.type_as_ReduceScatterOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    tensorRef = opContext.type_as_MeshShardOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::MeshPartitionOp: {
    tensorRef = opContext.type_as_MeshPartitionOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::AllToAllCombineOp: {
    tensorRef = opContext.type_as_AllToAllCombineOp()->out();
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
  case ::tt::target::ttnn::OpType::ConstantOp: {
    tensorRef = opContext.type_as_ConstantOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    tensorRef = opContext.type_as_FillCacheOp()->cache();
    break;
  }
  case ::tt::target::ttnn::OpType::PagedFillCacheOp: {
    tensorRef = opContext.type_as_PagedFillCacheOp()->cache();
    break;
  }
  case ::tt::target::ttnn::OpType::UpdateCacheOp: {
    tensorRef = opContext.type_as_UpdateCacheOp()->cache();
    break;
  }
  case ::tt::target::ttnn::OpType::PagedUpdateCacheOp: {
    tensorRef = opContext.type_as_PagedUpdateCacheOp()->cache();
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
  case ::tt::target::ttnn::OpType::RotaryEmbeddingOp: {
    tensorRef = opContext.type_as_RotaryEmbeddingOp()->out();
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
  case ::tt::target::ttnn::OpType::ScaledDotProductAttentionDecodeOp: {
    tensorRef = opContext.type_as_ScaledDotProductAttentionDecodeOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::PagedScaledDotProductAttentionDecodeOp: {
    tensorRef =
        opContext.type_as_PagedScaledDotProductAttentionDecodeOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::ScaledDotProductAttentionOp: {
    tensorRef = opContext.type_as_ScaledDotProductAttentionOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::NLPConcatHeadsDecodeOp: {
    tensorRef = opContext.type_as_NLPConcatHeadsDecodeOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::CpuOp:
  case ::tt::target::ttnn::OpType::BatchNormTrainingOp:
  case ::tt::target::ttnn::OpType::MaxPool2dWithIndicesOp:
  case ::tt::target::ttnn::OpType::SortOp:
  case ::tt::target::ttnn::OpType::LoadCachedOp:
  case ::tt::target::ttnn::OpType::GetDeviceOp:
  case ::tt::target::ttnn::OpType::DeallocateOp:
  case ::tt::target::ttnn::OpType::FuncCallOp:
  case ::tt::target::ttnn::OpType::WriteTensorOp:
  case ::tt::target::ttnn::OpType::EndTraceCaptureOp:
  case ::tt::target::ttnn::OpType::ExecuteTraceOp:
  case ::tt::target::ttnn::OpType::CaptureOrExecuteTraceOp:
  case ::tt::target::ttnn::OpType::NLPCreateQKVHeadsDecodeOp:
  case ::tt::target::ttnn::OpType::SplitQueryKeyValueAndSplitHeadsOp:
  case ::tt::target::ttnn::OpType::AllToAllDispatchOp:
  case ::tt::target::ttnn::OpType::MoeExpertTokenRemapOp:
  case ::tt::target::ttnn::OpType::DumpTensorOp:
  case ::tt::target::ttnn::OpType::TopKOp:
  case ::tt::target::ttnn::OpType::BreakpointOp:
  case ::tt::target::ttnn::OpType::PrintOp:
  case ::tt::target::ttnn::OpType::MemorySnapshotOp: {
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
  case ::tt::target::ttnn::OpType::AggregateTensorOp: {
    tensorRef = opContext.type_as_AggregateTensorOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::DistributeTensorOp: {
    tensorRef = opContext.type_as_DistributeTensorOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::AnnotateOp: {
    tensorRef = opContext.type_as_AnnotateOp()->result();
    break;
  }
  case ::tt::target::ttnn::OpType::RegionStartOp: {
    tensorRef = opContext.type_as_RegionStartOp()->result();
    break;
  }
  case ::tt::target::ttnn::OpType::RegionEndOp: {
    tensorRef = opContext.type_as_RegionEndOp()->result();
    break;
  }
  case ::tt::target::ttnn::OpType::CreateGlobalSemaphoreOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::ResetGlobalSemaphoreOp: {
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
  case ::tt::target::ttnn::OpType::DropoutOp: {
    tensorRefs = {opContext.type_as_DropoutOp()->in()};
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
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeScalarOp: {
    tensorRefs = {opContext.type_as_EltwiseBinaryCompositeScalarOp()->lhs()};
    break;
  }
  case ::tt::target::ttnn::OpType::ExperimentalEltwiseBinaryBackwardOp: {
    tensorRefs = {
        opContext.type_as_ExperimentalEltwiseBinaryBackwardOp()->grad(),
        opContext.type_as_ExperimentalEltwiseBinaryBackwardOp()->input()};
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
  case ::tt::target::ttnn::OpType::SparseMatmulOp: {
    tensorRefs = {opContext.type_as_SparseMatmulOp()->a(),
                  opContext.type_as_SparseMatmulOp()->b(),
                  opContext.type_as_SparseMatmulOp()->sparsity()};
    break;
  }
  case ::tt::target::ttnn::OpType::CumSumOp: {
    tensorRefs = {opContext.type_as_CumSumOp()->in()};
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
  case ::tt::target::ttnn::OpType::TopKOp: {
    tensorRefs = {opContext.type_as_TopKOp()->input_tensor()};
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
  case ::tt::target::ttnn::OpType::AssignOp: {
    tensorRefs = {opContext.type_as_AssignOp()->input()};
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    tensorRefs = utils::convertFbTensorRefsToVector(
        opContext.type_as_ConcatOp()->inputs());
    break;
  }
  case ::tt::target::ttnn::OpType::ScatterOp: {
    tensorRefs = {opContext.type_as_ScatterOp()->input(),
                  opContext.type_as_ScatterOp()->index(),
                  opContext.type_as_ScatterOp()->source()};
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
  case ::tt::target::ttnn::OpType::Conv3dOp: {
    tensorRefs = {opContext.type_as_Conv3dOp()->input()};
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
  case ::tt::target::ttnn::OpType::GlobalAvgPool2dOp: {
    tensorRefs = {opContext.type_as_GlobalAvgPool2dOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dWithIndicesOp: {
    tensorRefs = {opContext.type_as_MaxPool2dWithIndicesOp()->in()};
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
  case ::tt::target::ttnn::OpType::PrepareConvTranspose2dWeightsOp: {
    tensorRefs = {
        opContext.type_as_PrepareConvTranspose2dWeightsOp()->weight_tensor()};
    break;
  }
  case ::tt::target::ttnn::OpType::PrepareConvTranspose2dBiasOp: {
    tensorRefs = {
        opContext.type_as_PrepareConvTranspose2dBiasOp()->bias_tensor()};
    break;
  }
  case ::tt::target::ttnn::OpType::BatchNormInferenceOp: {
    tensorRefs = {opContext.type_as_BatchNormInferenceOp()->input(),
                  opContext.type_as_BatchNormInferenceOp()->running_mean(),
                  opContext.type_as_BatchNormInferenceOp()->running_var(),
                  opContext.type_as_BatchNormInferenceOp()->weight(),
                  opContext.type_as_BatchNormInferenceOp()->bias()};
    break;
  }
  case ::tt::target::ttnn::OpType::BatchNormTrainingOp: {
    tensorRefs = {opContext.type_as_BatchNormTrainingOp()->input(),
                  opContext.type_as_BatchNormTrainingOp()->running_mean(),
                  opContext.type_as_BatchNormTrainingOp()->running_var(),
                  opContext.type_as_BatchNormTrainingOp()->weight(),
                  opContext.type_as_BatchNormTrainingOp()->bias()};
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
  case ::tt::target::ttnn::OpType::DistributedRMSNormOp: {
    tensorRefs = {opContext.type_as_DistributedRMSNormOp()->input()};
    if (opContext.type_as_DistributedRMSNormOp()->weight()) {
      tensorRefs.push_back(opContext.type_as_DistributedRMSNormOp()->weight());
    }
    if (opContext.type_as_DistributedRMSNormOp()->residual()) {
      tensorRefs.push_back(
          opContext.type_as_DistributedRMSNormOp()->residual());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::LayerNormOp: {
    tensorRefs = {opContext.type_as_LayerNormOp()->input()};
    if (opContext.type_as_LayerNormOp()->weight()) {
      tensorRefs.push_back(opContext.type_as_LayerNormOp()->weight());
    }
    if (opContext.type_as_LayerNormOp()->bias()) {
      tensorRefs.push_back(opContext.type_as_LayerNormOp()->bias());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::GroupNormOp: {
    tensorRefs = {opContext.type_as_GroupNormOp()->input()};
    if (opContext.type_as_GroupNormOp()->input_mask()) {
      tensorRefs.push_back(opContext.type_as_GroupNormOp()->input_mask());
    }
    if (opContext.type_as_GroupNormOp()->weight()) {
      tensorRefs.push_back(opContext.type_as_GroupNormOp()->weight());
    }
    if (opContext.type_as_GroupNormOp()->bias()) {
      tensorRefs.push_back(opContext.type_as_GroupNormOp()->bias());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    tensorRefs = {opContext.type_as_AllGatherOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::AllReduceOp: {
    tensorRefs = {opContext.type_as_AllReduceOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    tensorRefs = {opContext.type_as_ReduceScatterOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    tensorRefs = {opContext.type_as_MeshShardOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::MeshPartitionOp: {
    tensorRefs = {opContext.type_as_MeshPartitionOp()->out()};
    break;
  }
  case ::tt::target::ttnn::OpType::AllToAllDispatchOp: {
    tensorRefs = {opContext.type_as_AllToAllDispatchOp()->input_tensor(),
                  opContext.type_as_AllToAllDispatchOp()->expert_indices(),
                  opContext.type_as_AllToAllDispatchOp()->expert_mapping()};
    break;
  }
  case ::tt::target::ttnn::OpType::AllToAllCombineOp: {
    tensorRefs = {opContext.type_as_AllToAllCombineOp()->input_tensor(),
                  opContext.type_as_AllToAllCombineOp()->expert_metadata(),
                  opContext.type_as_AllToAllCombineOp()->expert_mapping()};
    break;
  }
  case ::tt::target::ttnn::OpType::MoeExpertTokenRemapOp: {
    tensorRefs = {opContext.type_as_MoeExpertTokenRemapOp()->topk_tensor(),
                  opContext.type_as_MoeExpertTokenRemapOp()->expert_mapping(),
                  opContext.type_as_MoeExpertTokenRemapOp()->expert_metadata()};
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
  case ::tt::target::ttnn::OpType::PagedUpdateCacheOp: {
    tensorRefs = {opContext.type_as_PagedUpdateCacheOp()->cache(),
                  opContext.type_as_PagedUpdateCacheOp()->input(),
                  opContext.type_as_PagedUpdateCacheOp()->update_index(),
                  opContext.type_as_PagedUpdateCacheOp()->page_table()};
    break;
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    tensorRefs = {opContext.type_as_FillCacheOp()->cache(),
                  opContext.type_as_FillCacheOp()->input()};
    break;
  }
  case ::tt::target::ttnn::OpType::PagedFillCacheOp: {
    tensorRefs = {opContext.type_as_PagedFillCacheOp()->cache(),
                  opContext.type_as_PagedFillCacheOp()->input(),
                  opContext.type_as_PagedFillCacheOp()->page_table(),
                  opContext.type_as_PagedFillCacheOp()->batch_idx_tensor()};
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
  case ::tt::target::ttnn::OpType::SplitQueryKeyValueAndSplitHeadsOp: {
    tensorRefs = {
        opContext.type_as_SplitQueryKeyValueAndSplitHeadsOp()->in(),
        opContext.type_as_SplitQueryKeyValueAndSplitHeadsOp()->kv_input()};
    break;
  }
  case ::tt::target::ttnn::OpType::GenericOp: {
    for (const auto *input : *opContext.type_as_GenericOp()->io_tensors()) {
      tensorRefs.push_back(input);
    }
    break;
  }
  case ::tt::target::ttnn::OpType::ScaledDotProductAttentionDecodeOp: {
    tensorRefs = {
        opContext.type_as_ScaledDotProductAttentionDecodeOp()->query(),
        opContext.type_as_ScaledDotProductAttentionDecodeOp()->key(),
        opContext.type_as_ScaledDotProductAttentionDecodeOp()->value(),
        opContext.type_as_ScaledDotProductAttentionDecodeOp()->attention_mask(),
        opContext.type_as_ScaledDotProductAttentionDecodeOp()->cur_pos_tensor(),
        opContext.type_as_ScaledDotProductAttentionDecodeOp()
            ->attention_sink()};
    break;
  }
  case ::tt::target::ttnn::OpType::ScaledDotProductAttentionOp: {
    tensorRefs = {
        opContext.type_as_ScaledDotProductAttentionOp()->query(),
        opContext.type_as_ScaledDotProductAttentionOp()->key(),
        opContext.type_as_ScaledDotProductAttentionOp()->value(),
        opContext.type_as_ScaledDotProductAttentionOp()->attention_mask(),
        opContext.type_as_ScaledDotProductAttentionOp()->attention_sink()};
    break;
  }
  case ::tt::target::ttnn::OpType::PagedScaledDotProductAttentionDecodeOp: {
    tensorRefs = {
        opContext.type_as_PagedScaledDotProductAttentionDecodeOp()->query(),
        opContext.type_as_PagedScaledDotProductAttentionDecodeOp()->key(),
        opContext.type_as_PagedScaledDotProductAttentionDecodeOp()->value(),
        opContext.type_as_PagedScaledDotProductAttentionDecodeOp()
            ->page_table(),
        opContext.type_as_PagedScaledDotProductAttentionDecodeOp()
            ->attention_mask(),
        opContext.type_as_PagedScaledDotProductAttentionDecodeOp()
            ->cur_pos_tensor(),
        opContext.type_as_PagedScaledDotProductAttentionDecodeOp()
            ->attention_sink()};
    break;
  }
  case ::tt::target::ttnn::OpType::RotaryEmbeddingLlamaOp: {
    tensorRefs = {opContext.type_as_RotaryEmbeddingLlamaOp()->input(),
                  opContext.type_as_RotaryEmbeddingLlamaOp()->cos_cache(),
                  opContext.type_as_RotaryEmbeddingLlamaOp()->sin_cache(),
                  opContext.type_as_RotaryEmbeddingLlamaOp()->trans_mat()};
    break;
  }
  case ::tt::target::ttnn::OpType::RotaryEmbeddingOp: {
    tensorRefs = {opContext.type_as_RotaryEmbeddingOp()->input(),
                  opContext.type_as_RotaryEmbeddingOp()->cos_cache(),
                  opContext.type_as_RotaryEmbeddingOp()->sin_cache()};
    break;
  }
  case ::tt::target::ttnn::OpType::NLPCreateQKVHeadsDecodeOp: {
    tensorRefs = {
        opContext.type_as_NLPCreateQKVHeadsDecodeOp()->input(),
        opContext.type_as_NLPCreateQKVHeadsDecodeOp()->batch_offset()};
    break;
  }
  case ::tt::target::ttnn::OpType::DumpTensorOp: {
    tensorRefs = {opContext.type_as_DumpTensorOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::LoadTensorOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::AggregateTensorOp: {
    tensorRefs = {opContext.type_as_AggregateTensorOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::DistributeTensorOp: {
    tensorRefs = {opContext.type_as_DistributeTensorOp()->in()};
    break;
  }
  case ::tt::target::ttnn::OpType::AnnotateOp: {
    tensorRefs = {opContext.type_as_AnnotateOp()->operand()};
    break;
  }
  case ::tt::target::ttnn::OpType::RegionStartOp: {
    tensorRefs = {opContext.type_as_RegionStartOp()->operand()};
    break;
  }
  case ::tt::target::ttnn::OpType::RegionEndOp: {
    tensorRefs = {opContext.type_as_RegionEndOp()->operand()};
    break;
  }
  case ::tt::target::ttnn::OpType::BreakpointOp: {
    tensorRefs = {opContext.type_as_BreakpointOp()->operand()};
    break;
  }
  case ::tt::target::ttnn::OpType::PrintOp: {
    tensorRefs = {opContext.type_as_PrintOp()->operand()};
    break;
  }
  case ::tt::target::ttnn::OpType::MemorySnapshotOp: {
    tensorRefs = {opContext.type_as_MemorySnapshotOp()->operand()};
    break;
  }
  case ::tt::target::ttnn::OpType::CreateGlobalSemaphoreOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::ResetGlobalSemaphoreOp: {
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

std::unordered_map<std::string, tt::runtime::OpAttrValue>
getOpAttrs(OpContext opContextHandle, CallbackContext programContextHandle) {

  const auto &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);

  std::unordered_map<std::string, tt::runtime::OpAttrValue> attrs;

  switch (opContext.type_type()) {
  case ::tt::target::ttnn::OpType::ArangeOp: {
    const auto *op = opContext.type_as_ArangeOp();
    attrs["start"] = op->start();
    attrs["end"] = op->end();
    attrs["step"] = op->step();
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    const auto *op = opContext.type_as_EmptyOp();
    if (op->shape()) {
      attrs["shape"] =
          std::vector<int64_t>(op->shape()->begin(), op->shape()->end());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    const auto *op = opContext.type_as_FullOp();
    if (op->shape()) {
      attrs["shape"] =
          std::vector<int64_t>(op->shape()->begin(), op->shape()->end());
    }
    if (op->fill_value()) {
      if (op->fill_value_type() == ::tt::target::ttnn::NumberType::FP) {
        attrs["fill_value"] = op->fill_value_as_FP()->value();
      } else if (op->fill_value_type() == ::tt::target::ttnn::NumberType::I32) {
        attrs["fill_value"] = op->fill_value_as_I32()->value();
      }
    }
    break;
  }
  case ::tt::target::ttnn::OpType::ConstantOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::RandOp: {
    const auto *op = opContext.type_as_RandOp();
    if (op->size()) {
      attrs["shape"] =
          std::vector<int64_t>(op->size()->begin(), op->size()->end());
    }
    attrs["low"] = op->low();
    attrs["high"] = op->high();
    attrs["seed"] = static_cast<int32_t>(op->seed());
    break;
  }
  case ::tt::target::ttnn::OpType::DropoutOp: {
    const auto *op = opContext.type_as_DropoutOp();
    attrs["probability"] = op->prob();
    attrs["seed"] = static_cast<int32_t>(op->seed());
    break;
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseBinaryCompositeScalarOp: {
    const auto *op = opContext.type_as_EltwiseBinaryCompositeScalarOp();
    if (op->rhs()) {
      if (op->rhs_type() == ::tt::target::ttnn::NumberType::FP) {
        attrs["rhs"] = op->rhs_as_FP()->value();
      } else if (op->rhs_type() == ::tt::target::ttnn::NumberType::I32) {
        attrs["rhs"] = op->rhs_as_I32()->value();
      }
    }
    break;
  }
  case ::tt::target::ttnn::OpType::ExperimentalEltwiseBinaryBackwardOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseTernaryWhereOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseQuantizationOp: {
    const auto *op = opContext.type_as_EltwiseQuantizationOp();
    if (op->axis().has_value()) {
      attrs["axis"] = op->axis().value();
    }
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryOp: {
    const auto *op = opContext.type_as_EltwiseUnaryOp();
    if (op->params() && op->params_type() ==
                            ::tt::target::ttnn::EltwiseUnaryOpParams::
                                EltwiseOpWithFloatParams) {
      attrs["parameter"] =
          op->params_as_EltwiseOpWithFloatParams()->parameter();
    }
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseUnaryCompositeOp: {
    const auto *op = opContext.type_as_EltwiseUnaryCompositeOp();
    if (op->params()) {
      if (op->params_type() ==
          ::tt::target::ttnn::EltwiseUnaryCompositeOpParams::
              ClampScalarOpParams) {
        const auto *params = op->params_as_ClampScalarOpParams();
        if (params->min()) {
          if (params->min_type() == ::tt::target::ttnn::NumberType::FP) {
            attrs["min"] = params->min_as_FP()->value();
          } else if (params->min_type() ==
                     ::tt::target::ttnn::NumberType::I32) {
            attrs["min"] = params->min_as_I32()->value();
          }
        }
        if (params->max()) {
          if (params->max_type() == ::tt::target::ttnn::NumberType::FP) {
            attrs["max"] = params->max_as_FP()->value();
          } else if (params->max_type() ==
                     ::tt::target::ttnn::NumberType::I32) {
            attrs["max"] = params->max_as_I32()->value();
          }
        }
      }
    }
    break;
  }
  case ::tt::target::ttnn::OpType::LinearOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    const auto *op = opContext.type_as_MatmulOp();
    attrs["transpose_a"] = op->transpose_a();
    attrs["transpose_b"] = op->transpose_b();
    if (op->activation() && op->activation()->size() > 0) {
      attrs["activation"] = std::string(op->activation()->c_str());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::SparseMatmulOp: {
    const auto *op = opContext.type_as_SparseMatmulOp();
    attrs["is_input_a_sparse"] = op->is_input_a_sparse();
    attrs["is_input_b_sparse"] = op->is_input_b_sparse();
    attrs["nnz"] = static_cast<int64_t>(op->nnz());
    break;
  }
  case ::tt::target::ttnn::OpType::CumSumOp: {
    const auto *op = opContext.type_as_CumSumOp();
    attrs["dim"] = op->dim();
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionArgMaxOp: {
    const auto *op = opContext.type_as_ReductionArgMaxOp();
    if (op->dim().has_value()) {
      attrs["dim"] = op->dim().value();
    }
    attrs["keep_dim"] = op->keep_dim();
    attrs["use_multicore"] = op->use_multicore();
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionProdOp: {
    const auto *op = opContext.type_as_ReductionProdOp();
    if (op->dim_arg().has_value()) {
      attrs["dim_arg"] = op->dim_arg().value();
    }
    attrs["keep_dim"] = op->keep_dim();
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    const auto *op = opContext.type_as_ReductionOp();
    if (op->dim_arg()) {
      attrs["dim_arg"] =
          std::vector<int32_t>(op->dim_arg()->begin(), op->dim_arg()->end());
    }
    attrs["keep_dim"] = op->keep_dim();
    break;
  }
  case ::tt::target::ttnn::OpType::TopKOp: {
    const auto *op = opContext.type_as_TopKOp();
    attrs["k"] = op->k();
    attrs["dim"] = op->dim();
    attrs["largest"] = op->largest();
    attrs["sorted"] = op->sorted();
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingBackwardOp: {
    // EmbeddingBackwardOp doesn't have num_weights field in current schema
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    const auto *op = opContext.type_as_SoftmaxOp();
    attrs["dimension"] = op->dimension();
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    const auto *op = opContext.type_as_TransposeOp();
    attrs["dim0"] = op->dim0();
    attrs["dim1"] = op->dim1();
    break;
  }
  case ::tt::target::ttnn::OpType::PadOp: {
    const auto *op = opContext.type_as_PadOp();
    if (op->padding()) {
      attrs["padding"] =
          std::vector<uint32_t>(op->padding()->begin(), op->padding()->end());
    }
    attrs["value"] = op->value();
    attrs["use_multicore"] = op->use_multicore();
    break;
  }
  case ::tt::target::ttnn::OpType::AssignOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    const auto *op = opContext.type_as_ConcatOp();
    attrs["dim"] = op->dim();
    break;
  }
  case ::tt::target::ttnn::OpType::ScatterOp: {
    const auto *op = opContext.type_as_ScatterOp();
    attrs["dim"] = op->dim();
    attrs["scatter_reduce_type"] =
        static_cast<uint32_t>(op->scatter_reduce_type());
    break;
  }
  case ::tt::target::ttnn::OpType::PermuteOp: {
    const auto *op = opContext.type_as_PermuteOp();
    if (op->permutation()) {
      attrs["permutation"] = std::vector<int64_t>(op->permutation()->begin(),
                                                  op->permutation()->end());
    }
    attrs["pad_value"] = op->pad_value();
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    const auto *op = opContext.type_as_ReshapeOp();
    if (op->shape()) {
      attrs["shape"] =
          std::vector<int32_t>(op->shape()->begin(), op->shape()->end());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    const auto *op = opContext.type_as_SliceOp();
    if (op->params() &&
        op->params_type() ==
            ::tt::target::ttnn::SliceOpParams::SliceStaticOpParams) {
      const auto *params = op->params_as_SliceStaticOpParams();
      if (params->begins()) {
        attrs["begins"] = std::vector<int64_t>(params->begins()->begin(),
                                               params->begins()->end());
      }
      if (params->ends()) {
        attrs["ends"] = std::vector<int64_t>(params->ends()->begin(),
                                             params->ends()->end());
      }
    }
    if (op->step()) {
      attrs["step"] =
          std::vector<int64_t>(op->step()->begin(), op->step()->end());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatOp: {
    const auto *op = opContext.type_as_RepeatOp();
    for (size_t i = 0; i < op->repeat_dims()->size(); ++i) {
      std::cout << "Type of x1: " << typeid(op->repeat_dims()->Get(i)).name()
                << std::endl;
      std::cout
          << "Type of x2: "
          << typeid(static_cast<int64_t>(op->repeat_dims()->Get(i))).name()
          << std::endl;
    }
    if (op->repeat_dims()) {
      std::vector<int64_t> repeat_dims_vec;
      repeat_dims_vec.reserve(op->repeat_dims()->size());
      for (size_t i = 0; i < op->repeat_dims()->size(); ++i) {
        repeat_dims_vec.push_back(
            static_cast<int64_t>(op->repeat_dims()->Get(i)));
      }
      attrs["repeat_dims"] = tt::runtime::OpAttrValue(repeat_dims_vec);
    }
    break;
  }
  case ::tt::target::ttnn::OpType::RepeatInterleaveOp: {
    const auto *op = opContext.type_as_RepeatInterleaveOp();
    attrs["repeats"] = op->repeats();
    attrs["dim"] = op->dim();
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    const auto *op = opContext.type_as_Conv2dOp();
    attrs["in_channels"] = static_cast<uint32_t>(op->in_channels());
    attrs["out_channels"] = static_cast<uint32_t>(op->out_channels());
    attrs["batch_size"] = static_cast<uint32_t>(op->batch_size());
    attrs["input_height"] = static_cast<uint32_t>(op->input_height());
    attrs["input_width"] = static_cast<uint32_t>(op->input_width());
    if (op->kernel_size()) {
      attrs["kernel_size"] = std::vector<int32_t>(op->kernel_size()->begin(),
                                                  op->kernel_size()->end());
    }
    if (op->stride()) {
      attrs["stride"] =
          std::vector<int32_t>(op->stride()->begin(), op->stride()->end());
    }
    if (op->padding()) {
      attrs["padding"] =
          std::vector<int32_t>(op->padding()->begin(), op->padding()->end());
    }
    if (op->dilation()) {
      attrs["dilation"] =
          std::vector<int32_t>(op->dilation()->begin(), op->dilation()->end());
    }
    attrs["groups"] = static_cast<uint32_t>(op->groups());
    break;
  }
  case ::tt::target::ttnn::OpType::Conv3dOp: {
    const auto *op = opContext.type_as_Conv3dOp();
    attrs["in_channels"] = static_cast<uint32_t>(op->in_channels());
    attrs["out_channels"] = static_cast<uint32_t>(op->out_channels());
    attrs["batch_size"] = static_cast<uint32_t>(op->batch_size());
    attrs["input_depth"] = static_cast<uint32_t>(op->input_depth());
    attrs["input_height"] = static_cast<uint32_t>(op->input_height());
    attrs["input_width"] = static_cast<uint32_t>(op->input_width());
    if (op->kernel_size()) {
      attrs["kernel_size"] = std::vector<int32_t>(op->kernel_size()->begin(),
                                                  op->kernel_size()->end());
    }
    if (op->stride()) {
      attrs["stride"] =
          std::vector<int32_t>(op->stride()->begin(), op->stride()->end());
    }
    if (op->padding()) {
      attrs["padding"] =
          std::vector<int32_t>(op->padding()->begin(), op->padding()->end());
    }
    if (op->padding_mode()) {
      attrs["padding_mode"] = std::string(op->padding_mode()->c_str());
    }
    attrs["groups"] = static_cast<uint32_t>(op->groups());
    break;
  }
  case ::tt::target::ttnn::OpType::ConvTranspose2dOp: {
    const auto *op = opContext.type_as_ConvTranspose2dOp();
    attrs["in_channels"] = static_cast<uint32_t>(op->in_channels());
    attrs["out_channels"] = static_cast<uint32_t>(op->out_channels());
    attrs["batch_size"] = static_cast<uint32_t>(op->batch_size());
    attrs["input_height"] = static_cast<uint32_t>(op->input_height());
    attrs["input_width"] = static_cast<uint32_t>(op->input_width());
    if (op->kernel_size()) {
      attrs["kernel_size"] = std::vector<int32_t>(op->kernel_size()->begin(),
                                                  op->kernel_size()->end());
    }
    if (op->stride()) {
      attrs["stride"] =
          std::vector<int32_t>(op->stride()->begin(), op->stride()->end());
    }
    if (op->padding()) {
      attrs["padding"] =
          std::vector<int32_t>(op->padding()->begin(), op->padding()->end());
    }
    if (op->output_padding()) {
      attrs["output_padding"] = std::vector<int32_t>(
          op->output_padding()->begin(), op->output_padding()->end());
    }
    if (op->dilation()) {
      attrs["dilation"] =
          std::vector<int32_t>(op->dilation()->begin(), op->dilation()->end());
    }
    attrs["groups"] = static_cast<uint32_t>(op->groups());
    break;
  }
  case ::tt::target::ttnn::OpType::Pool2dOp: {
    const auto *op = opContext.type_as_Pool2dOp();
    attrs["batch_size"] = static_cast<uint32_t>(op->batch_size());
    attrs["input_height"] = static_cast<uint32_t>(op->input_height());
    attrs["input_width"] = static_cast<uint32_t>(op->input_width());
    attrs["channels"] = static_cast<uint32_t>(op->channels());
    if (op->kernel_size()) {
      attrs["kernel_size"] = std::vector<int32_t>(op->kernel_size()->begin(),
                                                  op->kernel_size()->end());
    }
    if (op->stride()) {
      attrs["stride"] =
          std::vector<int32_t>(op->stride()->begin(), op->stride()->end());
    }
    if (op->padding()) {
      attrs["padding"] =
          std::vector<int32_t>(op->padding()->begin(), op->padding()->end());
    }
    if (op->dilation()) {
      attrs["dilation"] =
          std::vector<int32_t>(op->dilation()->begin(), op->dilation()->end());
    }
    attrs["ceil_mode"] = op->ceil_mode();
    attrs["reallocate_halo_output"] = op->reallocate_halo_output();
    attrs["config_tensors_in_dram"] = op->config_tensors_in_dram();
    break;
  }
  case ::tt::target::ttnn::OpType::GlobalAvgPool2dOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dWithIndicesOp: {
    const auto *op = opContext.type_as_MaxPool2dWithIndicesOp();
    attrs["batch_size"] = static_cast<uint32_t>(op->batch_size());
    attrs["input_height"] = static_cast<uint32_t>(op->input_height());
    attrs["input_width"] = static_cast<uint32_t>(op->input_width());
    attrs["channels"] = static_cast<uint32_t>(op->channels());
    if (op->kernel_size()) {
      attrs["kernel_size"] = std::vector<int32_t>(op->kernel_size()->begin(),
                                                  op->kernel_size()->end());
    }
    if (op->stride()) {
      attrs["stride"] =
          std::vector<int32_t>(op->stride()->begin(), op->stride()->end());
    }
    if (op->padding()) {
      attrs["padding"] =
          std::vector<int32_t>(op->padding()->begin(), op->padding()->end());
    }
    if (op->dilation()) {
      attrs["dilation"] =
          std::vector<int32_t>(op->dilation()->begin(), op->dilation()->end());
    }
    attrs["ceil_mode"] = op->ceil_mode();
    attrs["reallocate_halo_output"] = op->reallocate_halo_output();
    attrs["config_tensors_in_dram"] = op->config_tensors_in_dram();
    break;
  }
  case ::tt::target::ttnn::OpType::PrepareConv2dWeightsOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::PrepareConv2dBiasOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::PrepareConvTranspose2dWeightsOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::PrepareConvTranspose2dBiasOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::BatchNormInferenceOp: {
    const auto *op = opContext.type_as_BatchNormInferenceOp();
    attrs["epsilon"] = op->epsilon();
    break;
  }
  case ::tt::target::ttnn::OpType::BatchNormTrainingOp: {
    const auto *op = opContext.type_as_BatchNormTrainingOp();
    attrs["epsilon"] = op->epsilon();
    attrs["momentum"] = op->momentum();
    break;
  }
  case ::tt::target::ttnn::OpType::RMSNormOp: {
    const auto *op = opContext.type_as_RMSNormOp();
    attrs["epsilon"] = op->epsilon();
    break;
  }
  case ::tt::target::ttnn::OpType::DistributedRMSNormOp: {
    const auto *op = opContext.type_as_DistributedRMSNormOp();
    attrs["epsilon"] = op->epsilon();
    break;
  }
  case ::tt::target::ttnn::OpType::LayerNormOp: {
    const auto *op = opContext.type_as_LayerNormOp();
    attrs["epsilon"] = op->epsilon();
    break;
  }
  case ::tt::target::ttnn::OpType::GroupNormOp: {
    const auto *op = opContext.type_as_GroupNormOp();
    attrs["num_groups"] = op->num_groups();
    attrs["epsilon"] = op->epsilon();
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    const auto *op = opContext.type_as_AllGatherOp();
    attrs["all_gather_dim"] = op->all_gather_dim();
    attrs["cluster_axis"] = static_cast<uint32_t>(op->cluster_axis());
    if (op->num_links().has_value()) {
      attrs["num_links"] = op->num_links().value();
    }
    break;
  }
  case ::tt::target::ttnn::OpType::AllReduceOp: {
    const auto *op = opContext.type_as_AllReduceOp();
    attrs["reduce_type"] = static_cast<uint32_t>(op->reduce_type());
    attrs["cluster_axis"] = static_cast<uint32_t>(op->cluster_axis());
    if (op->num_links().has_value()) {
      attrs["num_links"] = op->num_links().value();
    }
    break;
  }
  case ::tt::target::ttnn::OpType::ReduceScatterOp: {
    const auto *op = opContext.type_as_ReduceScatterOp();
    attrs["scatter_dim"] = op->scatter_dim();
    attrs["reduce_type"] = static_cast<uint32_t>(op->reduce_type());
    attrs["cluster_axis"] = static_cast<uint32_t>(op->cluster_axis());
    if (op->num_links().has_value()) {
      attrs["num_links"] = op->num_links().value();
    }
    break;
  }
  case ::tt::target::ttnn::OpType::MeshShardOp: {
    const auto *op = opContext.type_as_MeshShardOp();
    attrs["shard_direction"] = static_cast<uint32_t>(op->shard_direction());
    attrs["shard_type"] = static_cast<uint32_t>(op->shard_type());
    if (op->shard_shape()) {
      attrs["shard_shape"] = std::vector<int64_t>(op->shard_shape()->begin(),
                                                  op->shard_shape()->end());
    }
    if (op->shard_dims()) {
      attrs["shard_dims"] = std::vector<int64_t>(op->shard_dims()->begin(),
                                                 op->shard_dims()->end());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::MeshPartitionOp: {
    const auto *op = opContext.type_as_MeshPartitionOp();
    attrs["dim"] = op->dim();
    break;
  }
  case ::tt::target::ttnn::OpType::AllToAllDispatchOp: {
    const auto *op = opContext.type_as_AllToAllDispatchOp();
    attrs["num_devices"] = static_cast<uint32_t>(op->num_devices());
    attrs["cluster_axis"] = static_cast<uint32_t>(op->cluster_axis());
    break;
  }
  case ::tt::target::ttnn::OpType::AllToAllCombineOp: {
    const auto *op = opContext.type_as_AllToAllCombineOp();
    attrs["num_devices"] = static_cast<uint32_t>(op->num_devices());
    attrs["cluster_axis"] = static_cast<uint32_t>(op->cluster_axis());
    attrs["num_experts_per_tok"] =
        static_cast<uint32_t>(op->num_experts_per_tok());
    break;
  }
  case ::tt::target::ttnn::OpType::MoeExpertTokenRemapOp: {
    const auto *op = opContext.type_as_MoeExpertTokenRemapOp();
    attrs["reduction_size"] = static_cast<uint32_t>(op->reduction_size());
    break;
  }
  case ::tt::target::ttnn::OpType::UpsampleOp: {
    const auto *op = opContext.type_as_UpsampleOp();
    if (op->mode() && op->mode()->size() > 0) {
      attrs["mode"] = std::string(op->mode()->c_str());
    }
    // Note: scale_factor is a Scale2D union (UniformScale2D or
    // NonUniformScale2D) This would require more complex handling based on the
    // union type
    break;
  }
  case ::tt::target::ttnn::OpType::CpuOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    const auto *op = opContext.type_as_DeallocateOp();
    attrs["force"] = op->force();
    break;
  }
  case ::tt::target::ttnn::OpType::UpdateCacheOp: {
    const auto *op = opContext.type_as_UpdateCacheOp();
    attrs["batch_offset"] = static_cast<uint32_t>(op->batch_offset());
    break;
  }
  case ::tt::target::ttnn::OpType::PagedUpdateCacheOp: {
    const auto *op = opContext.type_as_PagedUpdateCacheOp();
    attrs["share_cache"] = op->share_cache();
    break;
  }
  case ::tt::target::ttnn::OpType::FillCacheOp: {
    const auto *op = opContext.type_as_FillCacheOp();
    attrs["batch_offset"] = static_cast<uint32_t>(op->batch_offset());
    break;
  }
  case ::tt::target::ttnn::OpType::PagedFillCacheOp: {
    // PagedFillCacheOp doesn't have any simple attributes to extract
    break;
  }
  case ::tt::target::ttnn::OpType::LoadCachedOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::SortOp: {
    const auto *op = opContext.type_as_SortOp();
    attrs["dim"] = op->dim();
    attrs["descending"] = op->descending();
    attrs["stable"] = op->stable();
    break;
  }
  case ::tt::target::ttnn::OpType::PointToPointOp: {
    const auto *op = opContext.type_as_PointToPointOp();
    if (op->sender_coord()) {
      attrs["sender_coord"] = std::vector<uint32_t>(op->sender_coord()->begin(),
                                                    op->sender_coord()->end());
    }
    if (op->receiver_coord()) {
      attrs["receiver_coord"] = std::vector<uint32_t>(
          op->receiver_coord()->begin(), op->receiver_coord()->end());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::NamedFullOp: {
    const auto *op = opContext.type_as_NamedFullOp();
    if (op->shape()) {
      attrs["shape"] =
          std::vector<int64_t>(op->shape()->begin(), op->shape()->end());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::FuncCallOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::WriteTensorOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::BeginTraceCaptureOp: {
    const auto *op = opContext.type_as_BeginTraceCaptureOp();
    attrs["cq_id"] = static_cast<uint32_t>(op->cq_id());
    break;
  }
  case ::tt::target::ttnn::OpType::EndTraceCaptureOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::ExecuteTraceOp: {
    attrs["cq_id"] =
        static_cast<uint32_t>(opContext.type_as_ExecuteTraceOp()->cq_id());
    attrs["blocking"] = opContext.type_as_ExecuteTraceOp()->blocking();
    break;
  }
  case ::tt::target::ttnn::OpType::CaptureOrExecuteTraceOp: {
    const auto *op = opContext.type_as_CaptureOrExecuteTraceOp();
    attrs["capture_program_id"] =
        static_cast<uint32_t>(op->capture_program_id());
    attrs["execute_program_id"] =
        static_cast<uint32_t>(op->execute_program_id());
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatenateHeadsOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::NLPConcatHeadsOp: {
    // NLPConcatHeadsOp doesn't have any simple attrs like num_heads
    break;
  }
  case ::tt::target::ttnn::OpType::NLPConcatHeadsDecodeOp: {
    const auto *op = opContext.type_as_NLPConcatHeadsDecodeOp();
    attrs["num_heads"] = static_cast<uint32_t>(op->num_heads());
    break;
  }
  case ::tt::target::ttnn::OpType::SplitQueryKeyValueAndSplitHeadsOp: {
    const auto *op = opContext.type_as_SplitQueryKeyValueAndSplitHeadsOp();
    attrs["num_heads"] = static_cast<uint32_t>(op->num_heads());
    if (op->num_kv_heads().has_value()) {
      attrs["num_kv_heads"] = op->num_kv_heads().value();
    }
    attrs["transpose_key"] = op->transpose_key();
    break;
  }
  case ::tt::target::ttnn::OpType::GenericOp: {
    // GenericOp schema has changed - no simple attributes to extract
    break;
  }
  case ::tt::target::ttnn::OpType::ScaledDotProductAttentionDecodeOp: {
    const auto *op = opContext.type_as_ScaledDotProductAttentionDecodeOp();
    if (op->scale().has_value()) {
      attrs["scale"] = op->scale().value();
    }
    attrs["is_causal"] = op->is_causal();
    break;
  }
  case ::tt::target::ttnn::OpType::ScaledDotProductAttentionOp: {
    const auto *op = opContext.type_as_ScaledDotProductAttentionOp();
    if (op->scale().has_value()) {
      attrs["scale"] = op->scale().value();
    }
    attrs["is_causal"] = op->is_causal();
    if (op->sliding_window_size().has_value()) {
      attrs["sliding_window_size"] = op->sliding_window_size().value();
    }
    break;
  }
  case ::tt::target::ttnn::OpType::PagedScaledDotProductAttentionDecodeOp: {
    const auto *op = opContext.type_as_PagedScaledDotProductAttentionDecodeOp();
    if (op->scale().has_value()) {
      attrs["scale"] = op->scale().value();
    }
    attrs["is_causal"] = op->is_causal();
    break;
  }
  case ::tt::target::ttnn::OpType::RotaryEmbeddingLlamaOp: {
    const auto *op = opContext.type_as_RotaryEmbeddingLlamaOp();
    attrs["is_decode_mode"] = op->is_decode_mode();
    break;
  }
  case ::tt::target::ttnn::OpType::RotaryEmbeddingOp: {
    const auto *op = opContext.type_as_RotaryEmbeddingOp();
    if (op->token_index().has_value()) {
      attrs["token_index"] = op->token_index().value();
    }
    break;
  }
  case ::tt::target::ttnn::OpType::NLPCreateQKVHeadsDecodeOp: {
    const auto *op = opContext.type_as_NLPCreateQKVHeadsDecodeOp();
    attrs["num_heads"] = static_cast<uint32_t>(op->num_heads());
    if (op->num_kv_heads().has_value()) {
      attrs["num_kv_heads"] = op->num_kv_heads().value();
    }
    if (op->overlap_qk_coregrid().has_value()) {
      attrs["overlap_qk_coregrid"] = op->overlap_qk_coregrid().value();
    }
    if (op->slice_size().has_value()) {
      attrs["slice_size"] = op->slice_size().value();
    }
    break;
  }
  case ::tt::target::ttnn::OpType::DumpTensorOp: {
    const auto *op = opContext.type_as_DumpTensorOp();
    if (op->file_path() && op->file_path()->size() > 0) {
      attrs["file_path"] = std::string(op->file_path()->c_str());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::LoadTensorOp: {
    const auto *op = opContext.type_as_LoadTensorOp();
    if (op->file_path() && op->file_path()->size() > 0) {
      attrs["file_path"] = std::string(op->file_path()->c_str());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::AggregateTensorOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::DistributeTensorOp: {
    break;
  }
  case ::tt::target::ttnn::OpType::AnnotateOp: {
    const auto *op = opContext.type_as_AnnotateOp();
    if (op->annotation() && op->annotation()->size() > 0) {
      attrs["annotation"] = std::string(op->annotation()->c_str());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::RegionStartOp: {
    const auto *op = opContext.type_as_RegionStartOp();
    if (op->region_id() && op->region_id()->size() > 0) {
      attrs["region_id"] = std::string(op->region_id()->c_str());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::RegionEndOp: {
    const auto *op = opContext.type_as_RegionEndOp();
    if (op->region_id() && op->region_id()->size() > 0) {
      attrs["region_id"] = std::string(op->region_id()->c_str());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::BreakpointOp: {
    // BreakpointOp doesn't have a name field in the schema
    break;
  }
  case ::tt::target::ttnn::OpType::PrintOp: {
    const auto *op = opContext.type_as_PrintOp();
    if (op->message() && op->message()->size() > 0) {
      attrs["message"] = std::string(op->message()->c_str());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::MemorySnapshotOp: {
    const auto *op = opContext.type_as_MemorySnapshotOp();
    if (op->file_path() && op->file_path()->size() > 0) {
      attrs["file_path"] = std::string(op->file_path()->c_str());
    }
    break;
  }
  case ::tt::target::ttnn::OpType::CreateGlobalSemaphoreOp: {
    const auto *op = opContext.type_as_CreateGlobalSemaphoreOp();
    if (op->initial_value().has_value()) {
      attrs["initial_value"] = op->initial_value().value();
    }
    break;
  }
  case ::tt::target::ttnn::OpType::ResetGlobalSemaphoreOp: {
    const auto *op = opContext.type_as_ResetGlobalSemaphoreOp();
    attrs["value"] = op->value();
    break;
  }
  case ::tt::target::ttnn::OpType::NONE: {
    LOG_FATAL("Invalid op type");
    break;
  }
  }

  return attrs;
}

void registerCallback() {
  PyGILState_STATE gstate = PyGILState_Ensure(); // Acquire GIL
  // Add the runtime directory to Python path
  PyObject *sys_path = PySys_GetObject("path");
  // Get the runtime python path from environment variable or construct it from
  // TT_MLIR_HOME
  const char *runtime_python_path = std::getenv("TTMLIR_RUNTIME_PYTHON_PATH");
  std::string path_to_add;
  if (runtime_python_path != nullptr) {
    path_to_add = runtime_python_path;
  } else {
    // Fallback: try to construct from TT_MLIR_HOME build directory
    const char *tt_mlir_home = std::getenv("TT_MLIR_HOME");
    if (tt_mlir_home != nullptr) {
      path_to_add = std::string(tt_mlir_home) + "/build/runtime/python";
    } else {
      // Last resort: try relative path (for backwards compatibility)
      path_to_add = "./build/runtime/python";
    }
  }

  PyObject *runtime_path = PyUnicode_FromString(path_to_add.c_str());
  PyList_Append(sys_path, runtime_path);
  Py_DECREF(runtime_path);

  // Import the golden module
  PyObject *golden_module = PyImport_ImportModule("python.scripts.golden");
  if (golden_module == nullptr) {
    PyErr_Print();
    PyGILState_Release(gstate);
    return;
  }

  // Get the log_message function
  PyObject *register_func = PyObject_GetAttrString(golden_module, "register");
  if (register_func == nullptr || !PyCallable_Check(register_func)) {
    PyErr_Print();
    Py_DECREF(golden_module);
    PyGILState_Release(gstate);
    return;
  }

  // Double check this is necessary: *********
  try {
    // Call the function with error handling
    PyObject *result = PyObject_CallFunction(register_func, "s", "Placeholder");
    if (result == nullptr) {
      PyErr_Print();
      // Print more detailed error info
      if (PyErr_Occurred()) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);

        if (pvalue) {
          PyObject *str = PyObject_Str(pvalue);
          if (str) {
            std::cout << "Python error: " << PyUnicode_AsUTF8(str) << std::endl;
            Py_DECREF(str);
          }
        }

        PyErr_Restore(ptype, pvalue, ptraceback);
      }
    } else {
      Py_DECREF(result);
    }
  } catch (...) {
    std::cout << "C++ exception in registerCallback" << std::endl;
  }

  // Clean up
  Py_DECREF(register_func);
  Py_DECREF(golden_module);

  PyGILState_Release(gstate); // Release GIL
}

std::vector<::tt::runtime::Tensor>
submit(Device deviceHandle, Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs,
       bool registerRuntimeGoldens) {

  if (registerRuntimeGoldens) {
    if (!Py_IsInitialized()) {
      std::cout << "Initializing New Python interpreter" << std::endl;
      Py_Initialize();
    } else {
      std::cout << "Python interpreter already initialized" << std::endl;
    }

    registerCallback();
  }
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  ::tt::runtime::utils::logMemoryStateIfNeeded(
      ::tt::runtime::ttnn::utils::getMemoryView, deviceHandle,
      ::tt::runtime::MemoryLogLevel::Program,
      "Device memory state before submit");
#endif

  std::unique_ptr<ProgramExecutor> executor = std::make_unique<ProgramExecutor>(
      deviceHandle, executableHandle, programIndex, inputs);

  executor->execute();
  std::vector<::tt::runtime::Tensor> outputTensors =
      executor->gatherOutputTensors();
  executor.reset();

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  ::tt::runtime::utils::logMemoryStateIfNeeded(
      ::tt::runtime::ttnn::utils::getMemoryView, deviceHandle,
      ::tt::runtime::MemoryLogLevel::Program,
      "Device memory state after submit");
#endif

  /*
  if (registerRuntimeGoldens) {
    // Finalize the Python interpreter if we initialized it in this function
    if (Py_IsInitialized()) {
      std::cout << "Finalizing Python interpreter" << std::endl;
      Py_Finalize();
    }
  }
  */
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
