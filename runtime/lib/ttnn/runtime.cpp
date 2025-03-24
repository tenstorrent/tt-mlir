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
static OwnedStorage createOwnedStorage(ElementType *ptr,
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

template <typename StorageType, typename ElementType>
static StorageType createStorage(ElementType *ptr, std::uint32_t numElements) {
  if constexpr (std::is_same_v<StorageType, BorrowedStorage>) {
    LOG_ASSERT(ptr != nullptr, "Cannot create borrowed storage from nullptr");
    return BorrowedStorage(
        ::tt::tt_metal::borrowed_buffer::Buffer<ElementType>(ptr, numElements),
        [] {}, [] {});
  } else if constexpr (std::is_same_v<StorageType, OwnedStorage>) {
    return createOwnedStorage(ptr, numElements);
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
  case ::tt::target::DataType::UInt8:
    return createStorage<StorageType>(static_cast<uint8_t *>(ptr), numElements);
  case ::tt::target::DataType::Int32:
    return createStorage<StorageType>(static_cast<int32_t *>(ptr), numElements);
  default:
    LOG_FATAL("Unsupported data type");
  }
}

static ::ttnn::Tensor
createOwnedTTNNTensor(std::shared_ptr<void> data,
                      std::vector<std::uint32_t> const &shape,
                      std::vector<std::uint32_t> const &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];

  return ::ttnn::Tensor(
      createStorage<OwnedStorage>(data.get(), numElements, dataType),
      ::ttnn::Shape(shape), utils::toTTNNDataType(dataType),
      ::ttnn::Layout::ROW_MAJOR);
}

Tensor createOwnedTensor(std::shared_ptr<void> data,
                         std::vector<std::uint32_t> const &shape,
                         std::vector<std::uint32_t> const &stride,
                         std::uint32_t itemsize,
                         ::tt::target::DataType dataType) {

  return utils::createRuntimeTensorFromTTNN(
      createOwnedTTNNTensor(data, shape, stride, itemsize, dataType));
}

static Tensor createNullTensor() {
  return Tensor(nullptr, nullptr, DeviceRuntime::TTNN);
}

static DeviceVariant getTargetDevice(::ttnn::MeshDevice &meshDevice) {
  if (meshDevice.num_devices() == 1) {
    return std::ref(*(meshDevice.get_device(
        ::tt::tt_metal::distributed::MeshCoordinate(0, 0))));
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

// Create a borrowed tensor from user-owned data
Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType) {
  std::uint32_t numElements = shape[0] * stride[0];

  auto tensor = std::make_shared<::ttnn::Tensor>(
      createStorage<BorrowedStorage>(data.get(), numElements, dataType),
      ::ttnn::Shape(shape), utils::toTTNNDataType(dataType),
      ::ttnn::Layout::ROW_MAJOR);
  return Tensor(std::static_pointer_cast<void>(tensor), nullptr,
                DeviceRuntime::TTNN);
}

// Create a owned multi-device host tensor from user-owned data
Tensor
createTensor(std::vector<std::shared_ptr<void>> &data,
             std::vector<std::uint32_t> const &shape,
             std::vector<std::uint32_t> const &stride, std::uint32_t itemsize,
             ::tt::target::DataType dataType,
             std::unordered_map<std::string, std::string> const &strategy) {
  std::vector<::ttnn::Tensor> tensorShards;
  tensorShards.reserve(data.size());
  std::transform(data.begin(), data.end(), std::back_inserter(tensorShards),
                 [&](std::shared_ptr<void> &dataShard) -> ::ttnn::Tensor {
                   return createOwnedTTNNTensor(dataShard, shape, stride,
                                                itemsize, dataType);
                 });
  DistributedTensorConfig distributionStrategy =
      ::tt::tt_metal::get_distributed_tensor_config(strategy);
  std::shared_ptr<::ttnn::Tensor> tensor = std::make_shared<::ttnn::Tensor>(
      ::ttnn::distributed::create_multi_device_tensor(
          tensorShards, ::ttnn::StorageType::MULTI_DEVICE_HOST,
          distributionStrategy));
  return Tensor(std::static_pointer_cast<void>(tensor), nullptr,
                DeviceRuntime::TTNN);
}

// Create an owned empty tensor on host/device
Tensor createTensor(Device device, Layout layout,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize) {
  const LayoutDesc &layoutDesc = layout.as<LayoutDesc>(DeviceRuntime::TTNN);
  if (layoutDesc.isOnHost()) {
    ::ttnn::Tensor tensor =
        createOwnedTTNNTensor(nullptr, shape, stride, itemsize,
                              utils::fromTTNNDataType(layoutDesc.dataType));
    Tensor out = utils::createRuntimeTensorFromTTNN(tensor);
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

tt::target::DataType getTensorDataType(Tensor tensor) {
  const ::ttnn::Tensor &nnTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  return utils::fromTTNNDataType(nnTensor.get_dtype());
}

size_t getNumAvailableDevices() {
  return ::tt::tt_metal::GetNumAvailableDevices();
}

Device openDevice(DeviceIds const &deviceIds, size_t numHWCQs,
                  std::optional<size_t> l1SmallSize,
                  std::optional<DispatchCoreType> dispatchCoreType,
                  std::optional<bool> enableAsyncTTNN) {

  ::tt::tt_metal::DispatchCoreType type =
      tt::runtime::common::getDispatchCoreType(dispatchCoreType);

  LOG_ASSERT(deviceIds.size(), "No devices specified");
  ::tt::tt_metal::distributed::MeshShape grid{
      1, static_cast<uint32_t>(deviceIds.size())};
  size_t l1SmallSizeValue = l1SmallSize.value_or(tt::constants::L1_SMALL_SIZE);
  std::shared_ptr<::ttnn::MeshDevice> meshDevice = ::ttnn::MeshDevice::create(
      ::tt::tt_metal::distributed::MeshDeviceConfig(grid), l1SmallSizeValue,
      DEFAULT_TRACE_REGION_SIZE, numHWCQs, type);

  CoreCoord logical_grid_size = meshDevice->compute_with_storage_grid_size();
  LOG_INFO("Grid size = { ", logical_grid_size.x, ", ", logical_grid_size.y,
           "}");

  bool enableAsyncValue = enableAsyncTTNN.value_or(false);
  for (::ttnn::IDevice *device : meshDevice->get_devices()) {
    device->enable_async(enableAsyncValue);
  }

  return Device(std::static_pointer_cast<void>(meshDevice),
                DeviceRuntime::TTNN);
}

void closeDevice(Device device) {
  ::ttnn::MeshDevice &ttnnMeshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
#if defined(TT_RUNTIME_ENABLE_PERF_TRACE)
  for (::ttnn::IDevice *ttnnDevice : ttnnMeshDevice.get_devices()) {
    ::tt::tt_metal::detail::DumpDeviceProfileResults(ttnnDevice);
  }
#endif

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

  auto dramMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      device, tt::tt_metal::BufferType::DRAM);
  auto l1MemoryView = ::tt::tt_metal::detail::GetMemoryView(
      device, tt::tt_metal::BufferType::L1);
  auto l1SmallMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      device, tt::tt_metal::BufferType::L1_SMALL);
  auto traceMemoryView = ::tt::tt_metal::detail::GetMemoryView(
      device, tt::tt_metal::BufferType::TRACE);

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

void wait(Tensor tensor) {
  LOG_ASSERT(tensor.matchesRuntime(DeviceRuntime::TTNN),
             "Expected ttnn tensor");
  ::tt::runtime::ttnn::wait(tensor.event);
}

void wait(std::vector<Tensor> const &tensors) {
  for (const Tensor &tensor : tensors) {
    ::tt::runtime::ttnn::wait(tensor);
  }
}

static Tensor toHostSingleTensor(Tensor tensor, bool untilize) {
  const ::ttnn::Tensor &deviceTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  std::shared_ptr<::ttnn::Tensor> hostTensor =
      std::make_shared<::ttnn::Tensor>(::ttnn::from_device(deviceTensor));

  if (untilize) {
    hostTensor = std::make_shared<::ttnn::Tensor>(::ttnn::to_layout(
        *hostTensor, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt,
        static_cast<::ttnn::IDevice *>(nullptr)));
  }

  return Tensor(std::static_pointer_cast<void>(hostTensor), nullptr,
                DeviceRuntime::TTNN);
}

std::vector<Tensor> toHost(Tensor tensor, bool untilize) {
  const ::ttnn::Tensor &multiDeviceTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  std::vector<Tensor> host_tensors;
  if (multiDeviceTensor.storage_type() ==
          ::ttnn::StorageType::MULTI_DEVICE_HOST ||
      multiDeviceTensor.storage_type() == ::ttnn::StorageType::MULTI_DEVICE) {
    std::vector<::ttnn::Tensor> single_tensors =
        ::ttnn::distributed::get_device_tensors(multiDeviceTensor);
    for (auto &tensor : single_tensors) {
      host_tensors.push_back(::tt::runtime::ttnn::toHostSingleTensor(
          Tensor(std::make_shared<::ttnn::Tensor>(tensor), nullptr,
                 DeviceRuntime::TTNN),
          untilize));
    }
  } else {
    host_tensors.push_back(
        ::tt::runtime::ttnn::toHostSingleTensor(tensor, untilize));
  }
  return host_tensors;
}

Tensor toLayout(Tensor tensor, Device device, Layout layout) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  const ::ttnn::StorageType &inputStorageType = ttnnTensor.storage_type();
  const ::ttnn::Layout &inputLayout = ttnnTensor.get_layout();
  const ::ttnn::DataType &inputDataType = ttnnTensor.get_dtype();
  LayoutDesc inputLayoutDesc(inputStorageType, inputLayout, inputDataType,
                             std::nullopt);

  const LayoutDesc &outputLayoutDesc =
      layout.as<LayoutDesc>(DeviceRuntime::TTNN);

  ::ttnn::MeshDevice &meshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  DeviceVariant targetDevice = getTargetDevice(meshDevice);
  if (workaround::Env::get().toLayoutAPIAssumeSingleChip) {
    targetDevice = std::ref(*meshDevice.get_device(
        ::tt::tt_metal::distributed::MeshCoordinate(0, 0)));
  }
  LayoutConverter converter(inputLayoutDesc, outputLayoutDesc);
  std::shared_ptr<::ttnn::Tensor> out = std::make_shared<::ttnn::Tensor>(
      converter.convertTensorLayout(ttnnTensor, targetDevice));

  return Tensor(std::static_pointer_cast<void>(out), nullptr,
                DeviceRuntime::TTNN);
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

void memcpy(void *dst, Tensor src) {
  const ::ttnn::Tensor &srcTensor = src.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  if (utils::isOnHost(srcTensor.storage_type())) {
    const void *srcPtr = ::tt::tt_metal::get_raw_host_data_ptr(srcTensor);
    size_t size = srcTensor.volume() * srcTensor.element_size();
    std::memcpy(dst, srcPtr, size);
  } else {
    ::tt::tt_metal::memcpy(dst, srcTensor);
  }
}

void memcpy(Tensor dst, Tensor src) {
  ::ttnn::Tensor &dstTensor = dst.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  const ::ttnn::Tensor &srcTensor = src.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
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

void deallocateTensor(Tensor &tensor, bool force) {
  ::ttnn::Tensor &ttnnTensor = tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
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

Tensor getOpOutputTensor(OpContext opContextHandle,
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
  case ::tt::target::ttnn::OpType::ZerosOp: {
    tensorRef = opContext.type_as_ZerosOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::OnesOp: {
    tensorRef = opContext.type_as_OnesOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    tensorRef = opContext.type_as_FullOp()->out();
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    tensorRef = opContext.type_as_EltwiseOp()->out();
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
    outPtr = &tensorPool.getAndValidate(tensorRef.value());
  } else {
    LOG_WARNING("Output tensor not found in tensor pool");
    return createNullTensor();
  }

  std::shared_ptr<::ttnn::Tensor> hostTensor =
      std::make_shared<::ttnn::Tensor>(::ttnn::to_layout(
          ::ttnn::from_device(*outPtr), ::ttnn::Layout::ROW_MAJOR, std::nullopt,
          std::nullopt, static_cast<::ttnn::IDevice *>(nullptr)));

  return Tensor(std::static_pointer_cast<void>(hostTensor), nullptr,
                DeviceRuntime::TTNN);
}

std::vector<std::byte> getTensorDataBuffer(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
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
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  std::vector<std::uint32_t> shape;
  for (size_t i = 0; i < ttnnTensor.logical_shape().size(); ++i) {
    shape.push_back(ttnnTensor.logical_shape()[i]);
  }
  return shape;
}

std::vector<std::uint32_t> getTensorStride(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  std::vector<std::uint32_t> stride;
  for (size_t i = 0; i < ttnnTensor.strides().size(); ++i) {
    stride.push_back(ttnnTensor.strides()[i]);
  }
  return stride;
}

std::uint32_t getTensorElementSize(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  return ttnnTensor.element_size();
}

std::uint32_t getTensorVolume(::tt::runtime::Tensor tensor) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
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

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> const &inputHandles) {
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);

  // Convert input tensors to the layout expected by the program
  std::vector<Tensor> inputsWithLayout;
  inputsWithLayout.reserve(inputHandles.size());
  std::transform(
      inputHandles.begin(), inputHandles.end(),
      std::back_inserter(inputsWithLayout), [&](const Tensor &input) -> Tensor {
        Layout inputLayout = ::tt::runtime::ttnn::getLayout(
            executableHandle, programIndex, inputsWithLayout.size());
        return ::tt::runtime::ttnn::toLayout(input, deviceHandle, inputLayout);
      });

  std::vector<::ttnn::Tensor *> ttnnInputs;
  ttnnInputs.reserve(inputsWithLayout.size());
  std::transform(inputsWithLayout.begin(), inputsWithLayout.end(),
                 std::back_inserter(ttnnInputs),
                 [](Tensor &input) -> ::ttnn::Tensor * {
                   return &input.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
                 });

  std::vector<Tensor> outputs = ::tt::runtime::ttnn::runProgram(
      meshDevice, executableHandle, programIndex, ttnnInputs);
  return outputs;
}

} // namespace tt::runtime::ttnn
