// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/detail/debug.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
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

static Tensor createNullTensor() {
  return Tensor(nullptr, nullptr, DeviceRuntime::TTNN);
}

static DeviceVariant getTargetDevice(::ttnn::MeshDevice &meshDevice) {
  if (meshDevice.num_devices() == 1) {
    return std::ref(*(meshDevice.get_device_index(0)));
  }
  return std::ref(meshDevice);
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

  ::tt::tt_metal::SmallVector<uint32_t> small_vector_shape(shape.begin(),
                                                           shape.end());

  auto tensor = std::make_shared<::ttnn::Tensor>(
      createStorage<BorrowedStorage>(data.get(), numElements, dataType),
      ::ttnn::Shape(small_vector_shape), utils::toTTNNDataType(dataType),
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
                   return createOwnedTensor(dataShard, shape, stride, itemsize,
                                            dataType);
                 });
  DistributedTensorConfig distributionStrategy =
      ::tt::tt_metal::get_distributed_tensor_config(strategy);
  std::shared_ptr<::ttnn::Tensor> tensor = std::make_shared<::ttnn::Tensor>(
      ::ttnn::distributed::api::create_multi_device_tensor(
          tensorShards, ::tt::tt_metal::StorageType::MULTI_DEVICE_HOST,
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
        createOwnedTensor(nullptr, shape, stride, itemsize,
                          utils::fromTTNNDataType(layoutDesc.dataType));
    Tensor out = utils::createRuntimeTensorFromTTNN(tensor);
    return toLayout(out, device, layout);
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

Tensor toHost(Tensor tensor, bool untilize) {
  const ::ttnn::Tensor &deviceTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  std::shared_ptr<::ttnn::Tensor> hostTensor =
      std::make_shared<::ttnn::Tensor>(::ttnn::from_device(deviceTensor));

  if (untilize) {
    hostTensor = std::make_shared<::ttnn::Tensor>(::ttnn::to_layout(
        *hostTensor, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt,
        static_cast<::ttnn::Device *>(nullptr)));
  }

  return Tensor(std::static_pointer_cast<void>(hostTensor), nullptr,
                DeviceRuntime::TTNN);
}

Tensor toLayout(Tensor tensor, Device device, Layout layout) {
  const ::ttnn::Tensor &ttnnTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  const ::ttnn::Layout &inputLayout = ttnnTensor.get_layout();
  const ::ttnn::DataType &inputDataType = ttnnTensor.get_dtype();
  LayoutDesc inputLayoutDesc(::ttnn::BufferType::SYSTEM_MEMORY, inputLayout,
                             inputDataType, std::nullopt);

  const LayoutDesc &outputLayoutDesc =
      layout.as<LayoutDesc>(DeviceRuntime::TTNN);

  ::ttnn::MeshDevice &meshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  DeviceVariant targetDevice = getTargetDevice(meshDevice);
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
  const ::tt::target::TensorRef *input = program->inputs()->Get(inputIndex);

  ::ttnn::BufferType inputBufferType = utils::toTTNNBufferType(
      input->desc()->layout()->memory_desc()->memory_space());
  ::ttnn::Layout inputLayout = utils::inferLayoutFromTileShape(input);
  ::ttnn::DataType inputDataType = utils::toTTNNDataType(
      input->desc()->layout()->memory_desc()->data_type());
  std::optional<::ttnn::MemoryConfig> inputMemoryConfig = std::nullopt;
  if (inputBufferType != ::ttnn::BufferType::SYSTEM_MEMORY) {
    inputMemoryConfig = utils::createMemoryConfig(input);
  }

  std::shared_ptr<LayoutDesc> layoutDesc = std::make_shared<LayoutDesc>(
      inputBufferType, inputLayout, inputDataType, inputMemoryConfig);

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
  std::int32_t globalId{-1};
  const ::ttnn::Tensor *outPtr = nullptr;

  switch (opContext.type_type()) {
  case ::tt::target::ttnn::OpType::GetDeviceOp: {
    globalId = opContext.type_as_GetDeviceOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    globalId = opContext.type_as_ToMemoryConfigOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ToLayoutOp: {
    globalId = opContext.type_as_ToLayoutOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::TypecastOp: {
    globalId = opContext.type_as_TypecastOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ToDeviceOp: {
    globalId = opContext.type_as_ToDeviceOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::FromDeviceOp: {
    globalId = opContext.type_as_FromDeviceOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    globalId = opContext.type_as_EmptyOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    globalId = opContext.type_as_FullOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    globalId = opContext.type_as_EltwiseOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    globalId = opContext.type_as_MatmulOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    globalId = opContext.type_as_ReductionOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::EmbeddingOp: {
    globalId = opContext.type_as_EmbeddingOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    globalId = opContext.type_as_SoftmaxOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    globalId = opContext.type_as_TransposeOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ConcatOp: {
    globalId = opContext.type_as_ConcatOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::ReshapeOp: {
    globalId = opContext.type_as_ReshapeOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::SliceOp: {
    globalId = opContext.type_as_SliceOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::Conv2dOp: {
    globalId = opContext.type_as_Conv2dOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::MaxPool2dOp: {
    globalId = opContext.type_as_MaxPool2dOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::AllGatherOp: {
    globalId = opContext.type_as_AllGatherOp()->out()->global_id();
    break;
  }
  case ::tt::target::ttnn::OpType::DeallocateOp: {
    LOG_WARNING("getting output tensor for DeallocateOp is not supported");
    return createNullTensor();
  }
  default: {
    LOG_FATAL("Unsupported operation type");
  }
  }

  if (tensorPool.contains(globalId)) {
    outPtr = &tensorPool.at(globalId);
  } else {
    LOG_WARNING("Output tensor not found in tensor pool");
    return createNullTensor();
  }

  ::ttnn::Tensor hostTensor = ::ttnn::from_device(*outPtr);
  ::ttnn::Tensor outCopy =
      ::ttnn::to_layout(hostTensor, ::ttnn::ROW_MAJOR_LAYOUT, std::nullopt,
                        std::nullopt, static_cast<::ttnn::Device *>(nullptr));

  void *src = ::tt::tt_metal::get_raw_host_data_ptr(outCopy);
  std::uint32_t outCopySize = outCopy.volume() * outCopy.element_size();
  std::shared_ptr<void> data = ::tt::runtime::utils::malloc_shared(outCopySize);
  std::memcpy(data.get(), src, outCopySize);

  auto tensor = std::make_shared<::ttnn::Tensor>(
      ttnn::createStorage<BorrowedStorage>(data.get(), outCopy.volume(),
                                           ::tt::target::DataType::Float32),
      outCopy.shape().value, ::ttnn::DataType::FLOAT32,
      ::ttnn::Layout::ROW_MAJOR);

  return Tensor(std::static_pointer_cast<void>(tensor), nullptr,
                DeviceRuntime::TTNN);
}

std::vector<float> getTensorData(Tensor tensor) {
  const ::ttnn::Tensor *nnTensor =
      static_cast<::ttnn::Tensor *>(tensor.handle.get());
  if (nnTensor == nullptr) {
    return {};
  }

  void *dataPtr = ::tt::tt_metal::get_raw_host_data_ptr(*nnTensor);
  return std::vector<float>(static_cast<float *>(dataPtr),
                            static_cast<float *>(dataPtr) + nnTensor->volume());
}

namespace legacy {

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex,
             std::vector<Tensor> const &inputHandles,
             std::vector<Tensor> const &outputHandles) {
  ::ttnn::MeshDevice &meshDevice =
      deviceHandle.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
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

  tt::runtime::ttnn::legacy::runProgram(meshDevice, executableHandle,
                                        programIndex, inputs, outputs);
  return Event(nullptr, DeviceRuntime::TTNN);
}
} // namespace legacy

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
