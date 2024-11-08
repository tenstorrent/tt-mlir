// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "tt/runtime/runtime.h"
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

namespace detail {
class LayoutConverter {
public:
  LayoutDesc inputDesc;
  LayoutDesc outputDesc;
  bool shouldTilize = false;
  bool shouldUntilize = false;
  bool shouldTypecast = false;
  bool shouldToDevice = false;
  bool shouldToMemoryConfig = false;
  LayoutConverter(const LayoutDesc &inputDesc, const LayoutDesc &outputDesc)
      : inputDesc(inputDesc), outputDesc(outputDesc) {
    shouldTilize = (inputDesc.layout == ::ttnn::Layout::ROW_MAJOR and
                    outputDesc.layout == ::ttnn::Layout::TILE);
    shouldUntilize = (inputDesc.layout == ::ttnn::Layout::TILE and
                      outputDesc.layout == ::ttnn::Layout::ROW_MAJOR);

    shouldTypecast = (inputDesc.dataType != outputDesc.dataType);

    shouldToDevice = (inputDesc.isOnHost() and outputDesc.isOnDevice());

    shouldToMemoryConfig =
        (not shouldToDevice and outputDesc.isOnDevice() and
         (inputDesc.memoryConfig != outputDesc.memoryConfig));
  }

  ::ttnn::Tensor
  convertTensorLayout(const ::ttnn::Tensor &input,
                      std::optional<DeviceVariant> targetDevice) {
    if (inputDesc.isOnHost()) {
      return convertHostTensorLayout(input, targetDevice);
    }
    return convertDeviceTensorLayout(input);
  }

private:
  ::ttnn::Tensor toLayoutIfNeeded(const ::ttnn::Tensor &input) {
    if (shouldTilize) {
      return ::ttnn::to_layout(input, ::ttnn::Layout::TILE, std::nullopt,
                               std::nullopt,
                               static_cast<::ttnn::Device *>(nullptr));
    } else if (shouldUntilize) {
      return ::ttnn::to_layout(input, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                               std::nullopt,
                               static_cast<::ttnn::Device *>(nullptr));
    }
    return input;
  }

  ::ttnn::Tensor typecastIfNeeded(const ::ttnn::Tensor &input) {
    if (shouldTypecast) {
      return ::ttnn::typecast(input, outputDesc.dataType);
    }
    return input;
  }

  ::ttnn::Tensor toDeviceIfNeeded(const ::ttnn::Tensor &input,
                                  std::optional<DeviceVariant> targetDevice,
                                  bool force = false) {
    if (shouldToDevice or force) {
      LOG_ASSERT(targetDevice.has_value());
      LOG_ASSERT(outputDesc.memoryConfig.has_value());
      return std::visit(
          [&](auto &&targetDevice) -> ::ttnn::Tensor {
            return ::ttnn::to_device(input, &(targetDevice.get()),
                                     outputDesc.memoryConfig.value());
          },
          targetDevice.value());
    }
    return input;
  }

  ::ttnn::Tensor toMemoryConfigIfNeeded(const ::ttnn::Tensor &input) {
    if (shouldToMemoryConfig) {
      LOG_ASSERT(outputDesc.memoryConfig.has_value());
      return ::ttnn::to_memory_config(input, outputDesc.memoryConfig.value());
    }
    return input;
  }

  //
  // Host input tensor APIs
  //
  ::ttnn::Tensor
  handleHostInputNoLayoutNoTypecast(const ::ttnn::Tensor &input,
                                    std::optional<DeviceVariant> targetDevice) {
    ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  ::ttnn::Tensor
  handleHostInputLayoutNoTypecast(const ::ttnn::Tensor &input,
                                  std::optional<DeviceVariant> targetDevice) {
    if (shouldUntilize) {
      ::ttnn::Tensor out = toLayoutIfNeeded(input);
      out = toDeviceIfNeeded(out, targetDevice);
      out = toMemoryConfigIfNeeded(out);
      return out;
    }

    else if (shouldTilize and
             outputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
      ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
      out = toLayoutIfNeeded(out);
      out = toMemoryConfigIfNeeded(out);
      return out;
    }

    else if (shouldTilize and
             outputDesc.dataType != ::ttnn::DataType::BFLOAT16) {
      ::ttnn::Tensor out = toLayoutIfNeeded(input);
      out = toDeviceIfNeeded(out, targetDevice);
      out = toMemoryConfigIfNeeded(out);
      return out;
    }
    LOG_FATAL("Unreachable code path");
  }

  ::ttnn::Tensor
  handleHostInputNoLayoutTypecast(const ::ttnn::Tensor &input,
                                  std::optional<DeviceVariant> targetDevice) {
    if (outputDesc.layout == ::ttnn::Layout::TILE) {
      ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
      out = typecastIfNeeded(out);
      out = toMemoryConfigIfNeeded(out);
      return out;
    }

    else if (outputDesc.layout != ::ttnn::Layout::TILE) {
      ::ttnn::Tensor out = typecastIfNeeded(input);
      out = toDeviceIfNeeded(out, targetDevice);
      out = toMemoryConfigIfNeeded(out);
      return out;
    }
    LOG_FATAL("Unreachable code path");
  }

  ::ttnn::Tensor
  handleHostInputLayoutTypecast(const ::ttnn::Tensor &input,
                                std::optional<DeviceVariant> targetDevice) {
    if (shouldUntilize) {
      ::ttnn::Tensor out = typecastIfNeeded(input);
      out = toLayoutIfNeeded(out);
      out = toDeviceIfNeeded(out, targetDevice);
      out = toMemoryConfigIfNeeded(out);
      return out;
    }

    else if (shouldTilize and
             inputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
      ::ttnn::Tensor out = toDeviceIfNeeded(input, targetDevice);
      out = toLayoutIfNeeded(out);
      out = typecastIfNeeded(out);
      out = toMemoryConfigIfNeeded(out);
      return out;
    }

    else if (shouldTilize and
             outputDesc.dataType == ::ttnn::DataType::BFLOAT16) {
      ::ttnn::Tensor out = typecastIfNeeded(input);
      out = toDeviceIfNeeded(out, targetDevice);
      out = toLayoutIfNeeded(input);
      out = toMemoryConfigIfNeeded(out);
      return out;
    } else if (shouldTilize and
               inputDesc.dataType != ::ttnn::DataType::BFLOAT16 and
               outputDesc.dataType != ::ttnn::DataType::BFLOAT16) {
      ::ttnn::Tensor out = typecastIfNeeded(input);
      out = toLayoutIfNeeded(out);
      out = toDeviceIfNeeded(out, targetDevice);
      out = toMemoryConfigIfNeeded(out);
      return out;
    }
    LOG_FATAL("Unreachable code path");
  }

  ::ttnn::Tensor
  convertHostTensorLayout(const ::ttnn::Tensor &input,
                          std::optional<DeviceVariant> targetDevice) {
    bool shouldToLayout = (shouldTilize or shouldUntilize);
    LOG_ASSERT(not shouldToDevice or targetDevice.has_value(),
               "Target device must be provided for ToDevice");
    if (not shouldToLayout and not shouldTypecast) {
      return handleHostInputNoLayoutNoTypecast(input, targetDevice);
    } else if (shouldToLayout and not shouldTypecast) {
      return handleHostInputLayoutNoTypecast(input, targetDevice);
    } else if (not shouldToLayout and shouldTypecast) {
      return handleHostInputNoLayoutTypecast(input, targetDevice);
    } else if (shouldToLayout and shouldTypecast) {
      return handleHostInputLayoutTypecast(input, targetDevice);
    }
    LOG_FATAL("Unreachable code path");
  }

  //
  // Device input tensor APIs
  //
  ::ttnn::Tensor
  handleDeviceInputNoLayoutNoTypecast(const ::ttnn::Tensor &input) {
    return toMemoryConfigIfNeeded(input);
  }

  ::ttnn::Tensor
  handleDeviceInputLayoutNoTypecast(const ::ttnn::Tensor &input) {
    if (shouldUntilize) {
      LOG_WARNING(
          "Device to device tilize is not fully supported on the ttnn side");
    }
    ::ttnn::Tensor out = toLayoutIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  ::ttnn::Tensor
  handleDeviceInputNoLayoutTypecast(const ::ttnn::Tensor &input) {
    ::ttnn::Tensor out = typecastIfNeeded(input);
    out = toMemoryConfigIfNeeded(out);
    return out;
  }

  ::ttnn::Tensor handleDeviceInputLayoutTypecast(const ::ttnn::Tensor &input) {
    if (shouldUntilize) {
      LOG_WARNING(
          "Device to device untilize is not fully supported on the ttnn side");
      ::ttnn::Tensor out = typecastIfNeeded(input);
      out = toLayoutIfNeeded(out);
      out = toMemoryConfigIfNeeded(out);
      return out;
    } else if (shouldTilize) {
      ::ttnn::Tensor out = toLayoutIfNeeded(input);
      out = typecastIfNeeded(out);
      out = toMemoryConfigIfNeeded(out);
      return out;
    }
    LOG_FATAL("Unreachable code path");
  }

  ::ttnn::Tensor convertDeviceTensorLayout(const ::ttnn::Tensor &input) {
    bool shouldToLayout = (shouldTilize or shouldUntilize);
    if (not shouldToLayout and not shouldTypecast) {
      return handleDeviceInputNoLayoutNoTypecast(input);
    } else if (shouldToLayout and not shouldTypecast) {
      return handleDeviceInputLayoutNoTypecast(input);
    } else if (not shouldToLayout and shouldTypecast) {
      return handleDeviceInputNoLayoutTypecast(input);
    } else if (shouldToLayout and shouldTypecast) {
      return handleDeviceInputLayoutTypecast(input);
    }
    LOG_FATAL("Unreachable code path");
  }
};

} // namespace detail

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

Tensor toDevice(Tensor tensor, Device device) {
  const ::ttnn::Tensor &hostTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  auto &meshDevice = device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  DeviceVariant targetDevice = getTargetDevice(meshDevice);
  std::shared_ptr<::ttnn::Tensor> deviceTensor = std::visit(
      [&hostTensor](auto &&device) -> std::shared_ptr<::ttnn::Tensor> {
        return std::make_shared<::ttnn::Tensor>(
            ::ttnn::to_device(hostTensor, &(device.get()), std::nullopt));
      },
      targetDevice);
  return Tensor(std::static_pointer_cast<void>(deviceTensor), nullptr,
                DeviceRuntime::TTNN);
}

Tensor toDevice(Tensor tensor, Device device, Layout layout) {
  const ::ttnn::Tensor &hostTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  LOG_ASSERT(utils::isOnHost(hostTensor.storage_type()),
             "Input tensor to ToDevice must be on host");
  const ::ttnn::Layout &inputLayout = hostTensor.get_layout();
  const ::ttnn::DataType &inputDataType = hostTensor.get_dtype();
  LayoutDesc inputLayoutDesc(::ttnn::BufferType::SYSTEM_MEMORY, inputLayout,
                             inputDataType, std::nullopt);
  const LayoutDesc &outputLayoutDesc =
      layout.as<LayoutDesc>(DeviceRuntime::TTNN);
  LOG_ASSERT(outputLayoutDesc.isOnDevice(), "Output layout must be on device");
  LOG_ASSERT(outputLayoutDesc.memoryConfig.has_value(),
             "Output layout must have memory config");
  ::ttnn::MeshDevice &meshDevice =
      device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
  DeviceVariant targetDevice = getTargetDevice(meshDevice);

  detail::LayoutConverter converter(inputLayoutDesc, outputLayoutDesc);
  std::shared_ptr<::ttnn::Tensor> out = std::make_shared<::ttnn::Tensor>(
      converter.convertTensorLayout(hostTensor, targetDevice));

  return Tensor(std::static_pointer_cast<void>(out), nullptr,
                DeviceRuntime::TTNN);
}

Tensor toLayout(Tensor tensor, Layout layout) {
  const ::ttnn::Tensor &inputTensor =
      tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  const ::ttnn::StorageType &inputStorageType = inputTensor.storage_type();
  const ::ttnn::BufferType &inputBufferType =
      utils::isOnHost(inputStorageType)
          ? ::ttnn::BufferType::SYSTEM_MEMORY
          : inputTensor.memory_config().buffer_type;
  const ::ttnn::Layout &inputLayout = inputTensor.get_layout();
  const ::ttnn::DataType &inputDataType = inputTensor.get_dtype();
  std::optional<::ttnn::MemoryConfig> inputMemoryConfig = std::nullopt;
  if (utils::isOnDevice(inputStorageType)) {
    inputMemoryConfig =
        std::make_optional<::ttnn::MemoryConfig>(inputTensor.memory_config());
  }
  LayoutDesc inputLayoutDesc(inputBufferType, inputLayout, inputDataType,
                             inputMemoryConfig);
  const LayoutDesc &outputLayoutDesc =
      layout.as<LayoutDesc>(DeviceRuntime::TTNN);
  LOG_ASSERT(inputLayoutDesc.isOnHost() == outputLayoutDesc.isOnHost(),
             "Input and output layout must be on the same memory space");

  detail::LayoutConverter converter(inputLayoutDesc, outputLayoutDesc);
  std::shared_ptr<::ttnn::Tensor> out = std::make_shared<::ttnn::Tensor>(
      converter.convertTensorLayout(inputTensor, std::nullopt));

  return Tensor(std::static_pointer_cast<void>(out), nullptr,
                DeviceRuntime::TTNN);
}

void deallocateTensor(Tensor tensor, bool force) {
  ::ttnn::Tensor &ttnnTensor = tensor.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
  ttnnTensor.deallocate(force);
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

  tt::runtime::ttnn::runProgram(meshDevice, executableHandle, programIndex,
                                inputs, outputs);
  return Event(nullptr, DeviceRuntime::TTNN);
}

void wait(Event event) {
  // Not implemented
  LOG_ASSERT(event.matchesRuntime(DeviceRuntime::TTNN));
}

std::string getOpDebugString(OpContext opContextHandle) {
  auto const &opContext =
      opContextHandle.as<::tt::target::ttnn::Operation>(DeviceRuntime::TTNN);
  return std::string(opContext.debug_info()->c_str());
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

  return Tensor(std::static_pointer_cast<void>(tensor), data,
                DeviceRuntime::TTNN);
}

std::vector<float> getTensorData(Tensor tensor) {
  ::ttnn::Tensor *nnTensor = static_cast<::ttnn::Tensor *>(tensor.handle.get());
  if (nnTensor == nullptr) {
    return {};
  }

  void *dataPtr = ::tt::tt_metal::get_raw_host_data_ptr(*nnTensor);
  return std::vector<float>(static_cast<float *>(dataPtr),
                            static_cast<float *>(dataPtr) + nnTensor->volume());
}

void wait(Tensor tensor) {
  LOG_ASSERT(tensor.matchesRuntime(DeviceRuntime::TTNN));
  LOG_ASSERT(tensor.event.matchesRuntime(DeviceRuntime::TTNN));
}

} // namespace tt::runtime::ttnn
