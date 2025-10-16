// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/controller/command_factory.h"
#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/types.h"
#include <atomic>

#define BUILD_COMMAND_IMPL(CommandName, fbb, builderFunc, ...)                 \
  [&]() -> uint64_t {                                                          \
    uint64_t commandId = nextCommandId();                                      \
    auto commandType = ::tt::runtime::distributed::flatbuffer::CommandType::   \
        CommandName##Command;                                                  \
                                                                               \
    auto commandOffset = (builderFunc)((fbb)__VA_OPT__(, ) __VA_ARGS__);       \
                                                                               \
    auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(      \
        (fbb), commandId, commandType, commandOffset.Union());                 \
                                                                               \
    ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer((fbb),         \
                                                                command);      \
                                                                               \
    debug::verifyFlatbuffer((fbb), verifyFn);                                  \
                                                                               \
    return commandId;                                                          \
  }()

#define BUILD_COMMAND(CommandName, fbb, ...)                                   \
  BUILD_COMMAND_IMPL(                                                          \
      CommandName, fbb,                                                        \
      ::tt::runtime::distributed::flatbuffer::Create##CommandName##Command,    \
      __VA_ARGS__)

#define BUILD_COMMAND_DIRECT(CommandName, fbb, ...)                            \
  BUILD_COMMAND_IMPL(CommandName, fbb,                                         \
                     ::tt::runtime::distributed::flatbuffer::                  \
                         Create##CommandName##CommandDirect,                   \
                     __VA_ARGS__)

namespace tt::runtime::distributed::controller {

using ::tt::runtime::DeviceRuntime;

static constexpr auto verifyFn =
    &::tt::runtime::distributed::flatbuffer::VerifyCommandBuffer;

static uint64_t nextCommandId() {
  static std::atomic<uint64_t> commandIdCounter = 0;
  return commandIdCounter.fetch_add(1, std::memory_order_relaxed);
}

uint64_t CommandFactory::buildGetSystemDescCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const std::optional<::tt::runtime::DispatchCoreType> &dispatchCoreType,
    const std::optional<::tt::runtime::Device> &deviceHandle) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  ::flatbuffers::Offset<::tt::target::DeviceRef> deviceRef = 0;
  if (deviceHandle.has_value()) {
    deviceRef =
        ::tt::target::CreateDeviceRef(fbb, deviceHandle.value().getGlobalId());
  }

  std::optional<::tt::runtime::DispatchCoreType> fbDispatchCoreType =
      std::nullopt;
  if (dispatchCoreType.has_value()) {
    fbDispatchCoreType = dispatchCoreType.value();
  }

  uint64_t commandId =
      BUILD_COMMAND(GetSystemDesc, fbb, deviceRef, fbDispatchCoreType);

  return commandId;
}

uint64_t CommandFactory::buildSetFabricConfigCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::FabricConfig &fabricConfig) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = BUILD_COMMAND(SetFabricConfig, fbb, fabricConfig);

  return commandId;
}

uint64_t CommandFactory::buildGetNumAvailableDevicesCommand(
    ::flatbuffers::FlatBufferBuilder &fbb) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = BUILD_COMMAND(GetNumAvailableDevices, fbb);

  return commandId;
}

uint64_t CommandFactory::buildOpenMeshDeviceCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Device &deviceHandle,
    const ::tt::runtime::MeshDeviceOptions &meshDeviceOptions) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  std::optional<::tt::runtime::DispatchCoreType> fbDispatchCoreType =
      std::nullopt;
  if (meshDeviceOptions.dispatchCoreType.has_value()) {
    fbDispatchCoreType = meshDeviceOptions.dispatchCoreType.value();
  }

  auto fbMeshDeviceOptions =
      ::tt::runtime::distributed::flatbuffer::CreateMeshDeviceOptionsDirect(
          fbb, &meshDeviceOptions.meshOffset, &meshDeviceOptions.deviceIds,
          meshDeviceOptions.numHWCQs, meshDeviceOptions.enableProgramCache,
          meshDeviceOptions.meshShape.has_value()
              ? &(meshDeviceOptions.meshShape.value())
              : nullptr,
          meshDeviceOptions.l1SmallSize, meshDeviceOptions.traceRegionSize,
          fbDispatchCoreType);

  uint64_t commandId = BUILD_COMMAND(
      OpenMeshDevice, fbb, deviceHandle.getGlobalId(), fbMeshDeviceOptions);

  return commandId;
}

uint64_t CommandFactory::buildCloseMeshDeviceCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Device &deviceHandle) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto deviceRef =
      ::tt::target::CreateDeviceRef(fbb, deviceHandle.getGlobalId());

  uint64_t commandId = BUILD_COMMAND(CloseMeshDevice, fbb, deviceRef);

  return commandId;
}

uint64_t CommandFactory::buildCreateSubMeshDeviceCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Device &parentMesh,
    const ::tt::runtime::Device &subMesh,
    const std::vector<uint32_t> &meshShape,
    const std::optional<const std::vector<uint32_t>> &meshOffset) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto parentMeshRef =
      ::tt::target::CreateDeviceRef(fbb, parentMesh.getGlobalId());
  const std::vector<uint32_t> *meshOffsetPtr = nullptr;
  if (meshOffset.has_value()) {
    meshOffsetPtr = &meshOffset.value();
  }

  uint64_t commandId =
      BUILD_COMMAND_DIRECT(CreateSubMeshDevice, fbb, parentMeshRef,
                           subMesh.getGlobalId(), &meshShape, meshOffsetPtr);

  return commandId;
}

uint64_t CommandFactory::buildReleaseSubMeshDeviceCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Device &subMesh) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto subMeshRef = ::tt::target::CreateDeviceRef(fbb, subMesh.getGlobalId());
  uint64_t commandId = BUILD_COMMAND(ReleaseSubMeshDevice, fbb, subMeshRef);
  return commandId;
}

uint64_t CommandFactory::buildGetMeshShapeCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Device &deviceHandle) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto deviceRef =
      ::tt::target::CreateDeviceRef(fbb, deviceHandle.getGlobalId());

  uint64_t commandId = BUILD_COMMAND(GetMeshShape, fbb, deviceRef);

  return commandId;
}

uint64_t CommandFactory::buildCreateHostTensorCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &outputTensor, const void *data,
    const std::vector<uint32_t> &shape, const std::vector<uint32_t> &stride,
    uint32_t itemSize, ::tt::target::DataType dataType) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  std::uint64_t numElements =
      std::accumulate(shape.begin(), shape.end(), static_cast<std::uint64_t>(1),
                      std::multiplies<std::uint64_t>());
  std::uint64_t numBytes = numElements * itemSize;
  auto dataVec =
      fbb.CreateVector<uint8_t>(static_cast<const uint8_t *>(data), numBytes);
  auto shapeVec = fbb.CreateVector<uint32_t>(shape.data(), shape.size());
  auto strideVec = fbb.CreateVector<uint32_t>(stride.data(), stride.size());

  uint64_t commandId =
      BUILD_COMMAND(CreateHostTensor, fbb, outputTensor.getGlobalId(), dataVec,
                    shapeVec, strideVec, itemSize, dataType);

  return commandId;
}

uint64_t CommandFactory::buildGetTensorVolumeCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &tensor) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId =
      BUILD_COMMAND(GetTensorVolume, fbb, tensor.getGlobalId());

  return commandId;
}

uint64_t CommandFactory::buildGetTensorRetainCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &tensor) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId =
      BUILD_COMMAND(GetTensorRetain, fbb, tensor.getGlobalId());

  return commandId;
}

uint64_t CommandFactory::buildSetTensorRetainCommand(
    ::flatbuffers::FlatBufferBuilder &fbb, const ::tt::runtime::Tensor &tensor,
    bool retain) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId =
      BUILD_COMMAND(SetTensorRetain, fbb, tensor.getGlobalId(), retain);

  return commandId;
}

uint64_t CommandFactory::buildGetLayoutCommand(
    ::flatbuffers::FlatBufferBuilder &fbb, const ::tt::runtime::Binary &binary,
    uint32_t programIndex, uint32_t inputIndex,
    const ::tt::runtime::Layout &outputLayout) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  std::vector<uint8_t> binaryBytes;
  binary.storeToMemory(binaryBytes);

  uint64_t commandId = BUILD_COMMAND_DIRECT(
      GetLayout, fbb, binary.id(), &binaryBytes, programIndex, inputIndex,
      outputLayout.getGlobalId());

  return commandId;
}

uint64_t CommandFactory::buildToLayoutCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &inputTensor,
    const ::tt::runtime::Device &device, const ::tt::runtime::Layout &layout,
    const ::tt::runtime::Tensor &outputTensor, std::optional<bool> retain) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t inputGlobalId = inputTensor.getGlobalId();
  uint64_t outputGlobalId = outputTensor.getGlobalId();

  auto deviceRef = ::tt::target::CreateDeviceRef(fbb, device.getGlobalId());

  uint64_t commandId =
      BUILD_COMMAND(ToLayout, fbb, inputGlobalId, outputGlobalId, deviceRef,
                    layout.getGlobalId(), retain);

  return commandId;
}

uint64_t CommandFactory::buildSubmitCommand(
    ::flatbuffers::FlatBufferBuilder &fbb, const ::tt::runtime::Device &device,
    const ::tt::runtime::Binary &executable, uint32_t programIndex,
    const std::vector<::tt::runtime::Tensor> &inputTensors,
    const std::vector<::tt::runtime::Tensor> &outputTensors) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  std::vector<uint64_t> inputGlobalIds;
  inputGlobalIds.reserve(inputTensors.size());
  std::transform(inputTensors.begin(), inputTensors.end(),
                 std::back_inserter(inputGlobalIds),
                 [](const auto &tensor) { return tensor.getGlobalId(); });

  std::vector<uint64_t> outputGlobalIds;
  outputGlobalIds.reserve(outputTensors.size());
  std::transform(outputTensors.begin(), outputTensors.end(),
                 std::back_inserter(outputGlobalIds),
                 [](const auto &tensor) { return tensor.getGlobalId(); });

  std::vector<uint8_t> binaryBytes;
  executable.storeToMemory(binaryBytes);

  auto deviceRef = ::tt::target::CreateDeviceRef(fbb, device.getGlobalId());

  uint64_t commandId = BUILD_COMMAND_DIRECT(
      Submit, fbb, &inputGlobalIds, &outputGlobalIds, executable.id(),
      &binaryBytes, programIndex, deviceRef);

  return commandId;
}

uint64_t
CommandFactory::buildGetNumShardsCommand(::flatbuffers::FlatBufferBuilder &fbb,
                                         const ::tt::runtime::Tensor &tensor) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = BUILD_COMMAND(GetNumShards, fbb, tensor.getGlobalId());

  return commandId;
}

uint64_t CommandFactory::buildToHostCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &inputTensor, bool untilize, bool blocking,
    const std::vector<::tt::runtime::Tensor> &outputTensors) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  std::vector<uint64_t> outputGlobalIds;
  outputGlobalIds.reserve(outputTensors.size());
  std::transform(outputTensors.begin(), outputTensors.end(),
                 std::back_inserter(outputGlobalIds),
                 [](const auto &tensor) { return tensor.getGlobalId(); });

  uint64_t commandId =
      BUILD_COMMAND_DIRECT(ToHost, fbb, inputTensor.getGlobalId(),
                           &outputGlobalIds, untilize, blocking);

  return commandId;
}

uint64_t CommandFactory::buildMemcpyCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &srcTensor,
    const std::optional<::tt::runtime::Tensor> &dstTensor,
    const std::optional<::tt::target::DataType> &dstDataType) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t srcGlobalId = srcTensor.getGlobalId();
  std::optional<uint64_t> dstGlobalId = std::nullopt;
  if (dstTensor.has_value()) {
    dstGlobalId = dstTensor.value().getGlobalId();
  }

  uint64_t commandId =
      BUILD_COMMAND(Memcpy, fbb, srcGlobalId, dstGlobalId, dstDataType);

  return commandId;
}

uint64_t
CommandFactory::buildShutdownCommand(::flatbuffers::FlatBufferBuilder &fbb) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = BUILD_COMMAND(Shutdown, fbb);

  return commandId;
}

#undef BUILD_COMMAND_IMPL
#undef BUILD_COMMAND
#undef BUILD_COMMAND_DIRECT

} // namespace tt::runtime::distributed::controller
