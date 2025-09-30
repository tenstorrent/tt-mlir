// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/controller/command_factory.h"
#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/types.h"
#include <atomic>

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

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::runtime::distributed::flatbuffer::CommandType::GetSystemDescCommand;

  ::flatbuffers::Offset<::tt::target::DeviceRef> deviceRef = 0;
  if (deviceHandle.has_value()) {
    deviceRef =
        ::tt::target::CreateDeviceRef(fbb, deviceHandle.value().getGlobalId());
  }

  std::optional<::tt::target::DispatchCoreType> fbDispatchCoreType =
      std::nullopt;
  if (dispatchCoreType.has_value()) {
    fbDispatchCoreType = ::tt::runtime::utils::fromRuntimeDispatchCoreType(
        dispatchCoreType.value());
  }

  auto getSystemDescCommand =
      ::tt::runtime::distributed::flatbuffer::CreateGetSystemDescCommand(
          fbb, deviceRef, fbDispatchCoreType);

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, getSystemDescCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

uint64_t CommandFactory::buildOpenMeshDeviceCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Device &deviceHandle,
    const ::tt::runtime::MeshDeviceOptions &meshDeviceOptions) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  std::optional<::tt::target::DispatchCoreType> fbDispatchCoreType =
      std::nullopt;
  if (meshDeviceOptions.dispatchCoreType.has_value()) {
    fbDispatchCoreType = ::tt::runtime::utils::fromRuntimeDispatchCoreType(
        meshDeviceOptions.dispatchCoreType.value());
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

  uint64_t commandId = nextCommandId();
  auto commandType = ::tt::runtime::distributed::flatbuffer::CommandType::
      OpenMeshDeviceCommand;

  auto openMeshDeviceCommand =
      ::tt::runtime::distributed::flatbuffer::CreateOpenMeshDeviceCommand(
          fbb, deviceHandle.getGlobalId(), fbMeshDeviceOptions);

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, openMeshDeviceCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

uint64_t CommandFactory::buildCloseMeshDeviceCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Device &deviceHandle) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType = ::tt::runtime::distributed::flatbuffer::CommandType::
      CloseMeshDeviceCommand;

  auto deviceRef =
      ::tt::target::CreateDeviceRef(fbb, deviceHandle.getGlobalId());

  auto closeMeshDeviceCommand =
      ::tt::runtime::distributed::flatbuffer::CreateCloseMeshDeviceCommand(
          fbb, deviceRef);

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, closeMeshDeviceCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

uint64_t CommandFactory::buildCreateHostTensorCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &outputTensor, const void *data,
    const std::vector<uint32_t> &shape, const std::vector<uint32_t> &stride,
    uint32_t itemSize, ::tt::target::DataType dataType) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType = ::tt::runtime::distributed::flatbuffer::CommandType::
      CreateHostTensorCommand;

  std::uint64_t numElements =
      std::accumulate(shape.begin(), shape.end(), static_cast<std::uint64_t>(1),
                      std::multiplies<std::uint64_t>());
  std::uint64_t numBytes = numElements * itemSize;
  auto dataVec =
      fbb.CreateVector<uint8_t>(static_cast<const uint8_t *>(data), numBytes);
  auto shapeVec = fbb.CreateVector<uint32_t>(shape.data(), shape.size());
  auto strideVec = fbb.CreateVector<uint32_t>(stride.data(), stride.size());

  auto createHostTensorCommand =
      ::tt::runtime::distributed::flatbuffer::CreateCreateHostTensorCommand(
          fbb, outputTensor.getGlobalId(), dataVec, shapeVec, strideVec,
          itemSize, dataType);

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, createHostTensorCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

uint64_t CommandFactory::buildGetLayoutCommand(
    ::flatbuffers::FlatBufferBuilder &fbb, const ::tt::runtime::Binary &binary,
    uint32_t programIndex, uint32_t inputIndex,
    const ::tt::runtime::Layout &outputLayout) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::runtime::distributed::flatbuffer::CommandType::GetLayoutCommand;

  std::vector<uint8_t> binaryBytes;
  binary.storeToMemory(binaryBytes);

  auto getLayoutCommand =
      ::tt::runtime::distributed::flatbuffer::CreateGetLayoutCommandDirect(
          fbb, binary.id(), &binaryBytes, programIndex, inputIndex,
          outputLayout.getGlobalId());

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, getLayoutCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

uint64_t CommandFactory::buildToLayoutCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &inputTensor,
    const ::tt::runtime::Device &device, const ::tt::runtime::Layout &layout,
    const ::tt::runtime::Tensor &outputTensor, std::optional<bool> retain) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::runtime::distributed::flatbuffer::CommandType::ToLayoutCommand;

  uint64_t inputGlobalId = inputTensor.getGlobalId();
  uint64_t outputGlobalId = outputTensor.getGlobalId();

  auto deviceRef = ::tt::target::CreateDeviceRef(fbb, device.getGlobalId());

  auto toLayoutCommand =
      ::tt::runtime::distributed::flatbuffer::CreateToLayoutCommand(
          fbb, inputGlobalId, outputGlobalId, deviceRef, layout.getGlobalId(),
          retain);

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, toLayoutCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

uint64_t CommandFactory::buildSubmitCommand(
    ::flatbuffers::FlatBufferBuilder &fbb, const ::tt::runtime::Device &device,
    const ::tt::runtime::Binary &executable, uint32_t programIndex,
    const std::vector<::tt::runtime::Tensor> &inputTensors,
    const std::vector<::tt::runtime::Tensor> &outputTensors) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::runtime::distributed::flatbuffer::CommandType::SubmitCommand;

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

  auto submitCommand =
      ::tt::runtime::distributed::flatbuffer::CreateSubmitCommandDirect(
          fbb, &inputGlobalIds, &outputGlobalIds, executable.id(), &binaryBytes,
          programIndex, deviceRef);

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, submitCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

uint64_t
CommandFactory::buildGetNumShardsCommand(::flatbuffers::FlatBufferBuilder &fbb,
                                         const ::tt::runtime::Tensor &tensor) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::runtime::distributed::flatbuffer::CommandType::GetNumShardsCommand;

  auto getNumShardsCommand =
      ::tt::runtime::distributed::flatbuffer::CreateGetNumShardsCommand(
          fbb, tensor.getGlobalId());

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, getNumShardsCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

uint64_t CommandFactory::buildToHostCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &inputTensor, bool untilize, bool blocking,
    const std::vector<::tt::runtime::Tensor> &outputTensors) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::runtime::distributed::flatbuffer::CommandType::ToHostCommand;

  std::vector<uint64_t> outputGlobalIds;
  outputGlobalIds.reserve(outputTensors.size());
  std::transform(outputTensors.begin(), outputTensors.end(),
                 std::back_inserter(outputGlobalIds),
                 [](const auto &tensor) { return tensor.getGlobalId(); });

  auto toHostCommand =
      ::tt::runtime::distributed::flatbuffer::CreateToHostCommandDirect(
          fbb, inputTensor.getGlobalId(), &outputGlobalIds, untilize, blocking);

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, toHostCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

uint64_t CommandFactory::buildMemcpyCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &srcTensor,
    const std::optional<::tt::runtime::Tensor> &dstTensor,
    const std::optional<::tt::target::DataType> &dstDataType) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::runtime::distributed::flatbuffer::CommandType::MemcpyCommand;

  uint64_t srcGlobalId = srcTensor.getGlobalId();
  std::optional<uint64_t> dstGlobalId = std::nullopt;
  if (dstTensor.has_value()) {
    dstGlobalId = dstTensor.value().getGlobalId();
  }

  auto memcpyCommand =
      ::tt::runtime::distributed::flatbuffer::CreateMemcpyCommand(
          fbb, srcGlobalId, dstGlobalId, dstDataType);

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, memcpyCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

uint64_t
CommandFactory::buildShutdownCommand(::flatbuffers::FlatBufferBuilder &fbb) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::runtime::distributed::flatbuffer::CommandType::ShutdownCommand;

  auto shutdownCommand =
      ::tt::runtime::distributed::flatbuffer::CreateShutdownCommand(fbb);

  auto command = ::tt::runtime::distributed::flatbuffer::CreateCommand(
      fbb, commandId, commandType, shutdownCommand.Union());
  ::tt::runtime::distributed::flatbuffer::FinishCommandBuffer(fbb, command);

  debug::verifyFlatbuffer(fbb, verifyFn);

  return commandId;
}

} // namespace tt::runtime::distributed::controller
