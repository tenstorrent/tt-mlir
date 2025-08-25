// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/distributed/server/command_factory.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::distributed::server {

using ::tt::runtime::DeviceRuntime;

static uint64_t nextCommandId() {
  static std::atomic<uint64_t> commandIdCounter = 0;
  return commandIdCounter.fetch_add(1, std::memory_order_relaxed);
}

static void verifyCommand(const ::flatbuffers::FlatBufferBuilder &fbb) {
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  bool verified =
      ::tt::target::ttnn::distributed::VerifyCommandBuffer(verifier);
  LOG_ASSERT(verified, "Failed to verify Command");
}

uint64_t CommandFactory::buildGetSystemDescCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::DispatchCoreType &dispatchCoreType,
    std::optional<uint32_t> deviceGlobalId) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::target::ttnn::distributed::CommandType::GetSystemDescCommand;

  ::flatbuffers::Offset<::tt::target::DeviceRef> deviceRef = 0;
  if (deviceGlobalId.has_value()) {
    deviceRef = ::tt::target::CreateDeviceRef(fbb, deviceGlobalId.value());
  }

  auto getSystemDescCommand =
      ::tt::target::ttnn::distributed::CreateGetSystemDescCommand(
          fbb, deviceRef,
          ::tt::runtime::utils::fromRuntimeDispatchCoreType(dispatchCoreType));

  auto command = ::tt::target::ttnn::distributed::CreateCommand(
      fbb, commandId, commandType, getSystemDescCommand.Union());
  fbb.Finish(command);

  verifyCommand(fbb);

  return commandId;
}

uint64_t CommandFactory::buildOpenMeshDeviceCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Device &deviceShell,
    const ::tt::runtime::MeshDeviceOptions &meshDeviceOptions) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  std::optional<::tt::target::DispatchCoreType> fbDispatchCoreType =
      std::nullopt;
  if (meshDeviceOptions.dispatchCoreType.has_value()) {
    fbDispatchCoreType = ::tt::runtime::utils::fromRuntimeDispatchCoreType(
        meshDeviceOptions.dispatchCoreType.value());
  }

  auto fbMeshDeviceOptions =
      ::tt::target::ttnn::distributed::CreateMeshDeviceOptionsDirect(
          fbb, &meshDeviceOptions.meshOffset, &meshDeviceOptions.deviceIds,
          meshDeviceOptions.numHWCQs, meshDeviceOptions.enableProgramCache,
          meshDeviceOptions.meshShape.has_value()
              ? &(meshDeviceOptions.meshShape.value())
              : nullptr,
          meshDeviceOptions.l1SmallSize, meshDeviceOptions.traceRegionSize,
          fbDispatchCoreType);

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::target::ttnn::distributed::CommandType::OpenMeshDeviceCommand;

  auto openMeshDeviceCommand =
      ::tt::target::ttnn::distributed::CreateOpenMeshDeviceCommand(
          fbb, deviceShell.getGlobalId(), fbMeshDeviceOptions);

  auto command = ::tt::target::ttnn::distributed::CreateCommand(
      fbb, commandId, commandType, openMeshDeviceCommand.Union());
  fbb.Finish(command);

  verifyCommand(fbb);

  return commandId;
}

uint64_t CommandFactory::buildCloseMeshDeviceCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Device &deviceShell) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::target::ttnn::distributed::CommandType::CloseMeshDeviceCommand;

  auto closeMeshDeviceCommand =
      ::tt::target::ttnn::distributed::CreateCloseMeshDeviceCommand(
          fbb, deviceShell.getGlobalId());

  auto command = ::tt::target::ttnn::distributed::CreateCommand(
      fbb, commandId, commandType, closeMeshDeviceCommand.Union());
  fbb.Finish(command);

  verifyCommand(fbb);

  return commandId;
}

uint64_t CommandFactory::buildCreateHostTensorCommand(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t outputGlobalId,
    const void *data, const std::vector<uint32_t> &shape,
    const std::vector<uint32_t> &stride, uint32_t itemSize,
    ::tt::target::DataType dataType) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::target::ttnn::distributed::CommandType::CreateHostTensorCommand;

  std::uint64_t numElements =
      std::accumulate(shape.begin(), shape.end(), static_cast<std::uint64_t>(1),
                      std::multiplies<std::uint64_t>());
  std::uint64_t numBytes = numElements * itemSize;
  auto dataVec =
      fbb.CreateVector<uint8_t>(static_cast<const uint8_t *>(data), numBytes);
  auto shapeVec = fbb.CreateVector<uint32_t>(shape.data(), shape.size());
  auto strideVec = fbb.CreateVector<uint32_t>(stride.data(), stride.size());

  auto createHostTensorCommand =
      ::tt::target::ttnn::distributed::CreateCreateHostTensorCommand(
          fbb, outputGlobalId, dataVec, shapeVec, strideVec, itemSize,
          dataType);

  auto command = ::tt::target::ttnn::distributed::CreateCommand(
      fbb, commandId, commandType, createHostTensorCommand.Union());
  fbb.Finish(command);

  verifyCommand(fbb);

  return commandId;
}

uint64_t CommandFactory::buildToLayoutCommand(
    ::flatbuffers::FlatBufferBuilder &fbb,
    const ::tt::runtime::Tensor &inputTensor,
    const ::tt::runtime::Device &device, const ::tt::runtime::Layout &layout,
    const ::tt::runtime::Tensor &outputTensor) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::target::ttnn::distributed::CommandType::ToLayoutCommand;

  uint64_t inputGlobalId = inputTensor.getGlobalId();
  uint64_t outputGlobalId = outputTensor.getGlobalId();

  const ::tt::runtime::ttnn::LayoutDesc &layoutDesc =
      layout.as<::tt::runtime::ttnn::LayoutDesc>(DeviceRuntime::TTNN);

  auto memoryDesc = layoutDesc.toMemoryDesc(fbb);

  auto deviceRef = ::tt::target::CreateDeviceRef(fbb, device.getGlobalId());

  auto toLayoutCommand = ::tt::target::ttnn::distributed::CreateToLayoutCommand(
      fbb, inputGlobalId, outputGlobalId, deviceRef, memoryDesc);

  auto command = ::tt::target::ttnn::distributed::CreateCommand(
      fbb, commandId, commandType, toLayoutCommand.Union());
  fbb.Finish(command);

  verifyCommand(fbb);

  return commandId;
}

uint64_t
buildSubmitCommand(::flatbuffers::FlatBufferBuilder &fbb,
                   ::tt::runtime::Device device,
                   ::tt::runtime::Binary executable, uint32_t programIndex,
                   const std::vector<::tt::runtime::Tensor> &inputTensors,
                   const std::vector<::tt::runtime::Tensor> &outputTensors) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::target::ttnn::distributed::CommandType::SubmitCommand;

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

  std::vector<uint8_t> ttnnBinaryBytes;
  executable.storeToMemory(ttnnBinaryBytes);

  auto deviceRef = ::tt::target::CreateDeviceRef(fbb, device.getGlobalId());

  auto submitCommand =
      ::tt::target::ttnn::distributed::CreateSubmitCommandDirect(
          fbb, &inputGlobalIds, &outputGlobalIds, &ttnnBinaryBytes,
          programIndex, deviceRef);

  auto command = ::tt::target::ttnn::distributed::CreateCommand(
      fbb, commandId, commandType, submitCommand.Union());
  fbb.Finish(command);

  verifyCommand(fbb);

  return commandId;
}

uint64_t
CommandFactory::buildShutdownCommand(::flatbuffers::FlatBufferBuilder &fbb) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  uint64_t commandId = nextCommandId();
  auto commandType =
      ::tt::target::ttnn::distributed::CommandType::ShutdownCommand;

  auto shutdownCommand =
      ::tt::target::ttnn::distributed::CreateShutdownCommand(fbb);

  auto command = ::tt::target::ttnn::distributed::CreateCommand(
      fbb, commandId, commandType, shutdownCommand.Union());
  fbb.Finish(command);

  verifyCommand(fbb);

  return commandId;
}

} // namespace tt::runtime::ttnn::distributed::server
