// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/worker/response_factory.h"
#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

#define BUILD_RESPONSE_IMPL(ResponseName, fbb, commandId, builderFunc, ...)    \
  do {                                                                         \
    auto responseType = fb::ResponseType::ResponseName##Response;              \
                                                                               \
    auto responseOffset = (builderFunc)((fbb)__VA_OPT__(, ) __VA_ARGS__);      \
                                                                               \
    auto response = fb::CreateResponse((fbb), (commandId), responseType,       \
                                       responseOffset.Union());                \
                                                                               \
    fb::FinishResponseBuffer((fbb), response);                                 \
                                                                               \
    debug::verifyFlatbuffer((fbb), verifyFn);                                  \
  } while (0)

#define BUILD_RESPONSE(ResponseName, fbb, commandId, ...)                      \
  BUILD_RESPONSE_IMPL(ResponseName, fbb, commandId,                            \
                      fb::Create##ResponseName##Response, __VA_ARGS__)

#define BUILD_RESPONSE_DIRECT(ResponseName, fbb, commandId, ...)               \
  BUILD_RESPONSE_IMPL(ResponseName, fbb, commandId,                            \
                      fb::Create##ResponseName##ResponseDirect, __VA_ARGS__)

namespace tt::runtime::distributed::worker {

using ::tt::runtime::DeviceRuntime;

namespace fb = ::tt::runtime::distributed::flatbuffer;

static constexpr auto verifyFn = &fb::VerifyResponseBuffer;

void ResponseFactory::buildErrorResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                         uint64_t commandId,
                                         const std::string &errorMessage) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE_DIRECT(Error, fbb, commandId, errorMessage.c_str());
}

void ResponseFactory::buildSetMemoryLogLevelResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(SetMemoryLogLevel, fbb, commandId);
}

void ResponseFactory::buildConfigureRuntimeContextResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(ConfigureRuntimeContext, fbb, commandId);
}

void ResponseFactory::buildGetSystemDescResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const ::tt::runtime::SystemDesc &systemDesc) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  std::vector<uint8_t> systemDescBuffer;
  systemDesc.storeToMemory(systemDescBuffer);

  BUILD_RESPONSE_DIRECT(GetSystemDesc, fbb, commandId, &systemDescBuffer);
}

void ResponseFactory::buildSetFabricConfigResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(SetFabricConfig, fbb, commandId);
}

void ResponseFactory::buildGetNumAvailableDevicesResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    size_t numDevices) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(GetNumAvailableDevices, fbb, commandId, numDevices);
}

void ResponseFactory::buildOpenMeshDeviceResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const ::tt::runtime::Device &device) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto deviceRef = ::tt::target::CreateDeviceRef(fbb, device.getGlobalId());

  BUILD_RESPONSE(OpenMeshDevice, fbb, commandId, deviceRef);
}

void ResponseFactory::buildCloseMeshDeviceResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(CloseMeshDevice, fbb, commandId);
}

void ResponseFactory::buildCreateSubMeshDeviceResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const ::tt::runtime::Device &subMesh) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto subMeshRef = ::tt::target::CreateDeviceRef(fbb, subMesh.getGlobalId());
  BUILD_RESPONSE(CreateSubMeshDevice, fbb, commandId, subMeshRef);
}

void ResponseFactory::buildReleaseSubMeshDeviceResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(ReleaseSubMeshDevice, fbb, commandId);
}

void ResponseFactory::buildGetMeshShapeResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const std::vector<uint32_t> &shape) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE_DIRECT(GetMeshShape, fbb, commandId, &shape);
}

void ResponseFactory::buildCreateHostTensorResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(CreateHostTensor, fbb, commandId);
}

void ResponseFactory::buildCreateMultiDeviceHostTensorFromShardsResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(CreateMultiDeviceHostTensorFromShards, fbb, commandId);
}

void ResponseFactory::buildIsTensorAllocatedResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId, bool allocated) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(IsTensorAllocated, fbb, commandId, allocated);
}

void ResponseFactory::buildGetTensorVolumeResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    uint32_t volume) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(GetTensorVolume, fbb, commandId, volume);
}

void ResponseFactory::buildGetTensorRetainResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId, bool retain) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(GetTensorRetain, fbb, commandId, retain);
}

void ResponseFactory::buildSetTensorRetainResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(SetTensorRetain, fbb, commandId);
}

void ResponseFactory::buildGetLayoutResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(GetLayout, fbb, commandId);
}

void ResponseFactory::buildToLayoutResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(ToLayout, fbb, commandId);
}

void ResponseFactory::buildSubmitResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                          uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(Submit, fbb, commandId);
}

void ResponseFactory::buildGetNumShardsResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    uint32_t numBuffers) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(GetNumShards, fbb, commandId, numBuffers);
}

void ResponseFactory::buildToHostResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                          uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(ToHost, fbb, commandId);
}

void ResponseFactory::buildMemcpyResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const std::optional<const std::vector<std::uint8_t>> &data) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  const std::vector<uint8_t> *dataPtr = nullptr;
  if (data.has_value()) {
    dataPtr = &data.value();
  }

  BUILD_RESPONSE_DIRECT(Memcpy, fbb, commandId, dataPtr);
}

void ResponseFactory::buildDeallocateTensorResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(DeallocateTensor, fbb, commandId);
}

void ResponseFactory::buildShutdownResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(Shutdown, fbb, commandId);
}

void ResponseFactory::buildGetTensorDescResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const ::tt::runtime::TensorDesc &tensorDesc) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  // Create FlatBuffer vectors for shape and stride
  auto shapeOffset = fbb.CreateVector(tensorDesc.shape);
  auto strideOffset = fbb.CreateVector(tensorDesc.stride);

  // Create TensorDescFlat FlatBuffer object
  auto tensorDescFlatOffset = fb::CreateTensorDescFlat(
      fbb, shapeOffset, tensorDesc.dataType, tensorDesc.itemsize, strideOffset,
      tensorDesc.physicalVolume);

  BUILD_RESPONSE(GetTensorDesc, fbb, commandId, tensorDescFlatOffset);
}

void ResponseFactory::buildHasLayoutResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId, bool hasLayout) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(HasLayout, fbb, commandId, hasLayout);
}

void ResponseFactory::buildIsProgramCacheEnabledResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId, bool isEnabled) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(IsProgramCacheEnabled, fbb, commandId, isEnabled);
}

void ResponseFactory::buildClearProgramCacheResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  BUILD_RESPONSE(ClearProgramCache, fbb, commandId);
}


void ResponseFactory::buildComputeMeshFabricConfigResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const ::tt::runtime::MeshFabricConfig &fabricConfig) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto perAxisVec = fbb.CreateVector(
      reinterpret_cast<const ::tt::runtime::FabricConfig *>(
          fabricConfig.perAxisConfig.data()),
      fabricConfig.perAxisConfig.size());

  auto responseType = fb::ResponseType::ComputeMeshFabricConfigResponse;
  auto responseOffset = fb::CreateComputeMeshFabricConfigResponse(
      fbb, fabricConfig.globalConfig, perAxisVec);
  auto response = fb::CreateResponse(fbb, commandId, responseType,
                                     responseOffset.Union());
  fb::FinishResponseBuffer(fbb, response);
  debug::verifyFlatbuffer(fbb, verifyFn);
}

#undef BUILD_RESPONSE_IMPL
#undef BUILD_RESPONSE
#undef BUILD_RESPONSE_DIRECT

} // namespace tt::runtime::distributed::worker
