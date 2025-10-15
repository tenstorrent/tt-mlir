// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/distributed/worker/response_factory.h"
#include "tt/runtime/debug.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/distributed/flatbuffer/flatbuffer.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"

namespace tt::runtime::distributed::worker {

using ::tt::runtime::DeviceRuntime;

static constexpr auto verifyFn =
    &::tt::runtime::distributed::flatbuffer::VerifyResponseBuffer;

void ResponseFactory::buildErrorResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                         uint64_t commandId,
                                         const std::string &errorMessage) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::runtime::distributed::flatbuffer::ResponseType::ErrorResponse;

  auto errorResponse =
      ::tt::runtime::distributed::flatbuffer::CreateErrorResponseDirect(
          fbb, errorMessage.c_str());

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, errorResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildGetSystemDescResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const ::tt::runtime::SystemDesc &systemDesc) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType = ::tt::runtime::distributed::flatbuffer::ResponseType::
      GetSystemDescResponse;

  std::vector<uint8_t> systemDescBuffer;
  systemDesc.storeToMemory(systemDescBuffer);

  auto getSystemDescResponse =
      ::tt::runtime::distributed::flatbuffer::CreateGetSystemDescResponseDirect(
          fbb, &systemDescBuffer);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, getSystemDescResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildOpenMeshDeviceResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const ::tt::runtime::Device &device) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto deviceRef = ::tt::target::CreateDeviceRef(fbb, device.getGlobalId());

  auto responseType = ::tt::runtime::distributed::flatbuffer::ResponseType::
      OpenMeshDeviceResponse;

  auto openMeshDeviceResponse =
      ::tt::runtime::distributed::flatbuffer::CreateOpenMeshDeviceResponse(
          fbb, deviceRef);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, openMeshDeviceResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildCloseMeshDeviceResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType = ::tt::runtime::distributed::flatbuffer::ResponseType::
      CloseMeshDeviceResponse;

  auto closeMeshDeviceResponse =
      ::tt::runtime::distributed::flatbuffer::CreateCloseMeshDeviceResponse(
          fbb);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, closeMeshDeviceResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildCreateHostTensorResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType = ::tt::runtime::distributed::flatbuffer::ResponseType::
      CreateHostTensorResponse;

  auto createHostTensorResponse =
      ::tt::runtime::distributed::flatbuffer::CreateCreateHostTensorResponse(
          fbb);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, createHostTensorResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildGetLayoutResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::runtime::distributed::flatbuffer::ResponseType::GetLayoutResponse;

  auto getLayoutResponse =
      ::tt::runtime::distributed::flatbuffer::CreateGetLayoutResponse(fbb);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, getLayoutResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildToLayoutResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::runtime::distributed::flatbuffer::ResponseType::ToLayoutResponse;

  auto toLayoutResponse =
      ::tt::runtime::distributed::flatbuffer::CreateToLayoutResponse(fbb);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, toLayoutResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildSubmitResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                          uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::runtime::distributed::flatbuffer::ResponseType::SubmitResponse;

  auto submitResponse =
      ::tt::runtime::distributed::flatbuffer::CreateSubmitResponse(fbb);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, submitResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildGetNumShardsResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    uint32_t numBuffers) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType = ::tt::runtime::distributed::flatbuffer::ResponseType::
      GetNumShardsResponse;

  auto getNumShardsResponse =
      ::tt::runtime::distributed::flatbuffer::CreateGetNumShardsResponse(
          fbb, numBuffers);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, getNumShardsResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildToHostResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                          uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::runtime::distributed::flatbuffer::ResponseType::ToHostResponse;

  auto toHostResponse =
      ::tt::runtime::distributed::flatbuffer::CreateToHostResponse(fbb);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, toHostResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildMemcpyResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const std::optional<const std::vector<std::uint8_t>> &data) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::runtime::distributed::flatbuffer::ResponseType::MemcpyResponse;

  const std::vector<uint8_t> *dataPtr = nullptr;
  if (data.has_value()) {
    dataPtr = &data.value();
  }

  auto memcpyResponse =
      ::tt::runtime::distributed::flatbuffer::CreateMemcpyResponseDirect(
          fbb, dataPtr);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, memcpyResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

void ResponseFactory::buildShutdownResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::runtime::distributed::flatbuffer::ResponseType::ShutdownResponse;

  auto shutdownResponse =
      ::tt::runtime::distributed::flatbuffer::CreateShutdownResponse(fbb);

  auto response = ::tt::runtime::distributed::flatbuffer::CreateResponse(
      fbb, commandId, responseType, shutdownResponse.Union());
  ::tt::runtime::distributed::flatbuffer::FinishResponseBuffer(fbb, response);

  debug::verifyFlatbuffer(fbb, verifyFn);
}

} // namespace tt::runtime::distributed::worker
