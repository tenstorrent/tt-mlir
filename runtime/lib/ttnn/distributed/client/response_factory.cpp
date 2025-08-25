// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/detail/ttnn/distributed/client/response_factory.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn::distributed::client {

using ::tt::runtime::DeviceRuntime;

static void verifyResponse(const ::flatbuffers::FlatBufferBuilder &fbb) {
  ::flatbuffers::Verifier verifier(fbb.GetBufferPointer(), fbb.GetSize());
  LOG_ASSERT(::tt::target::ttnn::distributed::VerifyResponseBuffer(verifier),
             "Failed to verify Response");
}

void ResponseFactory::buildErrorResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                         uint64_t commandId,
                                         std::string_view errorMessage) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::target::ttnn::distributed::ResponseType::ErrorResponse;

  auto errorResponse =
      ::tt::target::ttnn::distributed::CreateErrorResponseDirect(
          fbb, errorMessage.data());

  auto response = ::tt::target::ttnn::distributed::CreateResponse(
      fbb, commandId, responseType, errorResponse.Union());
  fbb.Finish(response);

  verifyResponse(fbb);
}

void ResponseFactory::buildGetSystemDescResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    ::flatbuffers::Offset<::tt::target::SystemDescRoot> systemDesc) {

  LOG_ASSERT(fbb.GetSize() > 0,
             "Flatbuffer builder should have system desc root buffer");

  auto responseType =
      ::tt::target::ttnn::distributed::ResponseType::GetSystemDescResponse;

  auto getSystemDescResponse =
      ::tt::target::ttnn::distributed::CreateGetSystemDescResponse(fbb,
                                                                   systemDesc);

  auto response = ::tt::target::ttnn::distributed::CreateResponse(
      fbb, commandId, responseType, getSystemDescResponse.Union());
  fbb.Finish(response);

  verifyResponse(fbb);
}

void ResponseFactory::buildOpenMeshDeviceResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
    const ::tt::runtime::Device &device) {

  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  std::vector<uint32_t> meshShape = ::tt::runtime::ttnn::getMeshShape(device);
  std::vector<int> deviceIds = ::tt::runtime::ttnn::getDeviceIds(device);
  size_t numHwCqs = ::tt::runtime::ttnn::getNumHwCqs(device);
  bool programCacheEnabled = ::tt::runtime::ttnn::isProgramCacheEnabled(device);
  size_t l1SmallSize = ::tt::runtime::ttnn::getL1SmallSize(device);
  size_t traceRegionSize = ::tt::runtime::ttnn::getTraceRegionSize(device);
  size_t numDramChannels = ::tt::runtime::ttnn::getNumDramChannels(device);
  size_t dramSizePerChannel =
      ::tt::runtime::ttnn::getDramSizePerChannel(device);
  size_t l1SizePerCore = ::tt::runtime::ttnn::getL1SizePerCore(device);

  auto meshDeviceDesc =
      ::tt::target::ttnn::distributed::CreateMeshDeviceDescDirect(
          fbb, &meshShape, &deviceIds, numHwCqs, programCacheEnabled,
          l1SmallSize, traceRegionSize, numDramChannels, dramSizePerChannel,
          l1SizePerCore);

  auto deviceRef = ::tt::target::CreateDeviceRef(fbb, device.getGlobalId());

  auto responseType =
      ::tt::target::ttnn::distributed::ResponseType::OpenMeshDeviceResponse;

  auto openMeshDeviceResponse =
      ::tt::target::ttnn::distributed::CreateOpenMeshDeviceResponse(
          fbb, deviceRef, meshDeviceDesc);

  auto response = ::tt::target::ttnn::distributed::CreateResponse(
      fbb, commandId, responseType, openMeshDeviceResponse.Union());
  fbb.Finish(response);

  verifyResponse(fbb);
}

void ResponseFactory::buildCloseMeshDeviceResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId, bool success) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::target::ttnn::distributed::ResponseType::CloseMeshDeviceResponse;

  auto closeMeshDeviceResponse =
      ::tt::target::ttnn::distributed::CreateCloseMeshDeviceResponse(fbb,
                                                                     success);

  auto response = ::tt::target::ttnn::distributed::CreateResponse(
      fbb, commandId, responseType, closeMeshDeviceResponse.Union());
  fbb.Finish(response);

  verifyResponse(fbb);
}

void ResponseFactory::buildCreateHostTensorResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId, bool success) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::target::ttnn::distributed::ResponseType::CreateHostTensorResponse;

  auto createHostTensorResponse =
      ::tt::target::ttnn::distributed::CreateCreateHostTensorResponse(fbb,
                                                                      success);

  auto response = ::tt::target::ttnn::distributed::CreateResponse(
      fbb, commandId, responseType, createHostTensorResponse.Union());
  fbb.Finish(response);

  verifyResponse(fbb);
}

void ResponseFactory::buildToLayoutResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId, bool success) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::target::ttnn::distributed::ResponseType::ToLayoutResponse;

  auto toLayoutResponse =
      ::tt::target::ttnn::distributed::CreateToLayoutResponse(fbb, success);

  auto response = ::tt::target::ttnn::distributed::CreateResponse(
      fbb, commandId, responseType, toLayoutResponse.Union());
  fbb.Finish(response);

  verifyResponse(fbb);
}

void ResponseFactory::buildSubmitResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                          uint64_t commandId, bool success) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::target::ttnn::distributed::ResponseType::SubmitResponse;

  auto submitResponse =
      ::tt::target::ttnn::distributed::CreateSubmitResponse(fbb, success);

  auto response = ::tt::target::ttnn::distributed::CreateResponse(
      fbb, commandId, responseType, submitResponse.Union());
  fbb.Finish(response);

  verifyResponse(fbb);
}

void ResponseFactory::buildShutdownResponse(
    ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId, bool success) {
  LOG_ASSERT(fbb.GetSize() == 0, "Flatbuffer builder must be empty");

  auto responseType =
      ::tt::target::ttnn::distributed::ResponseType::ShutdownResponse;

  auto shutdownResponse =
      ::tt::target::ttnn::distributed::CreateShutdownResponse(fbb, success);

  auto response = ::tt::target::ttnn::distributed::CreateResponse(
      fbb, commandId, responseType, shutdownResponse.Union());
  fbb.Finish(response);

  verifyResponse(fbb);
}

} // namespace tt::runtime::ttnn::distributed::client
