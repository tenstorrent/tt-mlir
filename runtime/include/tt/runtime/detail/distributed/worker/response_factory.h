// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_WORKER_RESPONSE_FACTORY_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_WORKER_RESPONSE_FACTORY_H

#include "flatbuffers/flatbuffers.h"
#include "tt/runtime/types.h"
#include <string_view>

namespace tt::runtime::distributed::worker {

class ResponseFactory {
public:
  static void buildErrorResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                 uint64_t commandId,
                                 const std::string &errorMessage);

  // The fbb should already have the system desc root buffer built.
  static void
  buildGetSystemDescResponse(::flatbuffers::FlatBufferBuilder &fbb,
                             uint64_t commandId,
                             const ::tt::runtime::SystemDesc &systemDesc);

  static void
  buildSetFabricConfigResponse(::flatbuffers::FlatBufferBuilder &fbb,
                               uint64_t commandId);

  static void
  buildGetNumAvailableDevicesResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                      uint64_t commandId, size_t numDevices);

  static void buildOpenMeshDeviceResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                          uint64_t commandId,
                                          const ::tt::runtime::Device &device);

  static void
  buildCloseMeshDeviceResponse(::flatbuffers::FlatBufferBuilder &fbb,
                               uint64_t commandId);

  static void
  buildCreateSubMeshDeviceResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                   uint64_t commandId,
                                   const ::tt::runtime::Device &subMesh);

  static void
  buildReleaseSubMeshDeviceResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                    uint64_t commandId);

  static void buildGetMeshShapeResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                        uint64_t commandId,
                                        const std::vector<uint32_t> &shape);

  static void
  buildCreateHostTensorResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                uint64_t commandId);

  static void
  buildGetTensorVolumeResponse(::flatbuffers::FlatBufferBuilder &fbb,
                               uint64_t commandId, uint32_t volume);

  static void
  buildGetTensorRetainResponse(::flatbuffers::FlatBufferBuilder &fbb,
                               uint64_t commandId, bool retain);

  static void
  buildSetTensorRetainResponse(::flatbuffers::FlatBufferBuilder &fbb,
                               uint64_t commandId);

  static void buildGetLayoutResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                     uint64_t commandId);

  static void buildToLayoutResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                    uint64_t commandId);

  static void buildSubmitResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                  uint64_t commandId);

  static void buildGetNumShardsResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                        uint64_t commandId,
                                        uint32_t numBuffers);

  static void buildToHostResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                  uint64_t commandId);

  static void buildMemcpyResponse(
      ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
      const std::optional<const std::vector<std::uint8_t>> &data);

  static void buildShutdownResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                    uint64_t commandId);
};

} // namespace tt::runtime::distributed::worker
#endif // TT_RUNTIME_DETAIL_DISTRIBUTED_WORKER_RESPONSE_FACTORY_H
