// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_RESPONSE_FACTORY_H
#define TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_RESPONSE_FACTORY_H

#include "flatbuffers/flatbuffers.h"
#include "tt/runtime/types.h"
#include <string_view>

namespace tt::runtime::ttnn::distributed::client {

class ResponseFactory {
public:
  static void buildErrorResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                 uint64_t commandId,
                                 std::string_view errorMessage);

  // The fbb should already have the system desc root buffer built.
  static void buildGetSystemDescResponse(
      ::flatbuffers::FlatBufferBuilder &fbb, uint64_t commandId,
      ::flatbuffers::Offset<::tt::target::SystemDescRoot> systemDesc);

  static void buildOpenMeshDeviceResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                          uint64_t commandId,
                                          const ::tt::runtime::Device &device);

  static void
  buildCloseMeshDeviceResponse(::flatbuffers::FlatBufferBuilder &fbb,
                               uint64_t commandId, bool success);

  static void
  buildCreateHostTensorResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                uint64_t commandId, bool success);

  static void buildToLayoutResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                    uint64_t commandId, bool success);

  static void buildSubmitResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                  uint64_t commandId, bool success);

  static void buildShutdownResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                    uint64_t commandId, bool success);
};

} // namespace tt::runtime::ttnn::distributed::client
#endif // TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_RESPONSE_FACTORY_H
