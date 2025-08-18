// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_TYPES_RESPONSE_FACTORY_H
#define TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_TYPES_RESPONSE_FACTORY_H

#include "flatbuffers/flatbuffers.h"
#include "tt/runtime/types.h"
#include <string_view>

namespace tt::runtime::ttnn::distributed::client {

class ResponseFactory {
public:
  static uint64_t buildErrorResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                     std::string_view errorMessage);

  // The fbb should already have the system desc root buffer built.
  static uint64_t buildGetSystemDescResponse(
      ::flatbuffers::FlatBufferBuilder &fbb,
      const ::flatbuffers::Offset<::tt::target::SystemDescRoot> &systemDesc);

  static uint64_t
  buildOpenMeshDeviceResponse(::flatbuffers::FlatBufferBuilder &fbb,
                              const ::tt::runtime::Device &device);

  static uint64_t
  buildCloseMeshDeviceResponse(::flatbuffers::FlatBufferBuilder &fbb,
                               bool success);

  static uint64_t
  buildCreateHostTensorResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                const ::tt::runtime::Tensor &hostTensor);

  static uint64_t
  buildToLayoutResponse(::flatbuffers::FlatBufferBuilder &fbb,
                        const ::tt::runtime::Tensor &outputTensor);

  static uint64_t
  buildSubmitResponse(::flatbuffers::FlatBufferBuilder &fbb,
                      const std::vector<::tt::runtime::Tensor> &outputTensors);

  static uint64_t buildShutdownResponse(::flatbuffers::FlatBufferBuilder &fbb,
                                        bool success);
};

} // namespace tt::runtime::ttnn::distributed::client
#endif // TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_CLIENT_TYPES_RESPONSE_FACTORY_H
