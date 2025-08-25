// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_SERVER_COMMAND_FACTORY_H
#define TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_SERVER_COMMAND_FACTORY_H

#include "flatbuffers/flatbuffers.h"
#include "tt/runtime/types.h"
#include <string_view>

namespace tt::runtime::ttnn::distributed::server {

class CommandFactory {
public:
  static uint64_t buildGetSystemDescCommand(
      ::flatbuffers::FlatBufferBuilder &fbb,
      const ::tt::runtime::DispatchCoreType &dispatchCoreType,
      std::optional<uint32_t> deviceGlobalId = std::nullopt);

  static uint64_t buildOpenMeshDeviceCommand(
      ::flatbuffers::FlatBufferBuilder &fbb,
      const ::tt::runtime::Device &deviceShell,
      const ::tt::runtime::MeshDeviceOptions &meshDeviceOptions);

  static uint64_t
  buildCloseMeshDeviceCommand(::flatbuffers::FlatBufferBuilder &fbb,
                              const ::tt::runtime::Device &deviceShell);

  static uint64_t buildCreateHostTensorCommand(
      ::flatbuffers::FlatBufferBuilder &fbb, uint64_t outputGlobalId,
      const void *data, const std::vector<uint32_t> &shape,
      const std::vector<uint32_t> &stride, uint32_t itemSize,
      ::tt::target::DataType dataType);

  static uint64_t
  buildToLayoutCommand(::flatbuffers::FlatBufferBuilder &fbb,
                       const ::tt::runtime::Tensor &inputTensor,
                       const ::tt::runtime::Device &device,
                       const ::tt::runtime::Layout &layout,
                       const ::tt::runtime::Tensor &outputTensor);

  static uint64_t
  buildSubmitCommand(::flatbuffers::FlatBufferBuilder &fbb,
                     ::tt::runtime::Device device,
                     ::tt::runtime::Binary executable, uint32_t programIndex,
                     const std::vector<::tt::runtime::Tensor> &inputTensors,
                     const std::vector<::tt::runtime::Tensor> &outputTensors);

  static uint64_t buildShutdownCommand(::flatbuffers::FlatBufferBuilder &fbb);
};

} // namespace tt::runtime::ttnn::distributed::server
#endif // TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_SERVER_COMMAND_FACTORY_H
