// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_SERVER_COMMAND_FACTORY_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_SERVER_COMMAND_FACTORY_H

#include "flatbuffers/flatbuffers.h"
#include "tt/runtime/types.h"
#include <string_view>

namespace tt::runtime::distributed::server {

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
      ::flatbuffers::FlatBufferBuilder &fbb,
      const ::tt::runtime::Tensor &outputTensor, const void *data,
      const std::vector<uint32_t> &shape, const std::vector<uint32_t> &stride,
      uint32_t itemSize, ::tt::target::DataType dataType);

  static uint64_t
  buildGetLayoutCommand(::flatbuffers::FlatBufferBuilder &fbb,
                        const ::tt::runtime::Binary &binary,
                        uint32_t programIndex, uint32_t inputIndex,
                        const ::tt::runtime::Layout &outputLayout);

  static uint64_t
  buildToLayoutCommand(::flatbuffers::FlatBufferBuilder &fbb,
                       const ::tt::runtime::Tensor &inputTensor,
                       const ::tt::runtime::Device &device,
                       const ::tt::runtime::Layout &layout,
                       const ::tt::runtime::Tensor &outputTensor,
                       std::optional<bool> retain = std::nullopt);

  static uint64_t
  buildSubmitCommand(::flatbuffers::FlatBufferBuilder &fbb,
                     ::tt::runtime::Device device,
                     ::tt::runtime::Binary executable, uint32_t programIndex,
                     const std::vector<::tt::runtime::Tensor> &inputTensors,
                     const std::vector<::tt::runtime::Tensor> &outputTensors);

  static uint64_t
  buildToHostCommand(::flatbuffers::FlatBufferBuilder &fbb,
                     ::tt::runtime::Tensor inputTensor, bool untilize,
                     bool blocking,
                     const std::vector<::tt::runtime::Tensor> &outputTensors);

  static uint64_t buildReadbackCommand(::flatbuffers::FlatBufferBuilder &fbb,
                                       ::tt::runtime::Tensor srcTensor);

  static uint64_t buildMemcpyCommand(
      ::flatbuffers::FlatBufferBuilder &fbb, ::tt::runtime::Tensor srcTensor,
      std::optional<::tt::runtime::Tensor> dstTensor = std::nullopt,
      std::optional<::tt::target::DataType> dstDataType = std::nullopt);

  static uint64_t buildShutdownCommand(::flatbuffers::FlatBufferBuilder &fbb);
};

} // namespace tt::runtime::distributed::server
#endif // TT_RUNTIME_DETAIL_DISTRIBUTED_SERVER_COMMAND_FACTORY_H
