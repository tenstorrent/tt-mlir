// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_CONTROLLER_COMMAND_FACTORY_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_CONTROLLER_COMMAND_FACTORY_H

#include "flatbuffers/flatbuffers.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/types.h"
#include <string_view>

namespace tt::runtime::distributed::controller {

class CommandFactory {
public:
  static uint64_t buildConfigureRuntimeContextCommand(
      ::flatbuffers::FlatBufferBuilder &fbb, const std::string &mlirHome,
      const std::string &metalHome,
      const ::tt::runtime::DeviceRuntime &currentDeviceRuntime);

  static uint64_t buildGetSystemDescCommand(
      ::flatbuffers::FlatBufferBuilder &fbb,
      const std::optional<::tt::runtime::DispatchCoreType> &dispatchCoreType =
          std::nullopt,
      const std::optional<::tt::runtime::Device> &deviceHandle = std::nullopt);

  static uint64_t
  buildSetFabricConfigCommand(::flatbuffers::FlatBufferBuilder &fbb,
                              const ::tt::runtime::FabricConfig &fabricConfig);

  static uint64_t
  buildGetNumAvailableDevicesCommand(::flatbuffers::FlatBufferBuilder &fbb);

  static uint64_t buildOpenMeshDeviceCommand(
      ::flatbuffers::FlatBufferBuilder &fbb,
      const ::tt::runtime::Device &deviceHandle,
      const ::tt::runtime::MeshDeviceOptions &meshDeviceOptions);

  static uint64_t
  buildCloseMeshDeviceCommand(::flatbuffers::FlatBufferBuilder &fbb,
                              const ::tt::runtime::Device &deviceHandle);

  static uint64_t buildCreateSubMeshDeviceCommand(
      ::flatbuffers::FlatBufferBuilder &fbb,
      const ::tt::runtime::Device &parentMesh,
      const ::tt::runtime::Device &subMesh,
      const std::vector<uint32_t> &meshShape,
      const std::optional<const std::vector<uint32_t>> &meshOffset =
          std::nullopt);

  static uint64_t
  buildReleaseSubMeshDeviceCommand(::flatbuffers::FlatBufferBuilder &fbb,
                                   const ::tt::runtime::Device &subMesh);

  static uint64_t
  buildGetMeshShapeCommand(::flatbuffers::FlatBufferBuilder &fbb,
                           const ::tt::runtime::Device &deviceHandle);

  static uint64_t buildCreateHostTensorCommand(
      ::flatbuffers::FlatBufferBuilder &fbb,
      const ::tt::runtime::Tensor &outputTensor, const void *data,
      const std::vector<uint32_t> &shape, const std::vector<uint32_t> &stride,
      uint32_t itemSize, ::tt::target::DataType dataType);

  static uint64_t buildCreateMultiDeviceHostTensorFromShardsCommand(
      ::flatbuffers::FlatBufferBuilder &fbb,
      const std::vector<::tt::runtime::Tensor> &inputTensors,
      const ::tt::runtime::Tensor &outputTensor,
      const std::unordered_map<std::string, std::string> &strategy,
      const std::vector<uint32_t> &meshShape);

  static uint64_t
  buildIsTensorAllocatedCommand(::flatbuffers::FlatBufferBuilder &fbb,
                                const ::tt::runtime::Tensor &tensor);

  static uint64_t
  buildGetTensorVolumeCommand(::flatbuffers::FlatBufferBuilder &fbb,
                              const ::tt::runtime::Tensor &tensor);

  static uint64_t
  buildGetTensorRetainCommand(::flatbuffers::FlatBufferBuilder &fbb,
                              const ::tt::runtime::Tensor &tensor);

  static uint64_t
  buildSetTensorRetainCommand(::flatbuffers::FlatBufferBuilder &fbb,
                              const ::tt::runtime::Tensor &tensor, bool retain);

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
                     const ::tt::runtime::Device &device,
                     const ::tt::runtime::Binary &executable,
                     uint32_t programIndex,
                     const std::vector<::tt::runtime::Tensor> &inputTensors,
                     const std::vector<::tt::runtime::Tensor> &outputTensors);

  static uint64_t
  buildGetNumShardsCommand(::flatbuffers::FlatBufferBuilder &fbb,
                           const ::tt::runtime::Tensor &tensor);

  static uint64_t
  buildToHostCommand(::flatbuffers::FlatBufferBuilder &fbb,
                     const ::tt::runtime::Tensor &inputTensor, bool untilize,
                     bool blocking,
                     const std::vector<::tt::runtime::Tensor> &outputTensors);

  static uint64_t buildMemcpyCommand(
      ::flatbuffers::FlatBufferBuilder &fbb,
      const ::tt::runtime::Tensor &srcTensor,
      const std::optional<::tt::runtime::Tensor> &dstTensor = std::nullopt,
      const std::optional<::tt::target::DataType> &dstDataType = std::nullopt);

  static uint64_t
  buildDeallocateTensorCommand(::flatbuffers::FlatBufferBuilder &fbb,
                               const ::tt::runtime::Tensor &tensor, bool force);

  static uint64_t buildShutdownCommand(::flatbuffers::FlatBufferBuilder &fbb);
};

} // namespace tt::runtime::distributed::controller
#endif // TT_RUNTIME_DETAIL_DISTRIBUTED_CONTROLLER_COMMAND_FACTORY_H
