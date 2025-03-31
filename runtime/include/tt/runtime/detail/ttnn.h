// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_H
#define TT_RUNTIME_DETAIL_TTNN_H

#define FMT_HEADER_ONLY
#include "hostdevcommon/common_values.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/memory_reporter.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/repeat/repeat.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/repeat_interleave.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/quantization/quantization.hpp"
#include "ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/kv_cache/kv_cache.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/host_buffer/owned_buffer.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

#include "tt/runtime/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn {

::tt::runtime::Tensor
createOwnedTensor(std::shared_ptr<void> data,
                  std::vector<std::uint32_t> const &shape,
                  std::vector<std::uint32_t> const &stride,
                  std::uint32_t itemsize, ::tt::target::DataType dataType);

::tt::runtime::Tensor createTensor(std::shared_ptr<void> data,
                                   std::vector<std::uint32_t> const &shape,
                                   std::vector<std::uint32_t> const &stride,
                                   std::uint32_t itemsize,
                                   ::tt::target::DataType dataType);

::tt::runtime::Tensor
createTensor(std::vector<std::shared_ptr<void>> &data,
             std::vector<std::uint32_t> const &shape,
             std::vector<std::uint32_t> const &stride, std::uint32_t itemsize,
             ::tt::target::DataType dataType,
             std::unordered_map<std::string, std::string> const &strategy);

::tt::runtime::Tensor createTensor(Device device, Layout layout,
                                   std::vector<std::uint32_t> const &shape,
                                   std::vector<std::uint32_t> const &stride,
                                   std::uint32_t itemsize);

inline ::tt::runtime::Tensor createTensor(std::shared_ptr<void> data,
                                          TensorDesc const &desc) {
  return ::tt::runtime::ttnn::createTensor(data, desc.shape, desc.stride,
                                           desc.itemsize, desc.dataType);
}

inline ::tt::runtime::Tensor
createTensor(std::vector<std::shared_ptr<void>> &data, TensorDesc const &desc,
             std::unordered_map<std::string, std::string> const &strategy) {
  return ::tt::runtime::ttnn::createTensor(
      data, desc.shape, desc.stride, desc.itemsize, desc.dataType, strategy);
}

inline ::tt::runtime::Tensor createTensor(Device device, Layout layout,
                                          TensorDesc const &desc) {
  return ::tt::runtime::ttnn::createTensor(device, layout, desc.shape,
                                           desc.stride, desc.itemsize);
}

bool isTensorAllocated(::tt::runtime::Tensor tensor);
tt::target::DataType getTensorDataType(::tt::runtime::Tensor tensor);
std::vector<std::byte> getTensorDataBuffer(::tt::runtime::Tensor tensor);
std::vector<std::uint32_t> getTensorShape(::tt::runtime::Tensor tensor);
std::vector<std::uint32_t> getTensorStride(::tt::runtime::Tensor tensor);
std::uint32_t getTensorElementSize(::tt::runtime::Tensor tensor);
std::uint32_t getTensorVolume(::tt::runtime::Tensor tensor);
TensorDesc getTensorDesc(::tt::runtime::Tensor tensor);
bool getTensorRetain(::tt::runtime::Tensor tensor);
void setTensorRetain(::tt::runtime::Tensor tensor, bool retain);

size_t getNumAvailableDevices();

Device openMeshDevice(const std::vector<uint32_t> &meshShape,
                      const MeshDeviceOptions &options = MeshDeviceOptions());

void closeMeshDevice(Device parentMesh);

Device createSubMeshDevice(
    Device parentMesh, const std::pair<uint32_t, uint32_t> &meshShape,
    const std::optional<const std::pair<uint32_t, uint32_t>> &meshOffset =
        std::nullopt);

void releaseSubMeshDevice(Device subMesh);

void deallocateBuffers(Device device);

void dumpMemoryReport(Device device);

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device device, int deviceID = 0);

void wait(Event event);

void wait(::tt::runtime::Tensor tensor);

void wait(std::vector<::tt::runtime::Tensor> const &tensors);

std::vector<::tt::runtime::Tensor> toHost(::tt::runtime::Tensor tensor,
                                          bool untilize = false);

::tt::runtime::Tensor toLayout(::tt::runtime::Tensor tensor, Device device,
                               Layout layout,
                               std::optional<bool> retain = std::nullopt);

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex);

void memcpy(void *dst, ::tt::runtime::Tensor src);

void memcpy(::tt::runtime::Tensor dst, ::tt::runtime::Tensor src);

void deallocateTensor(::tt::runtime::Tensor &tensor, bool force = false);

std::string getOpDebugString(OpContext opContextHandle);

std::string getOpLocInfo(OpContext opContextHandle);

::tt::runtime::Tensor getOpOutputTensor(OpContext opContextHandle,
                                        CallbackContext programContextHandle);

std::vector<::tt::runtime::Tensor>
submit(Device deviceHandle, Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs);

std::vector<::tt::runtime::Tensor>
runProgram(::ttnn::MeshDevice &meshDevice, Binary executableHandle,
           std::uint32_t programIndex,
           std::vector<::tt::runtime::Tensor> &inputs);

} // namespace tt::runtime::ttnn

#endif
