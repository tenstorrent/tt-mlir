// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_TTNN_H
#define TT_RUNTIME_DETAIL_TTNN_TTNN_H

#define FMT_HEADER_ONLY
#include "hostdevcommon/common_values.hpp"
#include "tt-metalium/hal.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/host_buffer.hpp"
#include "tt-metalium/memory_reporter.hpp"
#include "tt-metalium/mesh_device.hpp"
#include "tt-metalium/program_cache.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
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
#include "ttnn/operations/normalization/batch_norm/batch_norm.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

#include "tt/runtime/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn {

// Creates host tensor with borrowed storage (the buffer of the tensor is on the
// host and it was borrowed from an external buffer which is responsible for its
// allocation/deallocation).
::tt::runtime::Tensor
createBorrowedHostTensor(void *data, const std::vector<std::uint32_t> &shape,
                         const std::vector<std::uint32_t> &stride,
                         std::uint32_t itemsize,
                         ::tt::target::DataType dataType);

// Creates host tensor with owned storage (the buffer of the tensor is on the
// host and its allocation/deallocation is owned by this tensor instance).
::tt::runtime::Tensor
createOwnedHostTensor(const void *data, const std::vector<std::uint32_t> &shape,
                      const std::vector<std::uint32_t> &stride,
                      std::uint32_t itemsize, ::tt::target::DataType dataType);

// Creates multi-device host tensor with owned storage (buffers of the tensor
// are on the host and their allocation/deallocation is owned by this tensor
// instance).
::tt::runtime::Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
    ::tt::target::DataType dataType,
    const std::unordered_map<std::string, std::string> &strategy);

// Creates multi-device host tensor from already existing host tensor shards.
// Tensor shards can be host tensors with either owned or borrowed storage.
::tt::runtime::Tensor createMultiDeviceHostTensor(
    const std::vector<::tt::runtime::Tensor> &tensorShards,
    const std::unordered_map<std::string, std::string> &strategy);

::tt::runtime::Tensor createEmptyTensor(
    Device device, Layout layout, const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride, std::uint32_t itemsize);

inline ::tt::runtime::Tensor createOwnedHostTensor(const void *data,
                                                   const TensorDesc &desc) {
  return ::tt::runtime::ttnn::createOwnedHostTensor(
      data, desc.shape, desc.stride, desc.itemsize, desc.dataType);
}

inline ::tt::runtime::Tensor createBorrowedHostTensor(void *data,
                                                      const TensorDesc &desc) {
  return ::tt::runtime::ttnn::createBorrowedHostTensor(
      data, desc.shape, desc.stride, desc.itemsize, desc.dataType);
}

inline ::tt::runtime::Tensor createMultiDeviceHostTensor(
    const std::vector<const void *> &data, const TensorDesc &desc,
    const std::unordered_map<std::string, std::string> &strategy) {
  return ::tt::runtime::ttnn::createMultiDeviceHostTensor(
      data, desc.shape, desc.stride, desc.itemsize, desc.dataType, strategy);
}

inline ::tt::runtime::Tensor createEmptyTensor(Device device, Layout layout,
                                               const TensorDesc &desc) {
  return ::tt::runtime::ttnn::createEmptyTensor(device, layout, desc.shape,
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

Arch getArch();

size_t getNumAvailableDevices();

Device openMeshDevice(const std::vector<uint32_t> &meshShape,
                      const MeshDeviceOptions &options = MeshDeviceOptions());

void closeMeshDevice(Device parentMesh);

Device createSubMeshDevice(Device parentMesh,
                           const std::vector<uint32_t> &meshShape,
                           const std::optional<const std::vector<uint32_t>>
                               &meshOffset = std::nullopt);

void releaseSubMeshDevice(Device subMesh);

void reshapeMeshDevice(Device meshDevice,
                       const std::vector<uint32_t> &meshShape);

std::vector<uint32_t> getMeshShape(Device meshDevice);
std::vector<uint32_t> getMeshOffset(Device meshDevice);
std::vector<int> getDeviceIds(Device meshDevice);
size_t getNumHwCqs(Device meshDevice);
bool isProgramCacheEnabled(Device meshDevice);
size_t getL1SmallSize(Device meshDevice);
size_t getTraceRegionSize(Device meshDevice);
size_t getNumDramChannels(Device meshDevice);
size_t getDramSizePerChannel(Device meshDevice);
size_t getL1SizePerCore(Device meshDevice);

void deallocateBuffers(Device device);

void dumpMemoryReport(Device device);

void dumpDeviceProfileResults(Device device);

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device device);

void wait(Event event);

void wait(::tt::runtime::Tensor tensor);

void wait(const std::vector<::tt::runtime::Tensor> &tensors);

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
runProgram(std::shared_ptr<::ttnn::MeshDevice> meshDevice,
           Binary executableHandle, std::uint32_t programIndex,
           std::vector<::tt::runtime::Tensor> &inputs);

} // namespace tt::runtime::ttnn

#endif
