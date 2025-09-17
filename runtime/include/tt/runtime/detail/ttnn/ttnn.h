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
#include "tt-metalium/persistent_kernel_cache.hpp"
#include "tt-metalium/program_cache.hpp"
#include "ttnn/device.hpp"
#include "ttnn/events.hpp"
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
#include "ttnn/operations/data_movement/sort/sort.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/quantization/quantization.hpp"
#include "ttnn/operations/eltwise/ternary/where/where.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"
#include "ttnn/operations/generic/generic_op.hpp"
#include "ttnn/operations/kv_cache/kv_cache.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/moreh_cumsum.hpp"
#include "ttnn/operations/normalization/batch_norm/batch_norm.hpp"
#include "ttnn/operations/normalization/rmsnorm/rmsnorm.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/pool/upsample/upsample.hpp"
#include "ttnn/operations/rand/rand.hpp"
#include "ttnn/operations/reduction/argmax/argmax.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/operations/trace.hpp"
#include "ttnn/operations/transformer/concatenate_heads/concatenate_heads.hpp"
#include "ttnn/operations/transformer/sdpa_decode/sdpa_decode.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/serialization.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

#include "tt/runtime/types.h"
#include "ttmlir/Target/TTNN/Target.h"

#include <optional>
#include <vector>

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
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape);

// Creates multi-device host tensor from already existing host tensor shards.
// Tensor shards can be host tensors with either owned or borrowed storage.
::tt::runtime::Tensor createMultiDeviceHostTensor(
    const std::vector<::tt::runtime::Tensor> &tensorShards,
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape);

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
    const std::unordered_map<std::string, std::string> &strategy,
    const std::vector<uint32_t> &meshShape) {
  return ::tt::runtime::ttnn::createMultiDeviceHostTensor(
      data, desc.shape, desc.stride, desc.itemsize, desc.dataType, strategy,
      meshShape);
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

void enablePersistentKernelCache();
void disablePersistentKernelCache();

size_t getNumAvailableDevices();

Device openMeshDevice(const MeshDeviceOptions &options = {});

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

void releaseTrace(Device meshDevice, std::uint64_t binaryId,
                  size_t mainProgramId);

void deallocateBuffers(Device device);

void dumpMemoryReport(Device device);

void readDeviceProfilerResults(Device device);

std::unordered_map<tt::runtime::MemoryBufferType, tt::runtime::MemoryView>
getMemoryView(Device device);

void setFabricConfig(FabricConfig config);

void wait(Event event);

void wait(::tt::runtime::Tensor tensor,
          std::optional<uint8_t> cqId = std::nullopt);

void wait(const std::vector<::tt::runtime::Tensor> &tensors,
          std::optional<uint8_t> cqId = std::nullopt);

std::vector<::tt::runtime::Tensor> toHost(::tt::runtime::Tensor tensor,
                                          bool untilize = false,
                                          bool blocking = true);

::tt::runtime::Tensor toLayout(::tt::runtime::Tensor tensor, Device device,
                               Layout layout,
                               std::optional<bool> retain = std::nullopt);

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex);

void memcpy(void *dst, ::tt::runtime::Tensor src,
            std::optional<tt::target::DataType> dstDataType = std::nullopt);

void memcpy(::tt::runtime::Tensor dst, ::tt::runtime::Tensor src);

void deallocateTensor(::tt::runtime::Tensor &tensor, bool force = false);

std::string getOpDebugString(OpContext opContextHandle);

std::string getOpLocInfo(OpContext opContextHandle);

std::unordered_map<std::uint32_t, Tensor>
getOpOutputTensor(OpContext opContextHandle,
                  CallbackContext programContextHandle);

// Returns reference to the output tensor of the operation
// if the operation does not have an output tensor, returns std::nullopt
std::optional<tt::runtime::TensorRef>
getOpOutputRef(OpContext opContextHandle, CallbackContext programContextHandle);

// Returns list of references to the input tensors of the operation
// if the operation does not have any input tensors, returns empty vector
std::vector<tt::runtime::TensorRef>
getOpInputRefs(OpContext opContextHandle, CallbackContext programContextHandle);

// Returns tensor to which tensorRef refers
// In case that that tensor is not in the tensor pool, returns std::nullopt
// For now only supports single device tensors
std::optional<Tensor>
retrieveTensorFromPool(CallbackContext programContextHandle,
                       tt::runtime::TensorRef tensorRef, bool untilize);

// Update tensor to which tensorRef refers
// Prefered to be owned tensor to avoid unexpected behavior in case of
// deallocation
void updateTensorInPool(CallbackContext programContextHandle,
                        TensorRef tensorRef, Tensor srcTensor);

std::vector<::tt::runtime::Tensor>
submit(Device deviceHandle, Binary executableHandle, std::uint32_t programIndex,
       std::vector<::tt::runtime::Tensor> &inputs);

// Dumps tensor data to a file in binary format
void dumpTensor(::tt::runtime::Tensor tensor, const std::string &filePath);

// Loads tensor data from a binary file
::tt::runtime::Tensor loadTensor(const std::string &filePath,
                                 std::optional<Device> device = std::nullopt);

} // namespace tt::runtime::ttnn

#endif
