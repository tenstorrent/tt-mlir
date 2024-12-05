// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_H
#define TT_RUNTIME_DETAIL_TTNN_H

#define FMT_HEADER_ONLY
#include "distributed/mesh_device.hpp"
#include "host_api.hpp"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/device.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/copy.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "ttnn/operations/eltwise/ternary/where.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/embedding/embedding.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/normalization/softmax/softmax.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

#include "tt/runtime/types.h"
#include "ttmlir/Target/TTNN/Target.h"

namespace tt::runtime::ttnn {

// Default L1 small size to use for the ttnn runtime (32kb).
constexpr std::size_t kL1SmallSize = 1 << 15;

std::pair<SystemDesc, DeviceIds> getCurrentSystemDesc();

Tensor createTensor(std::shared_ptr<void> data,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize, ::tt::target::DataType dataType);

Tensor
createTensor(std::vector<std::shared_ptr<void>> &data,
             std::vector<std::uint32_t> const &shape,
             std::vector<std::uint32_t> const &stride, std::uint32_t itemsize,
             ::tt::target::DataType dataType,
             std::unordered_map<std::string, std::string> const &strategy);

Tensor createTensor(Device device, Layout layout,
                    std::vector<std::uint32_t> const &shape,
                    std::vector<std::uint32_t> const &stride,
                    std::uint32_t itemsize);

inline Tensor createTensor(std::shared_ptr<void> data, TensorDesc const &desc) {
  return ::tt::runtime::ttnn::createTensor(data, desc.shape, desc.stride,
                                           desc.itemsize, desc.dataType);
}

inline Tensor
createTensor(std::vector<std::shared_ptr<void>> &data, TensorDesc const &desc,
             std::unordered_map<std::string, std::string> const &strategy) {
  return ::tt::runtime::ttnn::createTensor(
      data, desc.shape, desc.stride, desc.itemsize, desc.dataType, strategy);
}

inline Tensor createTensor(Device device, Layout layout,
                           TensorDesc const &desc) {
  return ::tt::runtime::ttnn::createTensor(device, layout, desc.shape,
                                           desc.stride, desc.itemsize);
}

tt::target::DataType getTensorDataType(Tensor tensor);

size_t getNumAvailableDevices();

Device openDevice(DeviceIds const &deviceIds, size_t numHWCQs = 1);

void closeDevice(Device device);

void deallocateBuffers(Device device);

void wait(Event event);

void wait(Tensor tensor);

void wait(std::vector<Tensor> const &tensors);

Tensor toHost(Tensor tensor, bool untilize = false);

Tensor toLayout(Tensor tensor, Device device, Layout layout);

Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex);

void memcpy(void *dst, Tensor src);

void memcpy(Tensor dst, Tensor src);

void deallocateTensor(Tensor &tensor, bool force = false);

std::string getOpDebugString(OpContext opContextHandle);

std::string getOpLocInfo(OpContext opContextHandle);

Tensor getOpOutputTensor(OpContext opContextHandle,
                         CallbackContext programContextHandle);

std::vector<float> getTensorData(Tensor tensor);

namespace legacy {
/* Will be deprecated soon once FEs migrate to new API */

Event submit(Device deviceHandle, Binary executableHandle,
             std::uint32_t programIndex, std::vector<Tensor> const &inputs,
             std::vector<Tensor> const &outputs);

void runProgram(::ttnn::MeshDevice &meshDevice, Binary &executableHandle,
                std::uint32_t programIndex,
                std::vector<::ttnn::Tensor *> const &inputs,
                std::vector<::ttnn::Tensor *> const &outputs);
} // namespace legacy

std::vector<Tensor> submit(Device deviceHandle, Binary executableHandle,
                           std::uint32_t programIndex,
                           std::vector<Tensor> const &inputs);

std::vector<Tensor> runProgram(::ttnn::MeshDevice &meshDevice,
                               Binary executableHandle,
                               std::uint32_t programIndex,
                               std::vector<::ttnn::Tensor *> const &inputs);

} // namespace tt::runtime::ttnn

#endif
