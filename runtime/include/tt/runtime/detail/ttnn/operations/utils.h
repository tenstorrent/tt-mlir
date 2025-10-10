// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UTILS_H
#define TT_RUNTIME_DETAIL_TTNN_OPERATIONS_UTILS_H

#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "types_generated.h"
#include <concepts>
#include <cstdint>

namespace tt::runtime::ttnn::operations::utils {

void eventSync(::ttnn::MeshDevice *meshDevice, const ::ttnn::QueueId &recordCq,
               const ::ttnn::QueueId &waitCq);

bool isTilized(const ::tt::target::ttnn::TensorRef *tensorRef);

::ttnn::DataType getDataType(const ::tt::target::ttnn::TensorRef *tensorRef);

::ttnn::operations::unary::UnaryOpType
toTTNNUnaryOpType(::tt::target::ttnn::EltwiseUnaryOpType unaryOpType);

::ttnn::operations::unary::UnaryWithParam
toTTNNUnaryWithParam(const ::tt::target::ttnn::UnaryWithParam &unaryWithParam);

std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
createMatmulProgramConfigIfNeeded(const ::tt::target::ttnn::MatmulOp *op);

::ttnn::operations::conv::conv2d::Conv2dConfig
createConv2dConfig(const ::tt::target::ttnn::Conv2dConfig *memcfg);

::ttnn::operations::conv::conv2d::Conv2dSliceConfig
createConv2dSliceConfig(const ::tt::target::ttnn::Conv2dSliceConfig *config);

::ttnn::DeviceComputeKernelConfig createDeviceComputeKernelConfig(
    const ::tt::target::ttnn::DeviceComputeKernelConfig *config);

::ttnn::Tensor toTTNNTensor(const ::flatbuffers::Vector<uint8_t> *data,
                            const ::ttnn::Shape &shape,
                            const ::ttnn::DataType &dataType,
                            ::ttnn::MeshDevice *meshDevice,
                            const ::ttnn::Layout &layout,
                            const ::ttnn::MemoryConfig &memoryConfig);

::ttnn::Tensor
allocateTensorOnDevice(const ::tt::target::ttnn::TensorRef *tensorRef,
                       ::ttnn::MeshDevice &meshDevice);

template <std::integral T>
inline ::ttnn::Shape toTTNNShape(const flatbuffers::Vector<T> &vec) {
  std::vector<uint32_t> rawShape;
  rawShape.reserve(vec.size());
  std::transform(
      vec.begin(), vec.end(), std::back_inserter(rawShape),
      [](const T &x) -> uint32_t { return static_cast<uint32_t>(x); });
  return ::ttnn::Shape(rawShape);
}

} // namespace tt::runtime::ttnn::operations::utils
#endif
