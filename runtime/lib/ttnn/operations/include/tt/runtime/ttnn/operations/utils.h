// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_TTNN_OPERATIONS_UTILS_H
#define TT_RUNTIME_TTNN_OPERATIONS_UTILS_H

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "types_generated.h"
#include <concepts>
#include <cstdint>

namespace tt::runtime::ttnn::operations::utils {

bool isTilized(const ::tt::target::ttnn::TensorRef *tensorRef);

::ttnn::DataType getDataType(const ::tt::target::ttnn::TensorRef *tensorRef);

::tt::tt_metal::DistributedTensorConfig distributedTensorConfigFromFlatbuffer(
    const ::tt::target::ttnn::DistributionStrategy *strategy);

template <std::integral T>
inline ::ttnn::Shape toTTNNShape(const flatbuffers::Vector<T> &vec) {
  std::vector<uint32_t> rawShape;
  rawShape.reserve(vec.size());
  std::transform(
      vec.begin(), vec.end(), std::back_inserter(rawShape),
      [](const T &x) -> uint32_t { return static_cast<uint32_t>(x); });
  return ::ttnn::Shape(rawShape);
}

::ttnn::operations::unary::UnaryOpType
toTTNNUnaryOpType(::tt::target::ttnn::UnaryOpType unaryOpType);

::ttnn::operations::unary::UnaryWithParam
toTTNNUnaryWithParam(const ::tt::target::ttnn::UnaryWithParam &unaryWithParam);

std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
createMatmulProgramConfigIfNeeded(const ::tt::target::ttnn::MatmulOp *op);

::ttnn::operations::conv::conv2d::Conv2dConfig
createConv2dConfig(const ::tt::target::ttnn::Conv2dConfig *memcfg);

// Template function to create empty buffer for a specific type.
template <typename T>
static ::tt::tt_metal::OwnedBuffer createBuffer(const uint32_t size) {
  if constexpr (tt::runtime::ttnn::utils::IsHostTypeV<T>) {
    return tt::tt_metal::owned_buffer::create<T>(size);
  }

  LOG_FATAL("Unsupported data type");
  return {};
}

// Helper function for reading elements from flatbuffer data.
template <typename T>
static T getElement(const ::flatbuffers::Vector<uint8_t> *data, size_t i) {
  if constexpr (std::is_same_v<T, bfloat16>) {
    return bfloat16(
        ::flatbuffers::IndirectHelper<uint16_t>::Read(data->data(), i));
  }

  return ::flatbuffers::IndirectHelper<T>::Read(data->data(), i);
}

// Define function pointer types for buffer creation.
using BufferCreatorFn = ::tt::tt_metal::OwnedBuffer (*)(const uint32_t);

// Create lookup tables using index sequences.
template <size_t... Is>
inline constexpr auto createBufferTable(std::index_sequence<Is...>) {
  return std::array<BufferCreatorFn, sizeof...(Is)>{
      [](const uint32_t size) -> tt::tt_metal::OwnedBuffer {
        return createBuffer<tt::runtime::ttnn::utils::NativeDTypeT<static_cast<
            ::tt::target::DataType>(tt::runtime::ttnn::utils::DTypeMinV + Is)>>(
            size);
      }...};
}

// Instantiate the lookup tables.
inline constexpr auto bufferTable = createBufferTable(
    std::make_index_sequence<tt::runtime::ttnn::utils::DTypeCountV>());

// Helper function to create an empty buffer of the appropriate type.
inline ::tt::tt_metal::OwnedBuffer
createTypedBuffer(::tt::target::DataType dtype, const uint32_t size) {
  size_t idx = static_cast<size_t>(dtype) - tt::runtime::ttnn::utils::DTypeMinV;
  return bufferTable[idx](size);
}

// Helper function to create a buffer from data of the appropriate type.
inline ::tt::tt_metal::OwnedBuffer
createTypedBuffer(::tt::target::DataType dtype, const size_t size,
                  const ::flatbuffers::Vector<uint8_t> *data) {
  auto buffer = createTypedBuffer(dtype, size);

  std::visit(
      [data](auto &&buffer) {
        using T = std::decay_t<decltype(buffer)>::value_type;

        LOG_ASSERT(data->size() / sizeof(T) == buffer.size(),
                   "Unexpected data size");

        for (size_t i = 0; i < buffer.size(); ++i) {
          buffer[i] = getElement<T>(data, i);
        }
      },
      buffer);

  return buffer;
}

} // namespace tt::runtime::ttnn::operations::utils
#endif
