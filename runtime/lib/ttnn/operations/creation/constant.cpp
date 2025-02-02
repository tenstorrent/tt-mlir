// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/constant.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {

template <::tt::target::DataType DataType>
struct NativeDType {
  using type = std::monostate;
};
template <>
struct NativeDType<::tt::target::DataType::Float32> {
  using type = float;
};
template <>
struct NativeDType<::tt::target::DataType::BFloat16> {
  using type = bfloat16;
};
template <>
struct NativeDType<::tt::target::DataType::UInt32> {
  using type = uint32_t;
};
template <>
struct NativeDType<::tt::target::DataType::UInt16> {
  using type = uint16_t;
};
template <>
struct NativeDType<::tt::target::DataType::UInt8> {
  using type = uint8_t;
};

template <::tt::target::DataType DataType>
using NativeDTypeT = typename NativeDType<DataType>::type;

template <typename T>
constexpr bool IsHostTypeV =
    std::is_constructible_v<::tt::tt_metal::OwnedBuffer,
                            ::tt::tt_metal::owned_buffer::Buffer<T>>;

template <typename T>
static T getElement(const ::flatbuffers::Vector<uint8_t> *data, size_t i) {
  if constexpr (std::is_same_v<T, bfloat16>) {
    return bfloat16(
        ::flatbuffers::IndirectHelper<uint16_t>::Read(data->data(), i));
  } else {
    return ::flatbuffers::IndirectHelper<T>::Read(data->data(), i);
  }
}

template <typename T>
static ::tt::tt_metal::OwnedBuffer
makeBuffer(const ::flatbuffers::Vector<uint8_t> *data) {
  if constexpr (IsHostTypeV<T>) {
    size_t size = data->size() / sizeof(T);
    LOG_ASSERT(data->size() % sizeof(T) == 0, "Invalid data size");

    ::tt::tt_metal::owned_buffer::Buffer<T> ownedBuffer =
        tt::tt_metal::owned_buffer::create<T>(size);

    for (size_t i = 0; i < size; ++i) {
      ownedBuffer[i] = getElement<T>(data, i);
    }
    return ownedBuffer;
  } else {
    LOG_FATAL("Unsupported data type");
    return {};
  }
}

constexpr size_t DTypeMinV = static_cast<size_t>(tt::target::DataType::MIN);
constexpr size_t DTypeMaxV = static_cast<size_t>(tt::target::DataType::MAX);
constexpr size_t DTypeCountV = DTypeMaxV - DTypeMinV + 1;

using BufferCreatorFn =
    ::tt::tt_metal::OwnedBuffer (*)(const ::flatbuffers::Vector<uint8_t> *);

template <size_t... Is>
constexpr auto makeBufferTable(std::index_sequence<Is...>) {
  return std::array<BufferCreatorFn, sizeof...(Is)>{
      [](const ::flatbuffers::Vector<uint8_t> *data)
          -> tt::tt_metal::OwnedBuffer {
        return makeBuffer<
            NativeDTypeT<static_cast<::tt::target::DataType>(DTypeMinV + Is)>>(
            data);
      }...};
}

constexpr auto bufferTable =
    makeBufferTable(std::make_index_sequence<DTypeCountV>{});

static tt::tt_metal::OwnedBuffer
makeTypedBuffer(::tt::target::DataType dtype,
                const ::flatbuffers::Vector<uint8_t> *data) {
  return bufferTable[static_cast<size_t>(dtype)](data);
}

void run(const ::tt::target::ttnn::ConstantOp *op, ProgramContext &context) {
  ::ttnn::Shape shape(::tt::runtime::ttnn::utils::toShapeFromFBShape(
      *op->out()->desc()->shape()));

  ::tt::tt_metal::OwnedBuffer ownedBuffer = makeTypedBuffer(
      op->out()->desc()->layout()->memory_desc()->data_type(), op->data());

  ::ttnn::DataType dtype =
      ::tt::runtime::ttnn::operations::utils::getDataType(op->out());

  ::ttnn::Tensor out(::tt::tt_metal::OwnedStorage(ownedBuffer), shape, dtype,
                     ::ttnn::Layout::ROW_MAJOR);

  context.getTensorPool().insert_or_assign(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::creation
