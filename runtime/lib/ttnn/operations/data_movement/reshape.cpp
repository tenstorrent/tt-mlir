// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape.h"
#include "tt/runtime/detail/ttnn.h"

namespace tt::runtime::ttnn::operations::data_movement {

template <int32_t Rank>
static std::array<int32_t, Rank>
vectorToArray(const std::vector<int32_t> &vec) {
  if (vec.size() != Rank) {
    throw std::invalid_argument("Vector size does not match array size");
  }
  std::array<int32_t, Rank> arr;
  std::copy(vec.begin(), vec.end(), arr.begin());
  return arr;
}

template <int32_t Rank>
static ::ttnn::Tensor invoke_reshape(const ::ttnn::Tensor &tensor,
                                     const std::vector<int32_t> &shape) {
  return ::ttnn::reshape(tensor, vectorToArray<Rank>(shape));
}

void run(const ::tt::target::ttnn::ReshapeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  const auto *fbShape = op->shape();
  std::vector<int32_t> shape(fbShape->begin(), fbShape->end());
  constexpr int32_t Rank1 = 1;
  constexpr int32_t Rank2 = 2;
  constexpr int32_t Rank3 = 3;
  constexpr int32_t Rank4 = 4;
  constexpr int32_t Rank5 = 5;

  if (shape.size() == 1) {
    // if (in.get_layout() == ::ttnn::Layout::TILE) {
    //   throw std::runtime_error("Input tensor has TILE layout!!!");
    // }

    std::array<int32_t, Rank1> shape_arr = vectorToArray<Rank1>(shape);
    std::array<std::uint32_t, Rank1> new_shape{};
    std::copy(shape_arr.begin(), shape_arr.end(), new_shape.begin());
    ::ttnn::Shape target_shape(new_shape);

    in.set_layout(::ttnn::Layout::ROW_MAJOR);
    ::ttnn::Tensor out_tensor = in.reshape(target_shape.value);
    tensorPool.insert_or_assign(op->out()->global_id(), out_tensor);
    return;
  }

  ::ttnn::Tensor out;
  switch (fbShape->size()) {
  case Rank1:
    out = invoke_reshape<Rank1>(in, shape);
    break;
  case Rank2:
    out = invoke_reshape<Rank2>(in, shape);
    break;
  case Rank3:
    out = invoke_reshape<Rank3>(in, shape);
    break;
  case Rank4:
    out = invoke_reshape<Rank4>(in, shape);
    break;
  case Rank5:
    out = invoke_reshape<Rank5>(in, shape);
    break;
  default:
    throw std::invalid_argument("Unsupported rank for reshape");
  }

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
