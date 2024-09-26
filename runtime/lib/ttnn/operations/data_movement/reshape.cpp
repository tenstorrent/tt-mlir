// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"

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

static ::ttnn::Tensor tilize(::ttnn::Tensor const &input) {
  return ::ttnn::to_layout(input, ::ttnn::TILE_LAYOUT, std::nullopt,
                           std::nullopt,
                           static_cast<::ttnn::Device *>(nullptr));
}

static ::ttnn::Tensor untilize(::ttnn::Tensor const &input) {
  return ::ttnn::to_layout(input, ::ttnn::ROW_MAJOR_LAYOUT, std::nullopt,
                           std::nullopt,
                           static_cast<::ttnn::Device *>(nullptr));
}

static ::ttnn::Tensor prepareRank1ReshapeInput(const ::ttnn::Tensor &input) {
  ::ttnn::Tensor hostTensor = input.cpu();
  if (input.get_layout() != ::ttnn::ROW_MAJOR_LAYOUT) {
    hostTensor = untilize(hostTensor);
  }
  return hostTensor;
}

static ::ttnn::Tensor toOutputLayout(const ::ttnn::Tensor &input,
                                     ::ttnn::Device &device,
                                     const ::tt::target::TensorRef *outputRef) {
  ::ttnn::Tensor out = input;
  if (not utils::inSystemMemory(outputRef)) {
    out = ::ttnn::to_device(out, &device, std::nullopt);
    if (input.get_layout() == ::ttnn::Layout::TILE) {
      out = tilize(out);
    }
    out = ::ttnn::to_memory_config(out, utils::createMemoryConfig(outputRef),
                                   std::nullopt);
    return out;
  }
  if (input.get_layout() == ::ttnn::Layout::TILE) {
    out = tilize(out);
  }
  return out;
}

template <int32_t Rank>
static ::ttnn::Tensor invoke_reshape(const ::ttnn::Tensor &tensor,
                                     const std::vector<int32_t> &shape) {
  return ::ttnn::reshape(tensor, vectorToArray<Rank>(shape));
}

void run(const ::tt::target::ttnn::ReshapeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  const auto *fbShape = op->shape();
  std::vector<int32_t> shape(fbShape->begin(), fbShape->end());
  constexpr int32_t Rank1 = 1;
  constexpr int32_t Rank2 = 2;
  constexpr int32_t Rank3 = 3;
  constexpr int32_t Rank4 = 4;
  constexpr int32_t Rank5 = 5;

  ::ttnn::Tensor out;
  switch (fbShape->size()) {
  // Hack for handling special case where desired shape is 1
  case Rank1:
    out = invoke_reshape<Rank1>(prepareRank1ReshapeInput(in), shape);
    out = toOutputLayout(out, context.getFirstDevice(), op->out());
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
