// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <list>
#include <optional>
#include <unordered_map>

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/runtime.h"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "types_generated.h"
#include "utils.h"

#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

// It seems like `ttnn::to_layout` cannot be called inside of the
// `tt::runtime::ttnn` namespace.  TTNN uses a lot of metaprogramming and for
// some reason a static_assert fails when this is called from within our
// namespace.
ttnn::Tensor tilize(ttnn::Tensor const &input) {
  return ttnn::to_layout(input, ::ttnn::TILE_LAYOUT, std::nullopt, std::nullopt,
                         (Device *)nullptr);
}

ttnn::Tensor untilize(ttnn::Tensor const &input) {
  return ttnn::to_layout(input, ::ttnn::ROW_MAJOR_LAYOUT, std::nullopt,
                         std::nullopt, (Device *)nullptr);
}

namespace tt::runtime::ttnn {

static ::ttnn::Tensor convertDataType(const ::ttnn::Tensor &input,
                                      const ::ttnn::DataType &targetDataType) {
  const ::ttnn::StorageType storageType = input.storage_type();
  if (storageType == ::tt::tt_metal::StorageType::BORROWED) {
    return ::ttnn::to_dtype(input, targetDataType);
  } else if (storageType == ::tt::tt_metal::StorageType::DEVICE) {
    if (input.get_layout() != ::ttnn::TILE_LAYOUT) {
      // typecast op requires tilized tensor
      ::ttnn::Tensor converted =
          ::ttnn::typecast(::tilize(input), targetDataType);
      // untilize and return
      return ::untilize(converted);
    }
    return ::ttnn::typecast(input, targetDataType);
  } else {
    throw std::runtime_error("Unsupported storage type");
  }
}

/* TODO: Blocked by issue #272, ideal flow is to determine tilize/untilize with
 * tile_shape */
static ::ttnn::Tensor
updateLayoutAndDataType(const ::ttnn::Tensor &inputTensor,
                        const ::ttnn::DataType targetDataType,
                        const bool shouldTilize, const bool shouldUntilize) {
  ::ttnn::Tensor outputTensor = inputTensor;
  const bool shouldConvertDataType = inputTensor.get_dtype() != targetDataType;
  // const int targetTileX = targetTileShape->x();
  // const int targetTileY = targetTileShape->y();
  // const bool shouldTilize =
  //     targetTileX == 32 and targetTileY == 32 and
  //     inputTensor.get_layout() == ::ttnn::ROW_MAJOR_LAYOUT;
  // const bool shouldUntilize = (targetTileX != 32 or targetTileY != 32) and
  //                             inputTensor.get_layout() ==
  //                             ::ttnn::TILE_LAYOUT;
  if (shouldTilize) {
    outputTensor = ::tilize(outputTensor);
  } else if (shouldUntilize) {
    outputTensor = ::untilize(outputTensor);
  }
  if (shouldConvertDataType) {
    outputTensor = convertDataType(outputTensor, targetDataType);
  }
  return outputTensor;
}

// TODO: right now hardcoding tilize/untilize, should determine with tile shape
// blocked by issue #272
static void
run(::tt::target::ttnn::ToMemoryConfigOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  const ::ttnn::Tensor &inputTensor = *liveTensors.at(op->in0()->global_id());
  assert(inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED or
         inputTensor.storage_type() == ::tt::tt_metal::StorageType::DEVICE);

  const ::tt::target::Dim2d *targetTileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  TT_FATAL(utils::isValidTileShape(targetTileShape),
           "Invalid tile shape ({}, {})", targetTileShape->x(),
           targetTileShape->y());

  ::tt::target::DataType targetDataType =
      op->out()->desc()->layout()->memory_desc()->data_type();
  ::ttnn::DataType targetDataTypeTTNN = utils::toTTNNDataType(targetDataType);

  const ::tt::target::MemorySpace targetMemorySpace =
      op->out()->desc()->layout()->memory_desc()->memory_space();

  switch (targetMemorySpace) {
  case ::tt::target::MemorySpace::System:
  case ::tt::target::MemorySpace::SystemMMIO: {
    ::ttnn::Tensor result;
    if (inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED) {
      result =
          updateLayoutAndDataType(inputTensor, targetDataTypeTTNN, false, true);
    } else if (inputTensor.storage_type() ==
               ::tt::tt_metal::StorageType::DEVICE) {
      result = updateLayoutAndDataType(inputTensor.cpu(), targetDataTypeTTNN,
                                       false, true);
    }
    ::ttnn::Tensor &outputTensor = *liveTensors.at(op->out()->global_id());
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(result);
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(outputTensor);
    std::uint32_t size = result.volume() * result.element_size();
    std::memcpy(dst, src, size);
    break;
  }
  case ::tt::target::MemorySpace::DeviceDRAM: {
    ::tt::tt_metal::MemoryConfig memConfig = ::ttnn::DRAM_MEMORY_CONFIG;
    if (inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED) {
      ::ttnn::Tensor result = inputTensor;
      bool shouldTilize = true;
      // device tilize requires BFLOAT16, if not then tilize on host
      if (result.get_dtype() != ::ttnn::DataType::BFLOAT16) {
        result = ::tilize(result);
        shouldTilize = false;
      }
      result = ::ttnn::to_device(result, &device, memConfig);
      result = updateLayoutAndDataType(result, targetDataTypeTTNN, shouldTilize,
                                       false);
      tensorPool.push_back(result);
      liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    } else if (inputTensor.storage_type() ==
               ::tt::tt_metal::StorageType::DEVICE) {
      ::ttnn::Tensor result = updateLayoutAndDataType(
          inputTensor, targetDataTypeTTNN, false, false);
      result = ::ttnn::to_memory_config(result, memConfig, std::nullopt);
      tensorPool.push_back(result);
      liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    }
    break;
  }
  // Currently similar to ::tt::target::MemorySpace::DeviceDRAM
  // But will need it's own code path when we add support for sharding
  case ::tt::target::MemorySpace::DeviceL1: {
    ::tt::tt_metal::MemoryConfig memConfig = ::ttnn::L1_MEMORY_CONFIG;
    if (inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED) {
      ::ttnn::Tensor result = inputTensor;
      bool shouldTilize = true;
      // device tilize requires BFLOAT16, if not then tilize on host
      if (result.get_dtype() != ::ttnn::DataType::BFLOAT16) {
        result = ::tilize(result);
        shouldTilize = false;
      }
      result = ::ttnn::to_device(result, &device, memConfig);
      result = updateLayoutAndDataType(result, targetDataTypeTTNN, shouldTilize,
                                       false);
      tensorPool.push_back(result);
      liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    } else if (inputTensor.storage_type() ==
               ::tt::tt_metal::StorageType::DEVICE) {
      ::ttnn::Tensor result = updateLayoutAndDataType(
          inputTensor, targetDataTypeTTNN, false, false);
      result = ::ttnn::to_memory_config(result, memConfig, std::nullopt);
      tensorPool.push_back(result);
      liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    }
    break;
  }
  }
}

static void
run(::tt::target::ttnn::EmptyOp const *op, ::ttnn::device::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {

  ::ttnn::DataType targetDataTypeTTNN = utils::toTTNNDataType(
      op->out()->desc()->layout()->memory_desc()->data_type());
  // TODO: determine layout, hardcoding tile_layout for now
  auto desiredLayout = ::ttnn::Layout::TILE;
  // TODO: how do we determine shape from an int* and no known rank?
  // op->out()->desc()->shape()
  auto shape = ::ttnn::Shape(::tt::tt_metal::Shape({1, 1, 32, 32}));
  tensorPool.push_back(
      ::ttnn::empty(shape, targetDataTypeTTNN, desiredLayout, device));
  liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
}

static void
run(::tt::target::ttnn::EltwiseOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  switch (op->type()) {
  /* Eltwise Binary */
  case ::tt::target::ttnn::EltwiseOpType::Add: {
    assert(op->ins()->size() == 2 && "Unsupported number of inputs");
    auto &lhs = *liveTensors.at(op->ins()->Get(0)->global_id());
    auto &rhs = *liveTensors.at(op->ins()->Get(1)->global_id());
    tensorPool.push_back(::ttnn::add(lhs, rhs));
    liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Multiply: {
    assert(op->ins()->size() == 2 && "Unsupported number of inputs");
    auto &lhs = *liveTensors.at(op->ins()->Get(0)->global_id());
    auto &rhs = *liveTensors.at(op->ins()->Get(1)->global_id());
    tensorPool.push_back(::ttnn::multiply(lhs, rhs));
    liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Subtract: {
    assert(op->ins()->size() == 2 && "Unsupported number of inputs");
    auto &lhs = *liveTensors.at(op->ins()->Get(0)->global_id());
    auto &rhs = *liveTensors.at(op->ins()->Get(1)->global_id());
    tensorPool.push_back(::ttnn::subtract(lhs, rhs));
    liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::GreaterEqual: {
    assert(op->ins()->size() == 2 && "Unsupported number of inputs");
    ::ttnn::Tensor &lhs = *liveTensors.at(op->ins()->Get(0)->global_id());
    ::ttnn::Tensor &rhs = *liveTensors.at(op->ins()->Get(1)->global_id());
    tensorPool.push_back(::ttnn::ge(lhs, rhs));
    liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    break;
  }
  /* Eltwise Unary */
  case ::tt::target::ttnn::EltwiseOpType::Relu: {
    assert(op->ins()->size() == 1 && "Unsupported number of inputs");
    ::ttnn::Tensor &in = *liveTensors.at(op->ins()->Get(0)->global_id());
    tensorPool.push_back(::ttnn::relu(in));
    liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    break;
  }
  }
}

static void
run(::tt::target::ttnn::ReductionOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  switch (op->type()) {
  case ::tt::target::ttnn::ReductionOpType::Sum: {
    auto &in = *liveTensors.at(op->in()->global_id());

    const auto *dim_arg_fb_ptr = op->dim_arg();
    std::optional<vector<int>> dim_arg =
        dim_arg_fb_ptr ? std::make_optional(std::vector<int>(
                             dim_arg_fb_ptr->begin(), dim_arg_fb_ptr->end()))
                       : std::nullopt;

    tensorPool.push_back(::ttnn::sum(in, dim_arg, op->keep_dim()));

    liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Mean: {
    auto &in = *liveTensors.at(op->in()->global_id());

    const auto *dim_arg_fb_ptr = op->dim_arg();
    std::optional<vector<int>> dim_arg =
        dim_arg_fb_ptr ? std::make_optional(std::vector<int>(
                             dim_arg_fb_ptr->begin(), dim_arg_fb_ptr->end()))
                       : std::nullopt;

    tensorPool.push_back(::ttnn::mean(in, dim_arg, op->keep_dim()));

    liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    break;
  }
  }
}

static void
run(::tt::target::ttnn::SoftmaxOp const *op, ::ttnn::device::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  ::ttnn::Tensor &in = *liveTensors.at(op->in()->global_id());
  int32_t dimension = op->dimension();

  tensorPool.push_back(::ttnn::softmax(in, dimension));
  liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
}

static void
run(::tt::target::ttnn::TransposeOp const *op, ::ttnn::device::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  ::ttnn::Tensor &in = *liveTensors.at(op->in()->global_id());
  int32_t dimension1 = op->dimension1();
  int32_t dimension2 = op->dimension2();
  auto input_rank = in.get_shape().rank();
  std::vector<std::int64_t> dimensionOrder(input_rank);
  std::iota(dimensionOrder.begin(), dimensionOrder.end(), 0);
  if (dimension1 < 0) {
    dimension1 += input_rank;
  }
  if (dimension2 < 0) {
    dimension2 += input_rank;
  }
  std::swap(dimensionOrder[dimension1], dimensionOrder[dimension2]);
  tensorPool.push_back(::ttnn::permute(in, dimensionOrder));
  liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
}

// ANCHOR: adding_an_op_matmul_runtime
static void
run(::tt::target::ttnn::MatmulOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  auto &lhs = *liveTensors.at(op->in0()->global_id());
  auto &rhs = *liveTensors.at(op->in1()->global_id());
  tensorPool.push_back(::ttnn::operations::matmul::matmul(
      lhs, rhs, std::nullopt, ::tt::operations::primary::Matmul{}));
  liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
}
// ANCHOR_END: adding_an_op_matmul_runtime

static void
run(::tt::target::ttnn::Operation const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  switch (op->type_type()) {
  case ::tt::target::ttnn::OpType::OpenDeviceOp: {
    // Skip for now, do we want device externally supplied?
    break;
  }
  case ::tt::target::ttnn::OpType::CloseDeviceOp: {
    // Skip for now, do we want device externally supplied?
    break;
  }
  case ::tt::target::ttnn::OpType::ToMemoryConfigOp: {
    return run(op->type_as_ToMemoryConfigOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::EmptyOp: {
    return run(op->type_as_EmptyOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::FullOp: {
    // Skip for now, we need an empty op
    break;
  }
  case ::tt::target::ttnn::OpType::EltwiseOp: {
    return run(op->type_as_EltwiseOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::MatmulOp: {
    return run(op->type_as_MatmulOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::ReductionOp: {
    return run(op->type_as_ReductionOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::SoftmaxOp: {
    return run(op->type_as_SoftmaxOp(), device, liveTensors, tensorPool);
  }
  case ::tt::target::ttnn::OpType::TransposeOp: {
    return run(op->type_as_TransposeOp(), device, liveTensors, tensorPool);
  }
  default:
    throw std::runtime_error("Unsupported operation type");
  }
}

void runProgram(::ttnn::Device &device,
                ::tt::target::ttnn::Program const *program,
                std::vector<::ttnn::Tensor *> const &inputs,
                std::vector<::ttnn::Tensor *> const &outputs) {
  std::unordered_map<std::uint32_t, ::ttnn::Tensor *> liveTensors;
  std::list<::ttnn::Tensor> tensorPool;

  int inputIndex = 0;
  assert(program->inputs()->size() == inputs.size() &&
         "Mismatch between program inputs and input tensors");
  for (::tt::target::TensorRef const *input : *program->inputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(input->global_id(), inputs[inputIndex++]);
    assert(inserted && "Duplicate input tensor");
  }

  int outputIndex = 0;
  assert(program->outputs()->size() == outputs.size() &&
         "Mismatch between program outputs and output tensors");
  for (::tt::target::TensorRef const *output : *program->outputs()) {
    auto [iter, inserted] =
        liveTensors.try_emplace(output->global_id(), outputs[outputIndex++]);
    assert(inserted && "Duplicate output tensor");
  }

  for (::tt::target::ttnn::Operation const *op : *program->operations()) {
    run(op, device, liveTensors, tensorPool);
  }
}
} // namespace tt::runtime::ttnn
