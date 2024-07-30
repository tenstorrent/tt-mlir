// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <list>
#include <optional>
#include <unordered_map>

#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/runtime.h"
#include "ttnn/runtime/utils.h"

#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-qualifiers"
// Including this in ttnn.h causes multiple definition linker error
// due to non-inlined function definitions
#include "ttnn/operations/unary.hpp"
#pragma clang diagnostic pop

// It seems like `ttnn::to_layout` cannot be called inside of the
// `tt::runtime::ttnn` namespace.  TTNN uses a lot of metaprogramming and for
// some reason a static_assert fails when this is called from within our
// namespace.
ttnn::Tensor tilize(ttnn::Tensor const &input) {
  return ttnn::to_layout(input, ttnn::TILE_LAYOUT, std::nullopt, std::nullopt,
                         (Device *)nullptr);
}

namespace tt::runtime::ttnn {
static void
run(::tt::target::ttnn::ToMemoryConfigOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  ::tt::target::DataType targetDataType =
      op->out()->desc()->layout()->memory_desc()->data_type();
  assert(targetDataType == ::tt::target::DataType::Float32 or
         targetDataType == ::tt::target::DataType::BFloat16);
  ::ttnn::DataType targetDataTypeTTNN = utils::toTTNNDataType(targetDataType);
  const ::ttnn::Tensor &inputTensor = *liveTensors.at(op->in0()->global_id());
  assert(inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED or
         inputTensor.storage_type() == ::tt::tt_metal::StorageType::DEVICE);
  const ::tt::target::MemorySpace targetMemorySpace =
      op->out()->desc()->layout()->memory_desc()->memory_space();
  switch (targetMemorySpace) {
  case ::tt::target::MemorySpace::System:
  case ::tt::target::MemorySpace::SystemMMIO: {
    if (inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED) {
      ::ttnn::Tensor hostTensor = inputTensor.to(::ttnn::ROW_MAJOR_LAYOUT);
      hostTensor = ::ttnn::to_dtype(hostTensor, targetDataTypeTTNN);
      ::ttnn::Tensor &outputTensor = *liveTensors.at(op->out()->global_id());
      void *src = ::tt::tt_metal::get_raw_host_data_ptr(hostTensor);
      void *dst = ::tt::tt_metal::get_raw_host_data_ptr(outputTensor);
      std::uint32_t size = hostTensor.volume() * hostTensor.element_size();
      std::memcpy(dst, src, size);
    } else if (inputTensor.storage_type() ==
               ::tt::tt_metal::StorageType::DEVICE) {
      ::ttnn::Tensor hostTensor =
          ::ttnn::typecast(inputTensor, targetDataTypeTTNN);
      // Following the flow in core.py::to_torch - untilize on host
      hostTensor = hostTensor.cpu().to(::ttnn::ROW_MAJOR_LAYOUT);
      ::ttnn::Tensor &outputTensor = *liveTensors.at(op->out()->global_id());
      void *src = ::tt::tt_metal::get_raw_host_data_ptr(hostTensor);
      void *dst = ::tt::tt_metal::get_raw_host_data_ptr(outputTensor);
      std::uint32_t size = hostTensor.volume() * hostTensor.element_size();
      std::memcpy(dst, src, size);
    }
    break;
  }
  case ::tt::target::MemorySpace::DeviceDRAM: {
    ::tt::tt_metal::MemoryConfig memConfig = ::ttnn::DRAM_MEMORY_CONFIG;
    // Host tensor, currently only supports borrowed storage
    if (inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED) {
      // moving to device first allows us to use device tilize
      ::ttnn::Tensor deviceTensor =
          ::ttnn::to_device(inputTensor, &device, memConfig);
      deviceTensor = ::tilize(deviceTensor);
      if (deviceTensor.get_dtype() != targetDataTypeTTNN) {
        deviceTensor = ::ttnn::typecast(deviceTensor, targetDataTypeTTNN);
      }
      tensorPool.push_back(deviceTensor);
      liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
      // Device tensor, currently only support single-device storage
      // Since tensor already on device, update the memory config and break
    } else if (inputTensor.storage_type() ==
               ::tt::tt_metal::StorageType::DEVICE) {
      // Dram to L1 or Dram to Dram
      ::ttnn::Tensor deviceTensor =
          ::ttnn::to_memory_config(inputTensor, memConfig, targetDataTypeTTNN);
      if (deviceTensor.get_dtype() != targetDataTypeTTNN) {
        deviceTensor = ::ttnn::typecast(deviceTensor, targetDataTypeTTNN);
      }
      tensorPool.push_back(deviceTensor);
      liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    }
    break;
  }
  // Currently similar to ::tt::target::MemorySpace::DeviceDRAM
  // But will need it's own code path when we add support for sharding
  case ::tt::target::MemorySpace::DeviceL1: {
    ::tt::tt_metal::MemoryConfig memConfig = ::ttnn::L1_MEMORY_CONFIG;
    // Host tensor, currently only supports borrowed storage
    if (inputTensor.storage_type() == ::tt::tt_metal::StorageType::BORROWED) {
      // moving to device first allows us to use device tilize
      ::ttnn::Tensor deviceTensor =
          ::ttnn::to_device(inputTensor, &device, memConfig);
      deviceTensor = ::tilize(deviceTensor);
      if (deviceTensor.get_dtype() != targetDataTypeTTNN) {
        deviceTensor = ::ttnn::typecast(deviceTensor, targetDataTypeTTNN);
      }
      tensorPool.push_back(deviceTensor);
      liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
      // Device tensor, currently only support single-device storage
      // Since tensor already on device, update the memory config and break
    } else if (inputTensor.storage_type() ==
               ::tt::tt_metal::StorageType::DEVICE) {
      // L1 to Dram or L1 to L1
      ::ttnn::Tensor deviceTensor =
          ::ttnn::to_memory_config(inputTensor, memConfig, targetDataTypeTTNN);
      if (deviceTensor.get_dtype() != targetDataTypeTTNN) {
        deviceTensor = ::ttnn::typecast(deviceTensor, targetDataTypeTTNN);
      }
      tensorPool.push_back(deviceTensor);
      liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    }
    break;
  }
  }
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

// ANCHOR: adding_an_op_matmul_runtime
static void
run(::tt::target::ttnn::MatmulOp const *op, ::ttnn::Device &device,
    std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
    std::list<::ttnn::Tensor> &tensorPool) {
  auto &lhs = *liveTensors.at(op->in0()->global_id());
  auto &rhs = *liveTensors.at(op->in1()->global_id());
  tensorPool.push_back(
      ::ttnn::operations::matmul::matmul(lhs, rhs, std::nullopt));
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
