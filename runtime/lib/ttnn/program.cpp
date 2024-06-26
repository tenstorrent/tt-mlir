// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <list>
#include <unordered_map>

#include "tt/runtime/runtime.h"
#include "tt/runtime/detail/ttnn.h"

#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Version.h"

namespace tt::runtime::ttnn {
static void run(::tt::target::ttnn::ToMemoryConfigOp const *op,
                ::ttnn::Device &device,
                std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
                  std::list<::ttnn::Tensor> &tensorPool) {
  if (op->out()->desc()->layout()->memory_desc()->memory_space() ==
      ::tt::target::MemorySpace::System) {
    auto &inputTensor = *liveTensors.at(op->in0()->global_id());
    auto cpu = inputTensor.cpu();
    auto untilized = ::tt::tt_metal::tensor_impl::to_layout<float>(
        cpu, ::ttnn::ROW_MAJOR_LAYOUT);
    auto &outputTensor = *liveTensors.at(op->out()->global_id());
    void *src = ::tt::tt_metal::get_raw_host_data_ptr(untilized);
    void *dst = ::tt::tt_metal::get_raw_host_data_ptr(outputTensor);
    std::uint32_t size = untilized.volume() * untilized.element_size();
    std::memcpy(dst, src, size);
    return;
  }
  bool isL1 = op->in0()->desc()->layout()->memory_desc()->memory_space() ==
               ::tt::target::MemorySpace::DeviceL1;
  const auto memoryConfig =
      isL1 ? ::ttnn::L1_MEMORY_CONFIG : ::ttnn::DRAM_MEMORY_CONFIG;
  auto &inputTensor = *liveTensors.at(op->in0()->global_id());
  auto tilized = ::tt::tt_metal::tensor_impl::to_layout<float>(
      inputTensor, ::ttnn::TILE_LAYOUT);
  //::ttnn::to_layout(inputTensor, ::ttnn::TILE_LAYOUT, std::nullopt,
  //                  std::nullopt, (Device *)nullptr);
  auto deviceTensor = ::ttnn::to_device(tilized, &device, memoryConfig);
  tensorPool.push_back(deviceTensor);
  // auto [iter, inserted] =
  liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
  // assert(inserted && "Duplicate output tensor");
}

static void run(::tt::target::ttnn::EltwiseOp const *op, ::ttnn::Device &device,
                std::unordered_map<std::uint32_t, ::ttnn::Tensor *> &liveTensors,
                std::list<::ttnn::Tensor> &tensorPool) {
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Multiply: {
    assert(op->ins()->size() == 2 && "Unsupported number of inputs");
    auto &lhs = *liveTensors.at(op->ins()->Get(0)->global_id());
    auto &rhs = *liveTensors.at(op->ins()->Get(1)->global_id());
    tensorPool.push_back(::ttnn::multiply(lhs, rhs));
    // auto [iter, inserted] =
    liveTensors.try_emplace(op->out()->global_id(), &tensorPool.back());
    // assert(inserted && "Duplicate output tensor");
    break;
  }
  default:
    throw std::runtime_error("Unsupported elementwise operation type");
  }
}

static void run(::tt::target::ttnn::Operation const *op, ::ttnn::Device &device,
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
