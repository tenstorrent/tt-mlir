// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/eltwise/ternary/where.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <ttnn/tensor/tensor.hpp>

namespace tt::runtime::ttnn::operations::eltwise::ternary {

static void
runEltwiseTernaryWhereOp(const ::tt::target::ttnn::EltwiseTernaryWhereOp *op,
                         ProgramTensorPool &tensorPool) {

  const ::ttnn::Tensor &first =
      tensorPool.getTTNNTensorAndValidate(op->first());
  const ::ttnn::Tensor &second =
      tensorPool.getTTNNTensorAndValidate(op->second());
  const ::ttnn::Tensor &third =
      tensorPool.getTTNNTensorAndValidate(op->third());
  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  //::ttnn::Tensor out = ::ttnn::where(first, second, third,
  //: outputMemoryConfig);

  using std::vector;

  vector<float> first_cpu = first.to_vector<float>();
  vector<float> second_cpu = second.to_vector<float>();
  vector<float> third_cpu = third.to_vector<float>();

  if (first_cpu.size() != second_cpu.size() ||
      first_cpu.size() != third_cpu.size()) {
    LOG_ERROR("Input tensors must have the same size");
    abort();
  }

  vector<float> result;
  result.reserve(first_cpu.size());

  for (unsigned long i = 0; i < first_cpu.size(); ++i) {
    if (first_cpu[i] != 0) {
      result.push_back(second_cpu[i]);
    } else {
      result.push_back(third_cpu[i]);
    }
  }

  ::ttnn::TensorSpec out_spec = first.tensor_spec();

  ::ttnn::Tensor out_cpu =
      ::ttnn::Tensor::from_vector<float>(result, out_spec, first.mesh_device());

  tensorPool.insertTTNNTensorAndValidate(op->out(), out_cpu);
}

void run(const ::tt::target::ttnn::EltwiseTernaryWhereOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runEltwiseTernaryWhereOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::eltwise::ternary
