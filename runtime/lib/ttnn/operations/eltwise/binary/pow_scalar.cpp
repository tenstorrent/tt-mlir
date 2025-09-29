// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/pow_scalar.h"
#include "operations/data_movement/scatter.h"
#include "operations/eltwise/binary/binary.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include "ttmlir/Target/TTNN/operations/eltwise_generated.h"

namespace tt::runtime::ttnn::operations::eltwise::binary {

static void run(const ::tt::target::ttnn::PowScalarOp *op, auto &&exponent,
                ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor *input = &(tensorPool.getTTNNTensorAndValidate(op->input()));

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ::ttnn::pow(*input, exponent, outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::PowScalarOp *op, ProgramContext &context) {
  switch (op->exponent_type()) {
  case ::tt::target::ttnn::ExponentType::FP:
    run(op, op->exponent_as_FP()->value(), context);
    break;
  case ::tt::target::ttnn::ExponentType::UI32:
    run(op, op->exponent_as_UI32()->value(), context);
    break;
  default:
    LOG_FATAL("unknown fill value type");
  }
}

} // namespace tt::runtime::ttnn::operations::eltwise::binary
