// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/eltwise/binary/binary_composite.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/eltwise/binary/utils.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::binary::composite {

static void runEltwiseBinaryCompositeOp(
    const ::tt::target::ttnn::EltwiseOp *op, ProgramTensorPool &tensorPool,
    const std::function<
        ::ttnn::Tensor(const ::ttnn::Tensor &, const ::ttnn::Tensor &,
                       const std::optional<::ttnn::MemoryConfig> &)> &ttnnOp) {

  ::ttnn::Tensor lhs, rhs;
  getEltwiseBinaryOpInputTensors(op, tensorPool, lhs, rhs);

  // TODO (#2272): Support for int32 is added in #2272
  // However to_layout ops are not cannonicalized properly, blocking #2272
  // This is a hack to unblock metal uplifts for now until #2272 is merged
  if (lhs.get_dtype() == ::ttnn::DataType::UINT32) {
    lhs = ::ttnn::typecast(lhs, ::ttnn::DataType::INT32);
  }
  if (rhs.get_dtype() == ::ttnn::DataType::UINT32) {
    rhs = ::ttnn::typecast(rhs, ::ttnn::DataType::INT32);
  }

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  ::ttnn::Tensor out = ttnnOp(lhs, rhs, outputMemoryConfig);

  // TODO (#2272): Support for int32 is added in #2272
  // However to_layout ops are not cannonicalized properly, blocking #2272
  // This is a hack to unblock metal uplifts for now until #2272 is merged
  ::ttnn::DataType outputDataType = utils::getDataType(op->out());
  if (out.get_dtype() == ::ttnn::DataType::INT32 &&
      outputDataType == ::ttnn::DataType::UINT32) {
    out = ::ttnn::typecast(out, ::ttnn::DataType::UINT32);
  }

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

void run(const ::tt::target::ttnn::EltwiseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  switch (op->type()) {
  case ::tt::target::ttnn::EltwiseOpType::Maximum: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::maximum);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Minimum: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::minimum);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Remainder: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::remainder);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Scatter: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::scatter);
    break;
  }
  case ::tt::target::ttnn::EltwiseOpType::Power: {
    runEltwiseBinaryCompositeOp(op, tensorPool, ::ttnn::pow);
    break;
  }
  default:
    LOG_FATAL("Unsupported Eltwise Binary Composite operation");
  }
}

} // namespace tt::runtime::ttnn::operations::binary::composite
