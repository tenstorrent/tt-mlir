// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/eltwise/ternary/where.h"
#include "eltwise/ternary/unifiedEltwiseTernaryOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/Target/TTNN/operations/eltwise_generated.h"

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

  target::ttnn::EltwiseTernaryWhereOpT eltwiseTernaryWhereOpT;
  op->UnPackTo(&eltwiseTernaryWhereOpT);

  //   std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
  //       ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
  //           op->memory_config());
  //   LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
  //                  outputMemoryConfig.has_value(),
  //              "Memory config must exist for device tensors");

  unifiedOpLib::EltwiseTernaryOpResult result =
      unifiedOpLib::callEltwiseTernary(
          unifiedOpLib::CallType::EXECUTE, eltwiseTernaryWhereOpT,
          WRAP_OP(::ttnn::where), &first, &second, &third);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callEltwiseTernary execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  //   ::ttnn::Tensor out = ::ttnn::where(first, second, third,
  //   outputMemoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::EltwiseTernaryWhereOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runEltwiseTernaryWhereOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::eltwise::ternary
