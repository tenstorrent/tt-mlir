// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/eltwise/ternary/where.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/eltwise/ternary/eltwiseTernaryOp.h"
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

  ttnn_op_invoke::EltwiseTernaryOpResult result =
      ttnn_op_invoke::callEltwiseTernary(
          ttnn_op_invoke::CallType::EXECUTE, eltwiseTernaryWhereOpT,
          WRAP_OP(::ttnn::where), &first, &second, &third);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callEltwiseTernary execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::EltwiseTernaryWhereOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runEltwiseTernaryWhereOp(op, tensorPool);
}
} // namespace tt::runtime::ttnn::operations::eltwise::ternary
