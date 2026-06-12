// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/matmul/matmul.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttmlir/OpInvoke/TTNN/Matmul/MatmulOp.h"
#include "ttmlir/OpInvoke/TTNN/utils/utils.h"

#include <algorithm>
#include <optional>

namespace tt::runtime::ttnn::operations::matmul {

// ANCHOR: adding_an_op_matmul_runtime_operations
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.getTTNNTensorAndValidate(op->a());
  const ::ttnn::Tensor &rhs = tensorPool.getTTNNTensorAndValidate(op->b());

  target::ttnn::MatmulOpT matmulOpNative;
  op->UnPackTo(&matmulOpNative);

  ttnn_op_invoke::MatmulOpResult result = ttnn_op_invoke::callMatmul(
      ttnn_op_invoke::CallType::EXECUTE, matmulOpNative, &lhs, &rhs);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callMatmul execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
// ANCHOR_END: adding_an_op_matmul_runtime_operations

void run(const ::tt::target::ttnn::LinearOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.getTTNNTensorAndValidate(op->a());
  const ::ttnn::Tensor &rhs = tensorPool.getTTNNTensorAndValidate(op->b());
  std::optional<::ttnn::Tensor> bias =
      op->bias()
          ? std::make_optional(tensorPool.getTTNNTensorAndValidate(op->bias()))
          : std::nullopt;

  target::ttnn::LinearOpT linearOpNative;
  op->UnPackTo(&linearOpNative);

  ttnn_op_invoke::LinearOpResult result = ttnn_op_invoke::callLinear(
      ttnn_op_invoke::CallType::EXECUTE, linearOpNative, &lhs, &rhs,
      bias.has_value() ? std::make_optional(&bias.value()) : std::nullopt);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callLinear execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::SparseMatmulOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &a = tensorPool.getTTNNTensorAndValidate(op->a());
  const ::ttnn::Tensor &b = tensorPool.getTTNNTensorAndValidate(op->b());
  const ::ttnn::Tensor &sparsity =
      tensorPool.getTTNNTensorAndValidate(op->sparsity());

  target::ttnn::SparseMatmulOpT sparseMatmulOpNative;
  op->UnPackTo(&sparseMatmulOpNative);

  ttnn_op_invoke::SparseMatmulOpResult result =
      ttnn_op_invoke::callSparseMatmul(ttnn_op_invoke::CallType::EXECUTE,
                                       sparseMatmulOpNative, &a, &b, &sparsity);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callSparseMatmul execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::matmul
