// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/matmul/matmul.h"

#include "matmul/unifiedMatmulOp.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "utils/utils.h"

#include <algorithm>
#include <optional>

namespace tt::runtime::ttnn::operations::matmul {

// ANCHOR: adding_an_op_matmul_runtime_operations
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.getTTNNTensorAndValidate(op->a());
  const ::ttnn::Tensor &rhs = tensorPool.getTTNNTensorAndValidate(op->b());

  target::ttnn::MatmulOpT matmulOpT;
  op->UnPackTo(&matmulOpT);

  unifiedOpLib::MatmulOpResult result = unifiedOpLib::callMatmul(
      unifiedOpLib::CallType::EXECUTE, matmulOpT, &lhs, &rhs);

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

  target::ttnn::LinearOpT linearOpT;
  op->UnPackTo(&linearOpT);

  unifiedOpLib::LinearOpResult result = unifiedOpLib::callLinear(
      unifiedOpLib::CallType::EXECUTE, linearOpT, &lhs, &rhs,
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

  target::ttnn::SparseMatmulOpT sparseMatmulOpT;
  op->UnPackTo(&sparseMatmulOpT);

  unifiedOpLib::LinearOpResult result = unifiedOpLib::callSparseMatmul(
      unifiedOpLib::CallType::EXECUTE, sparseMatmulOpT, &a, &b, &sparsity);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected output Tensor from callSparseMatmul execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::matmul
