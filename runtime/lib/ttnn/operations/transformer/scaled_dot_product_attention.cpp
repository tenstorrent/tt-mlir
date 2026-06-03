// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/scaled_dot_product_attention.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "ttmlir/OpInvoke/TTNN/transformer/scaledDotProductAttentionOp.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include <variant>

namespace tt::runtime::ttnn::operations::transformer {
static void runScaledDotProductAttentionOp(
    const ::tt::target::ttnn::ScaledDotProductAttentionOp *op,
    ProgramTensorPool &tensorPool, ProgramContext &context) {
  const ::ttnn::Tensor &query =
      tensorPool.getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &key = tensorPool.getTTNNTensorAndValidate(op->key());
  const ::ttnn::Tensor &value =
      tensorPool.getTTNNTensorAndValidate(op->value());

  std::optional<::ttnn::Tensor> attentionMask = std::nullopt;
  if (op->attention_mask()) {
    attentionMask.emplace(
        tensorPool.getTTNNTensorAndValidate(op->attention_mask()));
  }

  std::optional<::ttnn::Tensor> attentionSink = std::nullopt;
  if (op->attention_sink()) {
    attentionSink.emplace(
        tensorPool.getTTNNTensorAndValidate(op->attention_sink()));
  }

  ::tt::target::ttnn::ScaledDotProductAttentionOpT opT;
  op->UnPackTo(&opT);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::ScaledDotProductAttentionOpResult result =
      ttnn_op_invoke::callScaledDotProductAttention(
          ttnn_op_invoke::CallType::EXECUTE, opT, &query, &key, &value,
          attentionMask.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*attentionMask)
              : std::nullopt,
          attentionSink.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*attentionSink)
              : std::nullopt,
          targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callScaledDotProductAttention execution");
  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::ScaledDotProductAttentionOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runScaledDotProductAttentionOp(op, tensorPool, context);
}

} // namespace tt::runtime::ttnn::operations::transformer
