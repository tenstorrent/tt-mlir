// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/paged_scaled_dot_product_attention_decode.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/PagedScaledDotProductAttentionDecodeOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::transformer {
static void runPagedScaledDotProductAttentionDecodeOp(
    const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOp *op,
    ProgramTensorPool &tensorPool, ProgramContext &context) {
  const ::ttnn::Tensor &query =
      tensorPool.getTTNNTensorAndValidate(op->query());
  const ::ttnn::Tensor &key = tensorPool.getTTNNTensorAndValidate(op->key());
  const ::ttnn::Tensor &value =
      tensorPool.getTTNNTensorAndValidate(op->value());
  const ::ttnn::Tensor &pageTable =
      tensorPool.getTTNNTensorAndValidate(op->page_table());

  std::optional<::ttnn::Tensor> attentionMask = std::nullopt;
  if (op->attention_mask()) {
    attentionMask.emplace(
        tensorPool.getTTNNTensorAndValidate(op->attention_mask()));
  }

  std::optional<::ttnn::Tensor> curPosTensor = std::nullopt;
  if (op->cur_pos_tensor()) {
    curPosTensor.emplace(
        tensorPool.getTTNNTensorAndValidate(op->cur_pos_tensor()));
  }

  std::optional<::ttnn::Tensor> attentionSink = std::nullopt;
  if (op->attention_sink()) {
    attentionSink.emplace(
        tensorPool.getTTNNTensorAndValidate(op->attention_sink()));
  }

  ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOpT
      pagedScaledDotProductAttentionDecodeOpNative;
  op->UnPackTo(&pagedScaledDotProductAttentionDecodeOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::PagedScaledDotProductAttentionDecodeOpResult result =
      ttnn_op_invoke::callPagedScaledDotProductAttentionDecode(
          ttnn_op_invoke::CallType::EXECUTE,
          pagedScaledDotProductAttentionDecodeOpNative, &query, &key, &value,
          &pageTable,
          attentionMask.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*attentionMask)
              : std::nullopt,
          curPosTensor.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*curPosTensor)
              : std::nullopt,
          attentionSink.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*attentionSink)
              : std::nullopt,
          &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from "
             "callPagedScaledDotProductAttentionDecode execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);
  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

void run(const ::tt::target::ttnn::PagedScaledDotProductAttentionDecodeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runPagedScaledDotProductAttentionDecodeOp(op, tensorPool, context);
}

} // namespace tt::runtime::ttnn::operations::transformer
