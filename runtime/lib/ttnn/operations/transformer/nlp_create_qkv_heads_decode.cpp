// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/nlp_create_qkv_heads_decode.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/NLPCreateQKVHeadsDecodeOp.h"
#include <tuple>
#include <variant>

namespace tt::runtime::ttnn::operations::transformer {

void run(const ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());

  std::optional<::ttnn::Tensor> batchOffset;
  if (op->batch_offset()) {
    batchOffset = tensorPool.getTTNNTensorAndValidate(op->batch_offset());
  }

  ::tt::target::ttnn::NLPCreateQKVHeadsDecodeOpT
      nlpCreateQkvHeadsDecodeOpNative;
  op->UnPackTo(&nlpCreateQkvHeadsDecodeOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::NLPCreateQKVHeadsDecodeOpResult result =
      ttnn_op_invoke::callNLPCreateQKVHeadsDecode(
          ttnn_op_invoke::CallType::EXECUTE, nlpCreateQkvHeadsDecodeOpNative,
          &input,
          batchOffset.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*batchOffset)
              : std::nullopt,
          &targetDevice);

  using QKVTuple = std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor>;
  LOG_ASSERT(
      std::holds_alternative<QKVTuple>(result),
      "Expected Tensor tuple from callNLPCreateQKVHeadsDecode execution");

  auto &[q, k, v] = std::get<QKVTuple>(result);

  tensorPool.insertTTNNTensorAndValidate(op->q_out(), q);
  tensorPool.insertTTNNTensorAndValidate(op->k_out(), k);
  tensorPool.insertTTNNTensorAndValidate(op->v_out(), v);
}

} // namespace tt::runtime::ttnn::operations::transformer
