// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/nlp_concat_heads_decode.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/NLPConcatHeadsDecodeOp.h"
#include <variant>

namespace tt::runtime::ttnn::operations::transformer {

void run(const ::tt::target::ttnn::NLPConcatHeadsDecodeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::tt::target::ttnn::NLPConcatHeadsDecodeOpT nlpConcatHeadsDecodeOpNative;
  op->UnPackTo(&nlpConcatHeadsDecodeOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::NLPConcatHeadsDecodeOpResult result =
      ttnn_op_invoke::callNLPConcatHeadsDecode(
          ttnn_op_invoke::CallType::EXECUTE, nlpConcatHeadsDecodeOpNative, &in,
          &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callNLPConcatHeadsDecode execution");

  ::ttnn::Tensor output = std::get<::ttnn::Tensor>(result);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}

} // namespace tt::runtime::ttnn::operations::transformer
