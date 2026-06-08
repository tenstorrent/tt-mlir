// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/transformer/split_query_key_value_and_split_heads.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/Transformer/SplitQueryKeyValueAndSplitHeadsOp.h"
#include <tuple>
#include <variant>

namespace tt::runtime::ttnn::operations::transformer {
static void runSplitQueryKeyValueAndSplitHeadsOp(
    const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOp *op,
    ProgramTensorPool &tensorPool, ProgramContext &context) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  std::optional<::ttnn::Tensor> kvInput;
  if (op->kv_input()) {
    kvInput.emplace(tensorPool.getTTNNTensorAndValidate(op->kv_input()));
  }

  ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOpT
      splitQueryKeyValueAndSplitHeadsOpNative;
  op->UnPackTo(&splitQueryKeyValueAndSplitHeadsOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::SplitQueryKeyValueAndSplitHeadsOpResult result =
      ttnn_op_invoke::callSplitQueryKeyValueAndSplitHeads(
          ttnn_op_invoke::CallType::EXECUTE,
          splitQueryKeyValueAndSplitHeadsOpNative, &in,
          kvInput.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*kvInput)
              : std::nullopt,
          &targetDevice);

  using QKVTuple = std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor>;
  LOG_ASSERT(std::holds_alternative<QKVTuple>(result),
             "Expected Tensor tuple from callSplitQueryKeyValueAndSplitHeads "
             "execution");

  auto &[q, k, v] = std::get<QKVTuple>(result);

  tensorPool.insertTTNNTensorAndValidate(op->q_out(), q);
  tensorPool.insertTTNNTensorAndValidate(op->k_out(), k);
  tensorPool.insertTTNNTensorAndValidate(op->v_out(), v);
}

void run(const ::tt::target::ttnn::SplitQueryKeyValueAndSplitHeadsOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  runSplitQueryKeyValueAndSplitHeadsOp(op, tensorPool, context);
}
} // namespace tt::runtime::ttnn::operations::transformer
