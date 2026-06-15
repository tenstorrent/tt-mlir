// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/fill_cache.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/KVCache/FillCacheOp.h"

namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::FillCacheOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &cache =
      tensorPool.getTTNNTensorAndValidate(op->cache());
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());

  ::tt::target::ttnn::FillCacheOpT fillCacheOpNative;
  op->UnPackTo(&fillCacheOpNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::FillCacheOpResult result = ttnn_op_invoke::callFillCache(
      ttnn_op_invoke::CallType::EXECUTE, fillCacheOpNative, &cache, &input,
      &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callFillCache execution");
}
} // namespace tt::runtime::ttnn::operations::kv_cache
