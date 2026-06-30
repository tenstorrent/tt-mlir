// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/paged_fill_cache.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/KVCache/PagedFillCacheOp.h"

namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::PagedFillCacheOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &cache =
      tensorPool.getTTNNTensorAndValidate(op->cache());
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &pageTable =
      tensorPool.getTTNNTensorAndValidate(op->page_table());

  std::optional<::ttnn::Tensor> batchIdxTensor;
  if (op->batch_idx_tensor()) {
    batchIdxTensor =
        tensorPool.getTTNNTensorAndValidate(op->batch_idx_tensor());
  }

  ::tt::target::ttnn::PagedFillCacheOpT opNative;
  op->UnPackTo(&opNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::PagedFillCacheOpResult result =
      ttnn_op_invoke::callPagedFillCache(
          ttnn_op_invoke::CallType::EXECUTE, opNative, &cache, &input,
          &pageTable,
          batchIdxTensor.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*batchIdxTensor)
              : std::nullopt,
          &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callPagedFillCache execution");
}
} // namespace tt::runtime::ttnn::operations::kv_cache
