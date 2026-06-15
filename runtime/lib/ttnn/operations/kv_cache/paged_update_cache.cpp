// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/paged_update_cache.h"
#include "tt/runtime/detail/common/logger.h"
#include "ttmlir/OpInvoke/TTNN/KVCache/PagedUpdateCacheOp.h"

namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::PagedUpdateCacheOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &cache =
      tensorPool.getTTNNTensorAndValidate(op->cache());
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const std::optional<::ttnn::Tensor> &updateIndexTensor =
      op->update_index()
          ? std::make_optional(context.getTensorPool().getTTNNTensorAndValidate(
                op->update_index()))
          : std::nullopt;

  const std::optional<::ttnn::Tensor> &pageTableTensor =
      op->page_table()
          ? std::make_optional(context.getTensorPool().getTTNNTensorAndValidate(
                op->page_table()))
          : std::nullopt;

  ::tt::target::ttnn::PagedUpdateCacheOpT opNative;
  op->UnPackTo(&opNative);

  ::ttnn::MeshDevice &targetDevice = context.getMeshDevice();

  ttnn_op_invoke::PagedUpdateCacheOpResult result =
      ttnn_op_invoke::callPagedUpdateCache(
          ttnn_op_invoke::CallType::EXECUTE, opNative, &cache, &input,
          updateIndexTensor.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*updateIndexTensor)
              : std::nullopt,
          pageTableTensor.has_value()
              ? std::optional<ttnn_op_invoke::TensorArg>(&*pageTableTensor)
              : std::nullopt,
          &targetDevice);

  LOG_ASSERT(std::holds_alternative<::ttnn::Tensor>(result),
             "Expected Tensor from callPagedUpdateCache execution");
}
} // namespace tt::runtime::ttnn::operations::kv_cache
