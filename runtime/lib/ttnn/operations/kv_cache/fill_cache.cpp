// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/fill_cache.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/debug_apis.h"
namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::FillCacheOp *op, ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &cache = tensorPool.getAndValidate(op->cache());
  const ::ttnn::Tensor &input = tensorPool.getAndValidate(op->input());

  ::ttnn::fill_cache(cache, input, op->batch_offset());
}
} // namespace tt::runtime::ttnn::operations::kv_cache
