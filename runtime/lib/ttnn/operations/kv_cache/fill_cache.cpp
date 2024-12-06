// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_cache.h"

namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::FillCacheOp *op, ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &cache = tensorPool.at(op->cache()->global_id());
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());

  ::ttnn::fill_cache(cache, input, op->batch_offset());
}
} // namespace tt::runtime::ttnn::operations::kv_cache
