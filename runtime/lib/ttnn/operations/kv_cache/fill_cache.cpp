// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/fill_cache.h"
#include "tt/runtime/detail/common/logger.h"

namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::FillCacheOp *op, ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &cache =
      tensorPool.getTTNNTensorAndValidate(op->cache());
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());

  ::ttnn::fill_cache(cache, input, op->batch_offset());
  ::ttnn::set_printoptions("full");
  auto cache_on_host = ::ttnn::from_device(cache);
}
} // namespace tt::runtime::ttnn::operations::kv_cache
