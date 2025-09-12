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
  if (const char* print_cache = std::getenv("PRINT_CACHE")) {
    ::ttnn::set_printoptions("full");
    std::cout <<"Fill cache op before:" <<std::endl;
    cache.print();
  }
  
  ::ttnn::fill_cache(cache, input, op->batch_offset());
  
  if (const char* print_cache = std::getenv("PRINT_CACHE")) {
    std::cout <<"Fill cache op after:" <<std::endl;
    cache.print();
  }
}
} // namespace tt::runtime::ttnn::operations::kv_cache
