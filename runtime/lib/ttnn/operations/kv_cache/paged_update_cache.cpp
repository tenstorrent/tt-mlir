// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/paged_update_cache.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/workarounds.h"
#include "ttnn/distributed/api.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::PagedUpdateCacheOp *op,
         ProgramContext &context) {

  auto cacheTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->cache());
  auto inputTensor =
      context.getTensorPool().getTTNNTensorAndValidate(op->input());
  auto updateIndexTensor =
      op->update_index()
          ? std::make_optional(context.getTensorPool().getTTNNTensorAndValidate(
                op->update_index()))
          : std::nullopt;
  bool shareCache = op->share_cache();
  auto pageTableTensor =
      op->page_table()
          ? std::make_optional(context.getTensorPool().getTTNNTensorAndValidate(
                op->page_table()))
          : std::nullopt;

  const std::vector<uint32_t> emptyUpdateIndex = {};
  ::ttnn::experimental::paged_update_cache(
      cacheTensor, inputTensor, emptyUpdateIndex, updateIndexTensor, shareCache,
      pageTableTensor,
      /*batch_offset=*/0,
      /*compute_kernel_config*/ std::nullopt,
      /*mesh_coords*/ std::nullopt);
}
} // namespace tt::runtime::ttnn::operations::kv_cache
