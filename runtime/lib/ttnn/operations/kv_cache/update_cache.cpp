// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/update_cache.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/workarounds.h"

namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::UpdateCacheOp *op, ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &cache = tensorPool.at(op->cache()->global_id());
  const ::ttnn::Tensor &input = tensorPool.at(op->input()->global_id());
  const ::ttnn::Tensor &updateIndex =
      tensorPool.at(op->update_index()->global_id());
  if (workaround::Env::get().readUpdateIndexFromDeviceForKVCache) {

    const ::ttnn::Tensor indexOnHost = ::ttnn::from_device(updateIndex);
    const auto storage = indexOnHost.get_storage();
    const auto ownedStorage = std::get<tt_metal::OwnedStorage>(storage);
    const auto buffer = ownedStorage.get_buffer();
    const auto buf = std::get<tt_metal::owned_buffer::Buffer<uint32_t>>(buffer);
    uint32_t upIdx = *buf.begin();

    ::ttnn::update_cache(cache, input, upIdx, op->batch_offset(), std::nullopt);
  } else {
    LOG_FATAL("Currently, the only way to execute ttnn::update_cache is to use "
              "the workaround enabled by the flag "
              "\"readUpdateIndexFromDeviceForKVCache\"");
  }
}
} // namespace tt::runtime::ttnn::operations::kv_cache
