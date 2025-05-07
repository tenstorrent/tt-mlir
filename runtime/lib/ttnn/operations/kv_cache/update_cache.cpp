// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/kv_cache/update_cache.h"

#include "tt/runtime/detail/logger.h"

#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::kv_cache {
void run(const ::tt::target::ttnn::UpdateCacheOp *op, ProgramContext &context) {

  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &cache =
      tensorPool.getTTNNTensorAndValidate(op->cache());
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &updateIndex =
      tensorPool.getTTNNTensorAndValidate(op->update_index());

  if (workaround::Env::get().readUpdateIndexFromDeviceForKVCache) {

    const ::ttnn::Tensor indexOnHost = ::ttnn::from_device(updateIndex);
    const auto &storage = indexOnHost.get_storage();
    const auto &multiDeviceHostStorage =
        std::get<tt_metal::MultiDeviceHostStorage>(storage);
    const auto &buffer = multiDeviceHostStorage.get_buffer(0);
    const auto &buf = buffer.view_as<uint32_t>();
    uint32_t upIdx = *buf.begin();

    ::ttnn::update_cache(cache, input, upIdx, op->batch_offset(), std::nullopt);
  } else {
    LOG_FATAL("Currently, the only way to execute ttnn::update_cache is to use "
              "the workaround enabled by the flag "
              "\"readUpdateIndexFromDeviceForKVCache\"");
  }
}
} // namespace tt::runtime::ttnn::operations::kv_cache
