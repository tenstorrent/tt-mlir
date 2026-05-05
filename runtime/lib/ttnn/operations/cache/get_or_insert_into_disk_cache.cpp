// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/get_or_insert_into_disk_cache.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/types/disk_tensor_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_impl.hpp>

namespace tt::runtime::ttnn::operations::cache {

using LogType = ::tt::runtime::logger::LogType;

void run(const ::tt::target::ttnn::GetOrInsertIntoDiskCacheOp *op,
         ProgramContext &context) {
  DiskTensorCache &cache = DiskTensorCache::getInstance();

  // If disk cache is disabled, pass through input unchanged
  if (!DiskTensorCache::isEnabled()) {
    const ::ttnn::Tensor &input =
        context.getTensorPool().getTTNNTensorAndValidate(op->input());
    context.getTensorPool().insertTTNNTensorAndValidate(op->out(), input);
    return;
  }

  std::string programHash = op->program_hash()->str();
  uint32_t argIndex = op->arg_index();
  auto cachePath = cache.getCachePath(programHash, argIndex);

  LOG_DEBUG(LogType::LogRuntimeTTNN, "GetOrInsertIntoDiskCacheOp for arg ",
            argIndex, " with hash: ", programHash);

  // Cache hit: load from disk
  if (cache.exists(programHash, argIndex)) {
    LOG_DEBUG(LogType::LogRuntimeTTNN, "Disk cache hit for arg ", argIndex,
              " at path: ", cachePath.string());

    ::ttnn::MeshDevice *device = context.getMeshDevicePtr().get();
    ::ttnn::Tensor out =
        ::tt::tt_metal::load_tensor_flatbuffer(cachePath.string(), device);
    context.getTensorPool().insertTTNNTensorAndValidate(op->out(), out);
    return;
  }

  // Cache miss: write to disk and pass through
  LOG_DEBUG(LogType::LogRuntimeTTNN, "Disk cache miss for arg ", argIndex,
            ", writing to: ", cachePath.string());

  cache.ensureDirectoryExists(programHash);

  const ::ttnn::Tensor &input =
      context.getTensorPool().getTTNNTensorAndValidate(op->input());
  ::tt::tt_metal::dump_tensor_flatbuffer(cachePath.string(), input);
  cache.markWritten(programHash, argIndex);

  // Pass through input as output
  context.getTensorPool().insertTTNNTensorAndValidate(op->out(), input);
}

} // namespace tt::runtime::ttnn::operations::cache
