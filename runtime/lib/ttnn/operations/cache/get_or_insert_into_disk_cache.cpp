// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/get_or_insert_into_disk_cache.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/types/disk_tensor_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/types.h"

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_impl.hpp>

namespace tt::runtime::ttnn::operations::cache {

using LogType = ::tt::runtime::logger::LogType;

void run(const ::tt::target::ttnn::GetOrInsertIntoDiskCacheOp *op,
         ProgramContext &context) {
  DiskTensorCache &cache = DiskTensorCache::getInstance();

  std::string programHash = op->program_hash()->str();
  uint32_t argIndex = op->arg_index();

  LOG_INFO("[DiskCache] GetOrInsertIntoDiskCacheOp: arg_index=", argIndex,
           ", program_hash=", programHash);

  // If disk cache is disabled, pass through input unchanged
  if (!DiskTensorCache::isEnabled()) {
    LOG_INFO("[DiskCache] TTMLIR_ENABLE_DISK_CACHE not set, passing through");
    const ::ttnn::Tensor &input =
        context.getTensorPool().getTTNNTensorAndValidate(op->input());
    context.getTensorPool().insertTTNNTensorAndValidate(op->out(), input);
    return;
  }

  auto cachePath = cache.getCachePath(programHash, argIndex);

  // L1 hit: return runtime tensor directly from in-memory cache
  if (const auto *cachedTensor =
          cache.getRuntimeTensor(programHash, argIndex)) {
    LOG_INFO("[DiskCache] L1 HIT - returning cached runtime tensor");
    context.getTensorPool().insertRuntimeTensorAndValidate(op->out(),
                                                           *cachedTensor);
    return;
  }

  // L2 hit (disk cache): load from disk, store in L1
  if (cache.exists(programHash, argIndex)) {
    bool loadToDevice = op->load_to_device();
    ::ttnn::MeshDevice *device =
        loadToDevice ? context.getMeshDevicePtr().get() : nullptr;

    LOG_INFO("[DiskCache] L2 HIT, L1 MISS - loading from disk to ",
             (loadToDevice ? "device" : "host"), ": ", cachePath.string());

    ::ttnn::Tensor ttnnTensor =
        ::tt::tt_metal::load_tensor_flatbuffer(cachePath.string(), device);

    // Insert into TensorPool with retain=true to prevent deallocation
    context.getTensorPool().insertTTNNTensorAndValidate(op->out(), ttnnTensor,
                                                        /*retain=*/true);

    // Get the runtime tensor and store in L1 cache for future executions
    ::tt::runtime::Tensor &runtimeTensor =
        context.getTensorPool().getRuntimeTensorAndValidate(op->out());
    cache.storeRuntimeTensor(programHash, argIndex, runtimeTensor);
    return;
  }

  // Both miss: write to disk, store in L1
  LOG_INFO("[DiskCache] L1 MISS, L2 MISS - writing to disk: ",
           cachePath.string());

  cache.ensureDirectoryExists(programHash);

  const ::ttnn::Tensor &input =
      context.getTensorPool().getTTNNTensorAndValidate(op->input());
  ::tt::tt_metal::dump_tensor_flatbuffer(cachePath.string(), input);
  cache.markWritten(programHash, argIndex);

  // Insert output (pass through input), mark as retained
  context.getTensorPool().insertTTNNTensorAndValidate(op->out(), input,
                                                      /*retain=*/true);

  // Get the runtime tensor and store in L1 cache for future executions
  ::tt::runtime::Tensor &runtimeTensor =
      context.getTensorPool().getRuntimeTensorAndValidate(op->out());
  cache.storeRuntimeTensor(programHash, argIndex, runtimeTensor);

  LOG_INFO("[DiskCache] successfully wrote tensor to disk and stored in L1");
}

} // namespace tt::runtime::ttnn::operations::cache
