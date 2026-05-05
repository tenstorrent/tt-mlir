// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/get_or_insert_into_disk_cache.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/types/disk_tensor_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_impl.hpp>

#include <sstream>

namespace tt::runtime::ttnn::operations::cache {

using LogType = ::tt::runtime::logger::LogType;

namespace {
std::string getTensorLayoutInfo(const ::ttnn::Tensor &tensor) {
  std::ostringstream ss;
  ss << "shape=" << tensor.get_shape();
  ss << ", dtype=" << tensor.get_dtype();
  ss << ", layout="
     << (tensor.get_layout() == ::ttnn::Layout::TILE ? "TILE" : "ROW_MAJOR");
  ss << ", storage_type=";
  switch (tensor.storage_type()) {
  case ::ttnn::StorageType::DEVICE:
    ss << "DEVICE";
    break;
  case ::ttnn::StorageType::MULTI_DEVICE:
    ss << "MULTI_DEVICE";
    break;
  case ::ttnn::StorageType::OWNED:
    ss << "OWNED";
    break;
  case ::ttnn::StorageType::BORROWED:
    ss << "BORROWED";
    break;
  case ::ttnn::StorageType::MULTI_DEVICE_HOST:
    ss << "MULTI_DEVICE_HOST";
    break;
  }
  if (tensor.is_allocated() &&
      (tensor.storage_type() == ::ttnn::StorageType::DEVICE ||
       tensor.storage_type() == ::ttnn::StorageType::MULTI_DEVICE)) {
    auto memConfig = tensor.memory_config();
    ss << ", memory_type="
       << (memConfig.memory_layout == ::tt::tt_metal::TensorMemoryLayout::INTERLEAVED
               ? "INTERLEAVED"
               : "SHARDED");
    ss << ", buffer_type="
       << (memConfig.buffer_type == ::tt::tt_metal::BufferType::DRAM ? "DRAM"
                                                                      : "L1");
  }
  return ss.str();
}
} // namespace

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
  LOG_INFO("[DiskCache] cache path: ", cachePath.string());

  // Cache hit: load from disk
  if (cache.exists(programHash, argIndex)) {
    LOG_INFO("[DiskCache] CACHE HIT - loading tensor from disk: ",
             cachePath.string());

    ::ttnn::MeshDevice *device = context.getMeshDevicePtr().get();
    ::ttnn::Tensor out =
        ::tt::tt_metal::load_tensor_flatbuffer(cachePath.string(), device);

    LOG_INFO("[DiskCache] loaded tensor layout: ", getTensorLayoutInfo(out));

    context.getTensorPool().insertTTNNTensorAndValidate(op->out(), out);
    return;
  }

  // Cache miss: write to disk and pass through
  LOG_INFO("[DiskCache] CACHE MISS - writing tensor to disk: ",
           cachePath.string());

  cache.ensureDirectoryExists(programHash);

  const ::ttnn::Tensor &input =
      context.getTensorPool().getTTNNTensorAndValidate(op->input());

  LOG_INFO("[DiskCache] dumping tensor layout: ", getTensorLayoutInfo(input));

  ::tt::tt_metal::dump_tensor_flatbuffer(cachePath.string(), input);
  cache.markWritten(programHash, argIndex);

  // Pass through input as output
  context.getTensorPool().insertTTNNTensorAndValidate(op->out(), input);
}

} // namespace tt::runtime::ttnn::operations::cache
