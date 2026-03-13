// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/load_cached.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/types/global_tensor_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/types.h"
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace tt::runtime::ttnn::operations::cache {

using LogType = ::tt::runtime::logger::LogType;

static size_t getProcessRSSBytes() {
  std::ifstream statm("/proc/self/statm");
  if (!statm.is_open()) {
    return 0;
  }
  size_t totalPages = 0;
  size_t residentPages = 0;
  statm >> totalPages >> residentPages;
  static const size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  return residentPages * pageSize;
}

static size_t getTensorHostBytes(const ::ttnn::Tensor &t) {
  if (!utils::isOnHost(t.storage_type())) {
    return 0;
  }
  return static_cast<size_t>(t.physical_volume()) *
         static_cast<size_t>(t.element_size());
}

void run(const ::tt::target::ttnn::LoadCachedOp *op, ProgramContext &context) {
  GlobalTensorCache &cache = GlobalTensorCache::getInstance();

  // Get the device ID from the parent mesh
  const int deviceId = context.getMeshDevice().id();
  const std::string &constEvalFuncname = op->callee_name()->str();

  std::vector<uint64_t> inputVersions;
  inputVersions.reserve(op->inputs()->size());
  // Extract versions for each input tensor.
  for (const auto *input : *op->inputs()) {
    const ::tt::runtime::ttnn::TTNNTensorWrapper &runtimeInput =
        context.getTensorPool().getTTNNTensorWrapperAndValidate(input);
    inputVersions.push_back(runtimeInput.getVersion());
  }

  const auto programHash = op->program_hash()->str();

  const auto cacheKey = CacheKey{deviceId, programHash, inputVersions};

  size_t rssBefore = getProcessRSSBytes();

  size_t totalInputBytes = 0;
  for (const auto *input : *op->inputs()) {
    const ::ttnn::Tensor &t =
        context.getTensorPool().getTTNNTensorAndValidate(input);
    size_t bytes = getTensorHostBytes(t);
    totalInputBytes += bytes;
    LOG_INFO("[ConstEvalMem] ", constEvalFuncname,
             " input id=", input->global_id(),
             " storage=", static_cast<int>(t.storage_type()),
             " volume=", t.physical_volume(), " elem_size=", t.element_size(),
             " host_bytes=", bytes);
  }

  LOG_INFO("[ConstEvalMem] ", constEvalFuncname,
           " total_input_host_bytes=", totalInputBytes,
           " rss_before=", rssBefore / 1024, "KB",
           " cache_entries=", cache.size());

  // Get the cached tensors, which will be empty if cache is invalid
  const std::vector<Tensor> *cachedOutputs = cache.getAll(cacheKey);

  if (cachedOutputs) {
    LOG_INFO("[ConstEvalMem] ", constEvalFuncname, " CACHE_HIT");

    LOG_ASSERT(cachedOutputs->size() == op->outputs()->size());
    for (size_t i = 0; i < cachedOutputs->size(); ++i) {
      context.getTensorPool().insertRuntimeTensorAndValidate(
          op->outputs()->Get(i), (*cachedOutputs)[i]);
    }

    size_t rssAfter = getProcessRSSBytes();
    LOG_INFO("[ConstEvalMem] ", constEvalFuncname,
             " rss_after=", rssAfter / 1024, "KB",
             " rss_delta=", static_cast<int64_t>(rssAfter - rssBefore) / 1024,
             "KB");
    return;
  }

  LOG_INFO("[ConstEvalMem] ", constEvalFuncname, " CACHE_MISS");

  // Collect the ::ttnn::Tensor objects for execution
  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(op->inputs()->size());
  for (const auto *input : *op->inputs()) {
    auto &tensor = context.getTensorPool().getRuntimeTensorAndValidate(input);
    inputs.emplace_back(tensor);
  }

  // Execute the function
  const size_t programIndex = op->program_idx();
  ProgramExecutor exec(context.getDeviceHandle(), context.getExecutableHandle(),
                       programIndex, inputs, /*constEvalProgram=*/true);
  exec.execute();
  LOG_DEBUG("executed sub-func: ", constEvalFuncname);
  std::vector<::tt::runtime::Tensor> outputs = exec.gatherOutputTensors();

  size_t totalOutputBytes = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    ::tt::runtime::ttnn::TTNNTensorWrapper &w =
        outputs[i].as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);
    const ::ttnn::Tensor &t = w.getTensor();
    size_t bytes = getTensorHostBytes(t);
    totalOutputBytes += bytes;
    LOG_INFO("[ConstEvalMem] ", constEvalFuncname, " output [", i,
             "] storage=", static_cast<int>(t.storage_type()),
             " volume=", t.physical_volume(), " elem_size=", t.element_size(),
             " host_bytes=", bytes);
  }

  LOG_INFO("[ConstEvalMem] ", constEvalFuncname,
           " total_output_host_bytes=", totalOutputBytes,
           " new_host_memory=", totalInputBytes + totalOutputBytes);

  // Const-eval outputs need to be retained
  for (::tt::runtime::Tensor &output : outputs) {
    ::tt::runtime::ttnn::TTNNTensorWrapper &outputWrapper =
        output.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
    outputWrapper.setRetain(true);
  }

  cache.store(cacheKey, inputs, outputs);

  for (size_t i = 0; i < outputs.size(); ++i) {
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), outputs[i]);
  }

  size_t rssAfter = getProcessRSSBytes();
  LOG_INFO("[ConstEvalMem] ", constEvalFuncname, " rss_after=", rssAfter / 1024,
           "KB",
           " rss_delta=", static_cast<int64_t>(rssAfter - rssBefore) / 1024,
           "KB", " cache_entries=", cache.size());
}
} // namespace tt::runtime::ttnn::operations::cache
