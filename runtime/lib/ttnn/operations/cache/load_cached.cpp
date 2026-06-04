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
#include <mutex>
#include <string_view>
#include <utility>
#include <vector>

namespace tt::runtime::ttnn::operations::cache {

using LogType = ::tt::runtime::logger::LogType;

namespace {
// One shared placeholder tile per (device, layout, dtype, memcfg) signature,
// swapped in for an evicted const-eval input. Sharing avoids a per-input tile
// (hundreds) that would fragment DRAM.
struct PlaceholderKey {
  int deviceId;
  ::ttnn::Layout layout;
  ::ttnn::DataType dtype;
  ::ttnn::MemoryConfig memcfg;

  bool operator==(const PlaceholderKey &o) const {
    return deviceId == o.deviceId && layout == o.layout && dtype == o.dtype &&
           memcfg == o.memcfg;
  }
};

::ttnn::Tensor getOrCreateSharedPlaceholder(const PlaceholderKey &key,
                                            ::ttnn::MeshDevice &meshDevice) {
  // Few distinct layouts, so a linear scan suffices; process-lifetime so a
  // swapped-in wrapper always points at a live tile.
  static std::vector<std::pair<PlaceholderKey, ::ttnn::Tensor>> pool;
  static std::mutex poolMutex;
  std::lock_guard<std::mutex> lock(poolMutex);
  for (const auto &entry : pool) {
    if (entry.first == key) {
      return entry.second;
    }
  }
  ::ttnn::Shape shape = (key.layout == ::ttnn::Layout::TILE)
                            ? ::ttnn::Shape({1, 32, 32})
                            : ::ttnn::Shape({1});
  ::ttnn::Tensor placeholder =
      ::ttnn::empty(shape, key.dtype, key.layout, &meshDevice, key.memcfg);
  pool.emplace_back(key, placeholder);
  return placeholder;
}
} // namespace

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

  LOG_DEBUG("Running LoadCachedOp for function ", constEvalFuncname,
            " with hash: ", op->program_hash()->str());

  // Get the cached tensors, which will be empty if cache is invalid
  const std::vector<Tensor> *cachedOutputs = cache.getAll(cacheKey);

  if (cachedOutputs) {
    LOG_DEBUG("Cache hit for function: ", constEvalFuncname.c_str());

    LOG_ASSERT(cachedOutputs->size() == op->outputs()->size());
    for (size_t i = 0; i < cachedOutputs->size(); ++i) {
      context.getTensorPool().insertRuntimeTensorAndValidate(
          op->outputs()->Get(i), (*cachedOutputs)[i]);
    }

    return;
  }

  LOG_DEBUG("Cache miss or invalid cache for function: ", constEvalFuncname);

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

  // Const-eval outputs need to be retained
  for (::tt::runtime::Tensor &output : outputs) {
    ::tt::runtime::ttnn::TTNNTensorWrapper &outputWrapper =
        output.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
    outputWrapper.setRetain(true);
  }

  cache.store(cacheKey, inputs, outputs);

  // Free the device buffer of each cached const-eval input that is a bf16 ->
  // bfp typecast: its bytes are never read again. Swap in a layout-matching
  // placeholder so PJRT's pre-execute layout check still passes, retained so a
  // shared tile is never freed while another wrapper aliases it.
  for (::tt::runtime::Tensor &input : inputs) {
    ::tt::runtime::ttnn::TTNNTensorWrapper &inputWrapper =
        input.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN);
    ::ttnn::Tensor &inputTensor = inputWrapper.getTensor();
    const ::ttnn::DataType inputDtype = inputTensor.dtype();

    // Only device-resident inputs hold the DRAM this reclaims; skip any the
    // compiler kept in system memory (the default const-eval path).
    if (inputTensor.storage_type() != ::ttnn::StorageType::DEVICE) {
      continue;
    }

    // Scope to bf16 -> bfp typecasts (unique bfp output dtype); skip if an
    // output shares the input dtype (it may alias the input storage).
    bool outputIsBlockFloat = false;
    bool maybeAliased = false;
    for (const ::tt::runtime::Tensor &output : outputs) {
      const ::ttnn::DataType od =
          output.as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
              .getTensor()
              .dtype();
      outputIsBlockFloat |= (od == ::ttnn::DataType::BFLOAT4_B ||
                             od == ::ttnn::DataType::BFLOAT8_B);
      maybeAliased |= (od == inputDtype);
    }
    if (!outputIsBlockFloat || maybeAliased) {
      continue;
    }

    PlaceholderKey key;
    key.deviceId = deviceId;
    key.layout = inputTensor.layout();
    key.dtype = inputDtype;
    key.memcfg = inputTensor.memory_config();

    if (inputTensor.is_allocated()) {
      ::ttnn::deallocate(inputTensor, /*force=*/true);
    }
    inputTensor = getOrCreateSharedPlaceholder(key, context.getMeshDevice());
    inputWrapper.setRetain(true);
    // Flag the reclaim so a device->host read-back of this weight errors
    // (it now points at the placeholder, not the original data).
    inputWrapper.setReclaimed(true);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::cache
