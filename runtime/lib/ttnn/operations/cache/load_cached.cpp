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
#include <string_view>
#include <vector>

namespace tt::runtime::ttnn::operations::cache {

using LogType = ::tt::runtime::logger::LogType;

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

  LOG_INFO("LoadCachedOp: function=", constEvalFuncname,
           " hash=", op->program_hash()->str(),
           " numInputs=", op->inputs()->size());
  for (size_t i = 0; i < inputVersions.size(); ++i) {
    LOG_INFO("  input[", i, "] version=", inputVersions[i]);
  }

  // Get the cached tensors, which will be empty if cache is invalid
  const std::vector<Tensor> *cachedOutputs = cache.getAll(cacheKey);

  if (cachedOutputs) {
    LOG_INFO("Cache HIT for function: ", constEvalFuncname,
             " numOutputs=", cachedOutputs->size());

    LOG_ASSERT(cachedOutputs->size() == op->outputs()->size());
    for (size_t i = 0; i < cachedOutputs->size(); ++i) {
      context.getTensorPool().insertRuntimeTensorAndValidate(
          op->outputs()->Get(i), (*cachedOutputs)[i]);
    }

    return;
  }

  LOG_INFO("Cache MISS for function: ", constEvalFuncname);

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
  std::vector<::tt::runtime::Tensor> outputs = exec.gatherOutputTensors();
  LOG_INFO("Executed sub-func: ", constEvalFuncname,
           " numOutputs=", outputs.size());

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
}
} // namespace tt::runtime::ttnn::operations::cache
