// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/load_cached.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/tensor_cache.h"
#include "tt/runtime/types.h"
#include <string_view>
#include <vector>

namespace tt::runtime::ttnn::operations::cache {

using LogType = ::tt::runtime::logger::LogType;

void run(const ::tt::target::ttnn::LoadCachedOp *op, ProgramContext &context) {
  std::shared_ptr<TensorCache> cache = context.getConstEvalTensorCache();
  LOG_ASSERT(cache, "Cache must be enabled to support const-eval ops.");

  // Get the device ID from the parent mesh
  const int deviceId = context.getMeshDevice().id();
  const std::string cacheKey =
      generateCacheOuterKey(deviceId, context.getProgramIndex());
  const std::string &constEvalFuncname = op->callee_name()->str();

  std::vector<uint64_t> inputVersions;
  inputVersions.reserve(op->inputs()->size());
  // Extract versions for each input tensor.
  for (const auto *input : *op->inputs()) {
    const ::tt::runtime::ttnn::TTNNTensorWrapper &runtimeInput =
        context.getTensorPool().getTTNNTensorWrapperAndValidate(input);
    inputVersions.push_back(runtimeInput.getVersion());
  }

  // Get the cached tensors, which will be empty if cache is invalid
  const std::vector<Tensor> *cachedOutputs =
      cache->getAll(cacheKey, constEvalFuncname, inputVersions);

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
    inputs.emplace_back(
        context.getTensorPool().getRuntimeTensorAndValidate(input));
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

  cache->store(cacheKey, constEvalFuncname, std::move(inputVersions), outputs);

  for (size_t i = 0; i < outputs.size(); ++i) {
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::cache
