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
  } else {
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
    ProgramExecutor exec(context.getDeviceHandle(),
                         context.getExecutableHandle(), programIndex, inputs,
                         /*constEvalProgram=*/true);
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

    for (size_t i = 0; i < outputs.size(); ++i) {
      context.getTensorPool().insertRuntimeTensorAndValidate(
          op->outputs()->Get(i), outputs[i]);
    }
  }

  // DEBUG: print KV cache tensor values after every forward call
  if (constEvalFuncname.find("kv_cache_const_eval") != std::string::npos) {
    for (size_t i = 0; i < op->outputs()->size(); ++i) {
      const ::ttnn::Tensor &t =
          context.getTensorPool().getTTNNTensorAndValidate(op->outputs()->Get(i));
      const auto &shape = t.logical_shape();
      std::string shapeStr = "[";
      for (size_t d = 0; d < shape.rank(); ++d) {
        if (d > 0) { shapeStr += ", "; }
        shapeStr += std::to_string(shape[d]);
      }
      shapeStr += "]";
      auto vals = t.to_vector<float>();
      constexpr size_t kMaxPrint = 16;
      std::string valStr;
      for (size_t v = 0; v < std::min(vals.size(), kMaxPrint); ++v) {
        if (v > 0) { valStr += ", "; }
        valStr += std::to_string(vals[v]);
      }
      if (vals.size() > kMaxPrint) { valStr += ", ..."; }
      LOG_INFO("[load_cached] kv_cache '", constEvalFuncname, "' output[", i,
               "] dtype=", toString(t.dtype()), " shape=", shapeStr,
               " vals=[", valStr, "]");
    }
  }
}
} // namespace tt::runtime::ttnn::operations::cache
