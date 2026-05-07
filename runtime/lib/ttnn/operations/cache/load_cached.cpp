// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/load_cached.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/types/global_tensor_cache.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include <cstdlib>
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
  // Extract versions for each input tensor. Use the non-validating pool
  // accessor so we don't trip is_allocated() on inputs whose buffers
  // were intentionally freed by an earlier load_cached + retain-clear
  // (TT_RUNTIME_FREE_CONST_EVAL_INPUTS path). The version is wrapper
  // metadata and does not depend on buffer allocation state; cache key
  // matching still works after the underlying bytes are gone.
  for (const auto *input : *op->inputs()) {
    const ::tt::runtime::Tensor &runtimeTensor =
        context.getTensorPool().getRuntimeTensor(input->global_id());
    const ::tt::runtime::ttnn::TTNNTensorWrapper &runtimeInput =
        runtimeTensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
            DeviceRuntime::TTNN);
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

  // Free const-eval input device buffers now that the transformed
  // outputs are cached. Future LoadCachedOp invocations only consult
  // `runtimeInput.getVersion()` (atomic metadata, no buffer access) and
  // pull outputs from the cache, so the input bytes are never touched
  // again.
  //
  // Critical for streaming-time bf16→bfp_bf4/bf8 conversion: without
  // this, the per-layer bf16 weights live on device alongside the
  // packed cache outputs (5× memory) until the executable is destroyed
  // — DeepSeek-V4-Pro OOMs at l~12 even with bfp_bf4 experts + bfp_bf8
  // attention configured. Freeing here drops bf16 immediately after the
  // typecast that produced the cache.
  //
  // SAFETY: some const_eval functions emit outputs that ALIAS the input
  // storage (e.g., reshape/view ops, where output is a metadata view of
  // input bytes). Deallocating the input would also tear down the
  // output. Heuristic: only dealloc when EVERY output has a DIFFERENT
  // dtype than the input. Same-dtype outputs are conservatively
  // assumed to be possible views — skip dealloc. Typecast bf16→bfp4
  // changes dtype, so the streaming use case is covered.
  //
  // Gated on `TT_RUNTIME_FREE_CONST_EVAL_INPUTS=1` so existing
  // workloads keep current behavior.
  static const bool freeConstEvalInputs = []() {
    const char *v = std::getenv("TT_RUNTIME_FREE_CONST_EVAL_INPUTS");
    return v != nullptr && v[0] == '1';
  }();
  if (freeConstEvalInputs) {
    // Strategy: clear retain=false on each const-eval input. The
    // compiler ALREADY emits a `ttnn.deallocate(%arg)` op immediately
    // after each `ttcore.load_cached(@const_eval_X, [%arg])`. With
    // retain=true (PJRT plugin's default for parameter inputs), that
    // op no-ops in `deletion::run`. Clearing retain here lets the
    // already-emitted dealloc actually free the buffer.
    //
    // We do NOT call deallocateTensor manually — that would race with
    // the compiler-emitted dealloc and trip
    // `DEBUG_ASSERT(is_allocated())` in `getRuntimeTensorAndValidate`.
    //
    // Aliasing safety: skip when any output shares the input's dtype
    // (likely reshape/view sharing input storage). For typecast
    // bf16→bfp_bf4, dtypes differ, so streaming use case is covered.
    for (::tt::runtime::Tensor &input : inputs) {
      ::tt::runtime::ttnn::TTNNTensorWrapper &inputWrapper =
          input.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
              DeviceRuntime::TTNN);
      const ::ttnn::Tensor &inputTensor = inputWrapper.getTensor();
      ::ttnn::DataType inputDtype = inputTensor.dtype();

      bool maybeAliased = false;
      for (const ::tt::runtime::Tensor &output : outputs) {
        const ::tt::runtime::ttnn::TTNNTensorWrapper &outputWrapper =
            output.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
                DeviceRuntime::TTNN);
        if (outputWrapper.getTensor().dtype() == inputDtype) {
          maybeAliased = true;
          break;
        }
      }
      if (maybeAliased) {
        LOG_DEBUG("Skipping const-eval input retain-clear for ",
                  constEvalFuncname,
                  " — output(s) share input dtype, possible view aliasing");
        continue;
      }

      inputWrapper.setRetain(false);
    }
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), outputs[i]);
  }
}
} // namespace tt::runtime::ttnn::operations::cache
