// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/load_cached.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/tensor_cache.h"
#include "tt/runtime/ttnn/program_executor.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/types.h"

#include <string_view>
#include <vector>

namespace tt::runtime::ttnn::operations::cache {

static ::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  LOG_ASSERT(isTTNN, "Unsupported binary format");
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

using LogType = ::tt::runtime::logger::LogType;

void run(const ::tt::target::ttnn::LoadCachedOp *op, ProgramContext &context) {
  std::shared_ptr<TensorCache> cache = context.getCache();
  LOG_ASSERT(cache, "Cache must be enabled to support const-eval ops.");

  // Get the binary UUID and generate the cache outer key
  const std::string outerKey = generateCacheOuterKey(
      context.getExecutableHandle().getUUID(), context.getProgramIndex());
  const std::string &constEvalFuncname = op->callee_name()->str();

  std::vector<uint64_t> inputVersions;
  inputVersions.reserve(op->inputs()->size());
  // Extract versions for each input tensor.
  for (const auto *input : *op->inputs()) {
    ::tt::runtime::Tensor runtimeInput =
        context.getTensorPool().getRuntimeTensorAndValidate(input);
    inputVersions.push_back(runtimeInput.version.load());
  }

  // Get the cached tensors, which will be empty if cache is invalid
  const std::vector<Tensor> *cachedOutputs =
      cache->getAll(outerKey, constEvalFuncname, inputVersions);

  if (cachedOutputs) {
    LOG_DEBUG("Cache hit for function: ", constEvalFuncname.c_str());

    assert(cachedOutputs->size() == op->outputs()->size());
    for (size_t i = 0; i < cachedOutputs->size(); ++i) {
      auto output = (*cachedOutputs)[i].as<::ttnn::Tensor>(DeviceRuntime::TTNN);
      context.getTensorPool().insertTTNNTensorAndValidate(op->outputs()->Get(i),
                                                          output);
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
  ::tt::target::ttnn::TTNNBinary const &fbb =
      *getBinary(context.getExecutableHandle());
  ::tt::target::ttnn::Program const *subProgram =
      fbb.programs()->Get(programIndex);
  ProgramExecutor exec(subProgram, context.getExecutableHandle(), inputs,
                       &context.getParentMesh(), nullptr);
  exec.execute();
  LOG_DEBUG("executed sub-func: ", constEvalFuncname);
  std::vector<Tensor> outputs = exec.gatherOutputTensors();

  cache->store(outerKey, constEvalFuncname, outputs, inputVersions);

  for (size_t i = 0; i < outputs.size(); ++i) {
    Tensor &runtimeOutput = outputs[i];
    ::ttnn::Tensor output =
        runtimeOutput.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
    context.getTensorPool().insertTTNNTensorAndValidate(op->outputs()->Get(i),
                                                        output);
  }
}
} // namespace tt::runtime::ttnn::operations::cache
