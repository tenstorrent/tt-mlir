// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/load_cached.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/tensor_cache.h"
#include "tt/runtime/ttnn/program.h"
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
  // TODO(vwells): think this through before PR

  std::shared_ptr<TensorCache> cache = context.getCache();
  LOG_ASSERT(cache, "Cache must be enabled to support const-eval ops.");

  // Extract function name
  const std::string &parentFuncName = context.getProgramName();
  const std::string &constEvalFuncname = op->callee_name()->str();

  // Initialize input versions array with the correct size
  std::vector<uint64_t> inputVersions(op->inputs()->size());

  // Extract versions from the context
  for (size_t i = 0; i < inputVersions.size(); ++i) {
    const size_t argId = op->inputs->Get(i)->global_id();
    std::optional<uint64_t> maybeVersion =
        context.getTensorPool().getVersion(argId);
    LOG_ASSERT(maybeVersion.has_value());
    inputVersions[i] = maybeVersion.value();
  }

  // Get the cached tensors, which will be empty if cache is invalid
  const std::vector<Tensor> *cachedOutputs =
      cache->getAll(parentFuncName, constEvalFuncname, inputVersions);

  if (cachedOutputs) {
    LOG_DEBUG("Cache hit for function: ", constEvalFuncname.c_str());

    assert(cachedOutputs->size() == op->outputs()->size());
    for (size_t i = 0; i < cachedOutputs->size(); ++i) {
      auto output = (*cachedOutputs)[i].as<::ttnn::Tensor>(DeviceRuntime::TTNN);
      context.getTensorPool().insertTTNNTensorAndValidate(
          op->outputs()->Get(i)->global_id(), output);
    }

    return;
  }

  LOG_DEBUG("Cache miss or invalid cache for function: ", constEvalFuncname);

  // Collect input tensor IDs for execution
  std::vector<uint32_t> funcInputIds =
      context.getTensorPool().getProgramInputIds();
  std::vector<uint32_t> inputIds(op->inputs_indexes()->size());
  for (size_t i = 0; i < inputIds.size(); ++i) {
    const size_t input = op->inputs_indexes()->Get(i);
    LOG_ASSERT(input < funcInputIds.size(),
               "Invalid arg index in load_cached op.");
    inputIds[i] = funcInputIds[input];
  }

  // Collect the ::ttnn::Tensor objects for execution
  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(op->inputs().size());
  for (const auto *input : op->inputs()) {
    inputs.emplace_back(
        context.getTensorPool().getRuntimeTensorAndValidate(input));
  }

  // Execute the function
  const size_t programIndex = op->program_idx();
  ::tt::target::ttnn::TTNNBinary const &fbb =
      *getBinary(context.getExecutableHandle());
  ::tt::target::ttnn::Program const *subProgram =
      fbb.programs()->Get(programIndex);
  executor::ProgramExecutor exec(subProgram, context.getExecutableHandle(),
                                 inputs, &context.getParentMesh());
  exec.execute();
  LOG_INFO("executed sub-func: ", constEvalFuncname);
  std::vector<Tensor> outputs = exec.gatherOutputTensors();

  cache->store(parentFuncName, constEvalFuncname, outputs, inputVersions);

  for (size_t i = 0; i < outputs.size(); ++i) {
    Tensor &runtimeOutput = outputs[i];
    ::ttnn::Tensor output =
        runtimeOutput.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
    context.getTensorPool().insertTTNNTensorAndValidate(
        op->outputs()->Get(i)->global_id(), output);
  }
}
} // namespace tt::runtime::ttnn::operations::cache
