// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/load_cached.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/tensor_cache.h"
#include "tt/runtime/ttnn/program.h"
#include "tt/runtime/ttnn/types.h"
#include "tt/runtime/types.h"

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

  // Get the appropriate cache for this device
  TensorCache &cache = context.getCache();

  // Extract function name
  const std::string functionName = op->callee_name()->str();

  // Initialize input versions array with the correct size
  std::vector<uint64_t> inputVersions(op->inputs_indexes()->size());
  const std::vector<uint64_t> &allArgVersions = context.getInputVersions();

  // Extract versions from the context
  for (size_t i = 0; i < inputVersions.size(); ++i) {
    const size_t argIdx = op->inputs_indexes()->Get(i);
    if (!allArgVersions.empty() && argIdx < allArgVersions.size()) {
      inputVersions[i] = allArgVersions[argIdx];
    }
  }

  // Get the cached tensors, which will be empty if cache is invalid
  const std::vector<Tensor> *cachedOutputs =
      cache.getAll(functionName, inputVersions);

  if (cachedOutputs) {
    LOG_INFO("Cache hit for function: ", functionName.c_str());

    assert(cachedOutputs->size() == op->outputs()->size());
    for (size_t i = 0; i < cachedOutputs->size(); ++i) {
      auto output = (*cachedOutputs)[i].as<::ttnn::Tensor>(DeviceRuntime::TTNN);
      context.getTensorPool().insertAndValidate(
          op->outputs()->Get(i)->global_id(), output);
    }

    return;
  }

  LOG_INFO("Cache miss or invalid cache for function: ", functionName);

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

  // Get the input runtime tensors for version checking
  std::vector<Tensor> inputTensors;
  inputTensors.reserve(inputIds.size());

  // Collect the ::ttnn::Tensor objects for execution
  std::vector<::ttnn::Tensor *> inputs(inputIds.size());
  for (size_t i = 0; i < inputIds.size(); ++i) {
    LOG_INFO("Looking for tensor with id: ", inputIds[i]);
    inputs[i] = &context.getTensorPool().getAndValidate(inputIds[i]);
    inputTensors.emplace_back(utils::createRuntimeTensorFromTTNN(*inputs[i]));
    // Updating the version feels correct, but isn't useful unless we support nested const-eval.
    inputTensors.back().version.store(inputVersions[i]);
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
  LOG_INFO("executed sub-func: ", functionName);
  std::vector<Tensor> outputs = exec.gatherOutputTensors();
  // Collect output tensor IDs
  std::vector<uint32_t> outputIds;
  for (const auto *output : *op->outputs()) {
    outputIds.push_back(output->global_id());
  }
  LOG_ASSERT(outputs.size() == outputIds.size());

  // Store the results in the cache with input versions
  if (inputTensors.size() == inputIds.size()) {
    // Only store versions if we have all input runtime tensors
    cache.store(functionName, outputs, inputVersions);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    Tensor &runtimeOutput = outputs[i];
    ::ttnn::Tensor output =
        runtimeOutput.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
    context.getTensorPool().insertAndValidate(outputIds[i], output);
  }
}
} // namespace tt::runtime::ttnn::operations::cache
