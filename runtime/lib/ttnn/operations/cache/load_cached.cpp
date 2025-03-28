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

  // Collect input tensor IDs for execution
  std::vector<uint32_t> inputIds;
  for (const auto *input : *op->inputs()) {
    inputIds.push_back(input->global_id());
  }

  // Collect output tensor IDs
  std::vector<uint32_t> outputIds;
  for (const auto *output : *op->outputs()) {
    outputIds.push_back(output->global_id());
  }

  // Get the input runtime tensors for version checking
  std::vector<Tensor> inputTensors;
  inputTensors.reserve(inputIds.size());

  // Collect the ::ttnn::Tensor objects for execution
  std::vector<::ttnn::Tensor *> inputs(inputIds.size());
  for (size_t i = 0; i < inputIds.size(); ++i) {
    LOG_INFO("Looking for tensor with id: ", inputIds[i]);
    inputs[i] = &context.getTensorPool().getAndValidate(inputIds[i]);

    // Get the original runtime::Tensor to track versions
    const Tensor *runtimeTensor =
        context.getTensorPool().getRuntimeTensor(inputIds[i]);
    if (runtimeTensor) {
      inputTensors.push_back(*runtimeTensor);
    } else {
      // If we don't have the runtime tensor, we can't check versions
      // Consider this a cache miss
      LOG_INFO("No runtime tensor found for input id: ", inputIds[i]);
      break;
    }
  }

  // Check if we have all input runtime tensors and the result is in cache and
  // valid
  if (inputTensors.size() == inputIds.size()) {
    // Get the cached tensors, which will be empty if cache is invalid
    auto outputs = cache.getAll(functionName, inputTensors);

    if (!outputs.empty()) {
      LOG_INFO("Cache hit for function: ", functionName);

      assert(outputs.size() == op->outputs()->size());
      for (size_t i = 0; i < outputs.size(); ++i) {
        auto output = outputs[i].as<::ttnn::Tensor>(DeviceRuntime::TTNN);
        context.getTensorPool().insertAndValidate(
            op->outputs()->Get(i)->global_id(), output);
      }

      return;
    }
  }

  LOG_INFO("Cache miss or invalid cache for function: ", functionName);

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
  LOG_ASSERT(outputs.size() == outputIds.size());

  // Store the results in the cache with input versions
  if (inputTensors.size() == inputIds.size()) {
    // Only store versions if we have all input runtime tensors
    cache.store(functionName, outputs, inputTensors);
  }

  for (size_t i = 0; i < outputs.size(); ++i) {
    Tensor runtimeOutput = outputs[i];
    // Store the runtime tensor for future version tracking
    context.getTensorPool().storeRuntimeTensor(outputIds[i], runtimeOutput);

    ::ttnn::Tensor output =
        runtimeOutput.as<::ttnn::Tensor>(DeviceRuntime::TTNN);
    context.getTensorPool().insertAndValidate(outputIds[i], output);
  }
}
} // namespace tt::runtime::ttnn::operations::cache
