// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/load_cached.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/program.h"
#include "tt/runtime/ttnn/tensor_cache.h"
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
  // TODO: think this through before PR

  // Get the appropriate cache for this device
  TensorCache &cache = context.getCache();

  // Extract function name
  const std::string functionName = op->callee_name()->str();

  // Collect input tensor IDs for the cache key
  std::vector<uint32_t> inputIds;
  for (const auto *input : *op->inputs()) {
    inputIds.push_back(input->global_id());
  }

  std::vector<uint32_t> outputIds;
  for (const auto *output : *op->outputs()) {
    outputIds.push_back(output->global_id());
  }

  // Create the cache key
  CacheKey cacheKey(functionName, inputIds);

  // Check if the result is already in the cache
  if (auto outputs = cache.getAll(cacheKey); !outputs.empty()) {
    LOG_INFO("Cache hit for function: ", functionName);

    // Get the cached tensor
    // Insert the cached tensor into the tensor pool for the output
    assert(outputs.size() == op->outputs()->size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      context.getTensorPool().insertAndValidate(
          op->outputs()->Get(i)->global_id(), *outputs[i].getTensor());
    }

    return;
  }

  LOG_INFO("Cache miss for function: ", functionName);

  // Execute the function
  std::vector<::ttnn::Tensor *> inputs(inputIds.size());
  for (size_t i = 0; i < inputIds.size(); ++i) {
    LOG_INFO("Looking for tensor with id: ", inputIds[i]);
    inputs[i] = &context.getTensorPool().getAndValidate(inputIds[i]);
  }
  const size_t programIndex = op->program_idx();
  ::tt::target::ttnn::TTNNBinary const &fbb =
      *getBinary(context.getExecutableHandle());
  ::tt::target::ttnn::Program const *subProgram =
      fbb.programs()->Get(programIndex);
  executor::ProgramExecutor exec(subProgram, context.getExecutableHandle(),
                                 inputs, &context.getParentMesh());
  exec.execute();
  LOG_INFO("executed sub-func: ", functionName);
  std::vector<::ttnn::Tensor *> outputs = exec.gatherTTNNOutputTensors();
  LOG_ASSERT(outputs.size() == outputIds.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    ::ttnn::Tensor *output = outputs[i];
    TensorPtrWrapper wrappedTensor(output);
    cache.add(cacheKey, output);
    context.getTensorPool().insertAndValidate(outputIds[i], *output);
  }
}
} // namespace tt::runtime::ttnn::operations::cache
