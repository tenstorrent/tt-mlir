// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/cache/load_cached.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/tensor_cache.h"
#include "tt/runtime/ttnn/types.h"
#include "ttmlir/Target/TTNN/Target.h"

#include <vector>

namespace tt::runtime::ttnn::operations::cache {

using LogType = ::tt::runtime::logger::LogType;

void run(const ::tt::target::ttnn::LoadCachedOp *op, ProgramContext &context) {
  // TODO: think this through before PR
  uint32_t meshId = 0;

  // Get the appropriate cache for this device
  TensorCache &cache = context.getCache(meshId);

  // Extract function name
  const std::string functionName = op->callee()->str();

  // Collect input tensor IDs for the cache key
  std::vector<uint32_t> inputIds;
  for (const auto *input : *op->inputs()) {
    inputIds.push_back(input->global_id());
  }

  // Create the cache key
  CacheKey cacheKey(functionName, inputIds);

  // Check if the result is already in the cache
  if (cache.contains(cacheKey)) {
    LOG_INFO("Cache hit for function: ", functionName);

    // Get the cached tensor
    ::ttnn::Tensor *cachedTensor = cache.get(cacheKey);

    // Insert the cached tensor into the tensor pool for the output
    for (const auto *out : *op->outputs()) {
      context.getTensorPool().insertAndValidate(out->global_id(),
                                                *cachedTensor);
    }

    return;
  }

  LOG_INFO("Cache miss for function: ", functionName);

  // Get the function to call
  const ::tt::target::ttnn::Operation *functionOp = op->function();

  // Execute the function
  // This requires dispatching to the appropriate operation handler
  // We need to determine the operation type and call the corresponding run
  // function

  // For now, we'll just log that we need to execute the function
  // The actual implementation would need to handle different operation types
  LOG_INFO("Executing function: ", functionName);

  // In a real implementation, we would:
  // 1. Determine the operation type
  // 2. Call the appropriate run function
  // 3. Get the result tensor
  // 4. Cache the result
  // 5. Return the result

  // For demonstration, we'll assume we have a result tensor
  // In a real implementation, this would come from executing the function
  ::ttnn::Tensor resultTensor;

  // Add the result to the cache
  cache.add(cacheKey, resultTensor);

  // Insert the result into the tensor pool for the output
  context.getTensorPool().insertAndValidate(op->out(), resultTensor);
}
} // namespace tt::runtime::ttnn::operations::cache
