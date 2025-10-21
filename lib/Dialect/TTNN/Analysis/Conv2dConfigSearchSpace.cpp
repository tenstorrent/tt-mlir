// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Analysis/Conv2dConfigSearchSpace.h"

#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Support/Logger.h"

namespace mlir::tt::ttnn {

::mlir::tt::ttnn::Conv2dConfigAttr Conv2dConfigGenerator::getNextConfig() {
  // If isDone is true, it means all combinations for active fields were
  // exhausted.
  if (isDone) {
    return nullptr;
  }

  // Copy base config.
  Conv2dConfigAttr generatedAttr = baseConfig;

  // Override with current search values from activeSearchFields.
  for (const ActiveFieldEntry &fieldEntry : activeSearchFields) {
    generatedAttr = fieldEntry.updateConfig(generatedAttr, fieldEntry.info);
  }

  // Advance the state for the next call. This works like an odometer:
  // Start with the least significant "digit" (the last active search
  // field). Try to advance it.
  // - If it advances without wrapping (e.g., '1' becomes '2'), we're done
  //   with this iteration, and we have the next configuration.
  // - If it wraps (e.g., '9' becomes '0'), "carry over" to the next
  //   more significant "digit" (the previous active search field) and
  //   repeat the process.
  // If all "digits" wrap, it means we've exhausted all combinations.
  int currentFieldToAdvance = activeSearchFields.size() - 1;
  while (currentFieldToAdvance >= 0) {
    // Try to advance the current field. If it wraps, move to the next
    // field.
    if (activeSearchFields[currentFieldToAdvance].info.advance()) {
      currentFieldToAdvance--;
    } else {
      break;
    }
  }

  if (currentFieldToAdvance < 0) {
    isDone = true;
  }

  TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Next conv2d config: {}",
               generatedAttr);

  if (filterOutFn(generatedAttr)) {
    TTMLIR_TRACE(ttmlir::LogComponent::Optimizer, "Filtered out {}",
                 generatedAttr);
    return getNextConfig();
  }

  return generatedAttr;
}

} // namespace mlir::tt::ttnn
