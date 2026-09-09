// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/ccl/allocate_moe_compute_semaphore.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/experimental/ccl/moe_compute/moe_compute.hpp"

namespace tt::runtime::ttnn::operations::ccl {

void run(const ::tt::target::ttnn::AllocateMoeComputeSemaphoreOp *op,
         ProgramContext &context) {
  ProgramGlobalSemaphorePool &globalSemaphorePool =
      context.getGlobalSemaphorePool();
  ::ttnn::MeshDevice &meshDevice = context.getMeshDevice();

  // num_data_parallel_cores = largest divisor of hidden_tiles <= 4 (must match
  // the moe_compute device op).
  uint32_t hiddenSize = op->hidden_size();
  uint32_t hiddenTiles = hiddenSize / 32;
  uint32_t numDataParallelCores = 1;
  for (uint32_t d = 4; d >= 1; --d) {
    if (hiddenTiles % d == 0) {
      numDataParallelCores = d;
      break;
    }
  }

  ::ttnn::CoreRangeSet muxCoreRangeSet =
      ::tt::runtime::ttnn::utils::toTTNNCoreRangeSet(*op->core_range_set());

  // Query the (dynamically-placed) combine cores and create the semaphore
  // there.
  std::vector<::ttnn::CoreCoord> combineCores =
      ::ttnn::experimental::get_moe_combine_cores(
          &meshDevice, op->output_height_shard_dim(), numDataParallelCores,
          hiddenSize, muxCoreRangeSet);
  std::vector<::ttnn::CoreRange> combineRanges;
  combineRanges.reserve(combineCores.size());
  for (const ::ttnn::CoreCoord &c : combineCores) {
    combineRanges.emplace_back(c, c);
  }
  ::ttnn::CoreRangeSet combineCoreRangeSet(combineRanges);

  ::ttnn::GlobalSemaphore combineSemaphore =
      ::ttnn::global_semaphore::create_global_semaphore(
          &meshDevice, combineCoreRangeSet,
          op->initial_value().has_value() ? op->initial_value().value() : 0,
          ::tt::tt_metal::BufferType::L1);

  globalSemaphorePool.insertTTNNGlobalSemaphoreAndValidate(op->out(),
                                                           combineSemaphore);
}
} // namespace tt::runtime::ttnn::operations::ccl
