// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/grid_sample.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::pool {

// grid_sample accepts INTERLEAVED or HEIGHT_SHARDED operands; any other
// sharding (block / width) has to be collected first.
static bool needsDeshard(const ::ttnn::Tensor &t) {
  return t.memory_config().is_sharded() &&
         t.memory_config().memory_layout() !=
             ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED;
}

void run(const ::tt::target::ttnn::GridSampleOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::MemoryConfig dramInterleaved{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                       ::ttnn::BufferType::DRAM};

  // The kernel mandates ROW_MAJOR for both operands. Desharding must happen
  // before to_layout, because to_layout on a sharded tensor keeps it sharded.
  ::ttnn::Tensor input = tensorPool.getTTNNTensorAndValidate(op->input());
  if (needsDeshard(input)) {
    input = ::ttnn::to_memory_config(input, dramInterleaved);
  }
  if (input.layout() != ::ttnn::Layout::ROW_MAJOR) {
    input = ::ttnn::to_layout(input, ::ttnn::Layout::ROW_MAJOR);
  }

  // Keep a HEIGHT_SHARDED grid in place. The compiler's GridSample layout
  // optimizer deliberately parks the grid in L1 HEIGHT_SHARDED so the sampling
  // kernel reads it out of L1; collecting it back to DRAM here would both undo
  // that and cost an extra dispatch.
  ::ttnn::Tensor grid = tensorPool.getTTNNTensorAndValidate(op->grid());
  if (needsDeshard(grid)) {
    grid = ::ttnn::to_memory_config(grid, dramInterleaved);
  }
  if (grid.layout() != ::ttnn::Layout::ROW_MAJOR) {
    grid = ::ttnn::to_layout(grid, ::ttnn::Layout::ROW_MAJOR);
  }

  std::string mode = op->mode()->str();
  std::string paddingMode = op->padding_mode()->str();
  bool alignCorners = op->align_corners();
  bool batchOutputChannels = op->batch_output_channels();
  bool usePrecomputedGrid = op->use_precomputed_grid();

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config()
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config())
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));

  ::ttnn::Tensor output =
      ::ttnn::grid_sample(input, grid, mode, paddingMode, alignCorners,
                          usePrecomputedGrid, batchOutputChannels, memoryConfig);

  // With a sharded grid the op derives the output shard spec from the grid and
  // ignores the requested memory config, so the result can disagree with the
  // type the flatbuffer declares. Reconcile only when they actually differ —
  // an unconditional collect would add a dispatch on every inference.
  if (memoryConfig && output.memory_config() != *memoryConfig) {
    output = ::ttnn::to_memory_config(output, *memoryConfig);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::pool
