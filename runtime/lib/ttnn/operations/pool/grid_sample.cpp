// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/grid_sample.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::pool {
void run(const ::tt::target::ttnn::GridSampleOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  // Fetch from pool; enforce ROW_MAJOR non-sharded layout required by the
  // kernel. At opt_level_2 the Memory Layout Analysis pass may assign
  // HEIGHT_SHARDED layout to the input tensor.  Desharding must happen before
  // to_layout because to_layout on a sharded tensor keeps it sharded.
  ::ttnn::MemoryConfig dramInterleaved{
      ::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM};

  ::ttnn::Tensor input = tensorPool.getTTNNTensorAndValidate(op->input());
  if (input.memory_config().is_sharded()) {
    input = ::ttnn::to_memory_config(input, dramInterleaved);
  }
  if (input.layout() != ::ttnn::Layout::ROW_MAJOR) {
    input = ::ttnn::to_layout(input, ::ttnn::Layout::ROW_MAJOR);
  }

  // Grid may also be sharded by MLA; deshard before use.
  ::ttnn::Tensor grid = tensorPool.getTTNNTensorAndValidate(op->grid());
  if (grid.memory_config().is_sharded()) {
    grid = ::ttnn::to_memory_config(grid, dramInterleaved);
  }

  std::string mode = op->mode()->str();
  std::string paddingMode = op->padding_mode()->str();
  bool alignCorners = op->align_corners();
  bool batchOutputChannels = op->batch_output_channels();
  // When use_precomputed_grid=true, PrepareGridSampleGridOp has already
  // transformed the grid tensor into the precomputed coordinate format
  // expected by the kernel.  For bilinear without align_corners the grid
  // is passed raw and use_precomputed_grid=false.
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

  // Nearest mode grid_sample produces HEIGHT_SHARDED L1 output which is
  // incompatible with subsequent layout conversion ops; collect to DRAM.
  if (output.memory_config().is_sharded()) {
    output = ::ttnn::to_memory_config(output, dramInterleaved);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::pool
