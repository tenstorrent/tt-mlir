// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/grid_sample.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/pool/grid_sample/grid_sample.hpp"

namespace tt::runtime::ttnn::operations::pool {
void run(const ::tt::target::ttnn::GridSampleOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->input());
  ::ttnn::Tensor &grid = tensorPool.getTTNNTensorAndValidate(op->grid());

  std::string mode = op->mode() ? op->mode()->str() : std::string("bilinear");
  std::string paddingMode =
      op->padding_mode() ? op->padding_mode()->str() : std::string("zeros");
  bool alignCorners = op->align_corners();

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config()
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config())
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(
                    op->out()));

  // PyTorch grid_sample uses NCHW, but ttnn::grid_sample requires NHWC
  // (last dimension must be channels, divisible by TILE_WIDTH=32).
  // Permute input NCHW -> NHWC before the kernel call, and output
  // NHWC -> NCHW after.
  ::ttsl::SmallVector<int64_t> nchwToNhwc = {0, 2, 3, 1};
  ::ttsl::SmallVector<int64_t> nhwcToNchw = {0, 3, 1, 2};

  ::ttnn::Tensor inputNhwc = ::ttnn::permute(input, nchwToNhwc, std::nullopt);

  ::ttnn::Tensor outputNhwc =
      ::ttnn::grid_sample(inputNhwc, grid, mode, paddingMode, alignCorners,
                          /*use_precomputed_grid=*/false,
                          /*batch_output_channels=*/false, memoryConfig);

  ::ttnn::Tensor output = ::ttnn::permute(outputNhwc, nhwcToNchw, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::pool
