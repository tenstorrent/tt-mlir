// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include <optional>

namespace tt::runtime::ttnn::operations::matmul {
// ANCHOR: adding_an_op_matmul_runtime
void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.tensorPool;
  const ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
  const ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());
  ::ttnn::DataType outputDataType = utils::getDataType(op->out());

  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  // PRINT
  std::cout << "OUTPUT FALTBUFFER" << std::endl;
  std::cout << op->out()->desc()->layout()->memory_desc()->tile_shape()
            << std::endl;

  std::optional<::ttnn::operations::matmul::MatmulProgramConfig>
      matmulOpConfig = std::nullopt;
  std::cout << "SHARD SPEC SHAPE: " << outputMemoryConfig.shard_spec->shape[0]
            << " " << outputMemoryConfig.shard_spec->shape[1] << std::endl;

  if (outputMemoryConfig.is_sharded()) {
    CoreCoord x =
        outputMemoryConfig.shard_spec->grid.ranges().begin()->end_coord;
    CoreCoord compute_with_storage_grid_size = CoreCoord{x.x + 1, x.y + 1};

    // PRINT SHARD SPEC SHAPE
    std::cout << "SHARD SPEC SHAPE: " << outputMemoryConfig.shard_spec->shape[0]
              << " " << outputMemoryConfig.shard_spec->shape[1] << std::endl;
    // Use llvm to print the shape of the shard spec

    matmulOpConfig =
        ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
            /*compute_with_storage_grid_size=*/compute_with_storage_grid_size,
            /*in0_block_w=*/1,
            /*out_subblock_h=*/1,
            /*out_subblock_w=*/1,
            /*per_core_M=*/1,
            /*per_core_N=*/1,
            false,
            std::nullopt,
            false};
  }

  ::ttnn::Tensor out = ::ttnn::operations::matmul::matmul(
      lhs, rhs, /*bias=*/std::nullopt,
      ::ttnn::operations::matmul::Matmul{/*program_config=*/std::nullopt,
                                         /*bcast_batch=*/std::nullopt,
                                         outputMemoryConfig, outputDataType});
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
// ANCHOR_END: adding_an_op_matmul_runtime

} // namespace tt::runtime::ttnn::operations::matmul
