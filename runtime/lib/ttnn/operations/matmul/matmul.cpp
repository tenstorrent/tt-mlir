// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include <cstdint>
#include <optional>
namespace tt::runtime::ttnn::operations::matmul {
    ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig createProgramConfig(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context){
        // this should be read from the output memory config
        uint32_t num_cores = 0;

        for (const auto &core_range : *op->out()->desc()->layout()->core_range_set()) {
            num_cores += core_range->size().x() * core_range->size().y();
        }
        // Print and check if equal to 8
        std::cout << "NUM CORES: " << num_cores << "  SHOULD BE 8" << std::endl;

        bool fuse_batch = true;  // required for sharded inputs

        ProgramTensorPool &tensorPool = context.tensorPool;
        const ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
        const ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());

        // note: use ttnn::Shape::value returns a legacy tt::tt_metal::Shape object which does take padding into account
        // VOLUMEN NEEDS TO BE TILISED AS WELL FOR THIS TO WORK
        uint32_t volume = 1;
        for (size_t i = 0; i < lhs.shape().rank(); i++) {
            volume *= lhs.shape().value[i];
        }
        
        uint32_t M = fuse_batch ? volume / lhs.shape().value[-1] : lhs.shape().value[-2];
        std::cout << "DEBUG lhs.shape().volume(): " << lhs.shape().volume() << std::endl;
        std::cout << "DEBUG lhs.shape().value[-1]: " << lhs.shape().value[-1] << std::endl;
        std::cout << "DEBUG lhs.shape().value[-2]: " << lhs.shape().value[-2] << std::endl;



        uint32_t K = lhs.shape().value[-1];
        uint32_t N = rhs.shape().value[-1];
        bool mcast_in0 = N > M;

        uint32_t per_core_M, per_core_N;

        if (mcast_in0) {
            per_core_M = M / tt::constants::TILE_HEIGHT;
            per_core_N = tt::div_up(tt::div_up(N, num_cores), tt::constants::TILE_WIDTH);
        } else {
            per_core_M = tt::div_up(tt::div_up(M, num_cores), tt::constants::TILE_HEIGHT);
            per_core_N = N / tt::constants::TILE_WIDTH;
        }

        std::cout << "DEBUG M: " << M << std::endl;
        std::cout << "DEBUG N: " << N << std::endl;
        std::cout << "DEBUG K: " << K << std::endl;
        std::cout << "DEBUG per_core_M: " << per_core_M << std::endl;
        std::cout << "DEBUG per_core_N: " << per_core_N << std::endl;

        uint32_t in0_block_w = K / tt::constants::TILE_WIDTH % 2 == 0 ? 2 : 1;

        // these should work in most cases, but there is a logic how we can optimize this later
        uint32_t out_subblock_h = 1, out_subblock_w = 1;

        std::cout << "DEBUG in0_block_w: " << in0_block_w << std::endl;
        std::cout << "DEBUG out_subblock_h: " << out_subblock_h << std::endl;
        std::cout << "DEBUG out_subblock_w: " << out_subblock_w << std::endl;

        return ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig{
            .compute_with_storage_grid_size = CoreCoord(1, 8),  // should be read from output memory config
            .in0_block_w = in0_block_w,
            .out_subblock_h = out_subblock_h,
            .out_subblock_w = out_subblock_w,
            .per_core_M = per_core_M,
            .per_core_N = per_core_N,
            .fuse_batch = true,
            .fused_activation = std::nullopt,
            .mcast_in0 = mcast_in0};
    };

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
    // print compute_grid
    std::cout << "COMPUTE GRID SIZE: " << compute_with_storage_grid_size.x
              << " " << compute_with_storage_grid_size.y << std::endl;


    matmulOpConfig = createProgramConfig(op, context);
    
    // PRINT SHARD SPEC SHAPE
    std::cout << "SHARD SPEC SHAPE: " << outputMemoryConfig.shard_spec->shape[0] << " " << outputMemoryConfig.shard_spec->shape[1] << std::endl;
    // Use llvm to print the shape of the shard spec

  }

//   // Print the outputMemoryConfig shard spec
//     std::cout << "SHARD SPEC SHAPE: " << outputMemoryConfig.shard_spec->num_cores()
//                 << " " << outputMemoryConfig.shard_spec->num_cores() << std::endl;

  ::ttnn::Tensor out = ::ttnn::operations::matmul::matmul(
      lhs, rhs, /*bias=*/std::nullopt,
      ::ttnn::operations::matmul::Matmul{/*program_config=*/matmulOpConfig,
                                         /*bcast_batch=*/std::nullopt,
                                         outputMemoryConfig, outputDataType});

  
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
// ANCHOR_END: adding_an_op_matmul_runtime

} // namespace tt::runtime::ttnn::operations::matmul
