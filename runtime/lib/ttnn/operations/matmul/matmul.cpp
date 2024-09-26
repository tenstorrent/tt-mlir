// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/detail/workarounds.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include <optional>

// ANCHOR: adding_an_op_matmul_runtime_operations
namespace tt::runtime::ttnn::operations::matmul {

// This is a workaround for the lack of program config selection in ttnn.matmul.
// The logic here is temporary and totaly incompleate.
static ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig
createProgramConfig(const ::tt::target::ttnn::MatmulOp *op,
                    ProgramContext &context,
                    ::tt::tt_metal::MemoryConfig outputMemoryConfig) {

  uint32_t numCores = outputMemoryConfig.shard_spec->grid.num_cores();
  bool fuseBatch = true; // required for sharded inputs

  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
  const ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());

  // note: use ttnn::Shape::value returns a legacy tt::tt_metal::Shape object
  // which does take padding into account.
  uint32_t volume = 1;
  for (size_t i = 0; i < lhs.shape().rank(); i++) {
    volume *= lhs.shape().value[i];
  }

  uint32_t M =
      fuseBatch ? volume / lhs.shape().value[-1] : lhs.shape().value[-2];
  // uint32_t K = lhs.shape().value[-1];
  uint32_t N = rhs.shape().value[-1];
  bool mcastIn0 = N >= M;

  uint32_t perCoreM, perCoreN;

  if (mcastIn0) {
    perCoreM = M / tt::constants::TILE_HEIGHT;
    perCoreN = tt::div_up(tt::div_up(N, numCores), tt::constants::TILE_WIDTH);
  } else {
    perCoreM = tt::div_up(tt::div_up(M, numCores), tt::constants::TILE_HEIGHT);
    perCoreN = N / tt::constants::TILE_WIDTH;
  }

  // uint32_t in0_block_w = (K / tt::constants::TILE_WIDTH) % 2 == 0 ? 2 : 1;
  uint32_t in0BlockW = 1;

  // These should work in most cases, but there is a logic how we can optimize
  // this later.
  uint32_t outSubblockH = 1, outSubblockW = 1;

  assert(outputMemoryConfig.shard_spec->grid.ranges().size() == 1);
  CoreCoord computeWithStorageGridSize =
      outputMemoryConfig.shard_spec->grid.ranges().begin()->grid_size();
  if (lhs.is_sharded()) {
    CoreCoord lhs_grid_size =
        lhs.shard_spec()->grid.ranges().begin()->grid_size();
    if (computeWithStorageGridSize < lhs_grid_size) {
      computeWithStorageGridSize = lhs_grid_size;
    }
  }

  return ::ttnn::operations::matmul::
      MatmulMultiCoreReuseMultiCast1DProgramConfig{
          .compute_with_storage_grid_size = computeWithStorageGridSize,
          .in0_block_w = in0BlockW,
          .out_subblock_h = outSubblockH,
          .out_subblock_w = outSubblockW,
          .per_core_M = perCoreM,
          .per_core_N = perCoreN,
          .fuse_batch = true,
          .fused_activation = std::nullopt,
          .mcast_in0 = mcastIn0};
};

void run(const ::tt::target::ttnn::MatmulOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &lhs = tensorPool.at(op->in0()->global_id());
  const ::ttnn::Tensor &rhs = tensorPool.at(op->in1()->global_id());
  DEBUG_ASSERT(lhs.is_allocated());
  DEBUG_ASSERT(rhs.is_allocated());
  ::ttnn::DataType outputDataType = utils::getDataType(op->out());
  ::tt::tt_metal::MemoryConfig outputMemoryConfig =
      utils::createMemoryConfig(op->out());

  std::optional<
      ::ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>
      programConfig = std::nullopt;

  // TODO(bug #891): ttnn::matmul doesn't chose correct program config.
  if (workaround::Env::get().setMatmul1DProgramConfig &&
      outputMemoryConfig.memory_layout ==
          ::tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
    programConfig = createProgramConfig(op, context, outputMemoryConfig);
  }

  const std::optional<const ::tt::tt_metal::MemoryConfig> memoryConfig =
      std::make_optional(outputMemoryConfig);

  const std::optional<const ::ttnn::DataType> dtype =
      std::make_optional(outputDataType);

  ::ttnn::Tensor out = ::ttnn::matmul(
      lhs, rhs, /*transposeA*/ false, /*transposeB*/ false, memoryConfig, dtype,
      programConfig, /*activation*/ std::nullopt,
      /*computeKernelConfig*/ std::nullopt, /*coreGrid*/ std::nullopt);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::matmul
// ANCHOR_END: adding_an_op_matmul_runtime_operations
