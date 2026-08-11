// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/prepare_grid_sample_grid.h"

#include "tt/runtime/detail/ttnn/ttnn.h"

namespace tt::runtime::ttnn::operations::pool {
void run(const ::tt::target::ttnn::PrepareGridSampleGridOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::MemoryConfig dramInterleaved{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                       ::ttnn::BufferType::DRAM};

  uintptr_t opKey = reinterpret_cast<uintptr_t>(op);
  const ::ttnn::Tensor &result = context.getOrCreatePreparedGrid(opKey, [&]() {
    ::ttnn::Tensor grid = tensorPool.getTTNNTensorAndValidate(op->grid());

    // prepare_grid_sample_grid is a CPU function; move to host first.
    if (grid.memory_config().is_sharded()) {
      grid = ::ttnn::to_memory_config(grid, dramInterleaved);
    }
    if (grid.storage_type() == ::ttnn::StorageType::DEVICE) {
      grid = ::ttnn::from_device(grid);
    }

    // The kernel requires ROW_MAJOR layout on host.
    if (grid.layout() != ::ttnn::Layout::ROW_MAJOR) {
      grid = ::ttnn::to_layout(grid, ::ttnn::Layout::ROW_MAJOR);
    }

    // prepare_grid_sample_grid requires float32 input.
    ::ttnn::Tensor hostGridF32 =
        (grid.dtype() == ::ttnn::DataType::FLOAT32)
            ? grid
            : ::ttnn::typecast(grid, ::ttnn::DataType::FLOAT32);

    std::vector<uint32_t> inputShape = {
        static_cast<uint32_t>(op->input_n()),
        static_cast<uint32_t>(op->input_h()),
        static_cast<uint32_t>(op->input_w()),
        static_cast<uint32_t>(op->input_c())};

    std::string mode = op->mode()->str();
    std::string paddingMode = op->padding_mode()->str();
    bool alignCorners = op->align_corners();

    ::ttnn::Tensor precomputedGrid = ::ttnn::prepare_grid_sample_grid(
        hostGridF32, inputShape, mode, paddingMode, alignCorners,
        ::ttnn::DataType::BFLOAT16);

    ::ttnn::MeshDevice &device = context.getMeshDevice();
    return ::ttnn::to_device(precomputedGrid, &device, dramInterleaved);
  });

  tensorPool.insertTTNNTensorAndValidate(op->out(), result);
}
} // namespace tt::runtime::ttnn::operations::pool
