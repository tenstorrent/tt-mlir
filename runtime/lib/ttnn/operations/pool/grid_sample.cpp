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

  // Fetch from pool; enforce ROW_MAJOR non-sharded layout required by the kernel.
  // At opt_level_2 the Memory Layout Analysis pass may assign HEIGHT_SHARDED
  // layout to the input tensor.  Desharding must happen before to_layout
  // because to_layout on a sharded tensor keeps it sharded.
  ::ttnn::MemoryConfig dramInterleaved{
      ::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM};

  ::ttnn::Tensor input = tensorPool.getTTNNTensorAndValidate(op->input());
  if (input.memory_config().is_sharded()) {
    input = ::ttnn::to_memory_config(input, dramInterleaved);
  }
  if (input.layout() != ::ttnn::Layout::ROW_MAJOR) {
    input = ::ttnn::to_layout(input, ::ttnn::Layout::ROW_MAJOR);
  }

  // Grid may also be sharded by MLA; desharded before moving to host.
  ::ttnn::Tensor grid = tensorPool.getTTNNTensorAndValidate(op->grid());
  if (grid.memory_config().is_sharded()) {
    grid = ::ttnn::to_memory_config(grid, dramInterleaved);
  }

  std::string mode = op->mode()->str();
  std::string paddingMode = op->padding_mode()->str();
  bool alignCorners = op->align_corners();

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config()
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config())
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));

  // The tt-metal bilinear kernel hardcodes the align_corners=False coordinate
  // formula. For align_corners=True, and for nearest mode (which always
  // requires use_precomputed_grid=true), precompute the grid on host first.
  bool needsPrecomputedGrid = (mode == "nearest") || alignCorners;

  if (needsPrecomputedGrid) {
    // Input is NHWC (N, H_in, W_in, C) — get shape for grid preprocessing.
    auto inputShape = input.logical_shape();
    std::vector<uint32_t> inputShapeNHWC = {
        static_cast<uint32_t>(inputShape[0]),
        static_cast<uint32_t>(inputShape[1]),
        static_cast<uint32_t>(inputShape[2]),
        static_cast<uint32_t>(inputShape[3])};

    ::ttnn::MeshDevice &device = context.getMeshDevice();

    // Use the flatbuffer op pointer as the stable cache key.  Unlike
    // grid.buffer()->address(), which changes when the grid is an intermediate
    // tensor re-allocated on each run, the op pointer is a fixed pointer into
    // the static flatbuffer binary and uniquely identifies this GridSampleOp
    // instance across the warmup and trace-capture calls.
    uintptr_t gridCacheKey = reinterpret_cast<uintptr_t>(op);

    // getOrCreateImplicitPrecomputedGrid: the factory runs only on first call
    // (warmup, where from_device is allowed).  The result is cached in the
    // root ProgramContext and reused during trace capture (where from_device
    // is forbidden).  parentContext forwarding in FuncCallOp ensures warmup
    // and trace-capture share the same root context entry.
    ::ttnn::Tensor precomputedGridDevice =
        context.getOrCreateImplicitPrecomputedGrid(
            gridCacheKey, [&]() -> ::ttnn::Tensor {
              // Grid is on device. Move to host for coordinate precomputation.
              ::ttnn::Tensor hostGrid = ::ttnn::from_device(grid);

              // The layout optimizer at opt_level_1+ may override the ROW_MAJOR
              // workaround and leave the grid in TILE layout on device.
              // prepare_grid_sample_grid requires ROW_MAJOR; enforce on host.
              if (hostGrid.layout() != ::ttnn::Layout::ROW_MAJOR) {
                hostGrid =
                    ::ttnn::to_layout(hostGrid, ::ttnn::Layout::ROW_MAJOR);
              }

              // prepare_grid_sample_grid requires float32 host input.
              // Typecast as a safety net in case it arrives as BF16.
              ::ttnn::Tensor hostGridF32 =
                  (hostGrid.dtype() == ::ttnn::DataType::FLOAT32)
                      ? hostGrid
                      : ::ttnn::typecast(hostGrid, ::ttnn::DataType::FLOAT32);

              // Precompute pixel coordinates and interpolation weights.
              // Returns (N, H_out, W_out, 6) for bilinear or
              // (N, H_out, W_out, 2) for nearest.
              ::ttnn::Tensor precomputedGrid = ::ttnn::prepare_grid_sample_grid(
                  hostGridF32, inputShapeNHWC, mode, paddingMode, alignCorners,
                  ::ttnn::DataType::BFLOAT16);

              // Move precomputed grid to device and return for caching.
              return ::ttnn::to_device(precomputedGrid, &device, dramInterleaved);
            });

    ::ttnn::Tensor output =
        ::ttnn::grid_sample(input, precomputedGridDevice, mode, paddingMode,
                            alignCorners, /*use_precomputed_grid=*/true,
                            /*batch_output_channels=*/false, memoryConfig);

    // Nearest mode grid_sample produces HEIGHT_SHARDED L1 output which is
    // incompatible with subsequent layout conversion ops (e.g. permute needs
    // tile-aligned shards). Collect to INTERLEAVED DRAM immediately.
    if (output.memory_config().is_sharded()) {
      output = ::ttnn::to_memory_config(output, dramInterleaved);
    }

    tensorPool.insertTTNNTensorAndValidate(op->out(), output);
  } else {
    ::ttnn::Tensor output =
        ::ttnn::grid_sample(input, grid, mode, paddingMode, alignCorners,
                            /*use_precomputed_grid=*/false,
                            /*batch_output_channels=*/false, memoryConfig);

    tensorPool.insertTTNNTensorAndValidate(op->out(), output);
  }
}
} // namespace tt::runtime::ttnn::operations::pool
