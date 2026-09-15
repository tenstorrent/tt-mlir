// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/reduction.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/quasar/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"

namespace tt::runtime::ttnn::operations::reduction {
void run(const ::tt::target::ttnn::ReductionOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));
  LOG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->out()) ||
                 outputMemoryConfig.has_value(),
             "Memory config must exist for device tensors");

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  const auto *fbDimArg = op->dim_arg();
  std::optional<::ttsl::SmallVector<int>> dimArg =
      fbDimArg ? std::make_optional(::ttsl::SmallVector<int>(fbDimArg->begin(),
                                                             fbDimArg->end()))
               : std::nullopt;

  std::optional<::ttnn::DeviceComputeKernelConfig> computeConfig;
  if (op->compute_config()) {
    computeConfig =
        utils::createDeviceComputeKernelConfig(op->compute_config());
  }

  // The Quasar reductions live under experimental/quasar/reduction/generic. They
  // have no python binding (sources.cmake calls them internal, used by the
  // Quasar avg_pool2d) but the C++ entry points are exported, and their leading
  // arguments match the mainline ones. Their dim argument is a variant rather
  // than a plain SmallVector, so it is built separately.
  const bool quasar = utils::isQuasar();
  std::optional<std::variant<int, int64_t, ::ttsl::SmallVector<int>>> quasarDimArg;
  if (quasar && dimArg.has_value()) {
    quasarDimArg = *dimArg;
  }

  // Quasar: reduce in ROW_MAJOR when the trailing dims are not tile-aligned.
  //
  // quasar::reduce clears the implicit tile padding before reducing, via mainline
  // ttnn::fill_implicit_tile_padding
  // (quasar/reduction/generic/generic_reductions.cpp:528). That op's program
  // factory builds a plain DataMovementKernel, which Quasar refuses outright
  // ("DataMovementKernel is not supported on Quasar", kernel.hpp:418), so every
  // reduction over a non-tile-aligned tiled tensor fails. ResNet-50's global
  // average pool is exactly that: mean over dim -2 of [1, 1, 49, 2048], where 49
  // pads to 64.
  //
  // That fill is guarded by `is_tiled` at the call site, so a ROW_MAJOR input
  // skips it entirely. quasar::pad is not a way out -- it reaches the same fill
  // through detail::invoke_tile -- but quasar::to_layout is, and it is measured
  // to preserve logical order exactly. So untilize, reduce, and restore the
  // input's layout afterwards.
  ::ttnn::Tensor input = in;
  bool untilizedForQuasar = false;

  // NOT DONE: reductions over the LAST axis (the W reduce) are refused on Quasar.
  //
  // Only the H, HW and CN reduce program factories were ported to Quasar's
  // KernelSpec+DFB API. The W factory
  // (quasar/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp)
  // is still on the old KernelDescriptor/FILE_PATH API, so it builds a plain
  // DataMovementKernel and Quasar refuses it, in TILE and ROW_MAJOR alike.
  //
  // Rewriting it as an H reduce between two swaps of the last two axes was tried
  // and REVERTED: it returns silently wrong results. The swap maps to
  // TransposeOpDim::WH, which is exact on Quasar only when both trailing extents
  // are tile-aligned; the shapes that need this have an unaligned one (e.g.
  // [1, 1, 64, 49], 49 padding to 64) and the transpose mixes the padding in.
  // Measured pcc 0.176574 on mean over dim 3 of [1, 1, 64, 49]. A degenerate
  // probe where the two axes have equal extent ([1, 1, 64, 64]) scored 0.999703
  // and hid it.
  //
  // ResNet-50 only needs this under the default pipeline, which lowers its
  // global average pool to mean(dim=[3]); the optimised pipeline lowers the same
  // pool to mean(dim=[-2]) and takes the working H path. Porting the W factory
  // is the real fix.

  if (quasar && input.layout() == ::ttnn::Layout::TILE &&
      input.logical_shape().rank() >= 2) {
    const ::ttnn::Shape &shape = input.logical_shape();
    const size_t rank = shape.rank();
    const bool trailingUnaligned =
        (shape[rank - 1] % ::tt::constants::TILE_WIDTH != 0) ||
        (shape[rank - 2] % ::tt::constants::TILE_HEIGHT != 0);
    if (trailingUnaligned) {
      input = ::ttnn::operations::experimental::quasar::to_layout(
          input, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
      untilizedForQuasar = true;
    }
  }

  ::ttnn::Tensor out;
  switch (op->type()) {
  case ::tt::target::ttnn::ReductionOpType::Sum: {
    out = quasar ? ::ttnn::operations::experimental::quasar::sum(
                       input, quasarDimArg, op->keep_dim(), outputMemoryConfig,
                       computeConfig)
                 : ::ttnn::sum(in, dimArg, op->keep_dim(), outputMemoryConfig,
                               computeConfig);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Mean: {
    out = quasar ? ::ttnn::operations::experimental::quasar::mean(
                       input, quasarDimArg, op->keep_dim(), outputMemoryConfig,
                       computeConfig)
                 : ::ttnn::mean(in, dimArg, op->keep_dim(), outputMemoryConfig,
                                computeConfig);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Max: {
    out = quasar ? ::ttnn::operations::experimental::quasar::max(
                       input, quasarDimArg, op->keep_dim(), outputMemoryConfig,
                       computeConfig)
                 : ::ttnn::max(in, dimArg, op->keep_dim(), outputMemoryConfig,
                               computeConfig);
    break;
  }
  case ::tt::target::ttnn::ReductionOpType::Min: {
    out = quasar ? ::ttnn::operations::experimental::quasar::min(
                       input, quasarDimArg, op->keep_dim(), outputMemoryConfig,
                       computeConfig)
                 : ::ttnn::min(in, dimArg, op->keep_dim(), outputMemoryConfig,
                               computeConfig);
    break;
  }
  }

  if (untilizedForQuasar && out.layout() != in.layout()) {
    out = ::ttnn::operations::experimental::quasar::to_layout(
        out, in.layout(), std::nullopt, std::nullopt);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::reduction
