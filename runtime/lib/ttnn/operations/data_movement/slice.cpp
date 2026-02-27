// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/slice.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <cstdint>

namespace tt::runtime::ttnn::operations::data_movement {

// Workaround for tt-metal#38016: ttnn::slice on height-sharded tensors may
// return the wrong logical shape when the sliced output maps to the same tiled
// layout as the input. Fix by using the low-level Tensor::reshape to update
// the logical shape while preserving the padded shape and underlying buffer.
static ::ttnn::Tensor
fixSliceOutputShape(::ttnn::Tensor &&out,
                    const ::tt::target::ttnn::SliceOp *op) {
  const auto *expectedShapeFb = op->out()->desc()->shape();
  auto actualShape = out.logical_shape();
  bool shapeMismatch = (actualShape.size() != expectedShapeFb->size());
  if (!shapeMismatch) {
    for (size_t i = 0; i < actualShape.size(); ++i) {
      if (actualShape[i] != static_cast<uint32_t>((*expectedShapeFb)[i])) {
        shapeMismatch = true;
        break;
      }
    }
  }
  if (shapeMismatch) {
    tt::tt_metal::Shape::Container newLogical;
    for (size_t i = 0; i < expectedShapeFb->size(); ++i) {
      newLogical.push_back(static_cast<uint32_t>((*expectedShapeFb)[i]));
    }
    // Keep the padded shape (tile-aligned) from the actual output; only fix
    // the logical shape so downstream ops see the correct unpadded dims.
    return out.reshape(tt::tt_metal::Shape(newLogical), out.padded_shape());
  }
  return std::move(out);
}

static void runSliceStaticOp(const ::tt::target::ttnn::SliceOp *op,
                             ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttsl::SmallVector<int32_t> begins(
      op->params_as<target::ttnn::SliceStaticOpParams>()->begins()->begin(),
      op->params_as<target::ttnn::SliceStaticOpParams>()->begins()->end());
  ::ttsl::SmallVector<int32_t> ends(
      op->params_as<target::ttnn::SliceStaticOpParams>()->ends()->begin(),
      op->params_as<target::ttnn::SliceStaticOpParams>()->ends()->end());
  ::ttsl::SmallVector<int32_t> step(op->step()->begin(), op->step()->end());

  ttsl::Span<const int32_t> beginsSpan(begins.data(), begins.size());
  ttsl::Span<const int32_t> endsSpan(ends.data(), ends.size());
  ttsl::Span<const int32_t> stepSpan(step.data(), step.size());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));

  ::ttnn::Tensor out =
      ::ttnn::slice(in, beginsSpan, endsSpan, stepSpan, memoryConfig);

  out = fixSliceOutputShape(std::move(out), op);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runSliceDynamicOp(const ::tt::target::ttnn::SliceOp *op,
                              ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  const ::ttnn::Tensor &begins = tensorPool.getTTNNTensorAndValidate(
      op->params_as<target::ttnn::SliceDynamicOpParams>()->begins());
  const ::ttnn::Tensor &ends = tensorPool.getTTNNTensorAndValidate(
      op->params_as<target::ttnn::SliceDynamicOpParams>()->ends());

  std::optional<::ttsl::SmallVector<uint32_t>> step;
  if (op->step() && op->step()->size() > 0) {
    step = ::ttsl::SmallVector<uint32_t>();
    step->resize(op->step()->size());
    std::transform(op->step()->begin(), op->step()->end(), step->begin(),
                   [](int32_t v) { return static_cast<uint32_t>(v); });
  }

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()));

  ::ttnn::Tensor out = ::ttnn::slice(in, begins, ends, step, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::SliceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  switch (op->type()) {
  case ::tt::target::ttnn::SliceOpType::SliceStaticOp:
    runSliceStaticOp(op, tensorPool);
    break;
  case ::tt::target::ttnn::SliceOpType::SliceDynamicOp:
    runSliceDynamicOp(op, tensorPool);
    break;
  }
}

} // namespace tt::runtime::ttnn::operations::data_movement
