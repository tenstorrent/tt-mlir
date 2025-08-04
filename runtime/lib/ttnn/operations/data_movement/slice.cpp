// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/slice.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <cstdint>

namespace tt::runtime::ttnn::operations::data_movement {
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

  ::ttnn::Tensor out = ::ttnn::slice(in, beginsSpan, endsSpan, stepSpan);

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

  ::ttnn::Tensor out = ::ttnn::slice(in, begins, ends, step);

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
