// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/slice.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "ttmlir/Target/TTNN/operations/data_movement_generated.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <cstdint>
#include <optional>

namespace tt::runtime::ttnn::operations::data_movement {
static void runSliceOp(const ::tt::target::ttnn::SliceXDOp *op, ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttsl::SmallVector<int32_t> begins(op->params_as<target::ttnn::SliceOpParams>()->begins()->begin(), op->params_as<target::ttnn::SliceOpParams>()->begins()->end());
  ::ttsl::SmallVector<int32_t> ends(op->params_as<target::ttnn::SliceOpParams>()->ends()->begin(), op->params_as<target::ttnn::SliceOpParams>()->ends()->end());
  ::ttsl::SmallVector<int32_t> step(op->step()->begin(), op->step()->end());

  ::ttnn::Tensor out = ::ttnn::slice(in, begins, ends, step);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

static void runSliceDynamicOp(const ::tt::target::ttnn::SliceXDOp *op, ProgramTensorPool &tensorPool) {
  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  const ::ttnn::Tensor &begins = tensorPool.getTTNNTensorAndValidate(op->params_as<target::ttnn::SliceDynamicOpParams>()->begins());
  const ::ttnn::Tensor &ends = tensorPool.getTTNNTensorAndValidate(op->params_as<target::ttnn::SliceDynamicOpParams>()->begins());
  ::ttsl::SmallVector<int32_t> step(op->step()->begin(), op->step()->end());

  ::ttnn::Tensor out = ::ttnn::slice(in, begins, ends, std::make_optional(step));

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

void run(const ::tt::target::ttnn::SliceXDOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttsl::SmallVector<int32_t> step(op->step()->begin(), op->step()->end());
  switch(op->type()) {
    case ::tt::target::ttnn::SliceXDOpType::SliceOp:
      runSliceOp(op, tensorPool);
      break;
    case ::tt::target::ttnn::SliceXDOpType::SliceDynamicOp:
      runSliceDynamicOp(op, tensorPool);
      break;
  }
}

} // namespace tt::runtime::ttnn::operations::data_movement
