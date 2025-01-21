// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/repeat.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::RepeatOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());
  const auto *fbShape = op->shape();
  const auto inShape = in.shape();

  bool isBcast = false;
  for (int i = 0; i < inShape.size(); i++) {
    if (inShape[i] == 1 && inShape[i] != fbShape[i]) {
      isBcast = true;
      break;
    }
  }

  if (!isBcast) {
    const std::vector<uint32_t> dims(fbShape->begin(), fbShape->end());
    ::ttnn::Shape shape(dims);
    ::ttnn::Tensor out = ::ttnn::repeat(in, shape);
    tensorPool.insert_or_assign(op->out()->global_id(), out);
  } else {
    // Broadcast
    const auto dtype =
        ::tt::runtime::ttnn::operations::utils::getDataType(op->out());
    const auto layout =
        ::tt::runtime::ttnn::utils::inferLayoutFromTileShape(op->out());
    const auto zeros = ::ttnn::full(fbShape, 0, dtype, layout);
    const auto out = ::ttnn::add(in, zeros);
    tensorPool.insert_or_assign(op->out()->global_id(), out);
  }
}
} // namespace tt::runtime::ttnn::operations::data_movement
