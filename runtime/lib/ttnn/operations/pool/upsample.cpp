// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample.h"

#include "tt/runtime/detail/logger.h"

namespace tt::runtime::ttnn::operations::pool {
void run(const ::tt::target::ttnn::UpsampleOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(input.is_allocated());

  std::array<uint32_t, 4> scaleFactor;
  std::copy(op->scale_factor()->begin(), op->scale_factor()->end(),
            scaleFactor.begin());
  std::string mode = op->mode()->str();
  ::ttnn::Tensor output = ::ttnn::upsample(input, scaleFactor, mode);

  tensorPool.insert_or_assign(op->out()->global_id(), std::move(output));
}
} // namespace tt::runtime::ttnn::operations::pool
