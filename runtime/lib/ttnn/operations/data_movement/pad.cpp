// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/pad.h"
#include "tt/runtime/detail/logger.h"
#include <tt-metalium/span.hpp>
#include <ttnn/tensor/types.hpp>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::PadOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(in.is_allocated());

  float padValue = op->value();

  ::ttnn::Tensor out;

  std::vector<std::pair<uint32_t, uint32_t>> padding;
  for (uint32_t i = 0; i < op->padding()->size(); i += 2) {
    padding.push_back(
        std::make_pair(op->padding()->Get(i), op->padding()->Get(i + 1)));
  }
  tt::stl::Span<const std::pair<uint32_t, uint32_t>> paddingSpan(
      padding.data(), padding.size());
  out = ::ttnn::pad(0, in, padding, padValue, true, std::nullopt);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
