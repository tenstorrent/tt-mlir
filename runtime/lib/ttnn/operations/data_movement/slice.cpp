// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "slice.h"
#include "tt/runtime/detail/ttnn.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <optional>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::SliceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  ::ttnn::Tensor in = tensorPool.at(op->in()->global_id());

  std::vector<uint32_t> start_idx_vec = {op->start_indices()->begin(), op->start_indices()->end()};
  ::tt::tt_metal::LegacyShape start_idx(start_idx_vec);
  std::vector<uint32_t> end_idx_vec = {op->end_indices()->begin(), op->end_indices()->end()};
  ::tt::tt_metal::LegacyShape end_idx(end_idx_vec);
  std::vector<uint32_t> strides_vec = {op->strides()->begin(), op->strides()->end()};
  ::tt::tt_metal::LegacyShape strides(strides_vec);

  ::ttnn::Tensor out = ::ttnn::slice(0, in, start_idx, end_idx, std::make_optional(strides), std::nullopt);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
