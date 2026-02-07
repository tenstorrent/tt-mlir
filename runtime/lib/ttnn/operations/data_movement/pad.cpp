// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/pad.h"
#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/workarounds.h"
#include "tt_stl/span.hpp"

#include "ttnn/operations/experimental/reshape/view.hpp"

#include <optional>

namespace tt::runtime::ttnn::operations::data_movement {
void run(const ::tt::target::ttnn::PadOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  float padValue = op->value();

  ::ttnn::Tensor out;

  ::ttsl::SmallVector<::ttnn::operations::data_movement::PadSpecDim> padding;
  for (uint32_t i = 0; i < op->padding()->size(); i += 2) {
    padding.emplace_back(op->padding()->Get(i), op->padding()->Get(i + 1));
  }

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());

  out = ::ttnn::pad(in, padding, padValue, op->use_multicore(),
                    outputMemoryConfig);

  // ttnn::pad may not update the logical shape to match the padded shape. Fix
  // by using a zero-cost view to synchronize them.
  if (out.logical_shape() != out.padded_shape()) {
    out =
        ::ttnn::experimental::view(out, out.padded_shape(), out.padded_shape());
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
