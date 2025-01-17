// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/downsample.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttmlir/Target/TTNN/program_generated.h"
#include <ttnn/operations/pool/downsample/downsample.hpp>

namespace tt::runtime::ttnn::operations::pool {
void run(const ::tt::target::ttnn::DownsampleOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(input.is_allocated());

  LOG_ASSERT(op->downsample_params()->size() == 5,
             "downsample_params expected to have 5 elements");
  std::array<uint32_t, 5> downsample_params;
  std::copy_n(op->downsample_params()->begin(), 5, downsample_params.begin());

  ::ttnn::Tensor output =
      ::ttnn::downsample(input, downsample_params, std::nullopt);

  tensorPool.insert_or_assign(op->out()->global_id(), output);
}
} // namespace tt::runtime::ttnn::operations::pool
