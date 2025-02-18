// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/pool/maxpool2d.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/types.hpp"
#include <optional>

namespace tt::runtime::ttnn::operations::pool {

void run(const ::tt::target::ttnn::MaxPool2dOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::operations::pool::Pool2DOp<
      ::ttnn::operations::pool::Pool2DType::MAX_POOL2D>
      operation = ::ttnn::operations::pool::Pool2DOp<
          ::ttnn::operations::pool::Pool2DType::MAX_POOL2D>();

  ::ttnn::Tensor input = tensorPool.at(op->in()->global_id());
  DEBUG_ASSERT(input.is_allocated());

  ::ttnn::MemoryConfig outMemConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfig(op->out());
  ::ttnn::Tensor out =
      operation.invoke(::ttnn::DefaultQueueId, input, op->batch_size(),
                       op->input_height(), op->input_width(), op->channels(),
                       {op->kernel_height(), op->kernel_width()},
                       {op->stride_height(), op->stride_width()},
                       {op->padding_height(), op->padding_width()},
                       {op->dilation_height(), op->dilation_width()},
                       outMemConfig, std::nullopt);

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}
} // namespace tt::runtime::ttnn::operations::pool
