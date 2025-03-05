// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/from_device.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/debug_apis.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::FromDeviceOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor = tensorPool.getAndValidate(op->in());
  DEBUG_ASSERT(!::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Calling ttnn::from_device on a host tensor");

  ::ttnn::Tensor out = ::ttnn::from_device(inputTensor);

  tensorPool.insertAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::layout
