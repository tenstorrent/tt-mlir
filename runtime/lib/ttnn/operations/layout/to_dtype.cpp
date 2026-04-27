// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/to_dtype.h"
#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/detail/ttnn/utils.h"

#include <ttnn/operations/core/to_dtype/to_dtype_op.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/types.hpp>

namespace tt::runtime::ttnn::operations::layout {

void run(const ::tt::target::ttnn::ToDtypeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  DEBUG_ASSERT(::tt::runtime::ttnn::utils::inSystemMemory(op->in()),
               "Calling ttnn::to_dtype on a device tensor; expected host");

  ::ttnn::DataType targetDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());

  ::ttnn::Tensor out = ::ttnn::to_dtype(inputTensor, targetDataType);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
