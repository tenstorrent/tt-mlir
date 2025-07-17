// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/typecast.h"
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/workarounds.h"
#include "ttnn/operations/core/core.hpp"

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::TypecastOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttnn::DataType targetDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());

  ::ttnn::Tensor out;

  // Special case: if typecasting from float32 or bfloat16 to int32, use
  // ttnn::round
  if ((inputTensor.dtype() == ::ttnn::DataType::FLOAT32 ||
       inputTensor.dtype() == ::ttnn::DataType::BFLOAT16) &&
      targetDataType == ::ttnn::DataType::INT32) {
    out = ::ttnn::round(inputTensor, 0);
    out = ::ttnn::typecast(out, targetDataType);
  } else {
    out = ::ttnn::typecast(inputTensor, targetDataType);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
