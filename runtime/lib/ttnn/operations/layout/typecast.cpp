// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/typecast.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/detail/workarounds.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"
#include "ttnn/operations/core/core.hpp"

namespace tt::runtime::ttnn::operations::layout {

void run(const ::tt::target::ttnn::TypecastOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());

  ::ttnn::DataType targetDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());

  ::ttnn::Tensor out;
  if (workaround::Env::get().toDtypeOnHost &&
      ::tt::runtime::ttnn::utils::isOnHost(inputTensor.storage_type())) {
    out = ::ttnn::to_dtype(inputTensor, targetDataType);
  } else {
    out = ::ttnn::typecast(inputTensor, targetDataType);
  }

  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
