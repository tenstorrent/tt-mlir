// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/creation/constant.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::creation {

void run(const ::tt::target::ttnn::ConstantOp *op, ProgramContext &context) {
  ::ttnn::Shape shape =
      operations::utils::toTTNNShape(*op->out()->desc()->shape());

  // Get data type from tensor descriptor
  ::tt::target::DataType targetDtype =
      op->out()->desc()->layout()->memory_desc()->data_type();

  ::ttnn::DataType ttnnDtype =
      ::tt::runtime::ttnn::utils::toTTNNDataType(targetDtype);

  ::ttnn::Tensor out = utils::toTTNNTensor(op->data(), shape, ttnnDtype);

  context.getTensorPool().insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::creation
