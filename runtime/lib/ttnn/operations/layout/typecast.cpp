// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast.h"
#include "tt/runtime/detail/ttnn.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::layout {

void run(const ::tt::target::ttnn::TypecastOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &inputTensor = tensorPool.at(op->in()->global_id());
  auto idt = inputTensor.dtype();
  auto ily = inputTensor.layout();
  auto shp = inputTensor.shape();
  auto lshp = inputTensor.legacy_shape();
  (void)idt;
  (void)ily;
  (void)shp;
  (void)lshp;
  ::ttnn::DataType targetDataType =
      ::tt::runtime::ttnn::utils::toTTNNDataType(op->dtype());

  // if (inputTensor.dtype() == targetDataType) {
  //   tensorPool.insert_or_assign(op->out()->global_id(), inputTensor);
  //   return;
  // }
  ::ttnn::Tensor out = ::ttnn::typecast(inputTensor, targetDataType);
  tensorPool.insert_or_assign(op->out()->global_id(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
