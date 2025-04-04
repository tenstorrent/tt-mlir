// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/print/print.h"
#include <iostream>
#include "tt/runtime/detail/logger.h"
#include "tt/runtime/ttnn/debug_apis.h"
#include "tt/runtime/ttnn/operations/utils.h"
#include "tt/runtime/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::print {
void run(const ::tt::target::ttnn::PrintOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor &input = tensorPool.getAndValidate(op->in());

  std::cerr << "-------this is tensor----------" << std::endl;
  ::ttnn::core::set_printoptions("FULL");
  std::cerr << input.write_to_string() << std::endl;
  std::cerr << "-------this is tensor----------" << std::endl;

  ::ttnn::Tensor output = input;

  tensorPool.insertAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::print
