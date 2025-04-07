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

  ::ttnn::core::set_printoptions("FULL");
  const std::string tensor_string = input.write_to_string();
  std::string probe_id = op->probe_id()->str();

  std::ofstream log_file("interpreter_log/" + probe_id + ".txt");
  if (log_file.is_open()) {
    log_file << tensor_string;
    log_file.close();
  } else {
    LOG_FATAL("Failed to open log file for probe_id: " + probe_id);
  }

  ::ttnn::Tensor output = input;

  tensorPool.insertAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::print
