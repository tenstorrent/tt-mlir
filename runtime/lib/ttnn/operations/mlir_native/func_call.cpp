// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/mlir_native/func_call.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/program_executor.h"
#include "tt/runtime/detail/ttnn/types/types.h"

namespace tt::runtime::ttnn::operations::mlir_native {

void run(const ::tt::target::ttnn::FuncCallOp *op, ProgramContext &context) {
  const size_t programIndex = op->program_id();
  std::vector<::tt::runtime::Tensor> inputs;
  inputs.reserve(op->inputs()->size());
  for (const auto *input : *op->inputs()) {
    inputs.emplace_back(
        context.getTensorPool().getRuntimeTensorAndValidate(input));
  }

  ProgramExecutor executor(context.getDeviceHandle(),
                           context.getExecutableHandle(), programIndex, inputs,
                           /*constEvalProgram=*/false);

  executor.execute();
  std::vector<::tt::runtime::Tensor> outputs = executor.gatherOutputTensors();

  LOG_ASSERT(outputs.size() == op->outputs()->size(),
             "Number of outputs does not match");
  for (size_t i = 0; i < op->outputs()->size(); i++) {
    context.getTensorPool().insertRuntimeTensorAndValidate(
        op->outputs()->Get(i), outputs[i]);
  }
}

} // namespace tt::runtime::ttnn::operations::mlir_native
