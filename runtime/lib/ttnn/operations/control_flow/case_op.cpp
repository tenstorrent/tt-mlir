// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/control_flow/case_op.h"
#include "operations/control_flow/sub_program.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/types/types.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::control_flow {

namespace {

// Reads the index back to host, the one device-to-host synchronization this op
// pays, and resolves it to a branch. The schema requires a host-resident
// single-element si32 index, so nothing is moved here.
//
// An index outside [0, numBranches) selects the last branch. Reading it signed
// is what makes a negative index land there rather than wrapping around to a
// valid one.
uint32_t selectBranch(const ::tt::target::ttnn::CaseOp *op,
                      ProgramContext &context, uint32_t numBranches) {
  const ::ttnn::Tensor &indexTensor =
      context.getTensorPool()
          .getRuntimeTensorAndValidate(op->index())
          .as<::tt::runtime::ttnn::TTNNTensorWrapper>(DeviceRuntime::TTNN)
          .getTensor();
  const int32_t index =
      ::tt::runtime::ttnn::utils::getScalarFromTensor<int32_t>(indexTensor);

  if (index < 0 || static_cast<uint32_t>(index) >= numBranches) {
    return numBranches - 1;
  }
  return static_cast<uint32_t>(index);
}

} // namespace

void run(const ::tt::target::ttnn::CaseOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const uint32_t numBranches = op->branch_program_ids()->size();
  LOG_ASSERT(numBranches > 0, "Case op must have at least one branch");

  const uint32_t selected = selectBranch(op, context, numBranches);

  // The branch programs take the captures as their inputs, in order.
  std::vector<::tt::runtime::Tensor> captures;
  captures.reserve(op->captures()->size());
  for (const auto *capture : *op->captures()) {
    captures.emplace_back(tensorPool.getRuntimeTensorAndValidate(capture));
  }

  std::vector<::tt::runtime::GlobalSemaphore> semaphores =
      utils::collectSemaphoreInputs(op->semaphore_inputs(), context);

  std::vector<::tt::runtime::Tensor> outputs = runSubProgram(
      op->branch_program_ids()->Get(selected), context, captures, semaphores);
  LOG_ASSERT(outputs.size() == op->outputs()->size(),
             "Case branch program returned ", outputs.size(),
             " values but the op has ", op->outputs()->size(), " outputs");

  for (size_t i = 0; i < op->outputs()->size(); i++) {
    tensorPool.insertRuntimeTensorAndValidate(op->outputs()->Get(i),
                                              outputs[i]);
  }
}

} // namespace tt::runtime::ttnn::operations::control_flow
