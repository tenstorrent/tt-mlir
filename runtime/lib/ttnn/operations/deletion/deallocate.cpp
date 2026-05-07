// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "operations/deletion/deallocate.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

namespace tt::runtime::ttnn::operations::deletion {
void run(const ::tt::target::ttnn::DeallocateOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  // Use the non-validating pool getter (`getRuntimeTensor`) so we don't
  // hit `DEBUG_ASSERT(is_allocated())` in `getRuntimeTensorAndValidate`
  // when the tensor was already freed by an earlier
  // `load_cached`-driven retain-clear (the compiler emits a
  // ttnn.deallocate immediately after each load_cached for const-eval
  // input args; with `TT_RUNTIME_FREE_CONST_EVAL_INPUTS=1`, that
  // earlier dealloc may already have fired). The pool erase below is
  // still safe — it's a metadata cleanup.
  const ::tt::runtime::Tensor &runtimeTensor =
      tensorPool.getRuntimeTensor(op->in()->global_id());
  ::tt::runtime::ttnn::TTNNTensorWrapper &tensorWrapper =
      const_cast<::tt::runtime::ttnn::TTNNTensorWrapper &>(
          runtimeTensor.as<::tt::runtime::ttnn::TTNNTensorWrapper>(
              DeviceRuntime::TTNN));
  ::ttnn::Tensor &ttnnTensor = tensorWrapper.getTensor();

  if (!tensorWrapper.shouldRetain() && ttnnTensor.is_allocated()) {
    ::ttnn::deallocate(ttnnTensor, op->force());
  } else if (tensorWrapper.shouldRetain()) {
    LOG_DEBUG("Tensor is retained thus not deallocating. To deallocate, set "
              "retain to false first");
  }
  // else: not retained AND not allocated → no-op (already freed by an
  // earlier path; pool erase below still removes the wrapper entry).

  tensorPool.erase(op->in());
}
} // namespace tt::runtime::ttnn::operations::deletion
