// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "cpu.h"

#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/common/dylib.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/utils.h"

#include <dlfcn.h>
#include <link.h>

namespace tt::runtime::ttnn::operations::cpu {

void run(const ::tt::target::ttnn::CpuOp *op, ProgramContext &context) {
  common::WrappedFunc fn = context.getDylibManager().getFunc(
      op->dylib_id(), op->func_name()->c_str());
  LOG_ASSERT(fn != nullptr);

  const auto *fbInputs = op->ins();

  std::vector<std::vector<int64_t>> allSizesAndStrides;

  std::function<void *(const tt::target::ttnn::TensorRef *)> getTensorDataPtr =
      [&context](const tt::target::ttnn::TensorRef *ref) -> void * {
    const auto &tens = context.getTensorPool().getTTNNTensorAndValidate(ref);
    return ::tt::runtime::ttnn::utils::getRawHostDataPtr(tens);
  };

  auto dylibInputs = tt::runtime::common::packTensors(
      fbInputs, getTensorDataPtr, allSizesAndStrides);

  // Last inputs are actually output destinations. We need to associate them
  // with tensors in the pool.
  for (size_t i = 0; i < op->outs()->size(); ++i) {
    const auto &fbOutput = op->outs()->Get(i);

    ::ttnn::Tensor outTensor = context.getTensorPool().getTTNNTensorAndValidate(
        fbInputs->Get(fbInputs->size() - op->outs()->size() + i));

    context.getTensorPool().insertTTNNTensorAndValidate(fbOutput, outTensor);
  }

  fn(dylibInputs.data());
  // We don't need to unpack any data from output, it should be written directly
  // to correct memory.
}

} // namespace tt::runtime::ttnn::operations::cpu
