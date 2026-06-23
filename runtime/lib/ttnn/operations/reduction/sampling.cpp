// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/reduction/sampling.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::reduction::sampling {
void run(const ::tt::target::ttnn::SamplingOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  ::ttnn::Tensor inputValues =
      tensorPool.getTTNNTensorAndValidate(op->input_values());
  ::ttnn::Tensor inputIndices =
      tensorPool.getTTNNTensorAndValidate(op->input_indices());
  ::ttnn::Tensor k = tensorPool.getTTNNTensorAndValidate(op->k());
  ::ttnn::Tensor p = tensorPool.getTTNNTensorAndValidate(op->p());
  ::ttnn::Tensor temp = tensorPool.getTTNNTensorAndValidate(op->temp());

  std::optional<uint32_t> seed;
  if (op->seed().has_value()) {
    seed = op->seed().value();
  }

  // Shape / dtype adaptation is now expressed explicitly in the IR (TTIRToTTNN
  // inserts reshape ops, TTNNWorkarounds pass inserts layout / typecast ops).
  // This handler is a thin 1:1 dispatch to the ttnn::sampling kernel.
  ::ttnn::Tensor output =
      ::ttnn::sampling(inputValues, inputIndices, k, p, temp, seed);

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::reduction::sampling
