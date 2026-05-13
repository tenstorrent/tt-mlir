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

  // ttnn::sampling kernel expects 4D input [N, C, H, W] where N*C*H == 32
  // (the kernel uses a fixed 32-user batch). The compiler always pads the
  // batch dimension to 32 before calling this op, so batch == 32 here.
  // Reshape 2D [batch, candidates] → [1, 1, batch, candidates] to satisfy
  // the 4D requirement.
  auto inputShape = inputValues.logical_shape();
  if (inputShape.rank() == 2) {
    uint32_t batch = inputShape[0];
    uint32_t candidates = inputShape[1];
    TT_FATAL(batch == 32,
             "ttnn::sampling requires batch == 32 (N*C*H == 32), got batch={}",
             batch);
    inputValues =
        ::ttnn::reshape(inputValues, ::ttnn::Shape({1, 1, batch, candidates}));
    inputIndices =
        ::ttnn::reshape(inputIndices, ::ttnn::Shape({1, 1, batch, candidates}));
  }

  // Workarounds pass requests ROW_MAJOR for index/param tensors but may be
  // skipped when the layout pass optimizes them away. Enforce defensively.
  auto toRowMajor = [](::ttnn::Tensor t) -> ::ttnn::Tensor {
    if (t.layout() != ::ttnn::Layout::ROW_MAJOR) {
      return ::ttnn::to_layout(t, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                               std::nullopt);
    }
    return t;
  };
  inputIndices = toRowMajor(inputIndices);
  k = toRowMajor(k);
  p = toRowMajor(p);
  temp = toRowMajor(temp);

  // Kernel requires k as UINT32.
  if (k.dtype() != ::tt::tt_metal::DataType::UINT32) {
    k = ::ttnn::typecast(k, ::tt::tt_metal::DataType::UINT32);
  }

  ::ttnn::Tensor output =
      ::ttnn::sampling(inputValues, inputIndices, k, p, temp, seed);

  // Reshape output from [1, 1, 1, batch] to match expected 1D output shape.
  auto outShape = output.logical_shape();
  if (outShape.rank() == 4 && outShape[0] == 1 && outShape[1] == 1 &&
      outShape[2] == 1) {
    output = ::ttnn::reshape(output, ::ttnn::Shape({outShape[3]}));
  }

  // ttnn::sampling returns UINT32. Typecast to INT32 to match compiler type.
  if (output.dtype() == ::tt::tt_metal::DataType::UINT32) {
    output = ::ttnn::typecast(output, ::tt::tt_metal::DataType::INT32);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::reduction::sampling
