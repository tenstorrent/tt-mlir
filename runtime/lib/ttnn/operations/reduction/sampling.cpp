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

  // ttnn::sampling kernel expects 4D input [N, C, H, W] where N*C*H == 32.
  // The compiler may pass 2D [batch, candidates]. Reshape to [1, 1, 32,
  // candidates].
  auto inputShape = inputValues.logical_shape();
  if (inputShape.rank() == 2) {
    uint32_t batch = inputShape[0];
    uint32_t candidates = inputShape[1];
    inputValues =
        ::ttnn::reshape(inputValues, ::ttnn::Shape({1, 1, batch, candidates}));
    inputIndices =
        ::ttnn::reshape(inputIndices, ::ttnn::Shape({1, 1, batch, candidates}));
  }

  // Ensure correct layouts: input_values=TILE, others=ROW_MAJOR.
  if (inputIndices.layout() != ::ttnn::Layout::ROW_MAJOR) {
    inputIndices = ::ttnn::to_layout(inputIndices, ::ttnn::Layout::ROW_MAJOR,
                                     std::nullopt, std::nullopt);
  }
  if (k.layout() != ::ttnn::Layout::ROW_MAJOR) {
    k = ::ttnn::to_layout(k, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                          std::nullopt);
  }
  if (p.layout() != ::ttnn::Layout::ROW_MAJOR) {
    p = ::ttnn::to_layout(p, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                          std::nullopt);
  }
  if (temp.layout() != ::ttnn::Layout::ROW_MAJOR) {
    temp = ::ttnn::to_layout(temp, ::ttnn::Layout::ROW_MAJOR, std::nullopt,
                             std::nullopt);
  }

  // Kernel requires k as UINT32, but compiler may pass INT32.
  if (k.dtype() != ::tt::tt_metal::DataType::UINT32) {
    k = ::ttnn::typecast(k, ::tt::tt_metal::DataType::UINT32);
  }
  // Kernel requires input_indices as UINT32 or INT32 (either is fine).
  if (inputIndices.dtype() != ::tt::tt_metal::DataType::UINT32 &&
      inputIndices.dtype() != ::tt::tt_metal::DataType::INT32) {
    inputIndices =
        ::ttnn::typecast(inputIndices, ::tt::tt_metal::DataType::INT32);
  }

  ::ttnn::Tensor output =
      ::ttnn::sampling(inputValues, inputIndices, k, p, temp, seed);

  // Reshape output from [1, 1, 1, batch] to match expected output shape.
  auto outShape = output.logical_shape();
  if (outShape.rank() == 4 && outShape[0] == 1 && outShape[1] == 1 &&
      outShape[2] == 1) {
    output = ::ttnn::reshape(output, ::ttnn::Shape({outShape[3]}));
  }

  // ttnn::sampling returns UINT32 but the compiler may expect a different type.
  // Typecast to INT32 (the closest supported signed type).
  if (output.dtype() == ::tt::tt_metal::DataType::UINT32) {
    output = ::ttnn::typecast(output, ::tt::tt_metal::DataType::INT32);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), output);
}
} // namespace tt::runtime::ttnn::operations::reduction::sampling
