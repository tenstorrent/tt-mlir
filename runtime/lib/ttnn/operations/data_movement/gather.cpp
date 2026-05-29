// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/gather.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

namespace tt::runtime::ttnn::operations::data_movement {

void run(const ::tt::target::ttnn::GatherOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input =
      tensorPool.getTTNNTensorAndValidate(op->input());
  const ::ttnn::Tensor &index =
      tensorPool.getTTNNTensorAndValidate(op->index());
  int32_t dim = op->dim();

  std::optional<::ttnn::MemoryConfig> outputMemoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());

  // Workaround to support Rank-1 input tensors.
  // tt-metal issue: https://github.com/tenstorrent/tt-metal/issues/45155
  const auto inputRank = input.logical_shape().rank();
  const auto indexRank = index.logical_shape().rank();
  if (inputRank < 2 || indexRank < 2) {
    std::vector<int32_t> inputShape2D = {1};
    for (size_t i = 0; i < inputRank; ++i) {
      inputShape2D.push_back(static_cast<int32_t>(input.logical_shape()[i]));
    }
    std::vector<int32_t> indexShape2D = {1};
    for (size_t i = 0; i < indexRank; ++i) {
      indexShape2D.push_back(static_cast<int32_t>(index.logical_shape()[i]));
    }

    ::ttnn::Tensor inputReshaped = ::ttnn::reshape(input, inputShape2D);
    ::ttnn::Tensor indexReshaped = ::ttnn::reshape(index, indexShape2D);

    // Map `dim` from the original rank into the unsqueezed (rank-2) layout.
    // Original dim d referenced in a rank-R tensor becomes d + (2 - R) under
    // the leading-1 prepend. Negative dims already index from the back and
    // remain valid.
    int32_t adjustedDim = dim;
    if (dim >= 0) {
      adjustedDim = dim + static_cast<int32_t>(2 - inputRank);
    }

    ::ttnn::Tensor outReshaped = ::ttnn::gather(
        inputReshaped, adjustedDim, indexReshaped,
        /*sparse_grad=*/false, outputMemoryConfig, std::nullopt, std::nullopt);

    std::vector<int32_t> outShape;
    outShape.reserve(indexRank);
    for (size_t i = 0; i < indexRank; ++i) {
      outShape.push_back(static_cast<int32_t>(index.logical_shape()[i]));
    }
    ::ttnn::Tensor out =
        ::ttnn::reshape(outReshaped, outShape, outputMemoryConfig);
    tensorPool.insertTTNNTensorAndValidate(op->out(), out);
    return;
  }

  ::ttnn::Tensor out =
      ::ttnn::gather(input, dim, index, /*sparse_grad=*/false,
                     outputMemoryConfig, std::nullopt, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
