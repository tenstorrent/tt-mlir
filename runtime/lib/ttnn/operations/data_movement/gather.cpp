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

  // TODO(tt-metal): Rank-1 inputs trip a TT_FATAL in
  // ttnn::squeeze_from_4D -> Shape::to_rank because the 4D padded shape has
  // tile padding on dim index 2, which must be dropped when squeezing back to
  // rank 1. Unsqueeze to rank 2 here so the squeeze-back only drops leading
  // ones. Remove this workaround once the upstream fix lands in tt-metal's
  // ttnn::gather (ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp's
  // post_gather_transform_tensor should recompute the padded shape from the
  // target rank instead of reusing the 4D padded shape).
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
    ::ttnn::Tensor out = ::ttnn::reshape(outReshaped, outShape);
    tensorPool.insertTTNNTensorAndValidate(op->out(), out);
    return;
  }

  // TODO(tt-metal): For rank > 4 with gather dim not being the last dim,
  // ttnn::gather's post_gather_transform_tensor mis-reverses its preprocessing.
  // Pre-transform does (transpose dim->last, then fold to 4D); post-transform
  // does (transpose-in-4D, then reshape to ND). For dim=0, rank=5 the
  // transpose computes dim_adj = dim + (4 - orig_rank) = -1, making
  // ttnn::transpose(out, -1, -1) a no-op, and the subsequent reshape
  // reinterprets still-transposed data under the original stride (PCC ~= 0).
  // Workaround: move the gather dim to last here so ttnn::gather takes its
  // working is_dim_last_idx path, then transpose the output back. Remove once
  // tt-metal fixes post_gather_transform_tensor for rank > 4 (should reshape
  // ND-first, then transpose in ND) at
  // ttnn/cpp/ttnn/operations/data_movement/gather/gather.cpp.
  int32_t normalizedDim = dim < 0 ? dim + static_cast<int32_t>(inputRank) : dim;
  if (inputRank > 4 && normalizedDim != static_cast<int32_t>(inputRank) - 1) {
    int32_t lastDim = static_cast<int32_t>(inputRank) - 1;
    ::ttnn::Tensor inputTransposed =
        ::ttnn::transpose(input, normalizedDim, lastDim, std::nullopt);
    ::ttnn::Tensor indexTransposed =
        ::ttnn::transpose(index, normalizedDim, lastDim, std::nullopt);

    ::ttnn::Tensor outTransposed = ::ttnn::gather(
        inputTransposed, /*dim=*/-1, indexTransposed,
        /*sparse_grad=*/false, outputMemoryConfig, std::nullopt, std::nullopt);

    ::ttnn::Tensor out = ::ttnn::transpose(outTransposed, normalizedDim,
                                           lastDim, outputMemoryConfig);
    tensorPool.insertTTNNTensorAndValidate(op->out(), out);
    return;
  }

  ::ttnn::Tensor out =
      ::ttnn::gather(input, dim, index, /*sparse_grad=*/false,
                     outputMemoryConfig, std::nullopt, std::nullopt);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
