// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/reshape.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/utils.h"

#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::data_movement {

static bool
isInPlaceReshape(const ::ttnn::Tensor &in, const std::vector<int32_t> &shape,
                 const std::optional<::ttnn::MemoryConfig> &memoryConfig) {
  // The logic below is identical to the logic used internally in ttnn to
  // determine if a reshape is a view.
  int64_t tensorShapeLastDim =
      in.logical_shape().rank() >= 1 ? in.logical_shape()[-1] : 1;
  int64_t shapeLastDim = shape.size() >= 1 ? shape.back() : 1;
  int64_t tensorShapeSecondLastDim =
      in.logical_shape().rank() >= 2
          ? in.logical_shape()[in.logical_shape().size() - 2]
          : 1;
  int64_t shapeSecondLastDim = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
  int64_t tileSecondDim = ::tt::constants::TILE_HEIGHT;
  int64_t tileFirstDim = ::tt::constants::TILE_WIDTH;
  ::ttnn::MemoryConfig memConfig = memoryConfig.value_or(in.memory_config());
  return (tensorShapeLastDim == shapeLastDim) &&
         (memConfig.is_sharded() == in.memory_config().is_sharded()) &&
         (memConfig.is_l1() == in.memory_config().is_l1()) &&
         ((in.layout() == ::ttnn::ROW_MAJOR_LAYOUT) || // It's row major
          (tensorShapeSecondLastDim ==
           shapeSecondLastDim) || // Second last dimension is the same
          (shapeSecondLastDim % tileSecondDim == 0 &&
           tensorShapeSecondLastDim % tileFirstDim ==
               0)); // There is no padding on the second last dimension
}

void run(const ::tt::target::ttnn::ReshapeOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  const auto *fbShape = op->shape();
  std::vector<int32_t> shape(fbShape->begin(), fbShape->end());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config() == 0
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()))
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config());
  ::ttnn::Tensor out;

  if (::tt::runtime::workaround::Env::get().forceOutOfPlaceReshape &&
      isInPlaceReshape(in, shape, memoryConfig)) {
    // If the reshape is a view, and we are forcing out-of-place reshapes, we
    // must clone the input tensor so that our `out` tensor is not the same
    // object as the `in` tensor.
    ::ttnn::Tensor clonedInput =
        ::ttnn::clone(in, std::nullopt, std::nullopt, std::nullopt);
    out = ::ttnn::reshape(clonedInput, shape, memoryConfig);
  } else {
    out = ::ttnn::reshape(in, shape, memoryConfig);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
