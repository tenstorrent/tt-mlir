// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/reshape.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/utils.h"

#include "tt/runtime/workarounds.h"

namespace tt::runtime::ttnn::operations::data_movement {
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

  if (::tt::runtime::workaround::Env::get().forceOutOfPlaceReshape) {

    // The logic below is identical to the logic used interally in ttnn to
    // determine if a reshape is a view.
    int64_t tensor_shape_last_dim =
        in.logical_shape().rank() >= 1 ? in.logical_shape()[-1] : 1;
    int64_t shape_last_dim = shape.size() >= 1 ? shape.back() : 1;
    int64_t tensor_shape_second_last_dim =
        in.logical_shape().rank() >= 2
            ? in.logical_shape()[in.logical_shape().size() - 2]
            : 1;
    int64_t shape_second_last_dim =
        shape.size() >= 2 ? shape[shape.size() - 2] : 1;
    int64_t tile_second_dim = ::tt::constants::TILE_HEIGHT;
    int64_t tile_first_dim = ::tt::constants::TILE_WIDTH;
    ::ttnn::MemoryConfig mem_config = memoryConfig.value_or(in.memory_config());
    bool this_is_view =
        (tensor_shape_last_dim == shape_last_dim) &&
        (mem_config.is_sharded() == in.memory_config().is_sharded()) &&
        (mem_config.is_l1() == in.memory_config().is_l1()) &&
        ((in.layout() == ::ttnn::ROW_MAJOR_LAYOUT) || // Its row major
         (tensor_shape_second_last_dim ==
          shape_second_last_dim) || // Second last dimension is the same
         (shape_second_last_dim % tile_second_dim == 0 &&
          tensor_shape_second_last_dim % tile_first_dim ==
              0)); // There is no padding on the second last dimension

    if (this_is_view) {
      // If the reshape is a view, and we are forcing out-of-place reshapes, we
      // must clone the input tensor so that our `out` tensor is not the same
      // object as the `in` tensor.
      ::ttnn::Tensor clonedInput =
          ::ttnn::clone(in, std::nullopt, std::nullopt, std::nullopt);
      out = ::ttnn::reshape(clonedInput, shape, memoryConfig);
    } else {
      out = ::ttnn::reshape(in, shape, memoryConfig);
    }
  } else {
    out = ::ttnn::reshape(in, shape, memoryConfig);
  }

  //   int64_t tensor_shape_last_dim = in.logical_shape().rank() >= 1 ?
  //   in.logical_shape()[-1] : 1; int64_t shape_last_dim = shape.size() >= 1 ?
  //   shape.back() : 1; int64_t tensor_shape_second_last_dim =
  //   in.logical_shape().rank() >= 2 ?
  //   in.logical_shape()[in.logical_shape().size() - 2] : 1; int64_t
  //   shape_second_last_dim = shape.size() >= 2 ? shape[shape.size() - 2] : 1;
  //   int64_t tile_second_dim = ::tt::constants::TILE_HEIGHT;
  //   int64_t tile_first_dim = ::tt::constants::TILE_WIDTH;
  //   ::ttnn::MemoryConfig mem_config =
  //   memoryConfig.value_or(in.memory_config()); bool this_is_view =
  //         (tensor_shape_last_dim == shape_last_dim) &&
  //         (mem_config.is_sharded() == in.memory_config().is_sharded()) &&
  //         (mem_config.is_l1() == in.memory_config().is_l1()) &&
  //         ((in.layout() == ::ttnn::ROW_MAJOR_LAYOUT) ||              // Its
  //         row major
  //          (tensor_shape_second_last_dim == shape_second_last_dim) ||  //
  //          Second last dimension is the same (shape_second_last_dim %
  //          tile_second_dim == 0 &&
  //           tensor_shape_second_last_dim % tile_first_dim == 0));  // There
  //           is no padding on the second last dimension

  //   bool is_kv_cache_input = in.logical_shape() == ::ttnn::Shape({1, 1, 12,
  //   32, 64}); const auto target_shape_small_vector =
  //   ::ttsl::SmallVector<uint32_t>(shape.begin(), shape.end());
  //   ::ttnn::Shape target_shape(target_shape_small_vector);
  //   if (this_is_view) {

  //       std::cout << "Reshape is a view" << std::endl;
  //       std::cout << "input shape: " << in.logical_shape() << std::endl;
  //       std::cout << "target shape: " << target_shape << std::endl;
  //       std::cout << "dtype: " << in.dtype() << std::endl;

  //       auto newIn = ::ttnn::clone(in, std::nullopt, std::nullopt,
  //       std::nullopt); out = ::ttnn::reshape(newIn, shape, memoryConfig);
  //   } else {
  //       if (is_kv_cache_input && this_is_view) {
  //         std::cout << "Reshape is a view but we're going to reshape it
  //         in-place anyway since its kv cache input" << std::endl; std::cout
  //         << "input shape: " << in.logical_shape() << std::endl; std::cout <<
  //         "target shape: " << target_shape << std::endl; std::cout << "dtype:
  //         " << in.dtype() << std::endl;
  //       }
  //       out = ::ttnn::reshape(in, shape, memoryConfig);
  //   }
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
