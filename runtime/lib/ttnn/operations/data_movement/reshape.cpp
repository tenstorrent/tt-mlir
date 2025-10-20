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

  if (::tt::runtime::workaround::Env::get().forceOutOfPlaceReshape &&
      utils::isInPlaceReshape(in, shape, memoryConfig)) {
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
