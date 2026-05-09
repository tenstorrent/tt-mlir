// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/slice_reshape.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttmlir/Target/TTNN/program_generated.h"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include <cstdint>
#include <vector>

namespace tt::runtime::ttnn::operations::data_movement {

// Fused slice + reshape: same kernel calls as the unfused form (one slice,
// one reshape), but only one program-executor case — saving one
// flatbuffer-dispatch round-trip per occurrence. Mirrors quetzal's
// fuse_slice_reshape pass which is the host-dispatch reduction.
void run(const ::tt::target::ttnn::SliceReshapeOp *op,
         ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();
  const ::ttnn::Tensor &input = tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttsl::SmallVector<int32_t> begins(op->begins()->begin(),
                                      op->begins()->end());
  ::ttsl::SmallVector<int32_t> ends(op->ends()->begin(), op->ends()->end());
  ::ttsl::SmallVector<int32_t> step(op->step()->begin(), op->step()->end());

  ttsl::Span<const int32_t> beginsSpan(begins.data(), begins.size());
  ttsl::Span<const int32_t> endsSpan(ends.data(), ends.size());
  ttsl::Span<const int32_t> stepSpan(step.data(), step.size());

  std::optional<::ttnn::MemoryConfig> memoryConfig =
      op->memory_config() == 0
          ? ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()))
          : ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                op->memory_config());

  ::ttnn::Tensor sliced =
      ::ttnn::slice(input, beginsSpan, endsSpan, stepSpan, memoryConfig);

  std::vector<int32_t> shape(op->shape()->begin(), op->shape()->end());
  ::ttnn::Tensor out = ::ttnn::reshape(sliced, shape, memoryConfig);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::data_movement
