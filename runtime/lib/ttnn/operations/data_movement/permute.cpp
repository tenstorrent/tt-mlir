// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/permute.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <vector>

#include "ttnn/operations/experimental/quasar/transpose/transpose.hpp"

namespace tt::runtime::ttnn::operations::data_movement {

namespace {

// Realise an arbitrary permutation on Quasar as a sequence of 2-axis swaps.
//
// Quasar exposes `transpose` (a single pair swap) but no general `permute`, and
// mainline ttnn::permute is not an option: its PermuteDeviceOperation builds a
// Gen1 compute config, which MakeGen2ComputeConfig rejects on Quasar
// ("generation mismatch", program_spec.cpp). Forge's NCHW<->NHWC permutes are
// 3-cycles ([0,2,3,1] and [0,3,1,2]), so one swap is never enough.
//
// Selection sort over the axes: walk the output positions, and for each one swap
// the required input axis into place. `current[i]` tracks which original axis
// currently sits at position i. This emits at most rank-1 transposes, and none
// at all for the identity.
//
// memoryConfig is applied only to the final transpose: forcing it on the
// intermediates would pay for a layout change that the next swap discards.
::ttnn::Tensor permuteViaTransposes(
    const ::ttnn::Tensor &in, const ::ttsl::SmallVector<int64_t> &permutation,
    const std::optional<::ttnn::MemoryConfig> &memoryConfig, float padValue) {
  const int64_t rank = static_cast<int64_t>(permutation.size());

  // Normalise so a negative axis compares equal to its positive form below.
  ::ttsl::SmallVector<int64_t> target;
  target.reserve(permutation.size());
  for (int64_t axis : permutation) {
    target.push_back(axis < 0 ? axis + rank : axis);
  }

  // The swaps to perform, collected first so the last one can carry the
  // memory config.
  std::vector<std::pair<int64_t, int64_t>> swaps;
  ::ttsl::SmallVector<int64_t> current;
  current.reserve(static_cast<size_t>(rank));
  for (int64_t i = 0; i < rank; ++i) {
    current.push_back(i);
  }

  for (int64_t i = 0; i < rank; ++i) {
    if (current[i] == target[i]) {
      continue;
    }
    int64_t j = i + 1;
    while (j < rank && current[j] != target[i]) {
      ++j;
    }
    LOG_ASSERT(j < rank, "Invalid permutation: axis ", target[i],
               " appears more than once or is out of range");
    std::swap(current[i], current[j]);
    swaps.emplace_back(i, j);
  }

  if (swaps.empty()) {
    // Identity. Still honour the requested memory config if there is one.
    if (memoryConfig.has_value()) {
      return ::ttnn::operations::experimental::quasar::transpose(
          in, 0, 0, memoryConfig, padValue);
    }
    return in;
  }

  ::ttnn::Tensor out = in;
  for (size_t s = 0; s < swaps.size(); ++s) {
    const bool isLast = (s + 1 == swaps.size());
    out = ::ttnn::operations::experimental::quasar::transpose(
        out, swaps[s].first, swaps[s].second,
        isLast ? memoryConfig : std::nullopt, padValue);
  }
  return out;
}

} // namespace

void run(const ::tt::target::ttnn::PermuteOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttsl::SmallVector<int64_t> permutation(op->permutation()->begin(),
                                           op->permutation()->end());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  float padValue = op->pad_value();

  ::ttnn::Tensor out =
      utils::isQuasar()
          ? permuteViaTransposes(in, permutation, memoryConfig, padValue)
          : ::ttnn::permute(in, permutation, memoryConfig, padValue);

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
