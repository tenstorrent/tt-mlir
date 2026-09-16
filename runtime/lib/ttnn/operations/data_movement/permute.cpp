// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/permute.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <vector>
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"
#include "ttnn/operations/experimental/quasar/transpose/transpose.hpp"

namespace tt::runtime::ttnn::operations::data_movement {

// Quasar has no permute entry point, only a two-axis transpose, so a general
// permutation has to be decomposed. Selection sort over the axes: walk the target
// order and swap the axis that belongs in each slot into place. That is at most
// rank-1 transposes, each of which is a Quasar op that works.
static ::ttnn::Tensor
permuteViaTransposes(const ::ttnn::Tensor &in,
                     const ::ttsl::SmallVector<int64_t> &permutation,
                     const std::optional<::ttnn::MemoryConfig> &memoryConfig,
                     float padValue) {
  const int64_t rank = static_cast<int64_t>(permutation.size());
  // current[i] is the ORIGINAL axis currently sitting at position i.
  std::vector<int64_t> current(rank);
  for (int64_t i = 0; i < rank; ++i) {
    current[i] = i;
  }
  ::ttnn::Tensor out = in;
  for (int64_t slot = 0; slot < rank; ++slot) {
    const int64_t want = permutation[slot];
    int64_t at = slot;
    while (at < rank && current[at] != want) {
      ++at;
    }
    LOG_ASSERT(at < rank, "permutation is not a permutation of the axes");
    if (at == slot) {
      continue;
    }
    out = ::ttnn::operations::experimental::quasar::transpose(out, slot, at,
                                                              memoryConfig,
                                                              padValue);
    std::swap(current[slot], current[at]);
  }
  return out;
}

void run(const ::tt::target::ttnn::PermuteOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &in = tensorPool.getTTNNTensorAndValidate(op->in());

  ::ttsl::SmallVector<int64_t> permutation(op->permutation()->begin(),
                                           op->permutation()->end());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
          op->memory_config());
  float padValue = op->pad_value();

  // A TILE tensor stores its trailing two dims in 32x32 tiles, so a transpose
  // that moves a non-tile-aligned extent through those slots drags tile padding
  // into the logical data. Measured on the emulator: [1,7,7,2048] with
  // permutation <0,3,1,2> returned pcc=0.931 -- plausible-looking and wrong.
  // Same failure mode, and same remedy, as the tiled reshape.
  ::ttnn::Tensor out;
  if (utils::isQuasar()) {
    const ::ttnn::Shape &inShape = in.logical_shape();
    const size_t rank = inShape.rank();
    auto tileAligned = [](int64_t v) { return v > 0 && (v % 32) == 0; };

    bool safeInPlace = in.layout() == ::ttnn::Layout::ROW_MAJOR;
    if (!safeInPlace && rank >= 2 && permutation.size() == rank) {
      const int64_t inW = static_cast<int64_t>(inShape[rank - 1]);
      const int64_t inH = static_cast<int64_t>(inShape[rank - 2]);
      // the extents that will land in the trailing two slots after permuting
      const int64_t outW =
          static_cast<int64_t>(inShape[permutation[rank - 1]]);
      const int64_t outH =
          static_cast<int64_t>(inShape[permutation[rank - 2]]);
      safeInPlace = tileAligned(inW) && tileAligned(inH) &&
                    tileAligned(outW) && tileAligned(outH);
    }

    if (safeInPlace) {
      out = permuteViaTransposes(in, permutation, memoryConfig, padValue);
    } else {
      // Permute in ROW_MAJOR, where a transpose is a pure index remap with no
      // tiles to straddle, then restore the original layout.
      const ::ttnn::Layout originalLayout = in.layout();
      ::ttnn::Tensor rowMajor =
          ::ttnn::operations::experimental::quasar::to_layout(
              in, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
      ::ttnn::Tensor permuted =
          permuteViaTransposes(rowMajor, permutation, std::nullopt, padValue);
      out = ::ttnn::operations::experimental::quasar::to_layout(
          permuted, originalLayout, std::nullopt, memoryConfig);
    }
  } else {
    out = ::ttnn::permute(in, permutation, memoryConfig, padValue);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
