// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/permute.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <vector>
#include <numeric>
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"
#include "ttnn/operations/experimental/quasar/transpose/transpose.hpp"

namespace tt::runtime::ttnn::operations::data_movement {

// Quasar: decompose a permutation into ADJACENT transposes.
//
// Mainline ttnn::permute returns an ALL-ZERO tensor on Quasar whenever an inner axis moves to
// last -- measured pcc 0.000000 / zerofrac 1.0000 for <0,2,3,1> (NCHW->NHWC) at every ResNet-50
// conv shape, in both ROW_MAJOR and TILE. <0,3,1,2> is fine, so it is specifically that
// direction. quasar::transpose is correct (pcc 0.999982-0.999996), and composing two ADJACENT
// swaps reproduces <0,2,3,1> exactly:
//     (0,1,2,3) -swap(1,2)-> (0,2,1,3) -swap(2,3)-> (0,2,3,1)
//
// An earlier helper decomposed via selection sort, which emits NON-adjacent swaps and still
// returned zeros. Adjacency is the part that matters, so this bubbles each axis into place one
// neighbour at a time.
static ::ttnn::Tensor
permuteViaAdjacentTransposes(const ::ttnn::Tensor &in,
                             const ::ttsl::SmallVector<int64_t> &permutation,
                             const std::optional<::ttnn::MemoryConfig> &memoryConfig,
                             float padValue) {
  const int64_t rank = static_cast<int64_t>(permutation.size());
  // current[i] is the ORIGINAL axis sitting at position i.
  std::vector<int64_t> current(rank);
  std::iota(current.begin(), current.end(), 0);

  ::ttnn::Tensor out = in;
  for (int64_t slot = 0; slot < rank; ++slot) {
    const int64_t want = permutation[slot];
    int64_t at = slot;
    while (at < rank && current[at] != want) {
      ++at;
    }
    LOG_ASSERT(at < rank, "permutation is not a permutation of the axes");
    // Walk it down to `slot` one neighbour at a time.
    for (int64_t k = at; k > slot; --k) {
      out = ::ttnn::operations::experimental::quasar::transpose(out, k - 1, k,
                                                                memoryConfig, padValue);
      std::swap(current[k - 1], current[k]);
    }
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
  // Quasar has no permute of its own, so this necessarily lands on mainline
  // ttnn::permute. Which program factory it picks is what matters:
  //
  //   ROW_MAJOR + last dim moved -> permute_rm_blocked_generic  <- HANGS on ZeBu
  //   TILE                       -> permute_tiled_{generic,row_invariant,invariant}
  //
  // The earlier decomposition-into-transposes needed a ROW_MAJOR round-trip to
  // avoid dragging tile padding into the data (measured pcc=0.931 without it) --
  // but that round-trip is exactly what forced the blocked-RM path, which then
  // hung on the emulator for every permute in ResNet-50.
  //
  // tt-metal's own tiled factories already handle the tile-padding case, so
  // handing the whole permute to ttnn::permute on the TILE tensor both keeps the
  // numerics right and stays off the hanging factory. The compute configs in
  // those factories are arch-correct as of the Gen1/Gen2 sweep.

  // Quasar: route every permute through adjacent quasar::transpose swaps. See the helper
  // above -- mainline ttnn::permute silently returns zeros for the NCHW->NHWC direction,
  // which is the permute forge wraps around every conv.
  if (utils::isQuasar() &&
      std::getenv("TTMLIR_QUASAR_PERMUTE_NO_TRANSPOSE") == nullptr) {
    ::ttnn::Tensor viaTranspose =
        permuteViaAdjacentTransposes(in, permutation, memoryConfig, padValue);
    tensorPool.insertTTNNTensorAndValidate(op->out(), viaTranspose);
    return;
  }

  ::ttnn::Tensor out;
  if (utils::isQuasar() && in.layout() == ::ttnn::Layout::ROW_MAJOR &&
      !permutation.empty() &&
      permutation.back() != static_cast<int64_t>(in.logical_shape().rank()) - 1) {
    // A ROW_MAJOR input whose last dim moves would select the blocked-RM factory.
    // Tilize first so a tiled factory is chosen instead, then restore the layout.
    ::ttnn::Tensor tiled = ::ttnn::operations::experimental::quasar::to_layout(
        in, ::ttnn::Layout::TILE, std::nullopt, std::nullopt);
    ::ttnn::Tensor permuted =
        ::ttnn::permute(tiled, permutation, std::nullopt, padValue);
    out = ::ttnn::operations::experimental::quasar::to_layout(
        permuted, ::ttnn::Layout::ROW_MAJOR, std::nullopt, memoryConfig);
  } else {
    out = ::ttnn::permute(in, permutation, memoryConfig, padValue);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
