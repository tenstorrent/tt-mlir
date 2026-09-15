// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/permute.h"

#include "tt/runtime/detail/common/logger.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include <vector>

#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"
#include "ttnn/operations/experimental/quasar/transpose/transpose.hpp"

#include <cstdlib>

#include <cmath>

#include <cstdio>

namespace tt::runtime::ttnn::operations::data_movement {

namespace {

// Realise an arbitrary permutation on Quasar as a sequence of ADJACENT 2-axis
// swaps.
//
// Quasar exposes `transpose` (a single pair swap) but no general `permute`, and
// mainline ttnn::permute is not an option: its PermuteDeviceOperation builds a
// Gen1 compute config, which MakeGen2ComputeConfig rejects on Quasar
// ("generation mismatch", program_spec.cpp). Forge's NCHW<->NHWC permutes are
// 3-cycles ([0,2,3,1] and [0,3,1,2]), so one swap is never enough.
//
// Only adjacent swaps may be emitted. Quasar's transpose maps an axis pair onto
// a TransposeOpDim, and only three of the six have a Quasar program factory:
// WH (2,3), HC (1,2) and CN (0,1) -- see transpose_{wh,hc,cn}_program_factory in
// ttnn/operations/experimental/quasar/transpose/device. The other three, NH
// (0,2), NW (0,3) and CW (1,3), fall through to mainline ttnn::permute
// (quasar/transpose/transpose.cpp:128-136), which is the op this function
// exists to avoid. It does not always fail loudly there: measured on
// [1,16,8,64] -> [1,64,16,8] it returned silently wrong data (max abs error
// 1.359375 against a CPU reference) and hung when run on its own.
//
// Adjacent transpositions generate the symmetric group, so a decomposition into
// neighbour swaps always exists. Insertion sort over the axes: for each output
// position, walk the required input axis down into place one neighbour at a
// time. `current[i]` tracks which original axis currently sits at position i.
// At most rank*(rank-1)/2 transposes, none for the identity, and for the
// NCHW->NHWC direction ([0,2,3,1]) it emits the same (1,2),(2,3) pair a
// non-adjacent decomposition would.
//
// memoryConfig is applied only to whichever op is genuinely last: forcing it on
// the intermediates would pay for a layout change that the next swap discards.
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
    // Walk axis target[i] down to position i one neighbour at a time, rather
    // than swapping i and j directly: a direct (i, j) swap can name a
    // non-adjacent pair, and those have no Quasar program factory.
    for (int64_t k = j; k > i; --k) {
      std::swap(current[k - 1], current[k]);
      swaps.emplace_back(k - 1, k);
    }
  }

  if (swaps.empty()) {
    // Identity. Still honour the requested memory config if there is one.
    if (memoryConfig.has_value()) {
      return ::ttnn::operations::experimental::quasar::transpose(
          in, 0, 0, memoryConfig, padValue);
    }
    return in;
  }

  // A swap of the last two axes maps to TransposeOpDim::WH, and on a ROW_MAJOR
  // interleaved tensor Quasar's transpose short-circuits that to
  // ttnn::prim::permute (quasar/transpose/transpose.cpp:142-145) -- mainline
  // again, and silently wrong here: measured 1.359375 max abs error on
  // [1,16,8,64] -> [1,16,64,8]. The same swap in TILE goes to
  // transpose_wh_program_factory and is exact. So tilize for the duration when
  // the sequence touches the last axis, and restore ROW_MAJOR afterwards.
  // Swaps that never touch the last axis (HC, CN) are correct as-is in ROW_MAJOR
  // and skip the round trip.
  bool touchesLastAxis = false;
  for (const auto &swap : swaps) {
    if (swap.second == rank - 1) {
      touchesLastAxis = true;
      break;
    }
  }
  const ::ttnn::Layout inputLayout = in.layout();
  const bool viaTile = touchesLastAxis &&
                       inputLayout == ::ttnn::Layout::ROW_MAJOR;

  ::ttnn::Tensor out = in;
  if (viaTile) {
    out = ::ttnn::operations::experimental::quasar::to_layout(
        out, ::ttnn::Layout::TILE, std::nullopt, std::nullopt);
  }
  for (size_t s = 0; s < swaps.size(); ++s) {
    // The memory config goes on whichever op is genuinely last: the closing
    // to_layout when there is one, otherwise the final transpose.
    const bool isLast = (s + 1 == swaps.size()) && !viaTile;
    out = ::ttnn::operations::experimental::quasar::transpose(
        out, swaps[s].first, swaps[s].second,
        isLast ? memoryConfig : std::nullopt, padValue);
  }
  if (viaTile) {
    out = ::ttnn::operations::experimental::quasar::to_layout(
        out, inputLayout, std::nullopt, memoryConfig);
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

  // A permute has an exact CPU reference: to_vector returns logical row-major
  // order whatever the physical layout, so out[i] must equal in[perm-mapped i]
  // bit for bit. No tolerance, no reference implementation to get wrong.
  if (std::getenv("TTMLIR_PERMUTE_CHECK") &&
      in.dtype() == ::ttnn::DataType::BFLOAT16 &&
      out.dtype() == ::ttnn::DataType::BFLOAT16) {
    const std::vector<bfloat16> a = in.to_vector<bfloat16>();
    const std::vector<bfloat16> b = out.to_vector<bfloat16>();
    const ::ttnn::Shape &is = in.logical_shape();
    const size_t rank = is.rank();
    std::vector<size_t> inStride(rank, 1);
    for (size_t i = rank - 1; i-- > 0;) {
      inStride[i] = inStride[i + 1] * is[i + 1];
    }
    // Output dim i is input dim permutation[i].
    std::vector<size_t> outShape(rank), outStride(rank, 1);
    for (size_t i = 0; i < rank; i++) {
      outShape[i] = is[permutation[i]];
    }
    for (size_t i = rank - 1; i-- > 0;) {
      outStride[i] = outStride[i + 1] * outShape[i + 1];
    }
    double maxAbs = 0.0;
    ssize_t firstBad = -1;
    size_t bad = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t idx = 0; idx < n; idx++) {
      size_t rem = idx, srcIdx = 0;
      for (size_t d = 0; d < rank; d++) {
        const size_t coord = rem / outStride[d];
        rem %= outStride[d];
        srcIdx += coord * inStride[permutation[d]];
      }
      if (srcIdx >= a.size()) { continue; }
      const double diff = std::abs(static_cast<float>(a[srcIdx]) -
                                   static_cast<float>(b[idx]));
      if (diff > maxAbs) { maxAbs = diff; }
      if (diff != 0.0) {
        bad++;
        if (firstBad < 0) { firstBad = static_cast<ssize_t>(idx); }
      }
    }
    std::fprintf(stderr,
                 "[permutecheck] n=%zu bad=%zu max_abs=%.6f first_bad=%zd\n",
                 n, bad, maxAbs, firstBad);
    std::fflush(stderr);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
