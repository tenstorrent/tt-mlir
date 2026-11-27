// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/reshape.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"

#include "tt/runtime/detail/ttnn/utils.h"
#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"

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
  // Quasar reimplements the op stack: the mainline reshape program factory builds a
  // DataMovementKernel, whose constructor TT_FATALs on Quasar. The Quasar reshape
  // takes the same leading arguments via its ttsl::Span<const int32_t> overload.
  //
  // But it is NOT valid for every reshape. A TILE-layout tensor stores its trailing
  // two dimensions in 32x32 tiles, so relabelling those dims only preserves logical
  // order when the change cannot make an element cross a tile boundary. Measured on
  // the emulator: [1,1,2048,1] -> [1,2048] returned pcc=0.004 -- executed happily and
  // produced garbage. A silently wrong answer is far worse than a refusal, so the
  // Quasar path is taken only when it is provably safe, and anything else asserts
  // with the shapes named rather than corrupting the tensor.
  ::ttnn::Tensor out;
  if (utils::isQuasar()) {
    const ::ttnn::Shape &inShape = in.logical_shape();
    const size_t inRank = inShape.rank();
    const size_t outRank = shape.size();
    auto tileAligned = [](int64_t v) { return v > 0 && (v % 32) == 0; };

    // A TILE tensor stores its trailing two dims in 32x32 tiles, so relabelling
    // those dims only preserves logical order when no element can cross a tile
    // boundary. Measured: [1,1,2048,1] -> [1,2048] in TILE returned pcc=0.004.
    bool safeInPlace = in.layout() == ::ttnn::Layout::ROW_MAJOR;
    if (!safeInPlace && inRank >= 2 && outRank >= 2) {
      const int64_t oldW = static_cast<int64_t>(inShape[inRank - 1]);
      const int64_t oldH = static_cast<int64_t>(inShape[inRank - 2]);
      const int64_t newW = static_cast<int64_t>(shape[outRank - 1]);
      const int64_t newH = static_cast<int64_t>(shape[outRank - 2]);
      safeInPlace = (oldW == newW && oldH == newH) ||
                    (tileAligned(oldW) && tileAligned(newW) &&
                     tileAligned(oldH) && tileAligned(newH));
    }

    if (safeInPlace) {
      out = ::ttnn::operations::experimental::quasar::reshape(in, shape,
                                                             memoryConfig);
    } else {
      // Round-trip through ROW_MAJOR, where a reshape is a pure relabelling with
      // no tiles to straddle. This is how tt-metal's own Quasar ResNet-50 uses
      // the op: its per-op test only ever calls quasar::reshape on ROW_MAJOR
      // tensors ("both resnet reshape call-sites operate on ROW_MAJOR tensors --
      // fold output / untilize output"), and that test passes on the emulator
      // where the tiled form returns garbage.
      const ::ttnn::Layout originalLayout = in.layout();
      ::ttnn::Tensor rowMajor =
          ::ttnn::operations::experimental::quasar::to_layout(
              in, ::ttnn::Layout::ROW_MAJOR, std::nullopt, std::nullopt);
      ::ttnn::Tensor reshaped =
          ::ttnn::operations::experimental::quasar::reshape(rowMajor, shape,
                                                            std::nullopt);
      out = ::ttnn::operations::experimental::quasar::to_layout(
          reshaped, originalLayout, std::nullopt, memoryConfig);
    }
  } else {
    out = ::ttnn::reshape(in, shape, memoryConfig);
  }
  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
