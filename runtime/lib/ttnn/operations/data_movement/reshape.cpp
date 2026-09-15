// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/data_movement/reshape.h"
#include "tt/runtime/detail/common/logger.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"

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
  // Mainline reshape is only safe on Quasar when it degenerates to a view. A
  // tiled reshape goes through reshape_tiled -> prim::reshape_view, whose
  // program factory builds a DataMovementKernel and TT_FATALs. Measured: the
  // reshape in the global-avg-pool and max-pool graphs takes exactly that path.
  // Quasar: a tiled reshape that has to repack across tile rows does not.
  //
  // A TILE tensor stores rows at prod(leading) * roundup(H, 32) + h, so the
  // implicit padding after a sub-tile H is part of the address. Whenever the
  // reshape changes H, elements have to move between tile rows -- and the Quasar
  // reshape only relabels. Measured: [1,2,2,256] -> [1,1,4,256] (two tile rows
  // merged into one) keeps the first two rows and returns someone else's data
  // for the other two, max_abs 0.978516 from element 512 on. That is the input
  // reshape of every NCHW convolution graph with sub-tile spatial dims, and the
  // error flows through the convolution to the second spatial row of its output.
  //
  // Free cases, kept on the device path: the last dim must be unchanged, and
  // then either H is unchanged (only the leading dims regroup, which does not
  // move a row) or H is tile-aligned on both sides (no implicit padding to
  // straddle).
  const bool tiledRepack = [&]() {
    if (!utils::isQuasar() || in.layout() != ::ttnn::Layout::TILE ||
        !::ttnn::is_device_tensor(in) ||
        in.dtype() != ::ttnn::DataType::BFLOAT16) {
      return false;
    }
    const ::ttnn::Shape &ls = in.logical_shape();
    if (ls.rank() < 2 || shape.size() < 2) {
      return false;
    }
    const uint32_t inH = ls[ls.rank() - 2];
    const uint32_t inW = ls[ls.rank() - 1];
    const uint32_t outH = static_cast<uint32_t>(shape[shape.size() - 2]);
    const uint32_t outW = static_cast<uint32_t>(shape[shape.size() - 1]);
    if (inW != outW) {
      return true;
    }
    if (inH == outH) {
      return false;
    }
    return inH % ::tt::constants::TILE_HEIGHT != 0 ||
           outH % ::tt::constants::TILE_HEIGHT != 0;
  }();

  // As in to_layout: on the repack route the device call's data is discarded, and
  // issuing the host-to-device write behind a just-launched program hangs the
  // emulator. Build the tensor from logical data and skip the device call.
  ::ttnn::Tensor out;
  if (tiledRepack) {
    const std::vector<bfloat16> data = in.to_vector<bfloat16>();
    std::vector<uint32_t> dims;
    dims.reserve(shape.size());
    for (int32_t d : shape) {
      dims.push_back(static_cast<uint32_t>(d));
    }
    const ::ttnn::MemoryConfig outMemoryConfig =
        memoryConfig.value_or(in.memory_config());
    const ::tt::tt_metal::TensorSpec spec(
        ::ttnn::Shape(dims),
        ::tt::tt_metal::TensorLayout(in.dtype(), in.layout(), outMemoryConfig));
    out = ::ttnn::Tensor::from_vector(std::move(data), spec, in.device());
  } else {
    out = utils::isQuasar()
              ? ::ttnn::operations::experimental::quasar::reshape(in, shape,
                                                                 memoryConfig)
              : ::ttnn::reshape(in, shape, memoryConfig);
  }
  // A reshape is a relabelling: to_vector returns logical row-major order
  // whatever the physical layout, so every element must survive in place. On a
  // TILE tensor whose second-last extent is sub-tile, splitting that axis has to
  // repack across tile rows, and a reshape that only relabels leaves everything
  // past the first tile row reading someone else's data.
  if (std::getenv("TTMLIR_RESHAPE_CHECK") &&
      in.dtype() == ::ttnn::DataType::BFLOAT16 &&
      out.dtype() == ::ttnn::DataType::BFLOAT16) {
    const std::vector<bfloat16> a = in.to_vector<bfloat16>();
    const std::vector<bfloat16> b = out.to_vector<bfloat16>();
    double maxAbs = 0.0;
    ssize_t firstBad = -1;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
      const double d =
          std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
      if (d > maxAbs) {
        maxAbs = d;
      }
      if (d != 0.0 && firstBad < 0) {
        firstBad = static_cast<ssize_t>(i);
      }
    }
    std::fprintf(stderr,
                 "[reshapecheck] n_in=%zu n_out=%zu in_layout=%d max_abs=%.6f "
                 "first_bad=%zd\n",
                 a.size(), b.size(), static_cast<int>(in.layout()), maxAbs,
                 firstBad);
    std::fflush(stderr);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}
} // namespace tt::runtime::ttnn::operations::data_movement
