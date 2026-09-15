// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "operations/layout/to_layout.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/ttnn/ttnn.h"

#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"

#include "ttnn/operations/experimental/quasar/to_layout/to_layout_op.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace tt::runtime::ttnn::operations::layout {
void run(const ::tt::target::ttnn::ToLayoutOp *op, ProgramContext &context) {
  ProgramTensorPool &tensorPool = context.getTensorPool();

  const ::ttnn::Tensor &inputTensor =
      tensorPool.getTTNNTensorAndValidate(op->in());
  const ::tt::target::Dim2d *targetTileShape =
      op->out()->desc()->layout()->memory_desc()->tile_shape();
  LOG_ASSERT(::tt::runtime::ttnn::utils::isValidTileShape(targetTileShape),
             "Invalid tile shape");

  ::ttnn::Layout layout =
      ::tt::runtime::ttnn::utils::toTTNNLayout(op->layout());
  std::optional<::ttnn::MemoryConfig> memoryConfig =
      ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(op->memcfg());
  std::optional<::ttnn::DataType> dtype = std::nullopt;

  if (op->dtype()) {
    dtype = ::tt::runtime::ttnn::utils::toTTNNDataType(*(op->dtype()));
  }

  // On Quasar, ttnn::to_layout reaches ttnn::tilize, whose program factory
  // builds a DataMovementKernel and TT_FATALs. See utils::isQuasar(). The
  // Quasar to_layout takes the same arguments.
  // Quasar: tilize through host logical data when the trailing dims are not
  // tile aligned, because the device conversion corrupts such tensors. The test
  // and the rebuild are shared with the conv tap path, which tilizes each tap
  // itself rather than through this op -- see operations/utils.h.
  const bool hostRoute =
      utils::quasarTilizeNeedsHostRoute(inputTensor, layout, dtype);

  // On the host route, do not run the device conversion at all.
  //
  // Its data is discarded, and issuing the host-to-device write behind a
  // just-launched program is what hung the ZeBu emulator. Traced to the call:
  //
  //   [optrace] 1 ToLayoutOp  %1 = "ttnn.to_layout"(%arg0) : tensor<1x256x2x2xbf16 ...
  //   [tilize] to_vector done n=1024
  //   [tilize] from_vector begin          <- never returns
  //
  // Building the tensor straight from logical data is the same route the 1x1
  // convolution path already takes, which is measured correct on the emulator.
  // The spec is assembled rather than taken from a device result: same shape,
  // the requested layout and dtype, and the memory config the graph declares for
  // this op's output.
  //
  // KNOWN LIMITATION. This always builds a DEVICE tensor. When op->out() is
  // declared in #system_memory the graph expects a HOST tensor and follows this
  // op with an explicit ttnn.to_device -- which is then handed a tensor that is
  // already resident and blocks forever on the emulator. Measured 2026-09-15:
  // three full-ResNet-50 runs on three different builds all stalled at op 16,
  // `to_device` on a [1,1,1,1024] const, with the ZeBu backend still healthy.
  // Single-op probes never hit it because their graphs keep every intermediate
  // on device, so none of them declares a system-memory to_layout output.
  //
  // Guarding on inSystemMemory(op->out()) and passing nullptr is NOT sufficient
  // on its own -- tried, and it broke craq-sim, because the assembled spec still
  // carried a device memory config. A correct fix has to make the spec's memory
  // space agree with the placement.
  ::ttnn::Tensor out;
  if (hostRoute) {
    const std::vector<bfloat16> hostData = inputTensor.to_vector<bfloat16>();
    const ::ttnn::MemoryConfig outMemoryConfig =
        memoryConfig.value_or(
            ::tt::runtime::ttnn::utils::createMemoryConfigIfNeeded(
                ::tt::runtime::ttnn::utils::getTensorRefMemoryConfig(op->out()))
                .value_or(inputTensor.memory_config()));
    const ::tt::tt_metal::TensorSpec spec(
        inputTensor.logical_shape(),
        ::tt::tt_metal::TensorLayout(dtype.value_or(inputTensor.dtype()),
                                     layout, outMemoryConfig));
    // Honour the memory space the graph declares. A ToLayoutOp whose output is
    // in system memory must hand back a HOST tensor -- the compiler follows it
    // with an explicit ttnn.to_device. Putting it on the device here instead made
    // that to_device receive a tensor that is already resident, and it blocked
    // forever on the emulator: all three full-model runs stalled at op 16,
    // `to_device` on a [1,1,1,1024] const, with the backend still healthy.
    out = ::ttnn::Tensor::from_vector(std::move(hostData), spec,
                                      inputTensor.device());
  } else {
    out = utils::isQuasar()
              ? ::ttnn::operations::experimental::quasar::to_layout(
                    inputTensor, layout, dtype, memoryConfig)
              : ::ttnn::to_layout(inputTensor, layout, dtype, memoryConfig);
  }

  // A layout change is not arithmetic: to_vector returns logical row-major order
  // whatever the physical layout, so every element must survive it exactly. The
  // host route only covers ROW_MAJOR->TILE; the untilize direction that the conv
  // graphs insert just before conv2d has never been checked.
  if (std::getenv("TTMLIR_LAYOUT_CHECK") &&
      inputTensor.dtype() == ::ttnn::DataType::BFLOAT16 &&
      out.dtype() == ::ttnn::DataType::BFLOAT16) {
    const std::vector<bfloat16> a = inputTensor.to_vector<bfloat16>();
    const std::vector<bfloat16> b = out.to_vector<bfloat16>();
    double maxAbs = 0.0;
    size_t bad = 0;
    ssize_t firstBad = -1;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
      const double d =
          std::abs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
      if (d > maxAbs) { maxAbs = d; }
      if (d != 0.0) { bad++; if (firstBad < 0) { firstBad = (ssize_t)i; } }
    }
    const ::ttnn::Shape &ls = inputTensor.logical_shape();
    std::fprintf(stderr,
                 "[layoutcheck] %dx%d in_layout=%d -> %d n=%zu bad=%zu "
                 "max_abs=%.6f first_bad=%zd host_route=%d\n",
                 ls.rank() >= 2 ? (int)ls[ls.rank() - 2] : 0,
                 ls.rank() >= 1 ? (int)ls[ls.rank() - 1] : 0,
                 (int)inputTensor.layout(), (int)layout, n, bad, maxAbs,
                 firstBad, (int)hostRoute);
    std::fflush(stderr);
  }

  tensorPool.insertTTNNTensorAndValidate(op->out(), out);
}

} // namespace tt::runtime::ttnn::operations::layout
