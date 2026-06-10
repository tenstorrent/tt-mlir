// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>

// Forward declaration only. The full definition lives in tt-metalium
// (tt-metalium/experimental/mock_device/mock_allocator.hpp) and is only
// available when TTMLIR_ENABLE_OPMODEL is defined. Keeping this header free of
// any tt-metalium include lets it compile in builds where the op model is
// disabled and MockAllocatorState has no definition; the shared_ptr member
// below only ever needs the incomplete type.
namespace tt::tt_metal::experimental {
class MockAllocatorState;
} // namespace tt::tt_metal::experimental

namespace mlir::tt::ttnn::op_model {

using MockAllocatorState = ::tt::tt_metal::experimental::MockAllocatorState;

/*
 * OpConstraints struct is used to store the constraints of an operation.
 * It is returned by the getOpConstraints method of the OpModel interface.
 * Note: The reason for separating the definition of this struct from
 * TTNNOpModel.h is to avoid coupling TTNNOpModelInterface to all the code in
 * TTNNOpModel.h.
 */

struct OpConstraints {
  size_t cbL1PeakSize;       // CB L1 peak allocation in bytes
  size_t tensorL1PeakSize;   // Tensor L1 peak allocation in bytes
  size_t peakL1MemorySize;   // Peak memory (CB+L1) allocation in bytes
  size_t outputL1BufferSize; // Output L1 buffer allocation in bytes
  llvm::SmallVector<tt::ttnn::TTNNLayoutAttr>
      outputLayouts; // Layouts of all output tensors (one layout per output
                     // tensor)
  // Allocator state produced by a stateful op-constraints query. Null when the
  // result came from the stateless query path (no initial state was threaded
  // in). Held by shared_ptr so this header stays decoupled from the
  // tt-metalium definition of MockAllocatorState (only an incomplete type is
  // needed here).
  std::shared_ptr<MockAllocatorState> newState;
  // ---------------------------------------------------------------------------
  // Parameterized constructor, should be used in most cases
  OpConstraints(size_t cbPeak, size_t tensorPeak, size_t peakMemory,
                size_t outputBuffer,
                llvm::SmallVector<tt::ttnn::TTNNLayoutAttr> layouts,
                std::shared_ptr<MockAllocatorState> newStateIn = nullptr)
      : cbL1PeakSize(cbPeak), tensorL1PeakSize(tensorPeak),
        peakL1MemorySize(peakMemory), outputL1BufferSize(outputBuffer),
        outputLayouts(std::move(layouts)), newState(std::move(newStateIn)) {}
  // ---------------------------------------------------------------------------
  // Default constructor, should be used only when the default value is intended
  // to be used, eg. when TTMLIR_ENABLE_OPMODEL is not defined.
  OpConstraints()
      : cbL1PeakSize(0), tensorL1PeakSize(0), peakL1MemorySize(0),
        outputL1BufferSize(0), outputLayouts({}), newState(nullptr) {}
};

} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H
