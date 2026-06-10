// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn::op_model {

// tt-mlir-side mirror of tt-metal's experimental::AllocationRecord. Kept as a
// plain POD of dialect/scalar types so this broadly-included header stays free
// of any tt-metalium include (which would pull a global ::tt into every
// translation unit). Conversion to/from the metal type happens behind the
// op-model boundary in TTNNOpModel.cpp.
struct OpModelAllocationRecord {
  tt::ttnn::BufferType bufferType = tt::ttnn::BufferType::L1;
  uint64_t address = 0;
  uint64_t sizePerBank = 0;
};

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
  // Per-output-buffer allocation records produced by a stateful
  // (build-from-records) query. Empty on the stateless path. The L1 spill path
  // keeps these for still-live tensors and rebuilds allocator state from them.
  llvm::SmallVector<OpModelAllocationRecord> outputAllocations;
  // ---------------------------------------------------------------------------
  // Parameterized constructor, should be used in most cases
  OpConstraints(size_t cbPeak, size_t tensorPeak, size_t peakMemory,
                size_t outputBuffer,
                llvm::SmallVector<tt::ttnn::TTNNLayoutAttr> layouts,
                llvm::SmallVector<OpModelAllocationRecord> allocations = {})
      : cbL1PeakSize(cbPeak), tensorL1PeakSize(tensorPeak),
        peakL1MemorySize(peakMemory), outputL1BufferSize(outputBuffer),
        outputLayouts(std::move(layouts)),
        outputAllocations(std::move(allocations)) {}
  // ---------------------------------------------------------------------------
  // Default constructor, should be used only when the default value is intended
  // to be used, eg. when TTMLIR_ENABLE_OPMODEL is not defined.
  OpConstraints()
      : cbL1PeakSize(0), tensorL1PeakSize(0), peakL1MemorySize(0),
        outputL1BufferSize(0), outputLayouts({}), outputAllocations({}) {}
};

} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H
