// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H
#define TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "llvm/ADT/SmallVector.h"

#include <limits>

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

// Per-core L1 usage as reported by a backend constraints query. Mirrors the
// four scalars of tt-metal's graph::ResourceUsage, kept metal-free so the
// underflow repair below can be unit tested without a device.
struct L1PeakUsage {
  size_t cbL1PeakSize = 0;       // CB L1 peak allocation in bytes
  size_t tensorL1PeakSize = 0;   // Tensor L1 peak allocation in bytes
  size_t peakL1MemorySize = 0;   // Peak memory (CB+L1) allocation in bytes
  size_t outputL1BufferSize = 0; // Output L1 buffer allocation in bytes
};

// True when a reported per-core L1 usage has wrapped around zero. Real per-core
// L1 is a few MiB, so the high bit can only be set by an unsigned underflow.
inline bool isUnderflowedL1Usage(size_t value) {
  return value > std::numeric_limits<size_t>::max() / 2;
}

// Repairs the L1 peaks of a *stateful* (build-from-records) query that
// tt-metal's graph-trace walker underflowed. Peaks that did not wrap are
// returned untouched.
//
// tt-metal derives the per-core peaks by walking the captured graph trace with
// unsigned running counters, subtracting on every buffer-deallocate node
// without a guard (`extract_resource_usage_per_core` in
// ttnn/core/graph/graph_trace_utils.cpp):
//
//     } else {  // kNodeBufferDeallocate
//       current_l1 -= alloc_size;
//       current_total -= alloc_size;
//     }
//
// On a stateless query the counters always balance: every buffer freed inside
// the trace was also allocated inside it. A stateful query pre-seeds the
// allocator with the caller's live L1 records, so the trace legitimately frees
// buffers it never allocated -- the counters wrap below zero and the next
// allocate latches the wrapped value through `peak = std::max(peak, current)`.
// The peak then comes back as 2^64 - N, with N a live per-core record size.
// Compared against the L1 limit as-is, such a peak reports an out-of-memory for
// any op holding live L1 inputs, at ~1% real occupancy, which the L1 spill pass
// then "recovers" from by spilling an activation to DRAM
// (https://github.com/tenstorrent/tt-mlir/issues/9069).
//
// The counters are deltas above the pre-seeded baseline, so a negative
// excursion means the op's trace-local footprint never rose above the buffers
// that were already live: zero, not 2^64, is the intended reading. The pre-wrap
// peak is not recoverable from the response, so substitute the tightest bound
// that cannot itself have wrapped:
//   - tensor peak  -> the output L1 buffer size, which is measured directly
//                     from the output tensors and never accumulated;
//   - overall peak -> cbL1PeakSize + tensor peak, the usual upper bound on
//                     max(current_cb + current_l1). cbL1PeakSize cannot wrap:
//                     its only subtraction (`current_total -= current_cb` on a
//                     CB-deallocate-all node) is exact by construction.
// No capacity check is lost by this: on the stateful path the pre-seeded
// allocator either places the op's own allocations or throws, and that
// exception is classified as an OOM by OpConstraintValidation -- it, not the
// byte comparison, is the authoritative verdict on this path.
inline L1PeakUsage repairUnderflowedL1Peaks(L1PeakUsage usage) {
  if (isUnderflowedL1Usage(usage.tensorL1PeakSize)) {
    usage.tensorL1PeakSize = usage.outputL1BufferSize;
  }
  if (isUnderflowedL1Usage(usage.peakL1MemorySize)) {
    usage.peakL1MemorySize = usage.cbL1PeakSize + usage.tensorL1PeakSize;
  }
  return usage;
}

} // namespace mlir::tt::ttnn::op_model

#endif // TTMLIR_OPMODEL_TTNN_TTNNOPCONSTRAINTS_H
