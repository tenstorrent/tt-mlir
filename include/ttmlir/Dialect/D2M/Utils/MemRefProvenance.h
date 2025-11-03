//===- MemRefProvenance.h - MemRef Source Tracking --------------*- C++-*-===//
//
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// Utility for tracking the provenance of memref values in D2M dialect.
// Distinguishes between circular buffers, temporary allocations, and DST register.
//
//===----------------------------------------------------------------------===//

#ifndef TTMLIR_DIALECT_D2M_UTILS_MEMREFPROVENANCE_H
#define TTMLIR_DIALECT_D2M_UTILS_MEMREFPROVENANCE_H

#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include <optional>

namespace mlir::tt::d2m::utils {

/// Enum representing the source/type of a memref value
enum class MemRefSource {
  CircularBuffer,   // BlockArgument from d2m.generic region params
  TempAllocation,   // memref.alloc from bufferized d2m.empty
  DstRegister,      // d2m.acquire_dst result
  StreamLayout,     // d2m.stream_layout result (wraps CB)
  Unknown           // Unable to determine
};

/// Information about a memref's provenance
struct ProvenanceInfo {
  MemRefSource source;
  Value rootValue;           // Original allocation/block arg
  Operation *allocSite;      // Where created (if applicable), nullptr otherwise

  ProvenanceInfo(MemRefSource src, Value root, Operation *alloc = nullptr)
      : source(src), rootValue(root), allocSite(alloc) {}
};

/// Trace a memref value through views to determine its source
///
/// Unwraps subview, collapse_shape, cast, wait, reserve operations
/// to find the original allocation or block argument.
///
/// \param memref The memref value to trace
/// \return Provenance information about the memref's source
ProvenanceInfo traceMemRefProvenance(Value memref);

/// Check if a memref is a circular buffer (from block argument)
bool isCircularBuffer(Value memref);

/// Check if a memref is a temporary allocation
bool isTempAllocation(Value memref);

/// Check if a memref is from DST register
bool isDstRegister(Value memref);

/// Try to get the circular buffer block argument from a memref
/// Returns nullopt if the memref is not CB-backed
std::optional<BlockArgument> tryGetCircularBufferArg(Value memref);

} // namespace mlir::tt::d2m::utils

#endif // TTMLIR_DIALECT_D2M_UTILS_MEMREFPROVENANCE_H
