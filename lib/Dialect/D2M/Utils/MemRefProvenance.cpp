//===- MemRefProvenance.cpp - MemRef Source Tracking ---------------------===//
//
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "ttmlir/Dialect/D2M/Utils/MemRefProvenance.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tt::d2m::utils {

ProvenanceInfo traceMemRefProvenance(Value memref) {
  Value current = memref;

  // Unwrap all view-like operations to find the root
  while (auto defOp = current.getDefiningOp()) {
    if (auto subview = mlir::dyn_cast<memref::SubViewOp>(defOp)) {
      current = subview.getSource();
    } else if (auto collapse = mlir::dyn_cast<memref::CollapseShapeOp>(defOp)) {
      current = collapse.getSrc();
    } else if (auto cast = mlir::dyn_cast<memref::CastOp>(defOp)) {
      current = cast.getSource();
    } else if (auto wait = mlir::dyn_cast<d2m::WaitOp>(defOp)) {
      current = wait.getCb();
    } else if (auto reserve = mlir::dyn_cast<d2m::ReserveOp>(defOp)) {
      current = reserve.getCb();
    } else {
      // Reached an allocation site or other terminal operation
      break;
    }
  }

  // Classify the root value
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(current)) {
    return ProvenanceInfo(MemRefSource::CircularBuffer, current, nullptr);
  }

  if (auto alloc = current.getDefiningOp<memref::AllocOp>()) {
    return ProvenanceInfo(MemRefSource::TempAllocation, current, alloc);
  }

  if (auto acquire = current.getDefiningOp<d2m::AcquireDstOp>()) {
    return ProvenanceInfo(MemRefSource::DstRegister, current, acquire);
  }

  if (auto stream = current.getDefiningOp<d2m::StreamLayoutOp>()) {
    // Stream layouts wrap circular buffers - trace through to storage
    return traceMemRefProvenance(stream.getStorage());
  }

  return ProvenanceInfo(MemRefSource::Unknown, current, current.getDefiningOp());
}

bool isCircularBuffer(Value memref) {
  auto info = traceMemRefProvenance(memref);
  return info.source == MemRefSource::CircularBuffer;
}

bool isTempAllocation(Value memref) {
  auto info = traceMemRefProvenance(memref);
  return info.source == MemRefSource::TempAllocation;
}

bool isDstRegister(Value memref) {
  auto info = traceMemRefProvenance(memref);
  return info.source == MemRefSource::DstRegister;
}

std::optional<BlockArgument> tryGetCircularBufferArg(Value memref) {
  auto info = traceMemRefProvenance(memref);
  if (info.source == MemRefSource::CircularBuffer) {
    return mlir::dyn_cast<BlockArgument>(info.rootValue);
  }
  return std::nullopt;
}

} // namespace mlir::tt::d2m::utils
