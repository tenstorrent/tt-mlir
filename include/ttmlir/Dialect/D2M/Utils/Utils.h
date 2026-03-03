// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_UTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_UTILS_H

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::tt::ttcore {
class DeviceAttr;
} // namespace mlir::tt::ttcore

namespace mlir::tt::d2m::utils {

// Discardable attribute names for propagating virtualGridMapping (inverse) and
// virtualGridForwardMapping (forward) through ops we don't own (e.g.
// memref.alloc).  Uses the dialect prefix so MLIR can verify they belong to
// D2M.
constexpr llvm::StringLiteral kVirtualGridInverseMappingAttr =
    "d2m.virtualGridInverseMapping";
constexpr llvm::StringLiteral kVirtualGridForwardMappingAttr =
    "d2m.virtualGridForwardMapping";

// Return a new RankedTensorType by reblocking its device shape to match a new
// grid shape.
RankedTensorType reblockTensor(RankedTensorType oldTensor,
                               ArrayRef<int64_t> newGridShape);

// Get square target grid shape.
llvm::SmallVector<int64_t>
getSquareTargetGrid(mlir::ArrayRef<int64_t> targetGridShape);

// Get the largest destination element type used in a region.
// Asserts if no DST-using ops are found.
Type getRegionLargestDstElemType(Region &region);

// Get the largest destination element type used in a region.
// Returns nullptr if no DST-using ops are found.
Type getRegionLargestDstElemTypeOrNull(Region &region);

// Computes dim constraints implied by the indexing maps and shapes. If
// successful, returns a vector of dim constraints for each dimension; a '0'
// indicates that the dimension is not constrained. If the shapes imply
// incompatible constraints, returns std::nullopt.
std::optional<SmallVector<int64_t>>
computeDimConstraints(mlir::ArrayRef<mlir::AffineMap> indexingMaps,
                      mlir::ArrayRef<mlir::SmallVector<int64_t>> shapes);

// Build grid dimension indices from an indexing map. For each result in the
// indexing map, translates arbitrary affine expressions into arith dialect
// operations to compute the index values. This supports all valid affine
// expressions including binary operations (add, mul, floordiv, ceildiv, mod).
SmallVector<Value> buildGridIndices(OpBuilder &builder, Location loc,
                                    AffineMap indexingMap);

// Gets the underlying physical grid shape corresponding to the tensor or
// memref. For views/streams, this 'physical' grid corresponds to the compute
// grid shape used if the tensor/memref was the output of a GenericOp.
SmallVector<int64_t> getPhysicalGridShape(Value tensorOrMemref);

// Returns the remapping associated with a value, if any.
// Traces back through the defining op to find a ViewLayoutOp or StreamLayoutOp
// and returns its remapping attribute. Returns std::nullopt if the value has
// no associated remapping.  After the virtual-grid refactor, these remappings
// are always reblockings — virtual grid info lives on EmptyOp attrs instead.
// Note: this is not recursive, it only checks immediate defining op.
std::optional<AffineMap> getAssociatedRemapping(Value val);

// Returns the virtualGridMapping (inverse map, physical→virtual) associated
// with a value, if any.  Traces through the def-use chain (ToLayoutOp →
// EmptyOp, StreamLayoutOp → storage EmptyOp, etc.) to find the underlying
// EmptyOp/AllocOp/CreateBufferOp and returns its virtualGridMapping attribute.
std::optional<AffineMap> getVirtualGridInverseMapping(Value val);

// Returns the virtualGridForwardMapping (forward map, virtual→physical)
// associated with a value, if any.  Traces the same def-use chain as
// getVirtualGridInverseMapping but returns the forward map attribute.
std::optional<AffineMap> getVirtualGridForwardMapping(Value val);

// Returns the effective affine map for a memref-typed value by resolving
// ViewLayoutAttr remappings (via applyViews) and falling back to the layout's
// getAffineMap() or an identity map.
AffineMap resolveEffectiveAffineMap(Value val, MemRefType memrefType);

// Compute the device memory map for a memref type. Returns an AffineMap
// that maps logical indices to physical device addresses (L1 or DRAM),
// handling core virtualization for ND or oversized grids.
AffineMap getMemoryMap(ttcore::DeviceAttr device, MemRefType memrefType,
                       size_t pageSize,
                       std::optional<AffineMap> view = std::nullopt,
                       size_t baseOffset = 0);

// Overload that accepts a Value so it can check whether the value carries a
// virtual grid mapping (via getVirtualGridInverseMapping).
AffineMap getMemoryMap(ttcore::DeviceAttr device, Value memrefValue,
                       size_t pageSize,
                       std::optional<AffineMap> view = std::nullopt,
                       size_t baseOffset = 0);

// Convenience overload accepting a (MemRefType, AffineMap) pair.
AffineMap getMemoryMap(ttcore::DeviceAttr device,
                       std::pair<MemRefType, AffineMap> memrefAndView,
                       size_t pageSize, size_t baseOffset = 0);

// Finds a 2D grid (y, x) such that y * x = gridVolume. The returned grid aims
// to be as square as possible while respecting the provided target grid shape
// bounds. If either MxN or NxM grids are feasible where M > N, MxN is chosen.
// Returns an empty vector if no valid grid is found.
llvm::SmallVector<int64_t>
findLegalPhysicalGridForVolume(int64_t gridVolume,
                               ArrayRef<int64_t> targetGridShape);

// Collapse an ND (or 2D) grid to a physical 2D grid that fits within
// deviceGridShape.  First tries collapseGridTo2D (which preserves the natural
// leading-dim collapse order).  If the result exceeds the device grid bounds,
// falls back to findLegalPhysicalGridForVolume to find a valid factorization.
llvm::SmallVector<int64_t, 2>
collapseToPhysicalGrid2D(ArrayRef<int64_t> gridShape,
                         ArrayRef<int64_t> deviceGridShape);

} // namespace mlir::tt::d2m::utils

#endif
