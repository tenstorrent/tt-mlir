// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_D2M_UTILS_UTILS_H
#define TTMLIR_DIALECT_D2M_UTILS_UTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"

#include <optional>
#include <utility>
#include <variant>

namespace mlir::tt::ttcore {
class DeviceAttr;
} // namespace mlir::tt::ttcore

namespace mlir::linalg {
class GenericOp;
} // namespace mlir::linalg

namespace mlir::tt::d2m {
class GenericOp;
} // namespace mlir::tt::d2m

namespace mlir::tt::d2m::utils {

// Discardable attribute names for propagating virtualGridMapping (inverse) and
// virtualGridForwardMapping (forward) through ops we don't own (e.g.
// memref.alloc).  Uses the dialect prefix so MLIR can verify they belong to
// D2M.
constexpr llvm::StringLiteral kVirtualGridInverseMappingAttr =
    "d2m.virtualGridInverseMapping";
constexpr llvm::StringLiteral kVirtualGridForwardMappingAttr =
    "d2m.virtualGridForwardMapping";
constexpr llvm::StringLiteral kReductionScalerAttr = "d2m.reduction_scaler";

inline bool isReductionScalerBuffer(Operation *op) {
  return op && op->hasAttr(kReductionScalerAttr);
}

// Return a new shaped type by reblocking its device shape to match a new grid
// shape.
ShapedType reblockShapedType(ShapedType oldType,
                             ArrayRef<int64_t> newGridShape);

// Rebuild `layout`'s logical shape from a [grid..., shard...] device shape, so
// that a re-split device shape keeps the same total extent (which the composite
// view and reblocking checks compare against). Every other layout field is
// carried over unchanged. `deviceShape` must have even rank.
ttcore::MetalLayoutAttr
rebuildLayoutForDeviceShape(ttcore::MetalLayoutAttr layout,
                            ArrayRef<int64_t> deviceShape);

// Build a sharded L1 MetalLayoutAttr for `logicalShape` where dim i is padded
// out to `tilesPerDim[i]` tiles. Uses default collapsed intervals.
ttcore::MetalLayoutAttr buildShardedTileLayout(MLIRContext *ctx,
                                               ArrayRef<int64_t> logicalShape,
                                               ArrayRef<int64_t> tilesPerDim,
                                               ttcore::MemorySpace memorySpace);

// True when `deviceShape` allocates more elements than `logicalShape` occupies
// along any dim, i.e. the shard carries a padding tail that must be masked.
bool deviceShapeNeedsPadding(ArrayRef<int64_t> deviceShape,
                             ArrayRef<int64_t> logicalShape);

// Clone a local shard type using the shard shape implied by a reference
// operand's device layout.
Type cloneWithShardShape(Value referenceOperand, Type typeToRetype);

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

// Derive generic block factors from operand grid shapes and indexing maps,
// mirroring GenericOp::build's reverse-flattened affine composition.
SmallVector<int64_t> deriveBlockFactorsFromOperandGrids(
    mlir::ArrayRef<mlir::AffineMap> indexingMaps,
    mlir::ArrayRef<mlir::SmallVector<int64_t>> operandGridShapes,
    mlir::ArrayRef<int64_t> outputGridShape);

// Build grid dimension indices from an indexing map. For each result in the
// indexing map, translates arbitrary affine expressions into arith dialect
// operations to compute the index values. This supports all valid affine
// expressions including binary operations (add, mul, floordiv, ceildiv, mod).
SmallVector<Value> buildGridIndices(OpBuilder &builder, Location loc,
                                    AffineMap indexingMap);

// Build this core's own coordinate as one `d2m.core_index` per grid dim.
// Unlike buildGridIndices, which projects loop indices through an indexing map,
// these address the core itself -- what a remote_load/remote_store wants when
// the access is local rather than a gather.
SmallVector<Value> buildCoreIndices(OpBuilder &builder, Location loc,
                                    std::size_t gridRank);

// Opt `generic` out of reblocking by clearing the attrs that pass keys off of.
// Needed for hand-built datamovement regions, whose constant operand maps
// reblocking would otherwise read as broadcasts and rebuild, discarding the
// hand-built region. Call after the region is fully populated.
void makeExplicitDatamovementForm(OpBuilder &builder, GenericOp generic);

// Emit a `linalg.generic` copying `input` into `output` tile-by-tile via
// TileTypecastOp. With `shardRedDim` set, a non-invertible `dim mod extent` map
// on that dim takes the loop bound from the narrow output while reading the
// wide input, copying only the leading tiles; without it the map is an identity
// over the whole shard.
linalg::GenericOp emitLeadingTileCopy(OpBuilder &builder, Location loc,
                                      Value input, Value output,
                                      std::optional<std::size_t> shardRedDim);

// Gets the underlying physical grid shape corresponding to the tensor or
// memref. For views/streams, this 'physical' grid corresponds to the compute
// grid shape used if the tensor/memref was the output of a GenericOp.
SmallVector<int64_t> getPhysicalGridShape(Value tensorOrMemref);

// N-dimensional axis-aligned bounding box (start and end inclusive).
// start.size() must equal end.size() (dimension).
struct BoundingBox {
  llvm::SmallVector<int64_t> start;
  llvm::SmallVector<int64_t> end;
};

// Maps `source` into the affine map's output space using its start/end corners.
BoundingBox getProjectedBoundingBox(const BoundingBox &source,
                                    mlir::AffineMap map);

// Returns the remapping associated with a value, if any.
// Traces back through the defining op to find a ViewLayoutOp and returns its
// remapping attribute. Returns std::nullopt if the value has no associated
// remapping.  After the virtual-grid refactor, these remappings are always
// reblockings — virtual grid info lives on EmptyOp attrs instead.
// Note: this is not recursive, it only checks immediate defining op.
std::optional<AffineMap> getAssociatedRemapping(Value val);

// Returns the virtualGridMapping (inverse map, physical→virtual) associated
// with a value, if any.  Traces through the def-use chain (ToLayoutOp →
// EmptyOp, etc.) to find the underlying EmptyOp/AllocOp/CreateBufferOp and
// returns its virtualGridMapping attribute.
std::optional<AffineMap> getVirtualGridInverseMapping(Value val);

// Returns the virtualGridForwardMapping (forward map, virtual→physical)
// associated with a value, if any.  Traces the same def-use chain as
// getVirtualGridInverseMapping but returns the forward map attribute.
std::optional<AffineMap> getVirtualGridForwardMapping(Value val);

// Derive GridAttr-compatible virtual grid maps for `gridShape` from the full
// tensor/memref virtual grid mapping carried by `val`. Returns std::nullopt
// when the stored mapping belongs to a different-rank view of the value.
std::optional<std::pair<AffineMap, AffineMap>>
getGridMapsFromVirtualGridMapping(Value val, ArrayRef<int64_t> gridShape);

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

// User-facing get memory map util function.
AffineMap getMemoryMap(ttcore::DeviceAttr device, Value input, bool isRemote);

template <typename Builder>
SmallVector<Value> applyMap(Builder &builder, Location loc, AffineMap map,
                            ValueRange index, bool isRemote);

std::tuple<SmallVector<Value>, SmallVector<Value>, SmallVector<Value>>
getLoopBounds(OpBuilder &builder, Location loc, ArrayRef<int64_t> shardShape);

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

AffineMap canonicalStridedMap(MLIRContext *context, ArrayRef<int64_t> shape,
                              Type elementType, AffineMap map);

// Return the NoC address alignment (also the minimum NoC transfer size) for a
// memory space.
int32_t getNocAddressAlignmentBytes(Operation *op,
                                    ttcore::MemorySpace memorySpace);

// Return the NoC alignment (also the minimum NoC transfer size) measured in
// number of tensor/memref elements.
int32_t
getNocElementAlignment(Operation *op, ttcore::MemorySpace memorySpace,
                       const std::variant<RankedTensorType, MemRefType> &type);

// Return the L1 NoC alignment (also the minimum L1 NoC transfer size) measured
// in number of tensor/memref elements.
int32_t getNocElementAlignmentL1(
    Operation *op, const std::variant<RankedTensorType, MemRefType> &type);

} // namespace mlir::tt::d2m::utils

#endif
