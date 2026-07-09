// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToD2M/TTIRToD2M.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/GridSelectionUtils.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace mlir::tt {

namespace {

/// True when the reduction touches a dim before the last two (tile C/R).
/// Those go through the D2M outer-reduction path and must not be decomposed.
template <typename TTIRReductionOp>
bool isOuterReduction(TTIRReductionOp op) {
  auto inputType = mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
  int64_t rank = inputType.getRank();
  std::optional<mlir::ArrayAttr> maybeDimArg = op.getDimArg();
  if (rank < 2 || !maybeDimArg.has_value()) {
    return false;
  }
  for (auto dimAttr : *maybeDimArg) {
    int64_t dim = mlir::cast<mlir::IntegerAttr>(dimAttr).getInt();
    if (dim < 0) {
      dim += rank;
    }
    if (dim < rank - 2) {
      return true;
    }
  }
  return false;
}

class D2MNamedRewriterCommon {
protected:
  using base = D2MNamedRewriterCommon;

  D2MNamedRewriterCommon(ttcore::MemorySpace defaultInputMemSpace,
                         ttcore::MemorySpace defaultOutputMemSpace,
                         bool ttnnMode, bool collapseTensors,
                         bool enableMulticastInference)
      : memorySpaces{defaultInputMemSpace, defaultOutputMemSpace},
        ttnnMode(ttnnMode), collapseTensors(collapseTensors),
        enableMulticastInference(enableMulticastInference) {}

  /// Attributes required to lower any `TTIR_ReductionOp` to D2M (used by both
  /// `D2MNamedAccumReductionRewriter` and
  /// `D2MNamedTileReduceRewriter`).
  /// Call once at the start of each match; downstream code assumes `dim_arg` is
  /// present and `keep_dim` is true. Valid indices inside `dim_arg` are
  /// enforced by the TTIR op verifier.
  template <typename TTIRReductionOp>
  static void checkTTIRReductionPreconditions(TTIRReductionOp op) {
    assert(op.getDimArg() &&
           "TTIR reduction op must have dim_arg for D2M lowering");
    assert(op.getKeepDimAttr().getValue() &&
           "TTIR reduction lowering expects keep_dim=true");
  }

  /// Normalize a possibly negative `dim_arg` entry into `[0, rank)`. `rank`
  /// must be non-zero (callers assert physical iterator rank >= 2 before
  /// lowering).
  static std::size_t normalizeReductionDimIndex(int64_t dim, std::size_t rank) {
    int64_t r = static_cast<int64_t>(rank);
    int64_t n = dim % r;
    if (n < 0) {
      n += r;
    }
    return static_cast<std::size_t>(n);
  }

  static void assertPhysicalIteratorRankForReduction(std::size_t physicalRank) {
    assert(physicalRank >= 2 &&
           "D2M reduction lowering expects at least two physical iterator "
           "dimensions (tile C and R) after layout");
  }

  static bool isTTNNTensor(Type type) {
    auto tensor = mlir::dyn_cast<RankedTensorType>(type);
    if (!tensor) {
      return false;
    }

    return mlir::isa_and_nonnull<ttnn::TTNNLayoutAttr>(tensor.getEncoding()) ||
           mlir::isa_and_nonnull<ttnn::TTNNNDLayoutAttr>(tensor.getEncoding());
  }

  template <typename LayoutAttr>
  void assertTTNNLayoutSupported(LayoutAttr ttnnLayout) const {
    assert(ttnnLayout.isDeviceBufferType() && "Must be a device tensor");

    // With these assumptions we can use the default alignment and dim
    // collapsing behavior in the MetalLayoutAttr.
    assert(ttnnLayout.isTiled() &&
           "Row major TTNN layouts are not supported yet");
    assert(
        mlir::cast<ttcore::TileType>(ttnnLayout.getElementType()).getHeight() ==
            ttcore::TileType::getDefaultShape()[0] &&
        "Only default tile shape is supported");
    assert(
        mlir::cast<ttcore::TileType>(ttnnLayout.getElementType()).getWidth() ==
            ttcore::TileType::getDefaultShape()[1] &&
        "Only default tile shape is supported");
  }

  llvm::SmallVector<int64_t>
  getImpliedNDGrid(ArrayRef<int64_t> tensorShape,
                   ttnn::TTNNNDLayoutAttr ttnnLayout) const {
    llvm::ArrayRef<int64_t> shardShape = ttnnLayout.getMemref().getShape();
    assert(shardShape.size() == tensorShape.size() &&
           "shard shape and tensor shape must have same rank");

    llvm::SmallVector<int64_t> impliedGrid;
    for (size_t i = 0; i < tensorShape.size(); ++i) {
      assert(shardShape[i] != 0 && "shard shape entry must not be zero");
      assert(tensorShape[i] % shardShape[i] == 0 &&
             "tensor dims must be divisible by shard dims for virtual grid");
      impliedGrid.push_back(tensorShape[i] / shardShape[i]);
    }

    // Divide out the tile shape for the last two dimensions.
    if (impliedGrid.size() == 1) {
      impliedGrid[0] /= ttcore::TileType::getDefaultShape()[1];
    } else {
      impliedGrid[impliedGrid.size() - 1] /=
          ttcore::TileType::getDefaultShape()[0];
      impliedGrid[impliedGrid.size() - 2] /=
          ttcore::TileType::getDefaultShape()[1];
    }

    return impliedGrid;
  }

  llvm::SmallVector<int64_t>
  getLegacyGrid(ttnn::TTNNLayoutAttr ttnnLayout) const {
    llvm::SmallVector<int64_t> ttnnGridShape(ttnnLayout.getGridShape());

    bool legacyWithVirtualGrid = ttnnLayout.getMemLayout().getValue() ==
                                     ttnn::TensorMemoryLayout::HeightSharded ||
                                 ttnnLayout.getMemLayout().getValue() ==
                                     ttnn::TensorMemoryLayout::WidthSharded;
    if (!legacyWithVirtualGrid) {
      return ttnnGridShape;
    }

    if (ttnnLayout.getMemLayout().getValue() ==
        ttnn::TensorMemoryLayout::HeightSharded) {
      return {ttnnGridShape[0] * ttnnGridShape[1], 1};
    }
    return {1, ttnnGridShape[0] * ttnnGridShape[1]};
  }

  llvm::SmallVector<int64_t>
  getGridForTTNNTensor(RankedTensorType tensorType) const {
    if (auto ttnnLayout =
            mlir::dyn_cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding())) {
      return getLegacyGrid(ttnnLayout);
    }

    if (auto ndLayout =
            mlir::dyn_cast<ttnn::TTNNNDLayoutAttr>(tensorType.getEncoding())) {
      return getImpliedNDGrid(tensorType.getShape(), ndLayout);
    }

    llvm_unreachable("Unsupported layout for TTNN Tensor");
  }

  template <typename LayoutAttr>
  std::tuple<ttcore::MemorySpace, Type, ttcore::TensorMemoryLayout>
  extractLayoutInfo(LayoutAttr layout) const {
    return {layout.getBufferType() == ttnn::BufferType::DRAM
                ? ttcore::MemorySpace::DeviceDRAM
                : ttcore::MemorySpace::DeviceL1,
            layout.getElementType(),
            layout.getMemLayout().getValue() ==
                    ttnn::TensorMemoryLayout::Interleaved
                ? ttcore::TensorMemoryLayout::Interleaved
                : ttcore::TensorMemoryLayout::Sharded};
  }

  RankedTensorType
  getMetalTensorFromTTNNTensor(mlir::ConversionPatternRewriter &rewriter,
                               Value value) const {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
    Attribute ttnnLayout = tensorType.getEncoding();

    auto [memSpace, elementType, memLayout] = [&]()
        -> std::tuple<ttcore::MemorySpace, Type, ttcore::TensorMemoryLayout> {
      if (auto ndLayout = mlir::dyn_cast<ttnn::TTNNNDLayoutAttr>(ttnnLayout)) {
        return extractLayoutInfo(ndLayout);
      }
      if (auto layout = mlir::dyn_cast<ttnn::TTNNLayoutAttr>(ttnnLayout)) {
        return extractLayoutInfo(layout);
      }
      llvm_unreachable("Unsupported layout for TTNN Tensor");
    }();

    auto optimalGrid = getGridForTTNNTensor(tensorType);

    llvm::SmallVector<int64_t> dimAlignments(tensorType.getShape().size(), 1);
    if (dimAlignments.size() == 1) {
      dimAlignments[0] = ttcore::TileType::getDefaultShape()[1];
    } else {
      dimAlignments[dimAlignments.size() - 1] =
          ttcore::TileType::getDefaultShape()[0];
      dimAlignments[dimAlignments.size() - 2] =
          ttcore::TileType::getDefaultShape()[1];
    }

    ttcore::MetalLayoutAttr metalLayout;
    if (mlir::isa<ttnn::TTNNNDLayoutAttr>(ttnnLayout)) {
      // There is no dim collapsing for ND layouts.
      auto emptyIntervalType = RankedTensorType::get(
          {0, 2}, IntegerType::get(rewriter.getContext(), 64));
      DenseIntElementsAttr emptyCollapseIntervals =
          DenseIntElementsAttr::get(emptyIntervalType, ArrayRef<int64_t>{});
      metalLayout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), tensorType.getShape(), memSpace, memLayout,
          emptyCollapseIntervals, dimAlignments);
    } else {
      metalLayout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), tensorType.getShape(), memSpace, memLayout,
          ttcore::MetalLayoutAttr::computeDefaultCollapsedIntervals(
              rewriter.getContext(), tensorType.getShape().size()),
          dimAlignments);
    }

    llvm::SmallVector<int64_t> unshardedShape =
        metalLayout.getPhysicalShape(ttcore::TileType::getDefaultShape());

    llvm::SmallVector<int64_t> shardedShape = metalLayout.getDeviceShape(
        optimalGrid, ttcore::TileType::getDefaultShape());

    return mlir::RankedTensorType::get(shardedShape, elementType, metalLayout);
  }

  static void copyVirtualGridAttrs(d2m::EmptyOp dst, d2m::EmptyOp src) {
    if (auto attr = src.getVirtualGridInverseMappingAttr()) {
      dst.setVirtualGridInverseMappingAttr(attr);
    }
    if (auto attr = src.getVirtualGridForwardMappingAttr()) {
      dst.setVirtualGridForwardMappingAttr(attr);
    }
  }

  bool createMaskOpIfNeeded(Value &value, d2m::EmptyOp layoutEmpty,
                            ttcore::OOBVal fillValue,
                            mlir::ConversionPatternRewriter &rewriter) const {
    if (fillValue == ttcore::OOBVal::Undef) {
      return false;
    }

    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    if (!tensorType || !ttcore::isTiled(tensorType)) {
      return false;
    }

    auto maskOutput = rewriter.create<d2m::EmptyOp>(
        value.getLoc(), tensorType.getShape(), tensorType.getElementType(),
        tensorType.getEncoding());
    copyVirtualGridAttrs(maskOutput, layoutEmpty);

    auto layout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        tensorType.getEncoding());
    ArrayRef<int64_t> logicalShape =
        layout ? layout.getLogicalShape() : tensorType.getShape();
    value = rewriter
                .create<d2m::MaskOp>(value.getLoc(), value, maskOutput,
                                     logicalShape, fillValue)
                .getResult();
    return true;
  }

  // Create a ToLayout operation for a value using the provided layout
  // information with a simple 1x1 grid; actual grid optimization and proper
  // dimension alignments are computed later in the D2MGridSelection pass.
  Value createOptimalLayoutOp(Value value, ttcore::MemorySpace memSpace,
                              bool tiled, bool noCollapse,
                              mlir::ConversionPatternRewriter &rewriter,
                              ttcore::OOBVal oobVal,
                              RankedTensorType logicalType = {}) const {
    bool isTTNN = isTTNNTensor(value.getType());
    if (isTTNN) {
      assert(ttnnMode && "Unexpected TTNN tensor as op operand");
      auto metalTensorType = getMetalTensorFromTTNNTensor(rewriter, value);
      auto metalCastOp = rewriter.create<ttir::TTNNMetalLayoutCastOp>(
          value.getLoc(), metalTensorType, value);

      // Propagate both VGM maps for height/width sharded TTNN layouts
      // so that downstream passes (GenericOp::build, getMemoryMap, etc.)
      // can discover the virtual-to-physical core mapping.
      auto inputEncoding =
          mlir::cast<RankedTensorType>(value.getType()).getEncoding();
      if (auto ttnnLayout =
              mlir::dyn_cast_if_present<ttnn::TTNNLayoutAttr>(inputEncoding)) {
        auto memLayout = ttnnLayout.getMemLayout().getValue();
        if (memLayout == ttnn::TensorMemoryLayout::HeightSharded ||
            memLayout == ttnn::TensorMemoryLayout::WidthSharded) {
          llvm::SmallVector<int64_t> ttnnGridShape(ttnnLayout.getGridShape());
          llvm::SmallVector<int64_t> virtualGrid;
          if (memLayout == ttnn::TensorMemoryLayout::HeightSharded) {
            virtualGrid = {ttnnGridShape[0] * ttnnGridShape[1], 1};
          } else {
            virtualGrid = {1, ttnnGridShape[0] * ttnnGridShape[1]};
          }
          auto [forwardMap, inverseMap] =
              ttmlir::d2m::utils::grids::createCoreVirtMaps(
                  rewriter.getContext(), virtualGrid, ttnnGridShape);
          metalCastOp.setVirtualGridInverseMappingAttr(
              AffineMapAttr::get(inverseMap));
          metalCastOp.setVirtualGridForwardMappingAttr(
              AffineMapAttr::get(forwardMap));
        }
      }

      ttcore::MemorySpace metalTensorMemSpace =
          mlir::cast<ttcore::MetalLayoutAttr>(metalTensorType.getEncoding())
              .getMemorySpace();
      if (metalTensorMemSpace == ttcore::MemorySpace::DeviceL1) {
        // Reblock L1 operand to unit grid to align with other operands while
        // preserving original TTNN tensor shape. These views will be removed in
        // GridSelection by insertTTNNDRAMViews().
        llvm::SmallVector<int64_t> unitGrid(
            metalTensorType.getShape().size() / 2, 1);
        auto [newTensorShape, reblockMap] =
            ttmlir::utils::calculateReblockMapForGrid(
                metalTensorType.getShape(), unitGrid,
                metalTensorType.getContext());
        auto unitGridType = RankedTensorType::get(
            newTensorShape, metalTensorType.getElementType(),
            metalTensorType.getEncoding());
        auto unitReblockingView = rewriter.create<d2m::ViewLayoutOp>(
            value.getLoc(), unitGridType, metalCastOp->getResult(0), reblockMap,
            /*reinterpretLayout=*/false);
        return unitReblockingView.getResult();
      }
      // For DRAM operands, we can return the metal cast result directly.
      return metalCastOp->getResult(0);
    }

    auto tensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
    // Relayouts of already-layouted values can still need the original logical
    // tensor shape, e.g. embedding indices staged from DRAM back to L1.
    if (!logicalType) {
      logicalType = tensorType;
    }
    ArrayRef<int64_t> logicalShape = logicalType.getShape();

    Type elementType = logicalType.getElementType();
    llvm::SmallVector<int64_t> tileShape;
    if (tiled) {
      constexpr std::array<int64_t, 2> defaultShape =
          ttcore::TileType::getDefaultShape();
      tileShape.assign(defaultShape.begin(), defaultShape.end());
      elementType = ttcore::TileType::get(elementType, tileShape);
    }

    ttcore::MetalLayoutAttr layout;
    if (!collapseTensors || noCollapse) {
      auto emptyIntervalType = RankedTensorType::get(
          {0, 2}, IntegerType::get(rewriter.getContext(), 64));

      DenseIntElementsAttr emptyCollapseIntervals =
          DenseIntElementsAttr::get(emptyIntervalType, ArrayRef<int64_t>{});

      layout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), logicalShape, memSpace,
          ttcore::TensorMemoryLayout::Sharded, emptyCollapseIntervals);

    } else {
      layout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), logicalShape, memSpace,
          ttcore::TensorMemoryLayout::Sharded);
    }

    // Get raw, unsharded physical shape.
    llvm::SmallVector<int64_t> unshardedShape =
        layout.getPhysicalShape(tileShape);

    // Use a placeholder, 1-filled tensor grid for this pass. The length is the
    // physical tensor rank, not a device-grid height.
    llvm::SmallVector<int64_t> placeholderTensorGrid(unshardedShape.size(), 1);

    llvm::SmallVector<int64_t> shardedShape =
        layout.getDeviceShape(placeholderTensorGrid, tileShape);

    auto emptyOp = rewriter.create<d2m::EmptyOp>(value.getLoc(), shardedShape,
                                                 elementType, layout);

    // For ND tensors (logicalShape.size() > 2), set placeholder virtual grid
    // mappings on the EmptyOp.  These will be replaced when GridSelection
    // optimizes the grid.
    if (logicalShape.size() > 2) {
      auto [forwardMap, inverseMap] =
          ttmlir::d2m::utils::grids::createCoreVirtMaps(
              rewriter.getContext(), placeholderTensorGrid, {1, 1});
      emptyOp.setVirtualGridInverseMappingAttr(AffineMapAttr::get(inverseMap));
      emptyOp.setVirtualGridForwardMappingAttr(AffineMapAttr::get(forwardMap));
    }

    auto toLayoutOp =
        rewriter.create<d2m::ToLayoutOp>(value.getLoc(), value, emptyOp);
    Value layoutResult = toLayoutOp.getResult(0);
    createMaskOpIfNeeded(layoutResult, emptyOp, oobVal, rewriter);
    return layoutResult;
  }

  Value createOptimalLayoutOp(Value value, ttcore::MemorySpace memSpace,
                              bool tiled, bool noCollapse,
                              mlir::ConversionPatternRewriter &rewriter) const {
    return createOptimalLayoutOp(value, memSpace, tiled, noCollapse, rewriter,
                                 ttcore::OOBVal::Undef);
  }

  // Insert ToLayout operations for a genericOp's operands and results,
  // including sharding and tilizing, with simple 1x1 grids; grid optimization
  // happens later in the D2MGridSelection pass.
  std::array<mlir::SmallVector<Value>, 2> toLayoutOperandsAndResults(
      mlir::ConversionPatternRewriter &rewriter,
      std::array<mlir::SmallVector<Value>, 2> operandsAndResults, bool tiled,
      bool noCollapse, ttcore::OOBVal oobVal) const {
    std::array<mlir::SmallVector<Value>, 2> result;

    for (Value operand : operandsAndResults[0]) {
      result[0].push_back(createOptimalLayoutOp(operand, memorySpaces[0], tiled,
                                                noCollapse, rewriter, oobVal));
    }
    // Outputs always use Undef: they are destination buffers being written
    // into, so their padding fill value is irrelevant.  Only inputs need
    // identity-element OOB to prevent padded tiles from corrupting reductions.
    for (Value operand : operandsAndResults[1]) {
      result[1].push_back(createOptimalLayoutOp(operand, memorySpaces[1], tiled,
                                                noCollapse, rewriter));
    }

    return result;
  }

  std::array<mlir::SmallVector<Value>, 2> toLayoutOperandsAndResults(
      mlir::ConversionPatternRewriter &rewriter,
      std::array<mlir::SmallVector<Value>, 2> operandsAndResults, bool tiled,
      bool noCollapse = false) const {
    return toLayoutOperandsAndResults(rewriter, operandsAndResults, tiled,
                                      noCollapse, ttcore::OOBVal::Undef);
  }

  Operation *unLayoutResult(mlir::ConversionPatternRewriter &rewriter,
                            Value fromValue, Type toResultType) const {
    if (isTTNNTensor(toResultType)) {
      assert(ttnnMode && "Unexpected TTNN tensor as op result");
      return rewriter.create<ttir::TTNNMetalLayoutCastOp>(
          fromValue.getLoc(), toResultType, fromValue);
    }
    auto output =
        rewriter.create<d2m::EmptyOp>(fromValue.getLoc(), toResultType,
                                      /*virtualGridInverseMapping=*/nullptr,
                                      /*virtualGridForwardMapping=*/nullptr);
    return rewriter.create<d2m::ToLayoutOp>(fromValue.getLoc(), fromValue,
                                            output);
  }

  static llvm::SmallVector<mlir::Value>
  createDpsOutputs(Location loc, OpBuilder builder,
                   ArrayRef<RankedTensorType> types) {
    llvm::SmallVector<mlir::Value> dpsOutputs;
    dpsOutputs.reserve(types.size());
    for (auto type : types) {
      ttir::EmptyOp empty = builder.create<ttir::EmptyOp>(
          loc, type.getShape(), type.getElementType(), type.getEncoding());
      dpsOutputs.push_back(empty);
    }
    return dpsOutputs;
  }

  static SmallVector<mlir::AffineMap>
  getIdentityAffineMapsArray(mlir::OpBuilder &builder, std::size_t arity,
                             std::size_t rank) {
    return SmallVector<mlir::AffineMap>(arity,
                                        builder.getMultiDimIdentityMap(rank));
  }

  // Convert from ttir enum to equivalent linalg enum.
  static SmallVector<mlir::utils::IteratorType>
  iteratorTypeTTIRToLinalg(mlir::OpBuilder &builder,
                           const SmallVector<mlir::Attribute> &iterators) {
    auto parallel = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Parallel);
    auto reduction = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Reduction);

    SmallVector<mlir::utils::IteratorType> r;
    for (auto iterator : iterators) {
      if (parallel == iterator) {
        r.emplace_back(mlir::utils::IteratorType::parallel);
      } else if (reduction == iterator) {
        r.emplace_back(mlir::utils::IteratorType::reduction);
      } else {
        llvm_unreachable("unexpected ttir iterator type");
      }
    }
    return r;
  }

  // Get grid dimension indices where multicast should happen.
  // Multicast is needed on grid dimensions where the indexing map has a
  // parallel iterator type. Returns empty vector if no multicast is needed.
  static SmallVector<int64_t> getMulticastGridDims(AffineMap indexingMap,
                                                   ArrayAttr iteratorTypes) {
    SmallVector<int64_t> mcastGridDims;

    // Iterate over the indexing map results (one per grid dimension).
    bool foundReductionDims = false;
    for (auto [gridDim, expr] : llvm::enumerate(indexingMap.getResults())) {
      if (auto dimExpr = mlir::dyn_cast<AffineDimExpr>(expr)) {
        int64_t iterDimPos = dimExpr.getPosition();

        // Check if this iterator dimension is a parallel dimension
        auto iterType =
            mlir::cast<ttcore::IteratorTypeAttr>(iteratorTypes[iterDimPos]);
        if (iterType.getValue() == ttcore::IteratorType::Parallel) {
          // This grid dimension needs multicast
          mcastGridDims.push_back(static_cast<int64_t>(gridDim));
        } else if (iterType.getValue() == ttcore::IteratorType::Reduction) {
          foundReductionDims = true;
        }
      }
    }

    // if no reduction dimensions are found, return empty vector to signal
    // multicast is not possible
    if (!foundReductionDims) {
      return SmallVector<int64_t>();
    }
    return mcastGridDims;
  }

  static std::pair<SmallVector<SmallVector<Value>>,
                   SmallVector<SmallVector<int64_t>>>
  createInputIndicesAndMcastGridDims(mlir::OpBuilder &builder,
                                     mlir::Location loc, d2m::GenericOp generic,
                                     bool enableMulticastInference) {
    SmallVector<SmallVector<Value>> inputIndices(generic.getNumOperands());
    SmallVector<SmallVector<int64_t>> mcastGridDims(generic.getNumOperands());
    for (size_t i = 0; i < generic.getNumOperands(); ++i) {
      inputIndices[i] =
          d2m::utils::buildGridIndices(builder, loc, generic.getIndexingMap(i));
      if (enableMulticastInference) {
        mcastGridDims[i] = getMulticastGridDims(generic.getIndexingMap(i),
                                                generic.getIteratorTypes());
      }
    }
    return std::make_pair(inputIndices, mcastGridDims);
  }

  static SmallVector<Value>
  createBlockArguments(mlir::OpBuilder &builder, mlir::Block *block,
                       mlir::Location loc, mlir::TypeRange inputs,
                       mlir::TypeRange outputs, d2m::GenericOp generic,
                       SmallVector<SmallVector<Value>> inputIndices,
                       SmallVector<SmallVector<int64_t>> mcastGridDims) {
    // Compute shard shapes from operand layouts.
    auto getShardType = [](Type t) -> RankedTensorType {
      auto tensorType = mlir::cast<mlir::RankedTensorType>(t);
      ttcore::MetalLayoutAttr layout =
          mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
      auto shardShape = layout.getShardShape(tensorType);
      return mlir::RankedTensorType::get(shardShape,
                                         tensorType.getElementType());
    };

    SmallVector<Value> operands;

    // Process input operands - create tensor.empty + remote_load operations.
    for (size_t i = 0; i < inputs.size(); ++i) {
      RankedTensorType shardType = getShardType(inputs[i]);

      // Get the generic operand (the remote memref/tensor)
      Value genericOperand = generic->getOperand(i);

      // Build grid indices from the indexing map
      SmallVector<Value> indices = inputIndices[i];

      // Create a buffer for the load result
      auto bufferOp = builder.create<tensor::EmptyOp>(
          loc, shardType.getShape(), shardType.getElementType());
      Value buffer = bufferOp.getResult();

      Value loadResult;
      if (!mcastGridDims[i].empty()) {
        // Build mcast dimension indices (constant Values) for the grid
        // dimensions that need multicast
        SmallVector<Value> mcastDims;
        for (int64_t gridDim : mcastGridDims[i]) {
          mcastDims.push_back(
              builder.create<arith::ConstantIndexOp>(loc, gridDim));
        }

        // Create remote_load with high-level multicast form
        loadResult =
            builder
                .create<d2m::RemoteLoadOp>(loc, shardType, buffer,
                                           genericOperand, indices, mcastDims)
                .getResult();
      } else {
        // Create remote_load without multicast (original behavior)
        loadResult = builder
                         .create<d2m::RemoteLoadOp>(loc, shardType, buffer,
                                                    genericOperand, indices)
                         .getResult();
      }

      operands.push_back(loadResult);
    }

    // Process output operands - create tensor.empty operations.
    for (size_t i = 0; i < outputs.size(); ++i) {
      RankedTensorType shardType = getShardType(outputs[i]);

      auto emptyOp = builder.create<tensor::EmptyOp>(
          loc, shardType.getShape(), shardType.getElementType());

      operands.push_back(emptyOp.getResult());
    }

    return operands;
  }

  /// Populates `generic`'s region: creates block arguments, calls `body` to
  /// get one computed result per output, then emits remote_store + yield.
  void withD2MGenericRegion(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      d2m::GenericOp generic, mlir::ValueRange inputs, mlir::ValueRange outputs,
      llvm::function_ref<SmallVector<Value>(mlir::ArrayRef<mlir::Value>)> body)
      const {
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);
      auto [inputIndices, mcastGridDims] = createInputIndicesAndMcastGridDims(
          rewriter, loc, generic, enableMulticastInference);
      SmallVector<Value> blockArgsVec = createBlockArguments(
          rewriter, block, loc, mlir::TypeRange(inputs),
          mlir::TypeRange(outputs), generic, inputIndices, mcastGridDims);

      SmallVector<Value> computedResults = body(blockArgsVec);
      assert(computedResults.size() == outputs.size());

      SmallVector<Value> storeResults;
      for (size_t outputIdx = 0; outputIdx < outputs.size(); ++outputIdx) {
        size_t operandIdx = inputs.size() + outputIdx;
        AffineMap indexingMap = generic.getIndexingMap(operandIdx);
        SmallVector<Value> indices =
            d2m::utils::buildGridIndices(rewriter, loc, indexingMap);
        Value genericOperand = generic->getOperand(operandIdx);
        Value storeResult =
            rewriter
                .create<d2m::RemoteStoreOp>(loc, genericOperand.getType(),
                                            genericOperand, indices,
                                            computedResults[outputIdx])
                .getResult();
        storeResults.push_back(storeResult);
      }
      rewriter.create<d2m::YieldOp>(loc, storeResults);
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);
  }

  /// Fills a single tile tensor with `fillValue` (e.g. mean scaler 1/N).
  mlir::Value createScaler(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Location loc, mlir::RankedTensorType inputType,
                           double fillValue = 1.0) const {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Type elementType = inputType.getElementType();
    mlir::Attribute encoding = nullptr;

    if (auto ttnnLayout = mlir::dyn_cast_if_present<ttnn::TTNNLayoutAttr>(
            inputType.getEncoding())) {
      llvm::SmallVector<int64_t> gridShape{1, 1};
      auto tileType = ttcore::TileType::get(
          elementType, ttcore::TileType::getDefaultShape());
      auto memref = mlir::MemRefType::get(
          {1, 1}, tileType, mlir::MemRefLayoutAttrInterface{},
          ttnnLayout.getMemref().getMemorySpace());

      auto memLayout = ttnnLayout.getMemLayout();
      ttnn::CoreRangeSetAttr coreRangeSet{};
      if (memLayout && isShardedMemoryLayout(memLayout.getValue())) {
        coreRangeSet = ttnn::CoreRangeSetAttr::get(
            ctx, ttnn::CoreRangeAttr::get(
                     ctx, ttnn::CoreCoordAttr::get(ctx, 0, 0),
                     ttnn::CoreCoordAttr::get(ctx, gridShape[1] - 1,
                                              gridShape[0] - 1)));
      }

      encoding = ttnn::TTNNLayoutAttr::get(
          rewriter.getContext(), rewriter.getMultiDimIdentityMap(2),
          /*gridShape=*/gridShape, memref, ttnnLayout.getMemLayout(),
          /*tensorMesh=*/nullptr, /*ignorePhysicalLayout=*/false, coreRangeSet);
    }

    mlir::RankedTensorType scalerType = mlir::RankedTensorType::get(
        ttcore::TileType::getDefaultShape(), elementType, encoding);

    assert(mlir::isa<mlir::FloatType>(elementType) &&
           "createScaler is float-only; integer reductions lower via "
           "tile_sfpu_reduce_* which take no scaler");
    mlir::Attribute fillAttr =
        mlir::FloatAttr::get(rewriter.getF32Type(), fillValue);

    mlir::FailureOr<mlir::Value> filled =
        lowerRankedTensorFillViaGeneric(rewriter, loc, scalerType, fillAttr);
    assert(mlir::succeeded(filled) && "scaler fill lowering failed");
    return *filled;
  }

  /// Lowers a constant fill via `d2m.generic` + `linalg.generic` with
  /// `tile_fill` / `remote_store`. Returns the result in `resultType` layout.
  mlir::FailureOr<mlir::Value> lowerRankedTensorFillViaGeneric(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::RankedTensorType resultType, mlir::Attribute fillAttr) const {

    SmallVector<Value> origInputs;
    SmallVector<Value> origOutputs =
        createDpsOutputs(loc, rewriter, {resultType});
    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {origInputs, origOutputs}, /*tiled*/ true);
    assert(outputs.size() == 1);
    Value output = outputs[0];

    auto outputRankedTy = mlir::cast<RankedTensorType>(output.getType());
    auto outputTileTy =
        mlir::cast<ttcore::TileType>(outputRankedTy.getElementType());
    Type scalarElemTy = outputTileTy.getElementType();
    if (mlir::isa<mlir::FloatType>(scalarElemTy)) {
      if (!mlir::isa<mlir::FloatAttr>(fillAttr)) {
        return mlir::failure();
      }
    } else if (mlir::isa<mlir::IntegerType>(scalarElemTy)) {
      if (!mlir::isa<mlir::IntegerAttr>(fillAttr)) {
        return mlir::failure();
      }
    } else {
      return mlir::failure();
    }

    const std::size_t physicalRank = outputRankedTy.getRank() / 2;

    SmallVector<AffineMap> indexingMaps =
        getIdentityAffineMapsArray(rewriter, 1, physicalRank);
    auto parallel = ttcore::IteratorTypeAttr::get(
        rewriter.getContext(), ttcore::IteratorType::Parallel);
    SmallVector<Attribute> iteratorTypes(physicalRank, parallel);

    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    withD2MGenericRegion(
        rewriter, loc, generic, inputs, outputs,
        [&](mlir::ArrayRef<mlir::Value> blockArgs) -> SmallVector<Value> {
          Value outputShard = blockArgs[0];

          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
              iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

          auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
              loc,
              llvm::to_vector(
                  mlir::ValueRange(blockArgs.take_back(1)).getTypes()),
              /*inputs=*/ValueRange{},
              /*outs=*/blockArgs.take_back(1), indexingMaps,
              linalgIteratorTypes,
              [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                  mlir::ValueRange) {
                auto shardTy =
                    mlir::cast<RankedTensorType>(outputShard.getType());
                auto tType =
                    mlir::cast<ttcore::TileType>(shardTy.getElementType());
                Type eType = tType.getElementType();
                TypedAttr scalarAttr;
                if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(eType)) {
                  scalarAttr = mlir::FloatAttr::get(
                      floatTy,
                      mlir::cast<mlir::FloatAttr>(fillAttr).getValueAsDouble());
                } else {
                  auto intTy = mlir::cast<mlir::IntegerType>(eType);
                  auto signlessIntTy = mlir::IntegerType::get(
                      bbBuilder.getContext(), intTy.getWidth());
                  scalarAttr = mlir::IntegerAttr::get(
                      signlessIntTy, mlir::cast<mlir::IntegerAttr>(fillAttr)
                                         .getValue()
                                         .getSExtValue());
                }
                Value fillScalar = bbBuilder.create<mlir::arith::ConstantOp>(
                    bbLoc, scalarAttr.getType(), scalarAttr);
                mlir::Value yieldTile =
                    bbBuilder.create<d2m::TileFillOp>(bbLoc, tType, fillScalar)
                        .getResult();
                bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yieldTile);
              });

          return {linalgGeneric.getResult(0)};
        });

    return unLayoutResult(rewriter, generic->getResult(0), resultType)
        ->getResult(0);
  }

  template <typename ConcreteOp>
  static ttcore::MemorySpace getDefaultMemorySpace(ConcreteOp op,
                                                   ttcore::MemorySpace dflt) {
    mlir::ModuleOp parent = op->template getParentOfType<mlir::ModuleOp>();
    if (!parent) {
      return dflt;
    }
    ttcore::MemorySpaceAttr defaultMemSpaceAttr =
        parent->getAttrOfType<ttcore::MemorySpaceAttr>(
            ttcore::MemorySpaceAttr::name);
    return defaultMemSpaceAttr ? defaultMemSpaceAttr.getValue() : dflt;
  }

protected:
  // Default memory spaces for {inputs, outputs}.
  std::array<ttcore::MemorySpace, 2> memorySpaces;

  // Translate TTNN Tensors to Metal Tensors.
  bool ttnnMode;

  // Automatically collapse higher-rank tensors to 2D.
  bool collapseTensors;

  // Enable automatic multicast inference for reduction operations.
  bool enableMulticastInference;
};
} // namespace

namespace {
// Maps a binary tile comparison op (used for floating-point lowering via the
// SFPU `*_binary_tile` API) to the corresponding unary compare-with-zero op
// (used for the integer fallback via `(a - b)` followed by `*z_int32`).
template <typename T>
struct ComparisonZTileOp;
template <>
struct ComparisonZTileOp<d2m::TileEqOp> {
  using type = d2m::TileEqzOp;
};
template <>
struct ComparisonZTileOp<d2m::TileNeOp> {
  using type = d2m::TileNezOp;
};
template <>
struct ComparisonZTileOp<d2m::TileGtOp> {
  using type = d2m::TileGtzOp;
};
template <>
struct ComparisonZTileOp<d2m::TileLtOp> {
  using type = d2m::TileLtzOp;
};
template <>
struct ComparisonZTileOp<d2m::TileGeOp> {
  using type = d2m::TileGezOp;
};
template <>
struct ComparisonZTileOp<d2m::TileLeOp> {
  using type = d2m::TileLezOp;
};

template <typename TileOp>
inline constexpr bool isBinaryComparisonTileOp =
    std::is_same_v<TileOp, d2m::TileEqOp> ||
    std::is_same_v<TileOp, d2m::TileNeOp> ||
    std::is_same_v<TileOp, d2m::TileGtOp> ||
    std::is_same_v<TileOp, d2m::TileLtOp> ||
    std::is_same_v<TileOp, d2m::TileGeOp> ||
    std::is_same_v<TileOp, d2m::TileLeOp>;

// ----------------------------------------------------------------------------
//
// Rewrite elementwise ops by emitting a matching D2M tile version of the op
// into a d2m.generic/linalg.generic nest.
template <typename ConcreteOp, typename TileOp>
class D2MNamedElementwiseRewriter final
    : public mlir::OpConversionPattern<ConcreteOp>,
      D2MNamedRewriterCommon {

public:
  D2MNamedElementwiseRewriter<ConcreteOp, TileOp>(
      const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
      ttcore::MemorySpace defaultInputMemSpace,
      ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
      bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

private:
  // Build outer (per-shard) implicit-broadcast indexing maps in `physicalRank`,
  // by comparing each input's physical shard shape (in tiles) to the output's.
  //
  // The d2m.generic / linalg.generic iteration domain is `physicalRank`, which
  // for TTNN-encoded operands may be smaller than the logical rank because the
  // TTNN layout collapses leading dims into a 2D physical memref. Building the
  // indexing maps in physical rank (instead of the logical output rank) keeps
  // the indexing-map domain in lock-step with the iteration domain regardless
  // of whether the layout collapses dims.
  //
  // Per physical dim:
  //   - input dim == output dim         -> identity dim expr
  //   - input dim == 1 && output dim > 1 -> constant 0 (broadcast)
  //   - any other mismatch              -> assertion failure
  //
  // In-tile broadcasts (TileBcastType::{Row, Col, Scalar}) are computed
  // independently from the original logical shapes via `getImplicitBcastInfo`
  // and applied inside the d2m.generic body via `d2m.tile_bcast`.
  static SmallVector<mlir::AffineMap> buildPhysicalImplicitBcastIndexingMaps(
      mlir::OpBuilder &builder, mlir::ArrayRef<Value> physicalInputs,
      Value physicalOutput, std::size_t physicalRank) {
    // Return an owning SmallVector rather than ArrayRef: even though the
    // current MetalLayoutAttr::getShardShape implementation returns a slice
    // backed by the MLIR type's stable storage, the interface contract
    // returns ArrayRef<int64_t> and a future implementation could return a
    // view into a temporary container, which would dangle. Copying once per
    // operand is cheap and removes that footgun.
    auto getShardTiles = [](Value v) -> SmallVector<int64_t> {
      auto tensorType = mlir::cast<mlir::RankedTensorType>(v.getType());
      auto layout =
          mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
      return SmallVector<int64_t>(layout.getShardShape(tensorType));
    };

    SmallVector<int64_t> outShard = getShardTiles(physicalOutput);
    TT_assertv(outShard.size() == physicalRank,
               "output shard rank ({}) must equal physicalRank ({})",
               outShard.size(), physicalRank);

    SmallVector<mlir::AffineMap> maps;
    maps.reserve(physicalInputs.size() + 1);

    for (Value input : physicalInputs) {
      SmallVector<int64_t> inShard = getShardTiles(input);
      TT_assertv(inShard.size() == physicalRank,
                 "input shard rank ({}) must equal physicalRank ({})",
                 inShard.size(), physicalRank);

      SmallVector<mlir::AffineExpr> exprs;
      exprs.reserve(physicalRank);
      for (std::size_t d = 0; d < physicalRank; ++d) {
        if (inShard[d] == outShard[d]) {
          exprs.push_back(builder.getAffineDimExpr(d));
        } else {
          TT_assertv(inShard[d] == 1,
                     "incompatible implicit bcast in physical dim {} "
                     "(input={}, output={})",
                     d, inShard[d], outShard[d]);
          exprs.push_back(builder.getAffineConstantExpr(0));
        }
      }
      maps.push_back(AffineMap::get(physicalRank, /*symbolCount=*/0, exprs,
                                    builder.getContext()));
    }

    maps.push_back(builder.getMultiDimIdentityMap(physicalRank));
    return maps;
  }

  static std::pair<SmallVector<mlir::AffineMap>,
                   SmallVector<d2m::TileBcastType>>
  getImplicitBcastInfo(mlir::OpBuilder &builder, ArrayRef<Value> inputs,
                       ArrayRef<Value> outputs) {
    const size_t numInputs = inputs.size();
    // Support binary (2 inputs) and ternary (3 inputs) ops.
    if ((numInputs != 2 && numInputs != 3) || outputs.size() != 1) {
      return {
          {},
          SmallVector<d2m::TileBcastType>(numInputs, d2m::TileBcastType::None)};
    }

    const auto outType =
        mlir::cast<mlir::RankedTensorType>(outputs[0].getType());
    const int outRank = static_cast<int>(outType.getRank());
    TT_assert(outRank >= 1);
    const auto outShape = outType.getShape();

    // Gather input types, ranks, and shapes.
    SmallVector<mlir::RankedTensorType> inputTypes;
    SmallVector<int> inputRanks;
    SmallVector<ArrayRef<int64_t>> inputShapes;
    int maxInputRank = 0;
    for (Value input : inputs) {
      auto type = mlir::cast<mlir::RankedTensorType>(input.getType());
      inputTypes.push_back(type);
      int rank = static_cast<int>(type.getRank());
      inputRanks.push_back(rank);
      inputShapes.push_back(type.getShape());
      maxInputRank = std::max(maxInputRank, rank);
    }
    TT_assert(outRank == maxInputRank);

    // Collapsing is disabled for implicit bcast, affine maps for both
    // d2m.generic and linalg.generic are derived from the logical shape.
    SmallVector<SmallVector<mlir::AffineExpr>> inputExprs(
        numInputs, SmallVector<mlir::AffineExpr>(outRank));

    // Deduce output shape and build affine indexing maps for broadcasting.
    // We iterate right-to-left (innermost to outermost) to align dimensions
    // per NumPy semantics. Lower-rank inputs are implicitly unsqueezed with
    // leading 1s. For each dim, we validate compatibility and mark inputs
    // needing broadcast with constant-0 affine exprs (locked index).
    SmallVector<int64_t> deducedShape(outRank, -1);
    for (int i = -1; i >= -outRank; i--) {
      const int outDim = outRank + i;

      // Gather dim sizes for all inputs (-1 for missing dims in lower-rank
      // tensors).
      SmallVector<int64_t> dimSizes;
      for (size_t j = 0; j < numInputs; ++j) {
        const int inputDim = inputRanks[j] + i;
        dimSizes.push_back((inputDim >= 0) ? inputShapes[j][inputDim] : -1);
      }

      // NumPy broadcasting: dims of 1 or -1 can broadcast, others must match.
      int64_t maxDimSize = -1;
      for (int64_t dimSize : dimSizes) {
        if (dimSize != -1 && dimSize != 1) {
          if (maxDimSize == -1) {
            maxDimSize = dimSize;
          } else {
            TT_assertv(dimSize == maxDimSize,
                       "Incompatible bcast dims {} & {}.", dimSize, maxDimSize);
          }
        }
      }
      if (maxDimSize == -1) {
        maxDimSize = 1;
      }

      // Set affine expr: constant 0 for broadcast dims, dim expr otherwise.
      for (size_t j = 0; j < numInputs; ++j) {
        const int inputDim = inputRanks[j] + i;
        const bool needsBcast =
            (dimSizes[j] == -1) || (dimSizes[j] == 1 && maxDimSize != 1);
        inputExprs[j][outDim] = needsBcast ? builder.getAffineConstantExpr(0)
                                           : builder.getAffineDimExpr(inputDim);
      }

      deducedShape[outDim] = maxDimSize;
    }

    TT_assert(llvm::equal(deducedShape, outShape));

    auto getTileBcastType =
        [](ArrayRef<mlir::AffineExpr> exprs) -> d2m::TileBcastType {
      const size_t rank = exprs.size();
      // Index locked for W -> Col/Scalar tile.
      const bool isColTile = mlir::isa<AffineConstantExpr>(exprs[rank - 1]);
      if (rank == 1) {
        return isColTile ? d2m::TileBcastType::Col : d2m::TileBcastType::None;
      }
      // Index locked for H -> Row/Scalar tile.
      const bool isRowTile = mlir::isa<AffineConstantExpr>(exprs[rank - 2]);

      if (isColTile && isRowTile) {
        return d2m::TileBcastType::Scalar;
      }
      if (isColTile) {
        return d2m::TileBcastType::Col;
      }
      if (isRowTile) {
        return d2m::TileBcastType::Row;
      }
      return d2m::TileBcastType::None;
    };

    SmallVector<d2m::TileBcastType> tileBcastTypes;
    SmallVector<mlir::AffineMap> indexingMaps;
    for (size_t j = 0; j < numInputs; ++j) {
      tileBcastTypes.push_back(getTileBcastType(inputExprs[j]));
      indexingMaps.push_back(
          AffineMap::get(outRank, 0, inputExprs[j], builder.getContext()));
    }
    indexingMaps.push_back(builder.getMultiDimIdentityMap(outRank));

    return {indexingMaps, tileBcastTypes};
  }

  void createComputeRegion(mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                           mlir::ValueRange bbArgs,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Location loc, const size_t numInputs,
                           const size_t numOutputs,
                           ArrayRef<d2m::TileBcastType> tileBcastTypes,
                           ArrayRef<NamedAttribute> opAttrs = {}) const {
    auto operands = llvm::to_vector(bbArgs.take_front(numInputs));
    mlir::TypeRange resultTypes = bbArgs.take_back(numOutputs);

    // Apply broadcast to all operands that need it.
    for (size_t i = 0; i < numInputs && i < tileBcastTypes.size(); ++i) {
      if (tileBcastTypes[i] != d2m::TileBcastType::None) {
        operands[i] = bbBuilder.create<d2m::TileBcastOp>(
            loc, resultTypes, operands[i], tileBcastTypes[i]);
      }
    }

    mlir::Value yield;
    if constexpr (std::is_same_v<ConcreteOp, ttir::ClampTensorOp>) {
      // Decompose into maximum(input, min) then minimum(result, max).
      yield = bbBuilder.create<d2m::TileMaximumOp>(
          loc, resultTypes, ValueRange{operands[0], operands[1]});
      yield = bbBuilder.create<d2m::TileMinimumOp>(
          loc, resultTypes, ValueRange{yield, operands[2]});
    } else if constexpr (std::is_same_v<ConcreteOp, ttir::ClampScalarOp> ||
                         std::is_same_v<ConcreteOp, ttir::SeluOp>) {
      // Unary ops with forwarded attributes (e.g. clamp min/max, selu
      // scale/alpha).
      yield =
          bbBuilder.create<TileOp>(loc, resultTypes[0], operands[0], opAttrs);
    } else if constexpr (std::is_same_v<ConcreteOp, ttir::LogicalAndOp>) {
      // LogicalAnd: NEZ(a) * NEZ(b) - both must be non-zero.
      auto nezA =
          bbBuilder.create<d2m::TileNezOp>(loc, resultTypes, operands[0]);
      auto nezB =
          bbBuilder.create<d2m::TileNezOp>(loc, resultTypes, operands[1]);
      yield = bbBuilder.create<d2m::TileMulOp>(loc, resultTypes,
                                               ValueRange{nezA, nezB});
    } else if constexpr (std::is_same_v<ConcreteOp, ttir::LogicalOrOp>) {
      // LogicalOr: NEZ(NEZ(a) + NEZ(b)) - at least one must be non-zero.
      auto nezA =
          bbBuilder.create<d2m::TileNezOp>(loc, resultTypes, operands[0]);
      auto nezB =
          bbBuilder.create<d2m::TileNezOp>(loc, resultTypes, operands[1]);
      auto sum = bbBuilder.create<d2m::TileAddOp>(loc, resultTypes,
                                                  ValueRange{nezA, nezB});
      yield = bbBuilder.create<d2m::TileNezOp>(loc, resultTypes, sum);
    } else if constexpr (std::is_same_v<ConcreteOp, ttir::LogicalXorOp>) {
      // LogicalXor: NEZ(NEZ(a) - NEZ(b)) - exactly one must be non-zero.
      auto nezA =
          bbBuilder.create<d2m::TileNezOp>(loc, resultTypes, operands[0]);
      auto nezB =
          bbBuilder.create<d2m::TileNezOp>(loc, resultTypes, operands[1]);
      auto diff = bbBuilder.create<d2m::TileSubOp>(loc, resultTypes,
                                                   ValueRange{nezA, nezB});
      yield = bbBuilder.create<d2m::TileNezOp>(loc, resultTypes, diff);
    } else if constexpr (isBinaryComparisonTileOp<TileOp>) {
      // The SFPU `*_binary_tile` API always writes fp32 1.0/0.0 into dst.
      // For floating-point operands that is the correct answer, but for
      // integer operands the result tensor's element type is an integer and
      // the fp32 bytes would be reinterpreted (1.0f -> 0x3F800000). Fall back
      // to the (sub + *z_int32) decomposition for integer-typed operands so
      // the kernel produces integer 0/1 directly.
      auto operandTileTy = mlir::cast<ttcore::TileType>(operands[0].getType());
      if (operandTileTy.getElementType().isInteger()) {
        auto sub = bbBuilder.create<d2m::TileSubOp>(loc, resultTypes, operands);
        yield = bbBuilder.create<typename ComparisonZTileOp<TileOp>::type>(
            loc, resultTypes, sub.getResult());
      } else {
        yield = bbBuilder.create<TileOp>(loc, resultTypes, operands);
      }
    } else {
      yield = bbBuilder.create<TileOp>(loc, resultTypes, operands);
    }

    bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
  }

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op->getLoc();

    auto origOutputs =
        createDpsOutputs(loc, rewriter, {op.getResult().getType()});
    SmallVector<Value> origInputs = adaptor.getOperands();

    // Compute logical-shape-derived bcast info. Only `tileBcastTypes` (used
    // for in-tile broadcasting via d2m.tile_bcast) and the `isImplicitBcast`
    // flag are consumed from this; the logical-rank affine maps are not used
    // to construct the d2m.generic, because they would not match the physical
    // (possibly collapsed) iteration domain. See
    // `buildPhysicalImplicitBcastIndexingMaps` below.
    SmallVector<mlir::AffineMap> logicalBcastMaps;
    SmallVector<d2m::TileBcastType> tileBcastTypes;
    std::tie(logicalBcastMaps, tileBcastTypes) =
        getImplicitBcastInfo(rewriter, origInputs, origOutputs);

    // Implicit bcast if tile-level bcast exists or any input indexing map is
    // not identity.
    const bool isImplicitBcast =
        !logicalBcastMaps.empty() &&
        llvm::any_of(ArrayRef<mlir::AffineMap>(logicalBcastMaps)
                         .take_front(origInputs.size()),
                     [](mlir::AffineMap map) { return !map.isIdentity(); });

    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {origInputs, origOutputs},
                                   /*tiled*/ true, isImplicitBcast);
    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;

    // Build indexing maps in the *physical* iteration domain. For implicit
    // broadcasts we cannot reuse the logical-rank maps from
    // `getImplicitBcastInfo` because TTNN-encoded operands may collapse leading
    // logical dims into a 2D physical layout, leaving the logical-rank maps
    // (e.g. 4D) misaligned with the physical iteration domain (e.g. 2D). The
    // physical-rank maps below are derived from per-operand physical shard
    // shape comparison and stay correct regardless of layout collapse.
    SmallVector<mlir::AffineMap> indexingMaps =
        isImplicitBcast
            ? buildPhysicalImplicitBcastIndexingMaps(rewriter, inputs,
                                                     outputs[0], physicalRank)
            : getAffineMapsArray(rewriter, numOperands, physicalRank);

    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, physicalRank);

    // Invariant required by getMulticastGridDims and downstream code: the
    // domain of every indexing map must match the size of iteratorTypes.
    // Use TT_assertv (always-on) instead of assert so a violation aborts
    // with a clear diagnostic in release builds rather than crashing later
    // with an out-of-bounds read into iteratorTypes.
    for (auto [i, m] : llvm::enumerate(indexingMaps)) {
      TT_assertv(m.getNumDims() == iteratorTypes.size(),
                 "indexing map #{} domain rank ({}) must match iterator "
                 "types count ({})",
                 i, m.getNumDims(), iteratorTypes.size());
    }

    // Create 'd2m.generic' accepting 'op's operands.
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    // Create one bb in 'generic''s region and set its arguments.
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block'.
      {
        auto [inputIndices, mcastGridDims] = createInputIndicesAndMcastGridDims(
            rewriter, loc, generic, enableMulticastInference);
        auto blockArgsVec = createBlockArguments(
            rewriter, block, loc, TypeRange(inputs), TypeRange(outputs),
            generic, inputIndices, mcastGridDims);
        ArrayRef<Value> blockArgs(blockArgsVec);

        // Create 'linalg.generic' accepting 'blockArgs'. The inner per-shard
        // tile iteration domain has the same rank as the d2m.generic outer
        // iteration domain (both `physicalRank`), and the same
        // shard-shape-derived broadcast pattern applies (broadcast inputs read
        // their single shard tile across the output's tile iteration). Reuse
        // the physical-rank maps to keep the two ranks/domains in sync.
        auto linalgIndexingMaps = indexingMaps;

        SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
            iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

        // Collect attributes to forward to tile ops (e.g., min/max for clamp).
        SmallVector<NamedAttribute> opAttrs;
        if constexpr (std::is_same_v<ConcreteOp, ttir::ClampScalarOp>) {
          opAttrs.push_back(rewriter.getNamedAttr("min", op.getMinAttr()));
          opAttrs.push_back(rewriter.getNamedAttr("max", op.getMaxAttr()));
        }
        if constexpr (std::is_same_v<ConcreteOp, ttir::SeluOp>) {
          opAttrs.push_back(rewriter.getNamedAttr("scale", op.getScaleAttr()));
          opAttrs.push_back(rewriter.getNamedAttr("alpha", op.getAlphaAttr()));
        }

        auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
            loc,
            /* result tensor types */
            llvm::to_vector(
                mlir::ValueRange(blockArgs.take_back(numOutputs)).getTypes()),
            /* inputs */ blockArgs.take_front(numInputs),
            /* outputs */ blockArgs.take_back(numOutputs), linalgIndexingMaps,
            linalgIteratorTypes,
            [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                mlir::ValueRange bbArgs) {
              createComputeRegion(bbBuilder, bbLoc, bbArgs, rewriter, loc,
                                  numInputs, numOutputs, tileBcastTypes,
                                  opAttrs);
            });

        // Insert remote_store operations for each output before yield
        SmallVector<Value> storeResults;
        for (size_t outputIdx = 0; outputIdx < numOutputs; ++outputIdx) {
          size_t operandIdx = numInputs + outputIdx;
          AffineMap indexingMap = generic.getIndexingMap(operandIdx);
          SmallVector<Value> indices =
              d2m::utils::buildGridIndices(rewriter, loc, indexingMap);
          Value genericOperand = generic->getOperand(operandIdx);
          Value result = linalgGeneric->getResult(outputIdx);
          Value storeResult =
              rewriter
                  .create<d2m::RemoteStoreOp>(loc, genericOperand.getType(),
                                              genericOperand, indices, result)
                  .getResult();
          storeResults.push_back(storeResult);
        }

        rewriter.create<d2m::YieldOp>(loc, storeResults);
      }
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);

    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op->getResult(0).getType()));
    return llvm::success();
  }

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, std::size_t arity,
                     std::size_t rank) {
    return getIdentityAffineMapsArray(builder, arity, rank);
  }

  static SmallVector<mlir::Attribute>
  getIteratorTypesArray(mlir::OpBuilder &builder, std::size_t rank) {
    auto parallel = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Parallel);
    return SmallVector<mlir::Attribute>(rank, parallel);
  }
};
} // namespace

namespace {
class D2MBroadcastRewriter final
    : public OpConversionPattern<ttir::BroadcastOp>,
      D2MNamedRewriterCommon {
public:
  D2MBroadcastRewriter(const TypeConverter &typeConverter,
                       mlir::MLIRContext *ctx,
                       ttcore::MemorySpace defaultInputMemSpace,
                       ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
                       bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ttir::BroadcastOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

private:
  RankedTensorType getUnitGridLayoutType(RankedTensorType logicalType,
                                         ttcore::MemorySpace memSpace,
                                         bool tiled, OpBuilder &builder) const {
    ArrayRef<int64_t> logicalShape = logicalType.getShape();

    Type elementType = logicalType.getElementType();
    llvm::SmallVector<int64_t> tileShape;
    if (tiled) {
      constexpr std::array<int64_t, 2> defaultShape =
          ttcore::TileType::getDefaultShape();
      tileShape.assign(defaultShape.begin(), defaultShape.end());
      elementType = ttcore::TileType::get(elementType, tileShape);
    }

    auto emptyIntervalType = RankedTensorType::get(
        {0, 2}, IntegerType::get(builder.getContext(), 64));
    DenseIntElementsAttr emptyCollapseIntervals =
        DenseIntElementsAttr::get(emptyIntervalType, ArrayRef<int64_t>{});
    auto layout = ttcore::MetalLayoutAttr::get(
        builder.getContext(), logicalShape, memSpace,
        ttcore::TensorMemoryLayout::Sharded, emptyCollapseIntervals);

    llvm::SmallVector<int64_t> unshardedShape =
        layout.getPhysicalShape(tileShape);
    llvm::SmallVector<int64_t> placeholderTensorGrid(unshardedShape.size(), 1);
    llvm::SmallVector<int64_t> shardedShape =
        layout.getDeviceShape(placeholderTensorGrid, tileShape);

    return RankedTensorType::get(shardedShape, elementType, layout);
  }

  static bool isTileBroadcastDim(int64_t rank, int64_t dim) {
    return rank == 1 ? dim == 0 : dim >= rank - 2;
  }

  static d2m::TileBcastType getTileBcastType(bool bcastRow, bool bcastCol) {
    if (bcastRow && bcastCol) {
      return d2m::TileBcastType::Scalar;
    }
    if (bcastCol) {
      return d2m::TileBcastType::Col;
    }
    if (bcastRow) {
      return d2m::TileBcastType::Row;
    }
    return d2m::TileBcastType::None;
  }

  static LogicalResult analyzeBroadcast(ttir::BroadcastOp op,
                                        ConversionPatternRewriter &rewriter,
                                        bool &hasOuterBroadcast,
                                        d2m::TileBcastType &tileBcastType) {
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    int64_t rank = inputType.getRank();
    if (rank < 1) {
      return rewriter.notifyMatchFailure(
          op, "D2M broadcast lowering expects ranked tensor with rank >= 1");
    }

    bool bcastRow = false;
    bool bcastCol = false;
    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t inputDim = inputShape[dim];
      int64_t outputDim = outputShape[dim];
      if (inputDim == outputDim) {
        continue;
      }
      if (ShapedType::isDynamic(inputDim) || ShapedType::isDynamic(outputDim)) {
        return rewriter.notifyMatchFailure(
            op, "D2M explicit broadcast lowering requires static shapes");
      }
      if (inputDim <= 0 || outputDim <= 0) {
        return rewriter.notifyMatchFailure(
            op, "D2M explicit broadcast lowering requires positive dims");
      }

      if (isTileBroadcastDim(rank, dim)) {
        if (inputDim != 1) {
          return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
            diag << "tile-dim explicit broadcast dim " << dim
                 << " requires input dim 1 (input=" << inputDim
                 << ", output=" << outputDim << ")";
          });
        }
        bcastRow |= rank > 1 && dim == rank - 2;
        bcastCol |= dim == rank - 1;
        continue;
      }

      if (outputDim % inputDim != 0) {
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
          diag << "outer-dim explicit broadcast dim " << dim
               << " requires output dim (" << outputDim
               << ") to be a positive multiple of input dim (" << inputDim
               << ")";
        });
      }
      hasOuterBroadcast = true;
    }

    tileBcastType = getTileBcastType(bcastRow, bcastCol);
    return success();
  }

  static AffineMap buildOuterBcastViewMap(OpBuilder &builder,
                                          RankedTensorType inputType,
                                          RankedTensorType outputType) {
    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> outputShape = outputType.getShape();
    TT_assert(inputShape.size() == outputShape.size());

    SmallVector<AffineExpr> inputExprs;
    inputExprs.reserve(outputShape.size());
    for (auto [dim, shapeDims] :
         llvm::enumerate(llvm::zip_equal(inputShape, outputShape))) {
      auto [inputDim, outputDim] = shapeDims;
      AffineExpr outputIndex = builder.getAffineDimExpr(dim);
      if (inputDim == outputDim) {
        inputExprs.push_back(outputIndex);
        continue;
      }

      bool validOuterBroadcastDim =
          inputDim > 0 && outputDim > 0 && outputDim % inputDim == 0;
      TT_assertv(validOuterBroadcastDim,
                 "explicit outer broadcast dim {} requires output dim ({}) "
                 "to be a positive multiple of input dim ({})",
                 dim, outputDim, inputDim);
      inputExprs.push_back(inputDim == 1 ? builder.getAffineConstantExpr(0)
                                         : outputIndex % inputDim);
    }

    return AffineMap::get(outputShape.size(), /*symbolCount=*/0, inputExprs,
                          builder.getContext());
  }

  LogicalResult rewriteOuterBroadcastWithViewLayout(
      ttir::BroadcastOp op, ttir::BroadcastOp::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());

    Value input =
        createOptimalLayoutOp(adaptor.getInput(), memorySpaces[0],
                              /*tiled=*/true, /*noCollapse=*/true, rewriter);
    auto viewType = getUnitGridLayoutType(outputType, memorySpaces[0],
                                          /*tiled=*/true, rewriter);
    auto remapping = buildOuterBcastViewMap(
        rewriter, mlir::cast<RankedTensorType>(input.getType()), viewType);
    auto view = rewriter.create<d2m::ViewLayoutOp>(
        loc, viewType, input, remapping, /*reinterpretLayout=*/false);

    rewriter.replaceOp(op,
                       unLayoutResult(rewriter, view.getResult(), outputType));
    return success();
  }

  static SmallVector<AffineMap>
  buildPhysicalBcastIndexingMaps(OpBuilder &builder, Value input, Value output,
                                 std::size_t physicalRank) {
    auto getShardTiles = [](Value v) -> SmallVector<int64_t> {
      auto tensorType = mlir::cast<RankedTensorType>(v.getType());
      auto layout =
          mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
      return SmallVector<int64_t>(layout.getShardShape(tensorType));
    };

    SmallVector<int64_t> inputShard = getShardTiles(input);
    SmallVector<int64_t> outputShard = getShardTiles(output);
    TT_assertv(inputShard.size() == physicalRank,
               "input shard rank ({}) must equal physicalRank ({})",
               inputShard.size(), physicalRank);
    TT_assertv(outputShard.size() == physicalRank,
               "output shard rank ({}) must equal physicalRank ({})",
               outputShard.size(), physicalRank);

    SmallVector<AffineExpr> inputExprs;
    inputExprs.reserve(physicalRank);
    for (std::size_t d = 0; d < physicalRank; ++d) {
      if (inputShard[d] == outputShard[d]) {
        inputExprs.push_back(builder.getAffineDimExpr(d));
        continue;
      }
      TT_assertv(inputShard[d] == 1,
                 "incompatible explicit broadcast in physical dim {} "
                 "(input={}, output={})",
                 d, inputShard[d], outputShard[d]);
      inputExprs.push_back(builder.getAffineConstantExpr(0));
    }

    return {
        AffineMap::get(physicalRank, /*symbolCount=*/0, inputExprs,
                       builder.getContext()),
        builder.getMultiDimIdentityMap(physicalRank),
    };
  }

  LogicalResult
  matchAndRewrite(ttir::BroadcastOp op, ttir::BroadcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    bool hasOuterBroadcast = false;
    d2m::TileBcastType tileBcastType = d2m::TileBcastType::None;
    if (failed(
            analyzeBroadcast(op, rewriter, hasOuterBroadcast, tileBcastType))) {
      return failure();
    }

    if (hasOuterBroadcast && tileBcastType == d2m::TileBcastType::None) {
      return rewriteOuterBroadcastWithViewLayout(op, adaptor, rewriter);
    }

    SmallVector<Value> origInputs = {adaptor.getInput()};
    SmallVector<Value> origOutputs =
        createDpsOutputs(loc, rewriter, {op.getResult().getType()});

    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {origInputs, origOutputs},
                                   /*tiled=*/true, /*noCollapse=*/true);
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);

    std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;
    SmallVector<AffineMap> indexingMaps = buildPhysicalBcastIndexingMaps(
        rewriter, inputs[0], outputs[0], physicalRank);

    auto parallel = ttcore::IteratorTypeAttr::get(
        rewriter.getContext(), ttcore::IteratorType::Parallel);
    SmallVector<Attribute> iteratorTypes(physicalRank, parallel);

    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    withD2MGenericRegion(
        rewriter, loc, generic, inputs, outputs,
        [&](ArrayRef<Value> blockArgs) -> SmallVector<Value> {
          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
              iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

          auto linalgGeneric = rewriter.create<linalg::GenericOp>(
              loc,
              /*resultTensorTypes=*/
              llvm::to_vector(ValueRange(blockArgs.take_back(1)).getTypes()),
              /*inputs=*/blockArgs.take_front(1),
              /*outputs=*/blockArgs.take_back(1), indexingMaps,
              linalgIteratorTypes,
              [&](OpBuilder &bbBuilder, Location bbLoc, ValueRange bbArgs) {
                Value yield = bbArgs[0];
                if (tileBcastType != d2m::TileBcastType::None) {
                  yield = bbBuilder.create<d2m::TileBcastOp>(
                      bbLoc, bbArgs[1].getType(), yield, tileBcastType);
                }
                bbBuilder.create<linalg::YieldOp>(bbLoc, yield);
              });

          return {linalgGeneric.getResult(0)};
        });

    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op.getResult().getType()));
    return success();
  }
};
} // namespace

// ----------------------------------------------------------------------------
//
// D2MNamedAccumReductionRewriter: outer logical-dim reductions
// lowered by accumulating full-tile binary ops (tile_add / tile_maximum /
// tile_minimum) across the reduction dim via d2m.generic with reduction
// iterators. Inner (tile C/R) reductions use the sibling
// D2MNamedTileReduceRewriter instead.
namespace {
template <typename ConcreteOp, typename TileAccumulateOp>
class D2MNamedAccumReductionRewriter final
    : public mlir::OpConversionPattern<ConcreteOp>,
      D2MNamedRewriterCommon {
public:
  D2MNamedAccumReductionRewriter(const TypeConverter &typeConverter,
                                 mlir::MLIRContext *ctx,
                                 ttcore::MemorySpace defaultInputMemSpace,
                                 ttcore::MemorySpace defaultOutputMemSpace,
                                 bool ttnnMode, bool collapseTensors,
                                 bool enableMulticastInference)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx, /*benefit=*/10),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

private:
  static mlir::RankedTensorType
  shardTypeForTiledOperand(mlir::Type operandType) {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(operandType);
    auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
    return mlir::RankedTensorType::get(layout.getShardShape(tensorType),
                                       tensorType.getElementType());
  }

  static mlir::Value
  createRemoteTensorSlice(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::RankedTensorType shardType,
                          mlir::Value loadBuffer, mlir::Value genericOperand,
                          mlir::ArrayRef<mlir::Value> gridIndices,
                          mlir::ArrayRef<int64_t> mcastGridDims) {
    if (!mcastGridDims.empty()) {
      mlir::SmallVector<mlir::Value> mcastDimValues;
      mcastDimValues.reserve(mcastGridDims.size());
      for (int64_t gridDim : mcastGridDims) {
        mcastDimValues.push_back(
            builder.create<mlir::arith::ConstantIndexOp>(loc, gridDim));
      }
      return builder
          .create<d2m::RemoteLoadOp>(loc, shardType, loadBuffer, genericOperand,
                                     gridIndices, mcastDimValues)
          .getResult();
    }
    return builder
        .create<d2m::RemoteLoadOp>(loc, shardType, loadBuffer, genericOperand,
                                   gridIndices)
        .getResult();
  }

  /// Per-shard linalg.generic for tile-level reduction: identity on the input
  /// map, projected-to-zero on reduced result dims (same `dim_arg` as TTIR).
  static std::pair<mlir::AffineMap, SmallVector<mlir::utils::IteratorType>>
  shardLinalgAccumulationSignature(mlir::OpBuilder &builder,
                                   mlir::ArrayAttr dimArg,
                                   std::size_t shardRank) {
    mlir::AffineExpr zero =
        mlir::getAffineConstantExpr(0, builder.getContext());
    mlir::MutableAffineMap accumulatorMap(
        builder.getMultiDimIdentityMap(shardRank));
    SmallVector<mlir::utils::IteratorType> iteratorTypes(
        shardRank, mlir::utils::IteratorType::parallel);
    for (mlir::Attribute reduceDimAttr : dimArg) {
      int64_t dim = mlir::cast<mlir::IntegerAttr>(reduceDimAttr).getInt();
      dim = (dim + static_cast<int64_t>(shardRank)) %
            static_cast<int64_t>(shardRank);
      accumulatorMap.setResult(dim, zero);
      iteratorTypes[dim] = mlir::utils::IteratorType::reduction;
    }
    return {accumulatorMap.getAffineMap(), std::move(iteratorTypes)};
  }

  /// Identity value for the outer-reduction scan (sum: 0, max: -inf / int min,
  /// min: +inf / int max) so padded / untouched elements do not affect the
  /// result.
  template <typename AccumOp>
  static mlir::Attribute
  initialFillAttrForOuterReduction(mlir::OpBuilder &builder,
                                   mlir::Type elemType) {
    if constexpr (std::is_same_v<AccumOp, d2m::TileAddOp>) {
      return builder.getZeroAttr(elemType);
    } else if constexpr (std::is_same_v<AccumOp, d2m::TileMaximumOp>) {
      if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemType)) {
        return mlir::FloatAttr::get(floatTy,
                                    -std::numeric_limits<double>::infinity());
      }
      if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        return mlir::IntegerAttr::get(
            intTy, llvm::APInt::getSignedMinValue(intTy.getWidth()));
      }
    } else if constexpr (std::is_same_v<AccumOp, d2m::TileMinimumOp>) {
      if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemType)) {
        return mlir::FloatAttr::get(floatTy,
                                    std::numeric_limits<double>::infinity());
      }
      if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        return mlir::IntegerAttr::get(
            intTy, llvm::APInt::getSignedMaxValue(intTy.getWidth()));
      }
    } else {
      static_assert(ttmlir::utils::always_false<AccumOp>(),
                    "Unhandled outer reduction accumulate op");
    }
    return {};
  }

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (!isOuterReduction(op)) {
      return failure();
    }
    checkTTIRReductionPreconditions(op);

    mlir::Location loc = op->getLoc();
    auto inputType =
        mlir::cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    auto outputType =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    mlir::ArrayAttr dimArg = *op.getDimArg();

    mlir::Attribute fillAttr =
        initialFillAttrForOuterReduction<TileAccumulateOp>(
            rewriter, outputType.getElementType());
    if (!fillAttr) {
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for outer reduction identity fill");
    }
    mlir::FailureOr<mlir::Value> filledOutput =
        lowerRankedTensorFillViaGeneric(rewriter, loc, outputType, fillAttr);
    if (mlir::failed(filledOutput)) {
      return failure();
    }

    mlir::SmallVector<mlir::Value> origInputs = adaptor.getOperands();
    origInputs.push_back(*filledOutput);
    SmallVector<Value> origOutputs =
        createDpsOutputs(loc, rewriter, {outputType});
    bool noCollapse = inputType.getRank() > 2;
    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {origInputs, origOutputs},
        /*tiled=*/true, noCollapse, getReductionOOBVal());

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;
    assertPhysicalIteratorRankForReduction(physicalRank);

    SmallVector<mlir::AffineMap> indexingMaps = {
        rewriter.getMultiDimIdentityMap(physicalRank),
        getOutputAffineMap(rewriter, op, physicalRank),
        getOutputAffineMap(rewriter, op, physicalRank)};
    mlir::SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, op, physicalRank);

    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/mlir::ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    mlir::Region &region = generic->getRegions().front();
    rewriter.createBlock(&region);

    mlir::RankedTensorType inputShardType =
        shardTypeForTiledOperand(inputs.front().getType());
    mlir::RankedTensorType outputShardType =
        shardTypeForTiledOperand(outputs.front().getType());

    mlir::AffineMap inputIndexingMap = generic.getIndexingMap(0);
    mlir::SmallVector<mlir::Value> inputIndices =
        d2m::utils::buildGridIndices(rewriter, loc, inputIndexingMap);
    mlir::Value inputLoadBuffer = rewriter.create<mlir::tensor::EmptyOp>(
        loc, inputShardType.getShape(), inputShardType.getElementType());
    mlir::SmallVector<int64_t> mcastGridDims;
    if (enableMulticastInference) {
      mcastGridDims =
          getMulticastGridDims(inputIndexingMap, generic.getIteratorTypes());
    }
    mlir::Value inputSlice = createRemoteTensorSlice(
        rewriter, loc, inputShardType, inputLoadBuffer, generic->getOperand(0),
        inputIndices, mcastGridDims);

    mlir::AffineMap outputIndexingMap = generic.getIndexingMap(1);
    mlir::SmallVector<mlir::Value> outputIndices =
        d2m::utils::buildGridIndices(rewriter, loc, outputIndexingMap);
    mlir::Value fillLoadBuffer = rewriter.create<mlir::tensor::EmptyOp>(
        loc, outputShardType.getShape(), outputShardType.getElementType());
    mlir::Value fillSlice =
        createRemoteTensorSlice(rewriter, loc, outputShardType, fillLoadBuffer,
                                generic->getOperand(1), outputIndices, {});
    mlir::Value outputBuffer = rewriter.create<mlir::tensor::EmptyOp>(
        loc, outputShardType.getShape(), outputShardType.getElementType());

    std::size_t shardRank = outputShardType.getRank();
    auto accumulationSig =
        shardLinalgAccumulationSignature(rewriter, dimArg, shardRank);
    mlir::AffineMap accumulatorMap = accumulationSig.first;
    SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
        std::move(accumulationSig.second);
    mlir::SmallVector<mlir::AffineMap> linalgIndexingMaps = {
        rewriter.getMultiDimIdentityMap(shardRank), accumulatorMap,
        accumulatorMap};

    auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{outputBuffer.getType()},
        mlir::ValueRange{inputSlice, fillSlice}, mlir::ValueRange{outputBuffer},
        linalgIndexingMaps, linalgIteratorTypes,
        [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
            mlir::ValueRange bbArgs) {
          // Check if we're on the first iteration of all reduction dimensions.
          // If so, use bbArgs[1] (init from fillSlice), otherwise use bbArgs[2]
          // (accumulated value from outputBuffer).
          mlir::Value c0 =
              bbBuilder.create<mlir::arith::ConstantIndexOp>(bbLoc, 0);
          mlir::Value isFirstIter = nullptr;
          for (size_t dim = 0; dim < shardRank; ++dim) {
            if (linalgIteratorTypes[dim] ==
                mlir::utils::IteratorType::reduction) {
              mlir::Value idx =
                  bbBuilder.create<mlir::linalg::IndexOp>(bbLoc, dim);
              mlir::Value isZero = bbBuilder.create<mlir::arith::CmpIOp>(
                  bbLoc, mlir::arith::CmpIPredicate::eq, idx, c0);
              if (!isFirstIter) {
                isFirstIter = isZero;
              } else {
                isFirstIter = bbBuilder.create<mlir::arith::AndIOp>(
                    bbLoc, isFirstIter, isZero);
              }
            }
          }

          // Select the accumulator: use init (bbArgs[1]) on first iteration,
          // otherwise use the output accumulator (bbArgs[2]).
          mlir::Value acc = bbArgs[2];
          if (isFirstIter) {
            acc = bbBuilder.create<mlir::arith::SelectOp>(bbLoc, isFirstIter,
                                                          bbArgs[1], bbArgs[2]);
          }

          mlir::Value reduced = bbBuilder.create<TileAccumulateOp>(
              bbLoc, mlir::TypeRange{acc.getType()},
              mlir::ValueRange{bbArgs[0], acc});
          bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, reduced);
        });

    mlir::SmallVector<mlir::Value> storeResults;
    storeResults.reserve(numOutputs);
    for (std::size_t outputIdx = 0; outputIdx < numOutputs; ++outputIdx) {
      std::size_t operandIdx = numInputs + outputIdx;
      mlir::AffineMap storeMap = generic.getIndexingMap(operandIdx);
      mlir::SmallVector<mlir::Value> indices =
          d2m::utils::buildGridIndices(rewriter, loc, storeMap);
      mlir::Value genericOperand = generic->getOperand(operandIdx);
      storeResults.push_back(
          rewriter
              .create<d2m::RemoteStoreOp>(loc, genericOperand.getType(),
                                          genericOperand, indices,
                                          linalgGeneric.getResult(outputIdx))
              .getResult());
    }
    rewriter.create<d2m::YieldOp>(loc, storeResults);

    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);

    if constexpr (std::is_same_v<ConcreteOp, ttir::MeanOp>) {
      return applyMeanScale(op, adaptor, rewriter, loc, generic, noCollapse);
    }

    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op->getResult(0).getType()));
    return mlir::success();
  }

  /// Applies the 1/N scaling for mean: sum(x) * (1/N).
  LogicalResult applyMeanScale(ConcreteOp op,
                               typename ConcreteOp::Adaptor adaptor,
                               mlir::ConversionPatternRewriter &rewriter,
                               mlir::Location loc, d2m::GenericOp generic,
                               bool noCollapse) const {
    mlir::RankedTensorType meanInputTy =
        mlir::cast<mlir::RankedTensorType>(adaptor.getInput().getType());
    mlir::ArrayAttr meanDimArg = *op.getDimArg();
    int64_t reductionSize = 1;
    for (mlir::Attribute dimAttr : meanDimArg) {
      int64_t dim = mlir::cast<mlir::IntegerAttr>(dimAttr).getInt();
      std::size_t n =
          normalizeReductionDimIndex(dim, meanInputTy.getShape().size());
      reductionSize *= meanInputTy.getShape()[n];
    }
    const double invN = 1.0 / static_cast<double>(reductionSize);

    mlir::RankedTensorType meanResultLogicalTy =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    mlir::Value sumAsLogical =
        unLayoutResult(rewriter, generic->getResult(0), meanResultLogicalTy)
            ->getResult(0);

    assert(mlir::isa<mlir::FloatType>(meanResultLogicalTy.getElementType()) &&
           "mean reduction requires float element types");
    mlir::Attribute invNFillAttr =
        mlir::FloatAttr::get(rewriter.getF32Type(), invN);
    mlir::FailureOr<mlir::Value> invNMetalFill =
        lowerRankedTensorFillViaGeneric(rewriter, loc, meanResultLogicalTy,
                                        invNFillAttr);
    if (mlir::failed(invNMetalFill)) {
      return failure();
    }
    mlir::Value invNAsLogical =
        unLayoutResult(rewriter, *invNMetalFill, meanResultLogicalTy)
            ->getResult(0);

    mlir::SmallVector<mlir::Value> scaleOrigInputs = {sumAsLogical,
                                                      invNAsLogical};
    mlir::SmallVector<mlir::Value> scaleOrigOutputs =
        createDpsOutputs(loc, rewriter, {op.getResult().getType()});
    auto [scaleInputs, scaleOutputs] = toLayoutOperandsAndResults(
        rewriter, {scaleOrigInputs, scaleOrigOutputs},
        /*tiled=*/true, noCollapse, ttcore::OOBVal::Zero);

    const std::size_t scalePhysRank =
        ttcore::getDeviceLayout(scaleOutputs[0]).getRank() / 2;
    mlir::SmallVector<mlir::AffineMap> scaleMaps(
        3, rewriter.getMultiDimIdentityMap(scalePhysRank));
    auto parallelIt = ttcore::IteratorTypeAttr::get(
        rewriter.getContext(), ttcore::IteratorType::Parallel);
    mlir::SmallVector<mlir::Attribute> scaleIterators(scalePhysRank,
                                                      parallelIt);

    auto scaleGeneric = rewriter.create<d2m::GenericOp>(
        loc, scaleInputs, scaleOutputs, mlir::ValueRange{},
        rewriter.getAffineMapArrayAttr(scaleMaps),
        rewriter.getArrayAttr(scaleIterators));

    withD2MGenericRegion(
        rewriter, loc, scaleGeneric, scaleInputs, scaleOutputs,
        [&](mlir::ArrayRef<mlir::Value> bbArgs) {
          mlir::SmallVector<mlir::utils::IteratorType> linalgIters(
              scalePhysRank, mlir::utils::IteratorType::parallel);
          auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
              loc, mlir::TypeRange{bbArgs.back().getType()},
              bbArgs.drop_back(1), bbArgs.take_back(1), scaleMaps, linalgIters,
              [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
                mlir::Value mul = b.create<d2m::TileMulOp>(
                    l, mlir::TypeRange{args[2].getType()},
                    mlir::ValueRange{args[0], args[1]});
                b.create<mlir::linalg::YieldOp>(l, mlir::ValueRange{mul});
              });
          return mlir::SmallVector<mlir::Value>{linalgGeneric.getResult(0)};
        });

    rewriter.replaceOp(op, unLayoutResult(rewriter, scaleGeneric->getResult(0),
                                          op->getResult(0).getType()));
    return mlir::success();
  }

  static constexpr ttcore::OOBVal getReductionOOBVal() {
    if constexpr (std::is_same_v<TileAccumulateOp, d2m::TileMaximumOp>) {
      return ttcore::OOBVal::NegInf;
    }
    if constexpr (std::is_same_v<TileAccumulateOp, d2m::TileMinimumOp>) {
      return ttcore::OOBVal::Inf;
    }
    return ttcore::OOBVal::Zero;
  }

  static mlir::AffineMap getOutputAffineMap(mlir::OpBuilder &builder,
                                            ConcreteOp op, std::size_t rank) {
    mlir::ArrayAttr dimArg = *op.getDimArg();
    mlir::AffineExpr zero =
        mlir::getAffineConstantExpr(0, builder.getContext());
    mlir::MutableAffineMap accumulator(builder.getMultiDimIdentityMap(rank));
    SmallVector<bool> dims(rank, false);
    for (auto reduceDim : dimArg) {
      int64_t dim = mlir::cast<IntegerAttr>(reduceDim).getInt();
      dims[normalizeReductionDimIndex(dim, rank)] = true;
    }
    for (std::size_t i = 0; i < rank; ++i) {
      if (dims[i]) {
        accumulator.setResult(i, zero);
      }
    }

    return accumulator.getAffineMap();
  }

  static SmallVector<mlir::Attribute>
  getIteratorTypesArray(mlir::OpBuilder &builder, ConcreteOp op,
                        std::size_t rank) {
    mlir::ArrayAttr dimArg = *op.getDimArg();
    auto parallel = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Parallel);
    auto reduction = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Reduction);
    SmallVector<mlir::Attribute> iterators(rank, parallel);
    for (auto reduceDim : dimArg) {
      int64_t dim = mlir::cast<IntegerAttr>(reduceDim).getInt();
      iterators[normalizeReductionDimIndex(dim, rank)] = reduction;
    }
    return iterators;
  }
};
} // namespace

// D2MNamedTileReduceRewriter lowers tile C/R reductions (the last two
// physical iterator dims) via tile_reduce_{sum,max,mean} for float element
// types (with a broadcast scaler tile encoding e.g. 1/N for mean) and via
// tile_sfpu_reduce_{sum,max} for integer element types (no scaler, SFPU
// lowering). `IntTileOp` may be `void` to indicate the int path is not
// supported (e.g. mean).
namespace {
template <typename ConcreteOp, typename FloatTileOp, typename IntTileOp = void>
class D2MNamedTileReduceRewriter final
    : public mlir::OpConversionPattern<ConcreteOp>,
      D2MNamedRewriterCommon {

public:
  D2MNamedTileReduceRewriter<ConcreteOp, FloatTileOp, IntTileOp>(
      const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
      ttcore::MemorySpace defaultInputMemSpace,
      ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
      bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

private:
  static constexpr bool kIntSupported = !std::is_void_v<IntTileOp>;

  // Return the identity OOB fill value for this reduction's tile op.
  // Padded elements must not affect the reduction result.
  static constexpr ttcore::OOBVal getReductionOOBVal() {
    if constexpr (std::is_same_v<FloatTileOp, d2m::TileReduceMaxOp>) {
      return ttcore::OOBVal::NegInf;
    } else if constexpr (std::is_same_v<FloatTileOp, d2m::TileReduceSumOp> ||
                         std::is_same_v<FloatTileOp, d2m::TileReduceMeanOp>) {
      return ttcore::OOBVal::Zero;
    } else {
      static_assert(ttmlir::utils::always_false<FloatTileOp>(),
                    "Unhandled reduction TileOp");
    }
  }

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    checkTTIRReductionPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto origInputs = adaptor.getOperands();
    auto origOutputs =
        createDpsOutputs(loc, rewriter, {op.getResult().getType()});
    auto inputTensorType =
        mlir::cast<RankedTensorType>(origInputs.front().getType());
    // Float reductions multiply A by a broadcast scaler tile inside
    // tile_reduce_*. Integer reductions lower through the SFPU, which
    // ignores the scaler entirely, so we emit tile_sfpu_reduce_* without a
    // scaler operand.
    const bool isFloat =
        mlir::isa<mlir::FloatType>(inputTensorType.getElementType());
    if (!isFloat && !kIntSupported) {
      return rewriter.notifyMatchFailure(
          op, "integer tile reduction is not supported for this op");
    }
    const bool hasScaler = isFloat;
    SmallVector<mlir::Value> newInputs(origInputs.begin(), origInputs.end());
    if (hasScaler) {
      newInputs.emplace_back(this->createScaler(rewriter, loc, inputTensorType,
                                                getScaleValue(op)));
    }
    bool noCollapse = (inputTensorType.getRank() > 2);

    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {newInputs, origOutputs},
        /*tiled*/ true, noCollapse, getReductionOOBVal());

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;
    assertPhysicalIteratorRankForReduction(physicalRank);

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, op, numOperands, physicalRank, hasScaler);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, op, physicalRank);

    // Create 'd2m.generic' accepting extended operands.
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    // Create one bb in 'generic''s region and set its arguments.
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block'.
      {
        auto [inputIndices, mcastGridDims] = createInputIndicesAndMcastGridDims(
            rewriter, loc, generic, enableMulticastInference);
        auto blockArgsVec = createBlockArguments(
            rewriter, block, loc, TypeRange(inputs), TypeRange(outputs),
            generic, inputIndices, mcastGridDims);
        ArrayRef<Value> blockArgs(blockArgsVec);

        // Create 'linalg.generic' accepting 'blockArgs'.

        SmallVector<mlir::AffineMap> linalgIndexingMaps = getAffineMapsArray(
            rewriter, op, numOperands, physicalRank, hasScaler);
        SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
            iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

        // Propagate attributes.

        auto reduceDimAttr =
            d2m::ReduceDimAttr::get(ctx, dimArgAsReduceDim(op, physicalRank));

        auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
            loc,
            /* result tensor types */
            llvm::to_vector(
                static_cast<mlir::ValueRange>(blockArgs.take_back(numOutputs))
                    .getTypes()),
            /* inputs */ blockArgs.take_front(numInputs),
            /* outputs */ blockArgs.take_back(numOutputs), linalgIndexingMaps,
            linalgIteratorTypes,
            [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                mlir::ValueRange bbArgs) {
              // bbArgs layout: [A, (scaler if hasScaler,) outputInit...].
              mlir::Value aArg = bbArgs.front();
              mlir::Value cArg = bbArgs[numInputs];
              mlir::Type resultType = cArg.getType();
              mlir::Value yield;
              if (hasScaler) {
                mlir::Value bArg = bbArgs[1];
                yield = bbBuilder.create<FloatTileOp>(
                    loc, resultType, aArg, bArg, cArg, reduceDimAttr);
              } else {
                if constexpr (kIntSupported) {
                  yield = bbBuilder.create<IntTileOp>(loc, resultType, aArg,
                                                      cArg, reduceDimAttr);
                } else {
                  llvm_unreachable("int path guarded against at entry");
                }
              }
              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
            });

        // Insert remote_store operations for each output before yield
        SmallVector<Value> storeResults;
        for (size_t outputIdx = 0; outputIdx < numOutputs; ++outputIdx) {
          size_t operandIdx = numInputs + outputIdx;
          AffineMap indexingMap = generic.getIndexingMap(operandIdx);
          SmallVector<Value> indices =
              d2m::utils::buildGridIndices(rewriter, loc, indexingMap);
          Value genericOperand = generic->getOperand(operandIdx);
          Value result = linalgGeneric->getResult(outputIdx);
          Value storeResult =
              rewriter
                  .create<d2m::RemoteStoreOp>(loc, genericOperand.getType(),
                                              genericOperand, indices, result)
                  .getResult();
          storeResults.push_back(storeResult);
        }

        rewriter.create<d2m::YieldOp>(loc, storeResults);
      }
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);

    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op->getResult(0).getType()));
    return llvm::success();
  }

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, ConcreteOp op, std::size_t arity,
                     std::size_t rank, bool hasScaler) {
    mlir::ArrayAttr dimArg = getDimArg(op);

    mlir::AffineExpr zero =
        mlir::getAffineConstantExpr(0, builder.getContext());

    mlir::MutableAffineMap accumulator(builder.getMultiDimIdentityMap(rank));
    forAllDims(rank, dimArg, [&](std::size_t index, bool dropped) {
      if (dropped) {
        accumulator.setResult(index, zero);
      }
    });
    // Final two (or one, if no scaler) maps are special: the scaler is
    // broadcast from a single tile via a zeros map, and the output uses the
    // accumulator map. All earlier inputs use the identity map.
    const std::size_t numIdentity = arity - 1 - (hasScaler ? 1 : 0);
    SmallVector<mlir::AffineMap> maps(numIdentity,
                                      builder.getMultiDimIdentityMap(rank));
    if (hasScaler) {
      std::array<mlir::AffineExpr, 2> zeros{zero, zero};
      maps.emplace_back(mlir::AffineMap::get(/* dimCount */ rank,
                                             /* symbolCount */ 0, zeros,
                                             builder.getContext()));
    }
    maps.emplace_back(accumulator.getAffineMap());

    return maps;
  }

  static SmallVector<mlir::Attribute>
  getIteratorTypesArray(mlir::OpBuilder &builder, ConcreteOp op,
                        std::size_t rank) {
    mlir::ArrayAttr dimArg = getDimArg(op);

    auto parallel = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Parallel);
    auto reduction = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Reduction);

    SmallVector<mlir::Attribute> iterators(rank, parallel);
    forAllDims(rank, dimArg, [&](std::size_t index, bool dropped) {
      if (dropped) {
        iterators[index] = reduction;
      }
    });
    return iterators;
  }

  // For mean reduction, the scaler must encode 1/N where N is the product of
  // the reduction dimension sizes. For all other reductions, the scaler is 1.0.
  static double getScaleValue(ConcreteOp op) {
    if constexpr (std::is_same_v<ConcreteOp, ttir::MeanOp>) {
      auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
      ArrayRef<int64_t> shape = inputType.getShape();
      mlir::ArrayAttr dimArg = getDimArg(op);
      int64_t reductionSize = 1;
      for (auto dimAttr : dimArg) {
        int64_t dim = mlir::cast<IntegerAttr>(dimAttr).getInt();
        std::size_t n = normalizeReductionDimIndex(dim, shape.size());
        reductionSize *= shape[n];
      }
      return 1.0 / static_cast<double>(reductionSize);
    }
    return 1.0;
  }

  /// Map `dim_arg` to a `ReduceDim` over the last two (tile C/R) dimensions.
  /// Outer-only reductions should have matched
  /// `D2MNamedAccumReductionRewriter`.
  static d2m::ReduceDim dimArgAsReduceDim(ConcreteOp op, std::size_t rank) {
    SmallVector<bool> dims(rank, false);
    forAllDims(rank, getDimArg(op),
               [&](std::size_t index, bool dropped) { dims[index] = dropped; });

    bool reduceSecondToLast = dims[rank - 2]; // "C" in tile terminology
    bool reduceLast = dims[rank - 1];         // "R" in tile terminology

    if (reduceSecondToLast && reduceLast) {
      return d2m::ReduceDim::RC;
    }
    if (reduceSecondToLast) {
      return d2m::ReduceDim::C;
    }
    if (reduceLast) {
      return d2m::ReduceDim::R;
    }
    llvm_unreachable(
        "D2MNamedTileReduceRewriter: dim_arg does not reduce tile "
        "C or R (last two physical iterator dims). For reductions over outer "
        "logical dims, D2MNamedAccumReductionRewriter must match "
        "before this pattern since it has higher benefit.");
  }

  static mlir::ArrayAttr getDimArg(ConcreteOp op) { return *op.getDimArg(); }

  template <typename F>
  static void forAllDims(std::size_t rank, mlir::ArrayAttr dimArg, F &&fn) {
    SmallVector<bool> dims(rank, false);
    for (auto reduceDim : dimArg) {
      int64_t dim = mlir::cast<IntegerAttr>(reduceDim).getInt();
      dims[normalizeReductionDimIndex(dim, rank)] = true;
    }
    for (std::size_t d = 0; d < rank; ++d) {
      std::forward<F>(fn)(d, dims[d]);
    }
  }
};
} // namespace

// ----------------------------------------------------------------------------
//
// D2MInnerMinDecompositionRewriter: rewrites inner-dim `ttir.min` as
// `f(max(f(x)))` where `f` is an order-reversing involution. No
// tile_reduce_min kernel exists; outer min uses `tile_minimum` via the
// accumulation path. Floats use `neg`; integers use `bitwise_not` since
// `~INT_MIN == INT_MAX` keeps it order-reversing at INT_MIN (unlike `neg`).
namespace {
class D2MInnerMinDecompositionRewriter final
    : public mlir::OpConversionPattern<ttir::MinOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MinOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (isOuterReduction(op)) {
      return failure();
    }
    auto inputType =
        mlir::cast<mlir::RankedTensorType>(op.getInput().getType());
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op.getResult().getType());
    mlir::Location loc = op.getLoc();
    const bool isInt = mlir::isa<mlir::IntegerType>(inputType.getElementType());

    auto invert = [&](mlir::Type type, mlir::Value value) -> mlir::Value {
      if (isInt) {
        return rewriter.create<ttir::BitwiseNotOp>(loc, type, value)
            .getResult();
      }
      return rewriter.create<ttir::NegOp>(loc, type, value).getResult();
    };

    mlir::Value invertedInput = invert(inputType, op.getInput());
    auto maxOp =
        rewriter.create<ttir::MaxOp>(loc, resultType, invertedInput,
                                     op.getKeepDimAttr(), op.getDimArgAttr());
    if (isInt) {
      rewriter.replaceOpWithNewOp<ttir::BitwiseNotOp>(op, resultType,
                                                      maxOp.getResult());
    } else {
      rewriter.replaceOpWithNewOp<ttir::NegOp>(op, resultType,
                                               maxOp.getResult());
    }
    return mlir::success();
  }
};
} // namespace

// ----------------------------------------------------------------------------
//
// Rewrite a MatmulOp into either a D2M TileMatmulOp or TileMatmulBlockOp
// (selected by TileOp template).
namespace {
template <typename TileOp>
class D2MMatmulRewriter final
    : public mlir::OpConversionPattern<ttir::MatmulOp>,
      D2MNamedRewriterCommon {

  using ConcreteOp = ttir::MatmulOp;
  static_assert(std::is_same_v<TileOp, d2m::TileMatmulBlockOp> ||
                    std::is_same_v<TileOp, d2m::TileMatmulOp>,
                "Unsupported Matmul TileOp");

public:
  D2MMatmulRewriter(const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
                    ttcore::MemorySpace defaultInputMemSpace,
                    ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
                    bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (op.getTransposeA()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected transpose_a to not be set");
    }
    checkPreconditions(op);

    const bool transposeB = op.getTransposeB();

    mlir::Location loc = op->getLoc();

    auto origOutputs =
        createDpsOutputs(loc, rewriter, {op.getResult().getType()});
    auto origInputs = adaptor.getOperands();

    // For higher-rank matmuls (rank > 2), don't collapse batch dimensions.
    // This preserves the ND structure for proper batch dimension handling.
    // Note: checkPreconditions() guarantees both inputs have the same rank.
    auto inputTensorType =
        mlir::cast<RankedTensorType>(origInputs[0].getType());
    bool noCollapse = (inputTensorType.getRank() > 2);

    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {origInputs, origOutputs}, /*tiled*/ true, noCollapse);

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    // Device layout doubles the rank (logical dimensions + device grid
    // dimensions).
    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;

    // transpose_b is handled via indexing maps and the transpose_b flag on
    // tile_matmul_block.
    SmallVector<mlir::AffineMap> indexingMaps = getAffineMapsArray(
        rewriter, numOperands, physicalRank, /*transposeB=*/transposeB);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, physicalRank);

    // Create 'd2m.generic' accepting 'op's operands.
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    // Create one bb in 'generic''s region and set its arguments.
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block'.
      {
        auto [inputIndices, mcastGridDims] = createInputIndicesAndMcastGridDims(
            rewriter, loc, generic, enableMulticastInference);
        auto blockArgsVec = createBlockArguments(
            rewriter, block, loc, TypeRange(inputs), TypeRange(outputs),
            generic, inputIndices, mcastGridDims);
        ArrayRef<Value> blockArgs(blockArgsVec);

        // Delegate next level of nesting to a "block" op.

        if constexpr (std::is_same_v<d2m::TileMatmulBlockOp, TileOp>) {
          rewriter.create<d2m::TileMatmulBlockOp>(
              loc, /* a */ blockArgs[0], /* b */ blockArgs[1],
              /* output */ blockArgs[2], /* transpose_b */ transposeB);

          // Insert remote_store operations for each output before yield
          SmallVector<Value> storeResults;
          for (size_t outputIdx = 0; outputIdx < numOutputs; ++outputIdx) {
            size_t operandIdx = numInputs + outputIdx;
            AffineMap indexingMap = generic.getIndexingMap(operandIdx);
            SmallVector<Value> indices =
                d2m::utils::buildGridIndices(rewriter, loc, indexingMap);
            Value genericOperand = generic->getOperand(operandIdx);
            Value result = blockArgs[numInputs + outputIdx];
            Value storeResult =
                rewriter
                    .create<d2m::RemoteStoreOp>(loc, genericOperand.getType(),
                                                genericOperand, indices, result)
                    .getResult();
            storeResults.push_back(storeResult);
          }

          // In pure tensor semantics, explicitly yield the output shard.
          rewriter.create<d2m::YieldOp>(loc, storeResults);

        } else if constexpr (std::is_same_v<d2m::TileMatmulOp, TileOp>) {

          static constexpr std::size_t tileOpNumOutputs = 1;

          // The generic-level indexing maps already encode the transposed
          // access pattern onto B; the linalg body then consumes per-iteration
          // tiles using the same iteration-space layout.
          SmallVector<mlir::AffineMap> linalgIndexingMaps = getAffineMapsArray(
              rewriter, numOperands, physicalRank, /*transposeB=*/transposeB);
          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
              iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

          auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
              loc,
              /* result tensor types */
              llvm::to_vector(
                  mlir::ValueRange(blockArgs.take_back(numOutputs)).getTypes()),
              /* inputs */ blockArgs.take_front(numInputs),
              /* outputs */ blockArgs.take_back(numOutputs), linalgIndexingMaps,
              linalgIteratorTypes,
              [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                  mlir::ValueRange bbArgs) {
                mlir::Value yield = bbBuilder.create<d2m::TileMatmulOp>(
                    loc, /* resultTypes */
                    bbArgs.take_back(tileOpNumOutputs).getTypes()[0],
                    /* a */ bbArgs[0], /* b */ bbArgs[1], /* c */ bbArgs[2]);

                bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
              });

          // Insert remote_store operations for each output before yield
          SmallVector<Value> storeResults;
          for (size_t outputIdx = 0; outputIdx < numOutputs; ++outputIdx) {
            size_t operandIdx = numInputs + outputIdx;
            AffineMap indexingMap = generic.getIndexingMap(operandIdx);
            SmallVector<Value> indices =
                d2m::utils::buildGridIndices(rewriter, loc, indexingMap);
            Value genericOperand = generic->getOperand(operandIdx);
            Value result = linalgGeneric->getResult(outputIdx);
            Value storeResult =
                rewriter
                    .create<d2m::RemoteStoreOp>(loc, genericOperand.getType(),
                                                genericOperand, indices, result)
                    .getResult();
            storeResults.push_back(storeResult);
          }

          rewriter.create<d2m::YieldOp>(loc, storeResults);
        }
      }
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);

    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op->getResult(0).getType()));
    return llvm::success();
  }

  static void checkPreconditions(ConcreteOp op) {
    assert(!op.getTransposeA() && "expected transpose_a to not be set");

    auto aType = mlir::cast<RankedTensorType>(op.getA().getType());
    auto bType = mlir::cast<RankedTensorType>(op.getB().getType());
    int64_t aRank = aType.getRank();
    int64_t bRank = bType.getRank();

    assert(aRank >= 2 && bRank >= 2 && "matmul operands must have rank >= 2");
    assert(aRank == bRank &&
           "matmul operands must have same rank for batched operations");
  }

  /// Creates affine maps for matmul operation.
  ///
  /// For 2D matmuls:
  ///   LHS: (M, K), RHS: (K, N), OUT: (M, N)
  ///   Iteration space: (M, N, K) where K is the contraction dimension
  ///
  /// For ND matmuls (N > 2):
  ///   LHS: (batch..., M, K), RHS: (batch..., K, N), OUT: (batch..., M, N)
  ///   Iteration space: (batch..., M, N, K)
  ///   - Batch dimensions are identity-mapped across all operands
  ///   - Last two logical dimensions follow standard matmul pattern
  ///
  /// When `transposeB` is true the RHS last two dims are swapped to (N, K) so
  /// that the physical RHS tensor of shape (..., N, K) is indexed correctly.
  /// The kernel is then asked to transpose B during compute via the tile_matmul
  /// block transpose flag.
  ///
  /// \param builder OpBuilder for creating affine expressions
  /// \param arity Number of operands (must be 3: LHS, RHS, OUT)
  /// \param rank Physical rank of the matmul operation (logical tensor rank)
  /// \param transposeB Whether the RHS operand is transposed
  /// \return Vector of affine maps for [LHS, RHS, OUT]
  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, std::size_t arity,
                     std::size_t rank, bool transposeB) {
    assert(arity == 3 && "expected 3 operands");
    assert(rank >= 2 && "matmul operation must have rank >= 2");
    mlir::MLIRContext *ctx = builder.getContext();

    // For higher ranks: batch dims are identity, last 2 dims follow matmul
    // pattern Matmul semantics: [...batch..., M, K] x [...batch..., K, N] ->
    // [...batch..., M, N]
    SmallVector<mlir::AffineExpr> lhsExprs, rhsExprs, outExprs;

    // Iteration space has rank+1 dimensions: (batch..., M, N, K)
    // where batch dimensions are [0, rank-2), M is rank-2, N is rank-1, K is
    // rank

    // Batch dimensions: identity mapping for all three operands
    for (unsigned i = 0; i < rank - 2; ++i) {
      lhsExprs.push_back(builder.getAffineDimExpr(i));
      rhsExprs.push_back(builder.getAffineDimExpr(i));
      outExprs.push_back(builder.getAffineDimExpr(i));
    }

    // LHS last two dimensions: [..., M, K]
    lhsExprs.push_back(builder.getAffineDimExpr(rank - 2)); // M (rows)
    lhsExprs.push_back(builder.getAffineDimExpr(rank));     // K (contraction)

    // RHS last two dimensions: [..., K, N], or [..., N, K] when transposed.
    if (transposeB) {
      rhsExprs.push_back(builder.getAffineDimExpr(rank - 1)); // N (rows)
      rhsExprs.push_back(builder.getAffineDimExpr(rank));     // K (columns)
    } else {
      rhsExprs.push_back(builder.getAffineDimExpr(rank));     // K (contraction)
      rhsExprs.push_back(builder.getAffineDimExpr(rank - 1)); // N (columns)
    }

    // OUT last two dimensions: [..., M, N]
    outExprs.push_back(builder.getAffineDimExpr(rank - 2)); // M (rows)
    outExprs.push_back(builder.getAffineDimExpr(rank - 1)); // N (columns)

    // Return affine maps with rank+1 total dimensions (batch + M + N + K)
    return SmallVector<mlir::AffineMap>{
        AffineMap::get(rank + 1, 0, lhsExprs, ctx),
        AffineMap::get(rank + 1, 0, rhsExprs, ctx),
        AffineMap::get(rank + 1, 0, outExprs, ctx)};
  }

  /// Creates iterator type attributes for matmul operation.
  ///
  /// The iteration space for an N-dimensional matmul has N+1 dimensions:
  ///   - Batch dimensions [0, N-2): parallel
  ///   - M dimension (N-2): parallel (result rows)
  ///   - N dimension (N-1): parallel (result columns)
  ///   - K dimension (N): reduction (contraction)
  ///
  /// \param builder OpBuilder for creating attributes
  /// \param rank Physical rank of the matmul operation (logical tensor rank)
  /// \return Vector of iterator type attributes
  static SmallVector<mlir::Attribute>
  getIteratorTypesArray(mlir::OpBuilder &builder, std::size_t rank) {
    assert(rank >= 2 && "matmul operation must have rank >= 2");
    auto parallel = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Parallel);
    auto reduction = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Reduction);

    SmallVector<mlir::Attribute> result;

    // All batch dimensions and result dimensions (M, N) are parallel
    for (unsigned i = 0; i < rank; ++i) {
      result.push_back(parallel);
    }

    // K (contraction dimension) is reduction
    result.push_back(reduction);

    return result;
  }

  static mlir::AffineMap makeAffineMap(mlir::MLIRContext *ctx,
                                       std::array<unsigned, 2> targets) {
    return mlir::AffineMap::getMultiDimMapWithTargets(3, targets, ctx);
  }
};
} // namespace

// ----------------------------------------------------------------------------
//
// Lower PermuteOp into a D2M ViewLayoutOp (to reblock into new tile-level
// shape) + GenericOp (to transpose individual tiles).
namespace {
class D2MPermuteRewriter final
    : public mlir::OpConversionPattern<ttir::PermuteOp>,
      D2MNamedRewriterCommon {

  using ConcreteOp = ttir::PermuteOp;

public:
  D2MPermuteRewriter(const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
                     ttcore::MemorySpace defaultInputMemSpace,
                     ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
                     bool /*collapseTensors*/, bool enableMulticastInference)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx, /*benefit=*/2),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, /*collapseTensors*/ false,
                               enableMulticastInference) {}

  LogicalResult
  matchAndRewrite(ttir::PermuteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto permutation = op.getPermutation();

    const int64_t permuteSize = static_cast<int64_t>(permutation.size());
    assert(permuteSize >= 2 && "Permute size must be >= 2");
    // Check if this is a pure inner permute (only last two dims swapped,
    // all outer dims are identity).
    const bool innerDimsSwapped =
        (permutation[permuteSize - 2] == permuteSize - 1 &&
         permutation[permuteSize - 1] == permuteSize - 2);
    bool outerDimsIdentity = true;
    for (int64_t i = 0; i < permuteSize - 2; ++i) {
      if (permutation[i] != i) {
        outerDimsIdentity = false;
        break;
      }
    }
    const bool isInnerPermute = innerDimsSwapped && outerDimsIdentity;
    if (isInnerPermute) {
      return permuteInnerDims(op, adaptor, rewriter);
    }
    assert(!(innerDimsSwapped && !outerDimsIdentity) &&
           "Complex permutes (both inner and outer permutations) are not "
           "supported.");
    // Unhandled conversion case.
    return failure();
  }

  // Handler for permutation of inner dims (i.e. transpose).
  LogicalResult
  permuteInnerDims(ttir::PermuteOp op, typename ConcreteOp::Adaptor adaptor,
                   mlir::ConversionPatternRewriter &rewriter) const {
    auto permutation = op.getPermutation();
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto origInputs = adaptor.getOperands();
    auto origOutputs =
        createDpsOutputs(loc, rewriter, {op.getResult().getType()});

    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {origInputs, origOutputs},
                                   /*tiled*/ true);

    const auto inputTensorType =
        mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
    const ArrayRef<int64_t> inputShape = inputTensorType.getShape();
    const unsigned deviceRank = static_cast<unsigned>(inputShape.size());
    auto inputLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(inputTensorType.getEncoding());

    // Compute permutation for all relevant attributes.
    auto permuted = computePermutation(
        rewriter, permutation, inputShape, deviceRank,
        inputLayout.getLogicalShape(), inputLayout.getDimAlignments());

    // Create the result layout by composing with input layout.
    auto resultLayout = ttcore::MetalLayoutAttr::get(
        ctx, permuted.logicalShape, inputLayout.getMemorySpace(),
        inputLayout.getMemoryLayout(), inputLayout.getCollapsedIntervals(),
        permuted.dimAlignments);

    auto viewType = mlir::RankedTensorType::get(
        permuted.physicalShape, inputTensorType.getElementType(), resultLayout);

    // For inner permute, we need a view to express the reblocking.
    // The allocator will later decide whether to insert a CB allocation
    // for the consuming GenericOp.
    auto view = rewriter.create<d2m::ViewLayoutOp>(loc, viewType, inputs[0],
                                                   permuted.transposeMap,
                                                   /*reinterpretLayout=*/false);
    inputs[0] = view.getResult();
    unsigned logicalRank = deviceRank / 2;
    // For inner permute, we alse need a GenericOp to transpose each individual
    // tile.

    // Capture values explicitly to avoid C++20 structured binding capture issue
    Value inputOperand = inputs[0];
    Value outputOperand = outputs[0];

    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        [&, inputOperand, outputOperand](OpBuilder &builder, Location bodyLoc,
                                         ValueRange blockArgs) {
          assert(blockArgs.size() == 2);
          auto identityMap = builder.getMultiDimIdentityMap(logicalRank);
          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes(
              logicalRank, mlir::utils::IteratorType::parallel);

          // blockArgs are tensor.empty results with shard shapes.
          auto inputShardType = blockArgs[0].getType();

          // Create remote_load for input using the tensor.empty as buffer.
          AffineMap inputIndexingMap = identityMap;
          SmallVector<Value> inputIndices =
              d2m::utils::buildGridIndices(builder, bodyLoc, inputIndexingMap);
          Value inputBuffer = blockArgs[0];
          Value input = builder
                            .create<d2m::RemoteLoadOp>(
                                bodyLoc, inputShardType, inputBuffer,
                                inputOperand, inputIndices)
                            .getResult();

          // Use the output tensor.empty directly.
          Value output = blockArgs[1];

          auto linalgGeneric = builder.create<mlir::linalg::GenericOp>(
              bodyLoc, output.getType(), input, output,
              SmallVector<mlir::AffineMap>{identityMap, identityMap},
              linalgIteratorTypes,
              [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                  mlir::ValueRange bbArgs) {
                mlir::Value yield = bbBuilder.create<d2m::TileTransposeOp>(
                    bbLoc, bbArgs.take_back(1).getTypes(),
                    bbArgs.take_front(1));
                bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
              });

          // Insert remote_store for output before yield
          AffineMap outputIndexingMap = identityMap;
          SmallVector<Value> outputIndices =
              d2m::utils::buildGridIndices(builder, bodyLoc, outputIndexingMap);
          Value result = linalgGeneric->getResult(0);
          Value storeResult = builder
                                  .create<d2m::RemoteStoreOp>(
                                      bodyLoc, outputOperand.getType(),
                                      outputOperand, outputIndices, result)
                                  .getResult();

          builder.create<d2m::YieldOp>(bodyLoc, storeResult);
        });

    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op->getResult(0).getType()));
    return success();
  }

private:
  // Apply permutation mapping to affine map, physical shape, logical shape, and
  // dimension alignments to get permuted versions.
  struct PermutationResult {
    AffineMap transposeMap;
    SmallVector<int64_t> physicalShape;
    SmallVector<int64_t> logicalShape;
    SmallVector<int64_t> dimAlignments;
  };

  static PermutationResult
  computePermutation(mlir::ConversionPatternRewriter &rewriter,
                     ArrayRef<int64_t> permutation,
                     ArrayRef<int64_t> inputPhysicalShape, unsigned deviceRank,
                     ArrayRef<int64_t> inputLogicalShape,
                     ArrayRef<int64_t> inputDimAlignments) {

    unsigned logicalRank = deviceRank / 2;
    assert(logicalRank == permutation.size());
    assert(inputLogicalShape.size() == permutation.size());
    assert(inputDimAlignments.size() == permutation.size());

    SmallVector<AffineExpr> results(deviceRank);
    SmallVector<int64_t> resultPhysicalShape(deviceRank);
    SmallVector<int64_t> resultLogicalShape(logicalRank);
    SmallVector<int64_t> resultDimAlignments(logicalRank);

    for (auto [dstIdx, srcIdx] : llvm::enumerate(permutation)) {
      // Permute grid mapping.
      results[dstIdx] = rewriter.getAffineDimExpr(srcIdx);
      // Permute shard mapping.
      results[logicalRank + dstIdx] =
          rewriter.getAffineDimExpr(logicalRank + srcIdx);

      // Permute grid shape.
      resultPhysicalShape[dstIdx] = inputPhysicalShape[srcIdx];
      // Permute shard shape.
      resultPhysicalShape[dstIdx + logicalRank] =
          inputPhysicalShape[srcIdx + logicalRank];

      // Permute logical shape and dimension alignments.
      resultLogicalShape[dstIdx] = inputLogicalShape[srcIdx];
      resultDimAlignments[dstIdx] = inputDimAlignments[srcIdx];
    }

    AffineMap transposeMap =
        AffineMap::get(deviceRank, 0, results, rewriter.getContext());
    return {transposeMap, resultPhysicalShape, resultLogicalShape,
            resultDimAlignments};
  }
};
} // namespace

namespace {
class D2MConcatRewriter final
    : public mlir::OpConversionPattern<ttir::ConcatOp>,
      D2MNamedRewriterCommon {

public:
  D2MConcatRewriter(const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
                    ttcore::MemorySpace defaultInputMemSpace,
                    ttcore::MemorySpace defaultOutputMemSpace,
                    bool /*ttnnMode*/, bool /*collapseTensors*/,
                    bool enableMulticastInference)
      : OpConversionPattern<ttir::ConcatOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               /*ttnnMode=*/false, /*collapseTensors=*/false,
                               enableMulticastInference) {}

  LogicalResult
  matchAndRewrite(ttir::ConcatOp op, ttir::ConcatOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto outType = mlir::cast<RankedTensorType>(op.getResult().getType());

    const int64_t rank = outType.getRank();
    TT_assert(rank >= 1);

    int32_t dim = op.getDim();
    if (dim < 0) {
      dim += rank;
    }

    // Check if we should do sub-tile H/W concat in the row-major layout.
    bool concatRowMajor = false;
    bool transposeRowMajor = false;
    const int32_t alignToElements =
        d2m::utils::getNocElementAlignmentL1(op, outType);

    if (dim >= rank - 2) {
      const int64_t tileSize =
          ttcore::TileType::getDefaultShape()[dim - rank + 2];
      const int nInputs = static_cast<int>(op.getInputs().size());
      for (int i = 0; i < nInputs; i++) {
        auto inType = mlir::cast<RankedTensorType>(op.getInputs()[i].getType());
        const int64_t dimSize = inType.getShape()[dim];

        // If any (other than the last) input has padding on the concat dim, do
        // row-major concat.
        if ((i != nInputs - 1) && (dimSize % tileSize != 0)) {
          concatRowMajor = true;
        }

        // For a row-major width concat, if at least one (including the last)
        // row's size violates the NoC constraints, use the
        // transpose-concat-transpose trick.
        if (rank >= 2 && concatRowMajor && (dim == rank - 1) &&
            (dimSize % alignToElements != 0)) {
          transposeRowMajor = true;
          break;
        }
      }
    }

    auto loc = op.getLoc();

    // Logical sizes on the concat dim is required for DMA generation in the
    // row-major case, record them in the CompositeView before the bufferization
    // pass discards the logical shapes info.
    DenseI64ArrayAttr logicalSizesAttr = nullptr;
    if (concatRowMajor) {
      SmallVector<int64_t> sizes;
      for (auto input : op.getInputs()) {
        auto inType = mlir::cast<RankedTensorType>(input.getType());
        sizes.push_back(inType.getShape()[dim]);
      }
      logicalSizesAttr = rewriter.getDenseI64ArrayAttr(sizes);
    }

    // Height <-> Width transpose indices.
    SmallVector<int64_t> hwTransposeIdx(rank);
    std::iota(hwTransposeIdx.begin(), hwTransposeIdx.end(), 0);
    if (rank >= 2) {
      std::swap(hwTransposeIdx[rank - 1], hwTransposeIdx[rank - 2]);
    }

    SmallVector<Value> effectiveInputs(adaptor.getOperands().begin(),
                                       adaptor.getOperands().end());
    auto effectiveOutputType = outType;

    if (transposeRowMajor) {
      TT_assert(dim == rank - 1);
      dim = rank - 2;

      effectiveInputs.clear();
      for (auto input : adaptor.getOperands()) {
        auto inType = mlir::cast<RankedTensorType>(input.getType());

        auto transposedInShape =
            ttmlir::utils::applyPermutation(inType.getShape(), hwTransposeIdx);
        auto transposedInType = RankedTensorType::get(
            transposedInShape, inType.getElementType(), inType.getEncoding());

        auto preTranspose = rewriter.create<ttir::PermuteOp>(
            loc, transposedInType, input, hwTransposeIdx);

        effectiveInputs.push_back(preTranspose.getResult());
      }

      auto transposedOutShape =
          ttmlir::utils::applyPermutation(outType.getShape(), hwTransposeIdx);
      effectiveOutputType = RankedTensorType::get(
          transposedOutShape, outType.getElementType(), outType.getEncoding());
    }

    TT_assert(effectiveInputs.size() > 1u);

    auto origOutputs = createDpsOutputs(loc, rewriter, {effectiveOutputType});

    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {effectiveInputs, origOutputs},
        /*tiled=*/!concatRowMajor, /*noCollapse=*/true);

    auto compositeView = rewriter.create<d2m::CompositeViewOp>(
        loc, outputs[0].getType(), inputs, dim, logicalSizesAttr);

    auto result = unLayoutResult(rewriter, compositeView->getResult(0),
                                 effectiveOutputType)
                      ->getResult(0);

    if (transposeRowMajor) {
      auto postTranspose = rewriter.create<ttir::PermuteOp>(
          loc, outType, result, hwTransposeIdx);
      result = postTranspose->getResult(0);
    }

    rewriter.replaceOp(op, result);

    return success();
  }
};
} // namespace

// Conversion for ttir.to_layout -> d2m.to_layout.
class D2MToLayoutOpRewriter : public D2MNamedRewriterCommon,
                              public OpConversionPattern<ttir::ToLayoutOp> {
public:
  D2MToLayoutOpRewriter(const TypeConverter &typeConverter,
                        MLIRContext *context, bool ttnnMode)
      // default values for memory spaces, collapseTensors,
      // enableMulticastInference. Only ttnnMode is used.
      : D2MNamedRewriterCommon(ttcore::MemorySpace::DeviceDRAM,
                               ttcore::MemorySpace::DeviceDRAM, ttnnMode, false,
                               false),
        OpConversionPattern<ttir::ToLayoutOp>(typeConverter, context) {}

  using D2MNamedRewriterCommon::getMetalTensorFromTTNNTensor;

  LogicalResult
  matchAndRewrite(ttir::ToLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    if (!ttnnMode) {
      // When ttnnMode is disabled, we can simply convert ttir.to_layout
      // directly to d2m.to_layout.
      Value empty = rewriter.create<d2m::EmptyOp>(
          op.getLoc(), outType.getShape(), outType.getElementType(),
          outType.getEncoding());
      auto newOp = rewriter.create<d2m::ToLayoutOp>(op.getLoc(),
                                                    adaptor.getInput(), empty);
      rewriter.replaceOp(op, newOp.getResult(0));
      return success();
    }
    return rewriteIfTTNNModeEnabled(op, adaptor, rewriter);
  }

private:
  // Compute the virtualGridInverseMapping (inverse) and
  // virtualGridForwardMapping (forward) for a TTNN legacy layout's shard
  // strategy.  Returns both maps as a pair, or nullopt if the layout does not
  // imply a virtual grid.
  static std::optional<std::pair<AffineMap, AffineMap>>
  computeLegacyVirtualGridMaps(MLIRContext *ctx,
                               ttnn::TTNNLayoutAttr ttnnLayout) {
    auto memLayout = ttnnLayout.getMemLayout().getValue();
    bool legacyWithVirtualGrid =
        memLayout == ttnn::TensorMemoryLayout::HeightSharded ||
        memLayout == ttnn::TensorMemoryLayout::WidthSharded;
    if (!legacyWithVirtualGrid) {
      return std::nullopt;
    }

    llvm::SmallVector<int64_t> ttnnGridShape(ttnnLayout.getGridShape());
    llvm::SmallVector<int64_t> virtualGrid;
    if (memLayout == ttnn::TensorMemoryLayout::HeightSharded) {
      virtualGrid = {ttnnGridShape[0] * ttnnGridShape[1], 1};
    } else {
      virtualGrid = {1, ttnnGridShape[0] * ttnnGridShape[1]};
    }

    auto [forwardMap, inverseMap] =
        ttmlir::d2m::utils::grids::createCoreVirtMaps(ctx, virtualGrid,
                                                      ttnnGridShape);
    return std::make_pair(forwardMap, inverseMap);
  }

  // Set both virtualGridInverseMapping (inverse) and virtualGridForwardMapping
  // (forward) on a TTNNMetalLayoutCastOp when the TTNN layout implies a
  // virtual grid.
  static void propagateVGMToCastOp(MLIRContext *ctx,
                                   ttir::TTNNMetalLayoutCastOp castOp,
                                   Attribute encoding) {
    auto ttnnLayout = mlir::dyn_cast_if_present<ttnn::TTNNLayoutAttr>(encoding);
    if (!ttnnLayout) {
      return;
    }
    if (auto maps = computeLegacyVirtualGridMaps(ctx, ttnnLayout)) {
      auto [forwardMap, inverseMap] = *maps;
      castOp.setVirtualGridInverseMappingAttr(AffineMapAttr::get(inverseMap));
      castOp.setVirtualGridForwardMappingAttr(AffineMapAttr::get(forwardMap));
    }
  }

  LogicalResult
  rewriteIfTTNNModeEnabled(ttir::ToLayoutOp op, OpAdaptor adaptor,
                           ConversionPatternRewriter &rewriter) const {
    /* Lowers ttir.to_layout with TTNN tensor operands when ttnnMode is enabled,
       to d2m.to_layout with Metal tensor operands. This is done by
       auto-inserting casts to/from tensors with MetalLayoutAttr, which
       downstream passes support. The conversion flow is:
       1. Cast TTNN input to Metal layout
       2. Create d2m.empty with TTNN output layout
       3. Cast the d2m.empty from TTNN to Metal layout
       4. Create d2m.to_layout with Metal input cast and d2m.empty cast
       5. Cast result back to TTNN layout
    */
    auto outType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    bool outputIsTTNN =
        mlir::isa_and_nonnull<ttnn::TTNNLayoutAttr>(outType.getEncoding());
    TT_assertv(
        outputIsTTNN,
        "expected output type to have TTNN layout when ttnnMode is enabled");
    // TTNN output handling.
    // Convert input to Metal layout if needed.
    Value metalInput = adaptor.getInput();
    auto inputType = mlir::cast<RankedTensorType>(adaptor.getInput().getType());
    if (mlir::isa_and_nonnull<ttnn::TTNNLayoutAttr>(inputType.getEncoding())) {
      auto inputMetalType =
          getMetalTensorFromTTNNTensor(rewriter, adaptor.getInput());
      auto inputCast = rewriter.create<ttir::TTNNMetalLayoutCastOp>(
          op.getLoc(), inputMetalType, adaptor.getInput());
      propagateVGMToCastOp(rewriter.getContext(), inputCast,
                           inputType.getEncoding());
      metalInput = inputCast.getResult();
    }
    auto outputMetalType =
        getMetalTensorFromTTNNTensor(rewriter, op.getOutput());
    // Create d2m.empty for TTNN layout.
    Value metalEmpty = rewriter.create<d2m::EmptyOp>(
        op.getLoc(), outType.getShape(), outType.getElementType(),
        outType.getEncoding());
    // Cast TTNN empty to Metal layout.
    auto metalCast = rewriter.create<ttir::TTNNMetalLayoutCastOp>(
        op.getLoc(), outputMetalType, metalEmpty);
    propagateVGMToCastOp(rewriter.getContext(), metalCast,
                         outType.getEncoding());
    // Create d2m.to_layout with Metal types.
    auto metalToLayout =
        rewriter.create<d2m::ToLayoutOp>(op.getLoc(), metalInput, metalCast);
    // Cast back to TTNN.
    auto ttnnResult = rewriter.create<ttir::TTNNMetalLayoutCastOp>(
        op.getLoc(), outType, metalToLayout.getResult(0));
    rewriter.replaceOp(op, ttnnResult.getResult());
    return success();
  }
};

// Simple conversion for ttir.empty -> d2m.empty.
class D2MEmptyOpRewriter : public OpConversionPattern<ttir::EmptyOp> {
  using OpConversionPattern<ttir::EmptyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::EmptyOp op, ttir::EmptyOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    auto tensorType = cast<RankedTensorType>(resultType);
    bool outputIsTTNN =
        mlir::isa_and_nonnull<ttnn::TTNNLayoutAttr>(resultType.getEncoding());

    if (outputIsTTNN) {
      // If a user of a ttir.empty is a ttir.to_layout, erase the ttir.empty
      // instead of converting to d2m.empty. The D2MToLayoutOpRewriter creates a
      // d2m.empty with the d2m.to_layout as a user, so this empty op is not
      // needed.
      for (Operation *user : op->getUsers()) {
        if (auto toLayoutOp = dyn_cast<ttir::ToLayoutOp>(user)) {
          if (toLayoutOp.getOutput() == op.getResult()) {
            rewriter.eraseOp(op);
            return success();
          }
        }
      }
    }

    // Create d2m.empty with same shape and element type.
    auto d2mEmpty = rewriter.create<d2m::EmptyOp>(
        op.getLoc(), tensorType.getShape(), tensorType.getElementType(),
        tensorType.getEncoding());

    rewriter.replaceOp(op, d2mEmpty.getResult());
    return success();
  }
};

/// Lowers `ttir.full`, `ttir.zeros`, and `ttir.ones` through
/// `lowerRankedTensorFillViaGeneric` (`d2m.tile_fill` + remote_store).
template <typename ConcreteOp>
class D2MConstantFillOpRewriter : public OpConversionPattern<ConcreteOp>,
                                  D2MNamedRewriterCommon {
  static_assert(std::is_same_v<ConcreteOp, ttir::FullOp> ||
                std::is_same_v<ConcreteOp, ttir::ZerosOp> ||
                std::is_same_v<ConcreteOp, ttir::OnesOp>);

  static mlir::Attribute getFillAttr(ConcreteOp op,
                                     RankedTensorType resultType) {
    if constexpr (std::is_same_v<ConcreteOp, ttir::FullOp>) {
      return op.getFillValueAttr();
    }
    Type elemTy = resultType.getElementType();
    constexpr bool kOnes = std::is_same_v<ConcreteOp, ttir::OnesOp>;
    if (auto floatTy = mlir::dyn_cast<mlir::FloatType>(elemTy)) {
      return mlir::FloatAttr::get(floatTy, kOnes ? 1.0 : 0.0);
    }
    if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(elemTy)) {
      return mlir::IntegerAttr::get(intTy, static_cast<int64_t>(kOnes ? 1 : 0));
    }
    return {};
  }

public:
  D2MConstantFillOpRewriter(const TypeConverter &typeConverter,
                            mlir::MLIRContext *ctx,
                            ttcore::MemorySpace defaultInputMemSpace,
                            ttcore::MemorySpace defaultOutputMemSpace,
                            bool ttnnMode, bool collapseTensors,
                            bool enableMulticastInference)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType resultType = op.getResult().getType();
    mlir::Attribute fillAttr = getFillAttr(op, resultType);
    if (!fillAttr) {
      return rewriter.notifyMatchFailure(
          op, "expected float or integer tensor element type (zeros/ones) or "
              "fill value attribute (full)");
    }

    mlir::FailureOr<mlir::Value> filled =
        lowerRankedTensorFillViaGeneric(rewriter, loc, resultType, fillAttr);
    if (mlir::failed(filled)) {
      return rewriter.notifyMatchFailure(
          op, "could not lower constant fill via tile_fill");
    }
    rewriter.replaceOp(op, *filled);
    return success();
  }
};

/// Lowers `ttir.rand` to a `d2m.generic` of `d2m.tile_rand` tiles, mapping
/// `[low, high)` to the kernel's `[from, from + scale)` form. Non-f32 outputs
/// are generated in f32 and then cast via a second `d2m.generic` wrapping
/// `d2m.tile_typecast`.
class D2MRandOpRewriter : public OpConversionPattern<ttir::RandOp>,
                          D2MNamedRewriterCommon {
public:
  D2MRandOpRewriter(const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
                    ttcore::MemorySpace defaultInputMemSpace,
                    ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
                    bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ttir::RandOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

  LogicalResult
  matchAndRewrite(ttir::RandOp op, ttir::RandOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType finalResultType = op.getResult().getType();
    Type finalElemType = finalResultType.getElementType();

    const float low = op.getLow().convertToFloat();
    const float high = op.getHigh().convertToFloat();
    const float scale = high - low;
    if (scale <= 0.0f) {
      return rewriter.notifyMatchFailure(
          op, "ttir.rand requires high > low for TTMetal lowering");
    }

    // `rand_tile` only produces f32 reliably; other dtypes rand in f32 and
    // then get cast via `buildTypecastGeneric` below.
    Type f32Ty = rewriter.getF32Type();
    const bool needsCast = finalElemType != f32Ty;
    RankedTensorType randResultType =
        needsCast ? RankedTensorType::get(finalResultType.getShape(), f32Ty,
                                          finalResultType.getEncoding())
                  : finalResultType;

    auto seedAttr = op.getSeedAttr();
    auto fromAttr = rewriter.getF32FloatAttr(low);
    auto scaleAttr = rewriter.getF32FloatAttr(scale);

    mlir::Value randResult = buildRandGeneric(rewriter, loc, randResultType,
                                              seedAttr, fromAttr, scaleAttr);

    mlir::Value result =
        needsCast
            ? buildTypecastGeneric(rewriter, loc, randResult, finalResultType)
            : randResult;

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  /// Fills a fresh `randResultType` tensor with `d2m.tile_rand` tiles.
  /// Returns the un-layouted logical result.
  mlir::Value buildRandGeneric(ConversionPatternRewriter &rewriter,
                               Location loc, RankedTensorType randResultType,
                               mlir::IntegerAttr seedAttr,
                               mlir::FloatAttr fromAttr,
                               mlir::FloatAttr scaleAttr) const {
    SmallVector<Value> origInputs;
    SmallVector<Value> origOutputs =
        createDpsOutputs(loc, rewriter, {randResultType});
    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {origInputs, origOutputs}, /*tiled*/ true);
    assert(outputs.size() == 1);

    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;
    SmallVector<AffineMap> indexingMaps =
        getIdentityAffineMapsArray(rewriter, 1, physicalRank);
    auto parallel = ttcore::IteratorTypeAttr::get(
        rewriter.getContext(), ttcore::IteratorType::Parallel);
    SmallVector<Attribute> iteratorTypes(physicalRank, parallel);

    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    withD2MGenericRegion(
        rewriter, loc, generic, inputs, outputs,
        [&](mlir::ArrayRef<mlir::Value> blockArgs) -> SmallVector<Value> {
          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
              iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

          auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
              loc,
              llvm::to_vector(
                  mlir::ValueRange(blockArgs.take_back(1)).getTypes()),
              /*inputs=*/ValueRange{},
              /*outs=*/blockArgs.take_back(1), indexingMaps,
              linalgIteratorTypes,
              [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                  mlir::ValueRange) {
                auto shardTy =
                    mlir::cast<RankedTensorType>(blockArgs.back().getType());
                auto tType =
                    mlir::cast<ttcore::TileType>(shardTy.getElementType());
                mlir::Value yieldTile =
                    bbBuilder
                        .create<d2m::TileRandOp>(bbLoc, tType, seedAttr,
                                                 fromAttr, scaleAttr)
                        .getResult();
                bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yieldTile);
              });

          return {linalgGeneric.getResult(0)};
        });

    return unLayoutResult(rewriter, generic->getResult(0), randResultType)
        ->getResult(0);
  }

  /// Casts `input` elementwise to `resultType` via `d2m.tile_typecast`.
  /// Emitted inline instead of going through `ttir.typecast` so we don't
  /// rely on the conversion driver revisiting a freshly-created op.
  mlir::Value buildTypecastGeneric(ConversionPatternRewriter &rewriter,
                                   Location loc, mlir::Value input,
                                   RankedTensorType resultType) const {
    SmallVector<Value> origInputs{input};
    SmallVector<Value> origOutputs =
        createDpsOutputs(loc, rewriter, {resultType});
    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {origInputs, origOutputs}, /*tiled*/ true);
    assert(inputs.size() == 1 && outputs.size() == 1);

    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;
    SmallVector<AffineMap> indexingMaps =
        getIdentityAffineMapsArray(rewriter, 2, physicalRank);
    auto parallel = ttcore::IteratorTypeAttr::get(
        rewriter.getContext(), ttcore::IteratorType::Parallel);
    SmallVector<Attribute> iteratorTypes(physicalRank, parallel);

    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    withD2MGenericRegion(
        rewriter, loc, generic, inputs, outputs,
        [&](mlir::ArrayRef<mlir::Value> blockArgs) -> SmallVector<Value> {
          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
              iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

          auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
              loc,
              llvm::to_vector(
                  mlir::ValueRange(blockArgs.take_back(1)).getTypes()),
              /*inputs=*/blockArgs.take_front(1),
              /*outs=*/blockArgs.take_back(1), indexingMaps,
              linalgIteratorTypes,
              [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                  mlir::ValueRange bbArgs) {
                auto outShardTy =
                    mlir::cast<RankedTensorType>(blockArgs.back().getType());
                auto outTileTy =
                    mlir::cast<ttcore::TileType>(outShardTy.getElementType());
                mlir::Value casted = bbBuilder
                                         .create<d2m::TileTypecastOp>(
                                             bbLoc, outTileTy, bbArgs.front())
                                         .getResult();
                bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, casted);
              });

          return {linalgGeneric.getResult(0)};
        });

    return unLayoutResult(rewriter, generic->getResult(0), resultType)
        ->getResult(0);
  }
};

class D2MArangeOpRewriter : public OpConversionPattern<ttir::ArangeOp>,
                            D2MNamedRewriterCommon {
public:
  D2MArangeOpRewriter(const TypeConverter &typeConverter,
                      mlir::MLIRContext *ctx,
                      ttcore::MemorySpace defaultInputMemSpace,
                      ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
                      bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ttir::ArangeOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

  LogicalResult
  matchAndRewrite(ttir::ArangeOp op, ttir::ArangeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    RankedTensorType resultType = op.getResult().getType();

    if (resultType.getRank() < 1 || resultType.getRank() > 2) {
      return rewriter.notifyMatchFailure(
          op, "D2M arange requires 1D or 2D tensor; decomposition pass should "
              "have handled other cases");
    }

    int64_t start = op.getStart();
    int64_t step = op.getStep();
    int64_t numElements = resultType.getShape().back();

    // Create output tensor with D2M layout (tiled).
    llvm::SmallVector<Value> origOutputs =
        createDpsOutputs(loc, rewriter, {resultType});
    SmallVector<Value> emptyInputs;
    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {emptyInputs, origOutputs}, /*tiled*/ true);
    Value output = outputs[0];

    auto outputTensorType = mlir::cast<RankedTensorType>(output.getType());
    auto outputLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(outputTensorType.getEncoding());
    const std::size_t physicalRank =
        ttcore::getDeviceLayout(output).getRank() / 2;

    // Create scratch tensor for index tile (single tile per core).
    auto outputTileType =
        mlir::cast<ttcore::TileType>(outputTensorType.getElementType());
    Type outputElemType = outputTileType.getElementType();
    llvm::ArrayRef<int64_t> gridShape =
        outputLayout.getGridShape(outputTensorType);
    SmallVector<int64_t> scratchShape(gridShape.begin(), gridShape.end());
    scratchShape.append({1, 1}); // One tile
    auto tileType = ttcore::TileType::get(outputElemType);
    SmallVector<int64_t> scratchLogicalShape = {1, 1};
    auto scratchLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), scratchLogicalShape,
        ttcore::MemorySpace::DeviceL1, ttcore::TensorMemoryLayout::Sharded);

    Value indexTileTensor =
        rewriter
            .create<d2m::EmptyOp>(loc, scratchShape, tileType, scratchLayout)
            .getResult();

    AffineMap identityMap = rewriter.getMultiDimIdentityMap(physicalRank);
    SmallVector<AffineExpr> zeroExprs(physicalRank,
                                      rewriter.getAffineConstantExpr(0));
    AffineMap constantMap =
        AffineMap::get(physicalRank, 0, zeroExprs, rewriter.getContext());

    SmallVector<AffineMap> indexingMaps = {constantMap, identityMap};
    auto parallel = ttcore::IteratorTypeAttr::get(
        rewriter.getContext(), ttcore::IteratorType::Parallel);
    SmallVector<Attribute> iteratorTypes(physicalRank, parallel);

    SmallVector<Value> genericInputs = {indexTileTensor};
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, genericInputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      auto [inputIndices, mcastGridDims] = createInputIndicesAndMcastGridDims(
          rewriter, loc, generic, enableMulticastInference);
      auto blockArgsVec = createBlockArguments(
          rewriter, block, loc, TypeRange(genericInputs), TypeRange(outputs),
          generic, inputIndices, mcastGridDims);
      ArrayRef<Value> blockArgs(blockArgsVec);
      Value indexTileTensor = blockArgs[0];
      Value outputTensor = blockArgs[1];

      // ArangeBlock operation will be decomposed in a later pass.
      Value arangeResult =
          rewriter
              .create<d2m::ArangeBlockOp>(loc, indexTileTensor, outputTensor,
                                          numElements, start, step)
              .getResult();

      AffineMap outputIndexingMap = generic.getIndexingMap(1);
      SmallVector<Value> indices =
          d2m::utils::buildGridIndices(rewriter, loc, outputIndexingMap);
      Value storeResult =
          rewriter
              .create<d2m::RemoteStoreOp>(loc, output.getType(), output,
                                          indices, arangeResult)
              .getResult();

      rewriter.create<d2m::YieldOp>(loc, storeResult);
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);
    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op->getResult(0).getType()));
    return success();
  }
};

class D2MTopKRewriter : public OpConversionPattern<ttir::TopKOp>,
                        D2MNamedRewriterCommon {
public:
  D2MTopKRewriter(const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
                  ttcore::MemorySpace defaultInputMemSpace,
                  ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
                  bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ttir::TopKOp>(typeConverter, ctx, /*benefit=*/10),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

  // Extracts top-k results from topkResult into extractOutput, collapsing
  // extractProjectDim to tile 0. Transposes (dim=1) or typecasts (dim=0)
  // each tile to handle row/column orientation.
  d2m::GenericOp createExtractGeneric(ConversionPatternRewriter &rewriter,
                                      Location loc, MLIRContext *ctx,
                                      Value topkResult, Value extractOutput,
                                      std::size_t extractProjectDim,
                                      int64_t outputReductionTiles,
                                      int64_t lastStride, int32_t dim) const {
    std::size_t extractRank =
        ttcore::getDeviceLayout(extractOutput).getRank() / 2;
    AffineMap extractIdentity = rewriter.getMultiDimIdentityMap(extractRank);
    SmallVector<Attribute> extractIters(
        extractRank,
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

    SmallVector<AffineExpr> inputMapExprs;
    for (std::size_t i = 0; i < extractRank; ++i) {
      inputMapExprs.push_back(i == extractProjectDim
                                  ? rewriter.getAffineConstantExpr(0)
                                  : rewriter.getAffineDimExpr(i));
    }
    AffineMap inputProjectedMap =
        AffineMap::get(extractRank, 0, inputMapExprs, ctx);

    SmallVector<Value> extractInputs = {topkResult};
    SmallVector<Value> extractOutputs = {extractOutput};
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, extractInputs, extractOutputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(
            SmallVector<AffineMap>{inputProjectedMap, extractIdentity}),
        rewriter.getArrayAttr(extractIters));
    // Keep the extract on one core: it reads the full topk partial shard.
    generic->setAttr("d2m.skip_grid_selection", rewriter.getUnitAttr());

    withD2MGenericRegion(
        rewriter, loc, generic, extractInputs, extractOutputs,
        [&](ArrayRef<Value> blockArgs) -> SmallVector<Value> {
          Value input = blockArgs[0];
          Value output = blockArgs[1];
          std::size_t outShardRank =
              cast<RankedTensorType>(output.getType()).getRank();
          int64_t inReductionExtent = cast<RankedTensorType>(input.getType())
                                          .getShape()[extractProjectDim];

          // The input map must stay non-invertible, or linalg infers the loop
          // bound from the full input extent and rejects the smaller output
          // shard. With lastStride=1, `dim * lastStride` folds to an identity
          // map, so we wrap it in `mod inReductionExtent` to stay
          // non-invertible without changing the in-range read indices
          AffineExpr projExpr =
              outputReductionTiles == 1
                  ? rewriter.getAffineConstantExpr(0)
                  : (rewriter.getAffineDimExpr(extractProjectDim) *
                     lastStride) %
                        inReductionExtent;
          SmallVector<AffineExpr> mapFirstExprs;
          for (std::size_t i = 0; i < outShardRank; ++i) {
            mapFirstExprs.push_back(i == extractProjectDim
                                        ? projExpr
                                        : rewriter.getAffineDimExpr(i));
          }
          AffineMap mapFirst =
              AffineMap::get(outShardRank, 0, mapFirstExprs, ctx);
          AffineMap outIdentity = rewriter.getMultiDimIdentityMap(outShardRank);
          SmallVector<mlir::utils::IteratorType> linalgIters(
              outShardRank, mlir::utils::IteratorType::parallel);

          auto linalgOp = rewriter.create<linalg::GenericOp>(
              loc, output.getType(), input, output,
              SmallVector<AffineMap>{mapFirst, outIdentity}, linalgIters,
              [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
                Value result;
                if (dim == 1) {
                  result = b.create<d2m::TileTransposeOp>(
                      bodyLoc, args[1].getType(), args[0]);
                } else {
                  // dim=0: copy tile via typecast
                  result = b.create<d2m::TileTypecastOp>(
                      bodyLoc, args[1].getType(), args[0]);
                }
                b.create<linalg::YieldOp>(bodyLoc, result);
              });
          return {linalgOp->getResult(0)};
        });

    return generic;
  }

  LogicalResult
  matchAndRewrite(ttir::TopKOp op, ttir::TopKOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto inputType = cast<RankedTensorType>(adaptor.getInputTensor().getType());
    auto valuesType = cast<RankedTensorType>(op.getValues().getType());
    auto indicesType = cast<RankedTensorType>(op.getIndices().getType());

    int32_t k = op.getK();
    assert(k <= 64 && "D2M topk only supports k <= 64");
    int32_t dim = op.getDim();
    int64_t rank = inputType.getRank();
    if (dim < 0) {
      dim += rank;
    }

    if (rank != 2 || (dim != 0 && dim != 1)) {
      return rewriter.notifyMatchFailure(
          op, "D2M topk only supports 2D reduction on dim 0 or dim 1");
    }

    // Multi-core sharding: a very wide reduction dim does not fit in a single
    // core's L1. Split it into G column bands ("slices"), run a local topk per
    // slice, then merge the sorted partials in a cross-core reduction tree.
    // First cut: k<=32 (small-k, 1-tile partials), dim==1, ht==1, and each
    // slice is capped to kMaxShardTiles tiles wide.
    constexpr int64_t kTileWidth = 32;
    constexpr int64_t kMaxShardTiles = 32; // L1-safe shard width.
    int64_t fullReductionElems = inputType.getShape()[dim];
    int64_t fullReductionTiles = fullReductionElems / kTileWidth;
    // G = number of slices needed so each slice is <= kMaxShardTiles tiles.
    int64_t numShards =
        (fullReductionTiles + kMaxShardTiles - 1) / kMaxShardTiles;
    bool multiCore = numShards > 1;

    if (fullReductionTiles < 2) {
      return rewriter.notifyMatchFailure(
          op,
          "D2M topk requires at least 2 tiles along the reduction dimension");
    }
    if (multiCore) {
      // Hard-fail (no TTNN fallback) for configs outside the first-cut scope.
      assert(k <= 32 && "multi-core topk first cut only supports k <= 32");
      assert(dim == 1 && "multi-core topk first cut only supports dim == 1");
      assert(inputType.getShape()[0] == kTileWidth &&
             "multi-core topk first cut only supports a single tile row "
             "(ht == 1)");
      assert(fullReductionElems % (kTileWidth * numShards) == 0 &&
             "multi-core topk first cut requires the reduction dim to divide "
             "evenly across shards");
    }

    // Reblocks a unit-grid ([1,1,ht,wt]) device tensor to grid 1xG
    // ([1,G,ht,wt/G]) via a ViewLayoutOp so the G reduction-dim bands land on G
    // distinct cores. createOptimalLayoutOp always builds a unit grid; this
    // splits the shard's reduction-dim tiles across grid columns without moving
    // data. GenericOp::build then auto-derives grid 1xG from the operand's grid
    // shape, distributing the local topk compute across cores. Same idiom as
    // the TTNN unit-grid reblock in createOptimalLayoutOp.
    auto reblockToGrid = [&](Value value, int64_t gridCols) -> Value {
      auto type = cast<RankedTensorType>(value.getType());
      // No-op if already at the requested grid (also covers the single-core
      // gridCols==1 path where the tensor is already unit grid).
      ArrayRef<int64_t> curShape = type.getShape();
      int64_t curGridCols = curShape[curShape.size() / 2 - 1];
      if (curGridCols == gridCols) {
        return value;
      }
      SmallVector<int64_t> newGrid = {1, gridCols};
      auto [newShape, reblockMap] = ttmlir::utils::calculateReblockMapForGrid(
          type.getShape(), newGrid, ctx);
      auto newType = RankedTensorType::get(newShape, type.getElementType(),
                                           type.getEncoding());
      return rewriter
          .create<d2m::ViewLayoutOp>(loc, newType, value, reblockMap,
                                     /*reinterpretLayout=*/false)
          .getResult();
    };

    // Emits the full single-shard topk (layout -> transpose -> arange/index ->
    // TopkBlockOp -> relayout) over a raw [ht, shardWidth] input, returning the
    // (values, indices) partials still in device layout (pre-extract).
    // arangeStart offsets the index buffer so slice g yields GLOBAL indices.
    // gridCols > 1 distributes the local topk compute across that many cores by
    // reblocking the reduction-dim tiles into grid columns (one band per core).
    auto emitShardTopk = [&](Value rawInput, int64_t arangeStart,
                             int64_t gridCols = 1) -> std::pair<Value, Value> {
      auto inputType = cast<RankedTensorType>(rawInput.getType());
      int64_t rank = inputType.getRank();
      int64_t reductionDimSize = inputType.getShape()[dim];

      // The value input's logical shape. When large-k pads the reduction dim to
      // a power-of-2 tile count, this grows to match the arange/index buffer,
      // since topk d2m.generic requires both operands to share a shard shape.
      SmallVector<int64_t> topkLogicalShape(inputType.getShape().begin(),
                                            inputType.getShape().end());

      Value layoutedInput =
          createOptimalLayoutOp(rawInput, memorySpaces[0], /*tiled=*/true,
                                /*noCollapse=*/false, rewriter);
      // Shard the input across gridCols cores up front: reblock the unit-grid
      // [1,1,ht,wt] layout to [1,gridCols,ht,wt/gridCols] so each core owns a
      // wt/gridCols-tile band. Deriving deviceShape/metalLayout from this
      // reblocked type makes the whole local-topk chain (transpose, arange,
      // bcast, topk_block) allocate and compute per-core on its band.
      layoutedInput = reblockToGrid(layoutedInput, gridCols);
      auto layoutedType = cast<RankedTensorType>(layoutedInput.getType());
      auto metalLayout =
          cast<ttcore::MetalLayoutAttr>(layoutedType.getEncoding());
      auto f32TileType = cast<ttcore::TileType>(layoutedType.getElementType());
      ArrayRef<int64_t> deviceShape = layoutedType.getShape();
      std::size_t physicalRank = deviceShape.size() / 2;

      int64_t ht = deviceShape[deviceShape.size() - 2];
      int64_t wt = deviceShape[deviceShape.size() - 1];
      int64_t numReductionTiles = (dim == 1) ? wt : ht;
      // The >=2 reduction-tile requirement is enforced on the full input before
      // this lambda runs (single-core) / guaranteed by 32-tile shards
      // (multi-core), so it is an invariant here.
      assert(numReductionTiles >= 2 &&
             "shard must have at least 2 tiles along the reduction dimension");
      // D2MDecomposeTopk's large-k left-fold keeps the accumulator at canonical
      // tiles (winner=0, loser=1) for any tile count, so no power-of-2 padding
      // (and no -inf mask) is needed on the reduction dim.

      Type f32Type = f32TileType.getElementType();
      auto topkValsEmpty = rewriter.create<d2m::EmptyOp>(
          loc, deviceShape, f32TileType, metalLayout);
      auto topkIdxEmpty = rewriter.create<d2m::EmptyOp>(
          loc, deviceShape, f32TileType, metalLayout);

      auto wrapInToLayout = [&](Value genericResult) -> Value {
        auto resultType = cast<RankedTensorType>(genericResult.getType());
        auto emptyOp = rewriter.create<d2m::EmptyOp>(
            loc, resultType.getShape(), resultType.getElementType(),
            resultType.getEncoding());
        return rewriter.create<d2m::ToLayoutOp>(loc, genericResult, emptyOp)
            .getResult(0);
      };

      auto parallel =
          ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel);
      SmallVector<Attribute> iteratorTypes(physicalRank, parallel);
      AffineMap identityMap = rewriter.getMultiDimIdentityMap(physicalRank);

      // Emits a generic that applies TileTransposeOp to every tile in the
      // shard. Used for both value and index when dim==1 to keep them in the
      // same orientation.
      auto transposeTiles = [&](Value src) -> Value {
        auto srcType = cast<RankedTensorType>(src.getType());
        auto srcTileType = cast<ttcore::TileType>(srcType.getElementType());
        auto srcLayout = cast<ttcore::MetalLayoutAttr>(srcType.getEncoding());
        auto transposeEmpty = rewriter.create<d2m::EmptyOp>(
            loc, srcType.getShape(), srcTileType, srcLayout);
        SmallVector<Value> transposeInputs = {src};
        SmallVector<Value> transposeOutputs = {transposeEmpty.getResult()};
        auto transposeGeneric = rewriter.create<d2m::GenericOp>(
            loc, transposeInputs, transposeOutputs,
            /*additionalArgs=*/ValueRange(),
            rewriter.getAffineMapArrayAttr(
                SmallVector<AffineMap>{identityMap, identityMap}),
            rewriter.getArrayAttr(iteratorTypes));
        // topk generics keep the whole reduction dim on one core: the block
        // ops (transpose/arange/bcast/topk) assume the full shard is local, and
        // GridSelection derives grids from operand shape (not iterator types),
        // so it would otherwise shard the reduction dim and corrupt the result.
        transposeGeneric->setAttr("d2m.skip_grid_selection",
                                  rewriter.getUnitAttr());

        withD2MGenericRegion(
            rewriter, loc, transposeGeneric, transposeInputs, transposeOutputs,
            [&](ArrayRef<Value> blockArgs) -> SmallVector<Value> {
              Value input = blockArgs[0];
              Value output = blockArgs[1];
              std::size_t shardRank =
                  cast<RankedTensorType>(input.getType()).getRank();
              AffineMap shardIdentity =
                  rewriter.getMultiDimIdentityMap(shardRank);
              SmallVector<mlir::utils::IteratorType> linalgIters(
                  shardRank, mlir::utils::IteratorType::parallel);
              // TileTransposeOp operates on a single tile. linalg::GenericOp
              // iterates over all tiles in the shard to apply it to each.
              auto linalgOp = rewriter.create<linalg::GenericOp>(
                  loc, output.getType(), input, output,
                  SmallVector<AffineMap>{shardIdentity, shardIdentity},
                  linalgIters,
                  [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
                    Value transposed = b.create<d2m::TileTransposeOp>(
                        bodyLoc, args[1].getType(), args[0]);
                    b.create<linalg::YieldOp>(bodyLoc, transposed);
                  });
              return {linalgOp->getResult(0)};
            });

        return wrapInToLayout(transposeGeneric->getResult(0));
      };

      // When dim=1, the sort dimension falls along tile rows, but TopkBlockOp
      // operates on tile columns, so we transpose the tiles first.
      Value topkInput = layoutedInput;
      if (dim == 1) {
        topkInput = transposeTiles(layoutedInput);
      }

      // Setting up the arange op.

      auto f32InputType = RankedTensorType::get(topkLogicalShape, f32Type,
                                                inputType.getEncoding());
      auto arangeOrigOutputs = createDpsOutputs(loc, rewriter, {f32InputType});
      auto [arangeins, arangeOutputs] = toLayoutOperandsAndResults(
          rewriter, {SmallVector<Value>{}, arangeOrigOutputs}, true,
          /*noCollapse=*/false);

      // Shard the index buffer across the same gridCols cores as the input so
      // arange/bcast allocate and run per-core. toLayoutOperandsAndResults
      // always builds a unit grid, so reblock to 1xgridCols here; the scratch
      // and downstream empties derive their grid from this output.
      arangeOutputs[0] = reblockToGrid(arangeOutputs[0], gridCols);
      auto arangeOutput = arangeOutputs[0];
      auto arangeTensorType =
          mlir::cast<RankedTensorType>(arangeOutput.getType());
      auto arangeLayout =
          mlir::cast<ttcore::MetalLayoutAttr>(arangeTensorType.getEncoding());
      auto arangeTileType =
          mlir::cast<ttcore::TileType>(arangeTensorType.getElementType());
      Type arangeElemType = arangeTileType.getElementType();
      ArrayRef<int64_t> arangeGridShape =
          arangeLayout.getGridShape(arangeTensorType);
      SmallVector<int64_t> arangeScratchShape(arangeGridShape.begin(),
                                              arangeGridShape.end());
      arangeScratchShape.append({1, 1});
      auto arangeScratchTileType = ttcore::TileType::get(arangeElemType);
      auto arangeScratchLayout = ttcore::MetalLayoutAttr::get(
          ctx, SmallVector<int64_t>{1, 1}, ttcore::MemorySpace::DeviceL1,
          ttcore::TensorMemoryLayout::Sharded);
      Value indexTileTensor =
          rewriter
              .create<d2m::EmptyOp>(loc, arangeScratchShape,
                                    arangeScratchTileType, arangeScratchLayout)
              .getResult();

      int64_t numElements =
          (dim == 0) ? topkLogicalShape[rank - 2] : topkLogicalShape[rank - 1];

      // Build affine maps: scratch input always at (0,0), output is identity.
      AffineExpr zero = rewriter.getAffineConstantExpr(0);
      AffineMap arangeConstMap =
          AffineMap::get(physicalRank, 0, {zero, zero}, ctx);
      AffineMap arangeIdentMap = rewriter.getMultiDimIdentityMap(physicalRank);
      SmallVector<AffineMap> arangeMaps = {arangeConstMap, arangeIdentMap};

      // All iterators are parallel for arange.
      SmallVector<Attribute> arangeIters(
          physicalRank,
          ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

      SmallVector<Value> arangeGenericInputs = {indexTileTensor};
      auto arange = rewriter.create<d2m::GenericOp>(
          loc, arangeGenericInputs, arangeOutputs,
          /*additionalArgs=*/ValueRange(),
          rewriter.getAffineMapArrayAttr(arangeMaps),
          rewriter.getArrayAttr(arangeIters));
      arange->setAttr("d2m.skip_grid_selection", rewriter.getUnitAttr());

      withD2MGenericRegion(
          rewriter, loc, arange, arangeGenericInputs, arangeOutputs,
          [&](ArrayRef<Value> blockArgs) -> SmallVector<Value> {
            Value idxTile = blockArgs[0];
            Value outTile = blockArgs[1];
            Value result = rewriter
                               .create<d2m::ArangeBlockOp>(
                                   loc, idxTile, outTile, numElements,
                                   /*start=*/arangeStart,
                                   /*step=*/1,
                                   /*colMajor=*/(dim == 0))
                               .getResult();
            return {result};
          });

      // Setting up the broadcast after arange.

      // Broadcast first row or column to full shape.
      d2m::TileBcastType postArangeBcastType =
          (dim == 1) ? d2m::TileBcastType::Row : d2m::TileBcastType::Col;
      d2m::TileBcastTypeAttr postArangeBcastTypeAttr =
          d2m::TileBcastTypeAttr::get(ctx, postArangeBcastType);
      SmallVector<Value> postArangeBcastInputs(arange.getResults().begin(),
                                               arange.getResults().end());
      auto postArangeBcastOrigOutputs =
          createDpsOutputs(loc, rewriter, {f32InputType});
      auto [postArangeBcastins, postArangeBcastOutputs] =
          toLayoutOperandsAndResults(
              rewriter, {SmallVector<Value>{}, postArangeBcastOrigOutputs},
              true,
              /*noCollapse=*/false);
      // Match the sharded (1xgridCols) arange input grid so the bcast generic's
      // operand grids agree and it too runs per-core.
      postArangeBcastOutputs[0] =
          reblockToGrid(postArangeBcastOutputs[0], gridCols);

      // Build affine maps: input is reduced, output is identity.
      mlir::MutableAffineMap postArangeInMap(
          rewriter.getMultiDimIdentityMap(physicalRank));
      switch (postArangeBcastType) {
      case d2m::TileBcastType::Row:
        postArangeInMap.setResult(physicalRank - 2, zero);
        break;
      case d2m::TileBcastType::Col:
        postArangeInMap.setResult(physicalRank - 1, zero);
        break;
      default:
        break;
      }
      AffineMap postArangeBcastInputMap = postArangeInMap.getAffineMap();
      AffineMap postArangeBcastOutputMap =
          rewriter.getMultiDimIdentityMap(physicalRank);
      SmallVector<AffineMap> postArangeBcastMaps = {postArangeBcastInputMap,
                                                    postArangeBcastOutputMap};

      SmallVector<Attribute> postArangeBcastIters(
          physicalRank,
          ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

      d2m::GenericOp postArangeBcast = rewriter.create<d2m::GenericOp>(
          loc, postArangeBcastInputs, postArangeBcastOutputs,
          /*additionalArgs=*/ValueRange(),
          rewriter.getAffineMapArrayAttr(postArangeBcastMaps),
          rewriter.getArrayAttr(postArangeBcastIters));
      postArangeBcast->setAttr("d2m.skip_grid_selection",
                               rewriter.getUnitAttr());

      withD2MGenericRegion(
          rewriter, loc, postArangeBcast, postArangeBcastInputs,
          postArangeBcastOutputs,
          [&](ArrayRef<Value> blockArgs) -> SmallVector<Value> {
            Value input = blockArgs[0];
            Value output = blockArgs[1];
            std::size_t shardRank =
                cast<RankedTensorType>(output.getType()).getRank();
            // Reuse the outer d2m.generic's projecting maps (input map zeroes
            // the broadcast axis). An identity map would make linalg infer the
            // broadcast-axis extent from the iteration domain, mismatching the
            // collapsed input shard when the non-target dim spans >1 tile (e.g.
            // ht=2 for dim=1).
            mlir::MutableAffineMap shardInMap(
                rewriter.getMultiDimIdentityMap(shardRank));
            switch (postArangeBcastType) {
            case d2m::TileBcastType::Row:
              shardInMap.setResult(shardRank - 2,
                                   rewriter.getAffineConstantExpr(0));
              break;
            case d2m::TileBcastType::Col:
              shardInMap.setResult(shardRank - 1,
                                   rewriter.getAffineConstantExpr(0));
              break;
            default:
              break;
            }
            AffineMap shardInputMap = shardInMap.getAffineMap();
            AffineMap shardIdentity =
                rewriter.getMultiDimIdentityMap(shardRank);
            SmallVector<mlir::utils::IteratorType> linalgIters(
                shardRank, mlir::utils::IteratorType::parallel);
            auto linalgOp = rewriter.create<linalg::GenericOp>(
                loc, output.getType(), input, output,
                SmallVector<AffineMap>{shardInputMap, shardIdentity},
                linalgIters,
                [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
                  Value bcast = b.create<d2m::TileBcastOp>(
                      bodyLoc, args[1].getType(), args[0],
                      postArangeBcastTypeAttr);
                  b.create<linalg::YieldOp>(bodyLoc, bcast);
                });
            return {linalgOp->getResult(0)};
          });

      Value indexOperand = postArangeBcast->getResult(0);
      Value idxBuf = wrapInToLayout(indexOperand);

      // TopkBlockOp reads value and index tiles in lockstep, so transpose the
      // index buffer the same way as the value input for dim==1. For dim==1,
      // arange+bcast yields idxBuf[row][col] = col, so after transpose the
      // column index varies along the tile's row axis, matching the value tile.
      if (dim == 1) {
        idxBuf = transposeTiles(idxBuf);
      }

      SmallVector<Value> topkInputs = {topkInput, idxBuf};
      SmallVector<Value> topkOutputs = {topkValsEmpty.getResult(),
                                        topkIdxEmpty.getResult()};
      auto topkGeneric = rewriter.create<d2m::GenericOp>(
          loc, topkInputs, topkOutputs, /*additionalArgs=*/ValueRange(),
          rewriter.getAffineMapArrayAttr(SmallVector<AffineMap>{
              identityMap, identityMap, identityMap, identityMap}),
          rewriter.getArrayAttr(iteratorTypes));
      topkGeneric->setAttr("d2m.skip_grid_selection", rewriter.getUnitAttr());

      withD2MGenericRegion(
          rewriter, loc, topkGeneric, topkInputs, topkOutputs,
          [&](ArrayRef<Value> blockArgs) -> SmallVector<Value> {
            auto topkBlock = rewriter.create<d2m::TopkBlockOp>(
                loc, blockArgs[0], blockArgs[1], blockArgs[2], blockArgs[3], k,
                reductionDimSize, /*stableSort=*/false, dim);
            return {topkBlock.getResultValues(), topkBlock.getResultIndices()};
          });

      // Gather the per-core (1xgridCols) topk results back to a unit grid so
      // the downstream extract/merge path (which assumes 1x1) is unchanged.
      // This is the single cross-core boundary; distributing the merge is
      // future work.
      Value topkValsRelayouted =
          reblockToGrid(wrapInToLayout(topkGeneric->getResult(0)), 1);
      Value topkIdxRelayouted =
          reblockToGrid(wrapInToLayout(topkGeneric->getResult(1)), 1);

      return {topkValsRelayouted, topkIdxRelayouted};
    }; // emitShardTopk

    // ----- Drive the shard topk(s): single-core or the multi-core tree. -----

    // physicalRank / lastStride / outputReductionTiles are needed by the shared
    // extract tail below and do not depend on which shard produced the result.
    // physicalRank equals the input rank (device layout is grid + shard, each
    // of input rank). For k>32 the result spans 2 tiles; D2MDecomposeTopk's
    // left-fold keeps the accumulator at canonical tiles (winner=0, loser=1),
    // so the extract reads tile j from input tile j (lastStride=1) for any tile
    // count, with no power-of-2 padding. For k<=32 lastStride is unused.
    Type elemF32Type = rewriter.getF32Type();
    std::size_t physicalRank = static_cast<std::size_t>(rank);
    int64_t outputReductionTiles = (k + 31) / 32;
    int64_t lastStride = 1;

    // to_layout wrapper usable at this (outer) scope for the merge tree.
    auto relayout = [&](Value genericResult) -> Value {
      auto resultType = cast<RankedTensorType>(genericResult.getType());
      auto emptyOp = rewriter.create<d2m::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType(),
                                                   resultType.getEncoding());
      return rewriter.create<d2m::ToLayoutOp>(loc, genericResult, emptyOp)
          .getResult(0);
    };
    auto parallelAttr =
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel);
    SmallVector<Attribute> mergeIters(physicalRank, parallelAttr);
    AffineMap mergeIdentity = rewriter.getMultiDimIdentityMap(physicalRank);
    // Shard-space index of the reduction dim (used for generic-level affine
    // maps, which operate on the [sh, sw] shard portion of rank physicalRank).
    std::size_t redPhysDim =
        (dim == 1) ? (physicalRank - 1) : (physicalRank - 2);
    // Device-tensor index of the reduction dim. The device layout is rank
    // 2*physicalRank ([grid..., shard...]); the reduction tiles live in the
    // shard portion, i.e. the last shard dim for dim==1 and the first shard dim
    // for dim==0. compactToTile/mergePair reshape the device tensor, so they
    // must index here, not in shard space.
    std::size_t deviceRedDim =
        (dim == 1) ? (2 * physicalRank - 1) : (2 * physicalRank - 2);

    // Rebuilds a MetalLayoutAttr from `layout` but with a logical shape that
    // matches `deviceShape`'s tile counts (tileRows*32 x tileCols*32). The
    // CompositeViewOp verifier checks logical shapes, so the layout's logical
    // shape must track the physical tile count as the merge tree grows/shrinks
    // the reduction dim.
    auto layoutForDeviceShape =
        [&](ttcore::MetalLayoutAttr layout,
            ArrayRef<int64_t> deviceShape) -> ttcore::MetalLayoutAttr {
      constexpr int64_t kTileDim = 32;
      int64_t shardH = deviceShape[deviceShape.size() - 2];
      int64_t shardW = deviceShape[deviceShape.size() - 1];
      SmallVector<int64_t> newLogical = {shardH * kTileDim, shardW * kTileDim};
      return ttcore::MetalLayoutAttr::get(
          ctx, newLogical, layout.getDimAlignments(),
          layout.getCollapsedIntervals(), layout.getMemorySpace(),
          layout.getMemoryLayout());
    };

    // Compacts tile 0 of a full-width partial into a fresh 1-tile f32
    // device-layout buffer, giving a distinct SSA value the merge tree can
    // CompositeView. Projects the reduction physical dim to constant 0 (copy of
    // tile 0), identity elsewhere.
    auto compactToTile = [&](Value fullPartial) -> Value {
      auto fullType = cast<RankedTensorType>(fullPartial.getType());
      auto tileType = cast<ttcore::TileType>(fullType.getElementType());
      auto layout = cast<ttcore::MetalLayoutAttr>(fullType.getEncoding());
      SmallVector<int64_t> oneTileShape(fullType.getShape().begin(),
                                        fullType.getShape().end());
      oneTileShape[deviceRedDim] = 1;
      // Fresh layout whose logical shape matches the compacted 1-tile physical
      // shape (the CompositeViewOp verifier reads logical shapes).
      auto oneTileLayout = layoutForDeviceShape(layout, oneTileShape);
      auto outEmpty = rewriter.create<d2m::EmptyOp>(loc, oneTileShape, tileType,
                                                    oneTileLayout);
      // Input map projects the reduction dim to 0; output identity.
      SmallVector<AffineExpr> inExprs;
      for (std::size_t i = 0; i < physicalRank; ++i) {
        inExprs.push_back(i == redPhysDim ? rewriter.getAffineConstantExpr(0)
                                          : rewriter.getAffineDimExpr(i));
      }
      AffineMap inProj = AffineMap::get(physicalRank, 0, inExprs, ctx);
      SmallVector<Value> ins = {fullPartial};
      SmallVector<Value> outs = {outEmpty.getResult()};
      auto generic = rewriter.create<d2m::GenericOp>(
          loc, ins, outs, /*additionalArgs=*/ValueRange(),
          rewriter.getAffineMapArrayAttr(
              SmallVector<AffineMap>{inProj, mergeIdentity}),
          rewriter.getArrayAttr(mergeIters));
      generic->setAttr("d2m.skip_grid_selection", rewriter.getUnitAttr());
      withD2MGenericRegion(
          rewriter, loc, generic, ins, outs,
          [&](ArrayRef<Value> blockArgs) -> SmallVector<Value> {
            Value input = blockArgs[0];
            Value output = blockArgs[1];
            std::size_t shardRank =
                cast<RankedTensorType>(output.getType()).getRank();
            // Copy of the (projected) input tile into the 1-tile output.
            SmallVector<AffineExpr> shardInExprs;
            for (std::size_t i = 0; i < shardRank; ++i) {
              shardInExprs.push_back(
                  i == (shardRank - (physicalRank - redPhysDim))
                      ? rewriter.getAffineConstantExpr(0)
                      : rewriter.getAffineDimExpr(i));
            }
            AffineMap shardInMap =
                AffineMap::get(shardRank, 0, shardInExprs, ctx);
            AffineMap shardIdentity =
                rewriter.getMultiDimIdentityMap(shardRank);
            SmallVector<mlir::utils::IteratorType> linalgIters(
                shardRank, mlir::utils::IteratorType::parallel);
            auto linalgOp = rewriter.create<linalg::GenericOp>(
                loc, output.getType(), input, output,
                SmallVector<AffineMap>{shardInMap, shardIdentity}, linalgIters,
                [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
                  Value copied = b.create<d2m::TileTypecastOp>(
                      bodyLoc, args[1].getType(), args[0]);
                  b.create<linalg::YieldOp>(bodyLoc, copied);
                });
            return {linalgOp->getResult(0)};
          });
      return relayout(generic->getResult(0));
    };

    // Merges two 1-tile (vals, idx) partials into one 1-tile partial by
    // CompositeView-ing them into a 2-tile buffer and running a TopkBlockOp
    // generic over it (result lands in tile 0), then compacting back to 1 tile.
    // Indices are CARRIED (not regenerated) so global indices propagate.
    auto mergePair = [&](std::pair<Value, Value> a,
                         std::pair<Value, Value> b) -> std::pair<Value, Value> {
      auto tileType = cast<ttcore::TileType>(
          cast<RankedTensorType>(a.first.getType()).getElementType());
      auto layout = cast<ttcore::MetalLayoutAttr>(
          cast<RankedTensorType>(a.first.getType()).getEncoding());
      SmallVector<int64_t> twoTileShape(
          cast<RankedTensorType>(a.first.getType()).getShape().begin(),
          cast<RankedTensorType>(a.first.getType()).getShape().end());
      twoTileShape[deviceRedDim] = 2;
      // Fresh layout whose logical shape matches the 2-tile physical shape so
      // the CompositeViewOp logical-shape verifier is satisfied (inputs are
      // 1 tile each along the composite dim; sum == 2 tiles == output).
      auto twoTileLayout = layoutForDeviceShape(layout, twoTileShape);
      auto twoTileType =
          RankedTensorType::get(twoTileShape, tileType, twoTileLayout);

      // CompositeViewOp's dim indexes the layout's logical shape (rank
      // physicalRank), so the composite axis is the logical reduction dim.
      auto valsComposite = rewriter.create<d2m::CompositeViewOp>(
          loc, twoTileType, ValueRange{a.first, b.first},
          static_cast<int32_t>(dim), /*logicalSizes=*/nullptr);
      auto idxComposite = rewriter.create<d2m::CompositeViewOp>(
          loc, twoTileType, ValueRange{a.second, b.second},
          static_cast<int32_t>(dim), /*logicalSizes=*/nullptr);

      // Materialize both composites into real 2-tile buffers before the merge
      // generic. The DMA-expansion pass supports only one composite view per
      // GenericOp (#7600); the merge needs both values and indices, so we copy
      // each composite into a plain buffer here and feed those to the generic.
      Value valsMaterialized = relayout(valsComposite.getResult());
      Value idxMaterialized = relayout(idxComposite.getResult());

      auto valsOut = rewriter.create<d2m::EmptyOp>(loc, twoTileShape, tileType,
                                                   twoTileLayout);
      auto idxOut = rewriter.create<d2m::EmptyOp>(loc, twoTileShape, tileType,
                                                  twoTileLayout);
      SmallVector<Value> mergeInputs = {valsMaterialized, idxMaterialized};
      SmallVector<Value> mergeOutputs = {valsOut.getResult(),
                                         idxOut.getResult()};
      auto mergeGeneric = rewriter.create<d2m::GenericOp>(
          loc, mergeInputs, mergeOutputs, /*additionalArgs=*/ValueRange(),
          rewriter.getAffineMapArrayAttr(SmallVector<AffineMap>{
              mergeIdentity, mergeIdentity, mergeIdentity, mergeIdentity}),
          rewriter.getArrayAttr(mergeIters));
      mergeGeneric->setAttr("d2m.skip_grid_selection", rewriter.getUnitAttr());
      withD2MGenericRegion(
          rewriter, loc, mergeGeneric, mergeInputs, mergeOutputs,
          [&](ArrayRef<Value> blockArgs) -> SmallVector<Value> {
            auto topkBlock = rewriter.create<d2m::TopkBlockOp>(
                loc, blockArgs[0], blockArgs[1], blockArgs[2], blockArgs[3], k,
                /*reductionDimSize=*/2 * 32, /*stableSort=*/false, dim);
            return {topkBlock.getResultValues(), topkBlock.getResultIndices()};
          });
      Value mergedVals = relayout(mergeGeneric->getResult(0));
      Value mergedIdx = relayout(mergeGeneric->getResult(1));
      return {compactToTile(mergedVals), compactToTile(mergedIdx)};
    };

    Value topkValsRelayouted, topkIdxRelayouted;
    if (!multiCore) {
      std::tie(topkValsRelayouted, topkIdxRelayouted) =
          emitShardTopk(adaptor.getInputTensor(), /*arangeStart=*/0);
    } else {
      // Run one local topk over the full input, distributed across numShards
      // cores: the reduction-dim tile bands are reblocked into grid columns so
      // each core runs topk_block on its own band. The per-core partials are
      // gathered back to a unit grid inside emitShardTopk; the merge across
      // bands is handled by the existing reduction tree below.
      SmallVector<std::pair<Value, Value>> partials;
      {
        auto [pv, pi] =
            emitShardTopk(adaptor.getInputTensor(),
                          /*arangeStart=*/0, /*gridCols=*/numShards);
        partials.push_back({compactToTile(pv), compactToTile(pi)});
      }

      // Reduction tree: fold pairs level by level; a ragged (odd) tail element
      // is carried unchanged to the next level (top-k merge is associative, so
      // any pairing order yields the same global top-k).
      while (partials.size() > 1) {
        SmallVector<std::pair<Value, Value>> next;
        for (std::size_t i = 0; i + 1 < partials.size(); i += 2) {
          next.push_back(mergePair(partials[i], partials[i + 1]));
        }
        if (partials.size() % 2 == 1) {
          next.push_back(partials.back());
        }
        partials = std::move(next);
      }
      topkValsRelayouted = partials[0].first;
      topkIdxRelayouted = partials[0].second;
    }

    // Index extract uses fp32 topk tiles; a typecast generic below converts
    // to the user index output element type (uint16) for width-preserving
    // untilize.
    auto f32IndicesType =
        RankedTensorType::get(indicesType.getShape(), elemF32Type);

    SmallVector<Value> origValOutputs =
        createDpsOutputs(loc, rewriter, {valuesType});
    SmallVector<Value> origIdxOutputs =
        createDpsOutputs(loc, rewriter, {f32IndicesType});
    Value extractValsLayout = createOptimalLayoutOp(
        origValOutputs[0], memorySpaces[1], /*tiled=*/true,
        /*noCollapse=*/false, rewriter);
    Value extractIdxLayout = createOptimalLayoutOp(
        origIdxOutputs[0], memorySpaces[1], /*tiled=*/true,
        /*noCollapse=*/false, rewriter);

    // Need 1 output tile for k <= 32 and 2 tiles for k > 32
    // outputReductionTiles and lastStride are computed once in the drive
    // section above (they do not depend on the shard).
    std::size_t extractProjectDim = (dim == 1) ? (physicalRank - 1) : 0;

    auto extractValsGeneric = createExtractGeneric(
        rewriter, loc, ctx, topkValsRelayouted, extractValsLayout,
        extractProjectDim, outputReductionTiles, lastStride, dim);
    auto extractIdxGeneric = createExtractGeneric(
        rewriter, loc, ctx, topkIdxRelayouted, extractIdxLayout,
        extractProjectDim, outputReductionTiles, lastStride, dim);

    // Typecast extracted fp32 indices to the output element type (uint16)
    // before untilize. One tile op per region so D2M->TTKernel store/DST-index
    // lookup finds a single terminal compute op.
    Value extractedIdx = extractIdxGeneric->getResult(0);
    auto extractedIdxType = cast<RankedTensorType>(extractedIdx.getType());
    Type idxOutElemType = indicesType.getElementType();
    auto extractIdxTileType =
        cast<ttcore::TileType>(extractedIdxType.getElementType());
    auto idxCastTileType =
        ttcore::TileType::get(idxOutElemType, extractIdxTileType.getShape());
    auto idxCastEmpty = rewriter.create<d2m::EmptyOp>(
        loc, extractedIdxType.getShape(), idxCastTileType,
        extractedIdxType.getEncoding());
    SmallVector<Value> castInputs = {extractedIdx};
    SmallVector<Value> castOutputs = {idxCastEmpty.getResult()};
    std::size_t castRank = extractedIdxType.getShape().size() / 2;
    AffineMap castIdentity = rewriter.getMultiDimIdentityMap(castRank);
    SmallVector<Attribute> castIters(
        castRank,
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));
    auto idxCastGeneric = rewriter.create<d2m::GenericOp>(
        loc, castInputs, castOutputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(
            SmallVector<AffineMap>{castIdentity, castIdentity}),
        rewriter.getArrayAttr(castIters));
    idxCastGeneric->setAttr("d2m.skip_grid_selection", rewriter.getUnitAttr());
    withD2MGenericRegion(
        rewriter, loc, idxCastGeneric, castInputs, castOutputs,
        [&](ArrayRef<Value> blockArgs) -> SmallVector<Value> {
          Value input = blockArgs[0];
          Value output = blockArgs[1];
          std::size_t shardRank =
              cast<RankedTensorType>(input.getType()).getRank();
          AffineMap shardIdentity = rewriter.getMultiDimIdentityMap(shardRank);
          SmallVector<mlir::utils::IteratorType> linalgIters(
              shardRank, mlir::utils::IteratorType::parallel);
          auto linalgOp = rewriter.create<linalg::GenericOp>(
              loc, output.getType(), input, output,
              SmallVector<AffineMap>{shardIdentity, shardIdentity}, linalgIters,
              [&](OpBuilder &b, Location bodyLoc, ValueRange args) {
                Value casted = b.create<d2m::TileTypecastOp>(
                    bodyLoc, args[1].getType(), args[0]);
                b.create<linalg::YieldOp>(bodyLoc, casted);
              });
          return {linalgOp->getResult(0)};
        });

    Operation *valResult =
        unLayoutResult(rewriter, extractValsGeneric->getResult(0), valuesType);
    Operation *idxResult =
        unLayoutResult(rewriter, idxCastGeneric->getResult(0), indicesType);

    rewriter.replaceOp(op, {valResult->getResult(0), idxResult->getResult(0)});
    return success();
  }
};

class D2MEmbeddingOpRewriter : public OpConversionPattern<ttir::EmbeddingOp>,
                               D2MNamedRewriterCommon {
public:
  D2MEmbeddingOpRewriter(const TypeConverter &typeConverter,
                         mlir::MLIRContext *ctx,
                         ttcore::MemorySpace defaultInputMemSpace,
                         ttcore::MemorySpace defaultOutputMemSpace,
                         bool ttnnMode, bool collapseTensors,
                         bool enableMulticastInference)
      : OpConversionPattern<ttir::EmbeddingOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

  LogicalResult
  matchAndRewrite(ttir::EmbeddingOp op, ttir::EmbeddingOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto indicesType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto weightType = mlir::cast<RankedTensorType>(op.getWeight().getType());
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());

    if (indicesType.getRank() < 2) {
      return rewriter.notifyMatchFailure(
          op, "D2M embedding currently requires at least 2D indices");
    }
    if (weightType.getRank() != 2) {
      return rewriter.notifyMatchFailure(op,
                                         "D2M embedding requires 2D weight");
    }

    int64_t numIndices = 1;
    for (int64_t dim : indicesType.getShape()) {
      if (ShapedType::isDynamic(dim)) {
        return rewriter.notifyMatchFailure(
            op, "D2M embedding requires static index shape");
      }
      numIndices *= dim;
    }
    int64_t embeddingDim = weightType.getShape().back();
    if (ShapedType::isDynamic(embeddingDim)) {
      return rewriter.notifyMatchFailure(
          op, "D2M embedding requires static embedding dimension");
    }

    Location loc = op.getLoc();
    Value weightInput =
        createOptimalLayoutOp(op.getWeight(), memorySpaces[0],
                              /*tiled=*/false, /*noCollapse=*/false, rewriter);
    Value indicesBoundaryInput =
        createOptimalLayoutOp(op.getInput(), memorySpaces[0],
                              /*tiled=*/false, /*noCollapse=*/false, rewriter);
    Value indicesInput =
        memorySpaces[0] == ttcore::MemorySpace::DeviceL1
            ? indicesBoundaryInput
            : createOptimalLayoutOp(
                  indicesBoundaryInput, ttcore::MemorySpace::DeviceL1,
                  /*tiled=*/false, /*noCollapse=*/false, rewriter,
                  ttcore::OOBVal::Undef, indicesType);
    SmallVector<Value> inputs{indicesInput, weightInput};
    SmallVector<Value> origOutputs =
        createDpsOutputs(loc, rewriter, {resultType});
    SmallVector<Value> outputs{
        createOptimalLayoutOp(origOutputs[0], memorySpaces[1],
                              /*tiled=*/false, /*noCollapse=*/false, rewriter)};

    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;
    if (physicalRank != 2) {
      return rewriter.notifyMatchFailure(
          op, "D2M embedding currently requires collapsed 2D layouts");
    }

    AffineMap identityMap = rewriter.getMultiDimIdentityMap(physicalRank);
    SmallVector<AffineExpr> indicesExprs = {rewriter.getAffineDimExpr(0),
                                            rewriter.getAffineConstantExpr(0)};
    AffineMap indicesMap = AffineMap::get(physicalRank, /*symbolCount=*/0,
                                          indicesExprs, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps = {indicesMap, identityMap,
                                           identityMap};
    auto parallel = ttcore::IteratorTypeAttr::get(
        rewriter.getContext(), ttcore::IteratorType::Parallel);
    SmallVector<Attribute> iteratorTypes(physicalRank, parallel);

    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes), d2m::ThreadType::Datamovement);

    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      Region &region = generic->getRegion(0);
      Block *block = rewriter.createBlock(&region);
      rewriter.setInsertionPointToStart(block);
      auto embedding = rewriter.create<d2m::EmbeddingOp>(
          loc, outputs[0].getType(), inputs[0], inputs[1], outputs[0],
          rewriter.getI64IntegerAttr(numIndices),
          rewriter.getI64IntegerAttr(embeddingDim),
          rewriter.getDenseI64ArrayAttr(indicesType.getShape()));
      rewriter.create<d2m::YieldOp>(loc, embedding.getResult());
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);

    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op->getResult(0).getType()));
    return success();
  }
};

class D2MMeshShardOpRewriter : public OpConversionPattern<ttir::MeshShardOp> {
  using OpConversionPattern<ttir::MeshShardOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::MeshShardOp op, ttir::MeshShardOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<d2m::MeshShardOp>(
        op, op.getResult().getType(), adaptor.getInput(), op.getShardType(),
        op.getShardDirection(), op.getShardShape(), op.getShardDims());
    return success();
  }
};

namespace {
class D2MMatmulBlockToLinalgGeneric final
    : public mlir::OpConversionPattern<d2m::TileMatmulBlockOp>,
      D2MNamedRewriterCommon {
public:
  D2MMatmulBlockToLinalgGeneric(const TypeConverter &typeConverter,
                                mlir::MLIRContext *ctx,
                                ttcore::MemorySpace defaultInputMemSpace,
                                ttcore::MemorySpace defaultOutputMemSpace,
                                bool ttnnMode, bool collapseTensors,
                                bool enableMulticastInference)
      : OpConversionPattern<d2m::TileMatmulBlockOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

private:
  LogicalResult
  matchAndRewrite(d2m::TileMatmulBlockOp op,
                  typename d2m::TileMatmulBlockOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    if (llvm::any_of(adaptor.getOperands(), [](Value operand) {
          RankedTensorType type =
              mlir::cast<RankedTensorType>(operand.getType());
          return !mlir::isa<ttcore::TileType>(type.getElementType());
        })) {
      return llvm::failure();
    }

    RankedTensorType tensorA =
        mlir::cast<RankedTensorType>(adaptor.getA().getType());
    const bool transposeB = op.getTransposeB();
    auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
        op.getLoc(), adaptor.getOutput().getType(),
        SmallVector<Value>{adaptor.getA(), adaptor.getB()}, adaptor.getOutput(),
        getAffineMapsArray(rewriter, adaptor.getOperands().size(),
                           tensorA.getRank(), transposeB),
        getIteratorTypesArray(rewriter, tensorA.getRank()),
        [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
            mlir::ValueRange bbArgs) {
          mlir::Value mm = bbBuilder.create<d2m::TileMatmulOp>(
              bbLoc, bbArgs.take_back(1).getTypes()[0], /*a=*/bbArgs[0],
              /*b=*/bbArgs[1], /*c=*/bbArgs[2]);
          bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, mm);
        });

    rewriter.replaceAllUsesExcept(adaptor.getOutput(),
                                  linalgGeneric.getResult(0), linalgGeneric);
    rewriter.eraseOp(op);

    return llvm::success();
  }

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, std::size_t arity,
                     std::size_t rank, bool transposeB) {
    assert(arity == 3 && "expected 3 operands");
    // TODO(#2592) for handling higher ranks if it's needed.
    assert(rank == 2 && "expected a rank 2 operation");
    mlir::MLIRContext *ctx = builder.getContext();

    // B indexing switches from (K, N) to (N, K) when transposed.
    std::array<unsigned, 2> bTargets = transposeB
                                           ? std::array<unsigned, 2>{1, 2}
                                           : std::array<unsigned, 2>{2, 1};
    return SmallVector<mlir::AffineMap>{makeAffineMap(ctx, {0, 2}),
                                        makeAffineMap(ctx, bTargets),
                                        makeAffineMap(ctx, {0, 1})};
  }

  static SmallVector<mlir::utils::IteratorType>
  getIteratorTypesArray(mlir::OpBuilder &builder, std::size_t rank) {
    assert(rank == 2 && "expected a rank 2 operation");
    return SmallVector<mlir::utils::IteratorType>{
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::parallel,
        mlir::utils::IteratorType::reduction,
    };
  }

  static mlir::AffineMap makeAffineMap(mlir::MLIRContext *ctx,
                                       std::array<unsigned, 2> targets) {
    return mlir::AffineMap::getMultiDimMapWithTargets(3, targets, ctx);
  }
};
} // namespace

class D2MAllGatherRewriter : public OpConversionPattern<ttir::AllGatherOp>,
                             D2MNamedRewriterCommon {
public:
  D2MAllGatherRewriter(const TypeConverter &typeConverter,
                       mlir::MLIRContext *ctx,
                       ttcore::MemorySpace defaultInputMemSpace,
                       ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
                       bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ttir::AllGatherOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}
  Value createToLayoutOp(Value originalValue,
                         mlir::ConversionPatternRewriter &rewriter,
                         llvm::SmallVector<int64_t> workerGridShape) const {
    auto tensorType =
        mlir::cast<mlir::RankedTensorType>(originalValue.getType());
    ArrayRef<int64_t> logicalShape = tensorType.getShape();
    Type tiledElementType = ttcore::TileType::get(
        tensorType.getElementType(), ttcore::TileType::getDefaultShape());
    ttcore::MetalLayoutAttr layout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), logicalShape, memorySpaces[0],
        ttcore::TensorMemoryLayout::Sharded);
    auto optimalGrid = d2m::utils::computeOptimalBlockShardedGrid(
        layout.getPhysicalShape(ttcore::TileType::getDefaultShape()),
        workerGridShape);
    llvm::SmallVector<int64_t> deviceShape =
        layout.getDeviceShape(optimalGrid, ttcore::TileType::getDefaultShape());
    auto emptyOp = rewriter.create<d2m::EmptyOp>(
        originalValue.getLoc(), deviceShape, tiledElementType, layout);
    auto toLayoutResult = rewriter
                              .create<d2m::ToLayoutOp>(originalValue.getLoc(),
                                                       originalValue, emptyOp)
                              ->getResult(0);
    return toLayoutResult;
  }

  Value createGlobalSemaphore(mlir::Location loc,
                              mlir::ConversionPatternRewriter &rewriter,
                              llvm::SmallVector<int64_t> workerGridShape,
                              uint32_t initialValue) const {
    Type elementType = mlir::IntegerType::get(rewriter.getContext(), 32,
                                              mlir::IntegerType::Unsigned);
    ttcore::MetalLayoutAttr layout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), workerGridShape, memorySpaces[0],
        ttcore::TensorMemoryLayout::Sharded,
        ttcore::MetalLayoutAttr::computeDefaultCollapsedIntervals(
            rewriter.getContext(), workerGridShape.size()),
        {1, 1});
    llvm::SmallVector<int64_t> deviceShape =
        layout.getDeviceShape(workerGridShape, {});
    auto emptyOp =
        rewriter.create<d2m::EmptyOp>(loc, deviceShape, elementType, layout);
    auto createGlobalSemaphoreOp =
        rewriter.create<d2m::CreateGlobalSemaphoreOp>(
            loc, d2m::GlobalSemaphoreType::get(rewriter.getContext()),
            emptyOp.getResult(),
            rewriter.getIntegerAttr(elementType, initialValue));
    return createGlobalSemaphoreOp.getResult();
  }

  LogicalResult
  matchAndRewrite(ttir::AllGatherOp op, ttir::AllGatherOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // multicast based AG
    mlir::Location loc = op->getLoc();

    auto device = ttcore::lookupDevice(op);
    auto workerGridShape = llvm::to_vector(device.getWorkerGrid().getShape());
    auto meshShape = device.getMeshShape();
    assert(meshShape.size() == 2 && "Mesh shape must be 2D");

    uint32_t clusterAxis = op.getClusterAxis();
    assert(clusterAxis <= meshShape.size() &&
           "Cluster axis must be one of the mesh dimensions");

    // Get the supported topology
    auto topology = device.getMeshTopology()[clusterAxis];
    assert(topology == ttcore::Topology::Ring &&
           "Only ring topology is supported for all gather");
    // TODO(sohaibnadeemTT): get num links from DeviceAttr (issue #7720)
    uint32_t num_links = 1;
    // We use unidir routing mode for ring topology so we get 2 cores for each
    // link
    int num_cores =
        topology == ttcore::Topology::Ring ? num_links * 2 : num_links * 1;
    int num_devices = meshShape[clusterAxis];

    // Create global semaphores for synchronization
    Value startSemaphore =
        createGlobalSemaphore(loc, rewriter, workerGridShape, 0);
    Value endSemaphore =
        createGlobalSemaphore(loc, rewriter, workerGridShape, 0);

    // create input and output
    auto origOutputs =
        createDpsOutputs(loc, rewriter, {op.getResult().getType()});
    SmallVector<Value> origInputs = adaptor.getOperands();
    auto input = createToLayoutOp(origInputs[0], rewriter, workerGridShape);
    auto output = createToLayoutOp(origOutputs[0], rewriter, workerGridShape);

    // Find dim to split work across cores and use it to calc view grids for
    // input and output Go through all dims in order and find one that where
    // num_cores divides the dim size
    uint32_t inputRank =
        mlir::cast<mlir::RankedTensorType>(origInputs[0].getType())
            .getShape()
            .size();
    ttcore::MetalLayoutAttr layout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(),
        mlir::cast<mlir::RankedTensorType>(origInputs[0].getType()).getShape(),
        memorySpaces[0], ttcore::TensorMemoryLayout::Sharded);
    llvm::SmallVector<int64_t> physicalShape =
        layout.getPhysicalShape(ttcore::TileType::getDefaultShape());
    uint32_t workerCoreSplitDim = inputRank;
    for (uint32_t i = 0; i < inputRank; i++) {
      if (physicalShape[i] % num_cores == 0) {
        workerCoreSplitDim = i;
        break;
      }
    }
    assert(workerCoreSplitDim != inputRank &&
           "No dim found for worker core split");

    llvm::SmallVector<int64_t> inputViewGrid(inputRank, 1);
    llvm::SmallVector<int64_t> outputViewGrid(inputRank, 1);
    inputViewGrid[workerCoreSplitDim] *= num_cores;
    outputViewGrid[workerCoreSplitDim] *= num_cores;
    outputViewGrid[op.getAllGatherDim()] *= num_devices;

    auto inputTensorType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto inputStreamTensorType =
        d2m::utils::reblockShapedType(inputTensorType, inputViewGrid);
    auto inputStreamReblockMap = ttmlir::utils::calculateReblockMap(
        inputTensorType.getShape(), inputStreamTensorType.getShape(),
        rewriter.getContext());
    Value inputStreamResult =
        rewriter
            .create<d2m::ViewLayoutOp>(op.getLoc(), inputStreamTensorType,
                                       input, inputStreamReblockMap)
            ->getResult(0);

    auto outputTensorType =
        mlir::cast<mlir::RankedTensorType>(output.getType());
    auto outputStreamTensorType =
        d2m::utils::reblockShapedType(outputTensorType, outputViewGrid);
    auto outputStreamReblockMap = ttmlir::utils::calculateReblockMap(
        outputTensorType.getShape(), outputStreamTensorType.getShape(),
        rewriter.getContext());
    Value outputStreamResult =
        rewriter
            .create<d2m::ViewLayoutOp>(op.getLoc(), outputStreamTensorType,
                                       output, outputStreamReblockMap)
            ->getResult(0);

    // Create generic in explicit form: block factors, indexing maps, and
    // iterator types are empty
    SmallVector<mlir::AffineMap> emptyIndexingMaps;
    SmallVector<mlir::Attribute> emptyIteratorTypes;
    ArrayRef<int64_t> emptyBlockFactors = {};
    llvm::SmallVector<int64_t> genericGridShape =
        llvm::to_vector(ttcore::getGridShape(inputStreamResult));
    auto fabricConnectionConfig = ttcore::FabricConnectionConfigAttr::get(
        rewriter.getContext(), ttcore::NocIndex::Noc0, topology, clusterAxis,
        ttcore::RoutingMode::UnidirRingTorus, num_links);
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, TypeRange(outputStreamResult), inputStreamResult,
        outputStreamResult, ValueRange({startSemaphore, endSemaphore}),
        ttcore::GridAttr::get(rewriter.getContext(), genericGridShape),
        rewriter.getI64ArrayAttr(emptyBlockFactors),
        rewriter.getAffineMapArrayAttr(emptyIndexingMaps),
        rewriter.getArrayAttr(emptyIteratorTypes),
        rewriter.getArrayAttr(
            rewriter.getAttr<d2m::ThreadAttr>(d2m::ThreadType::Unified)),
        fabricConnectionConfig, /*numRegions=*/1);

    // Create one bb in 'generic''s region and set its arguments.
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block'.
      {
        // get the start device on the cluster axis and shape to multicast along
        // that axis
        SmallVector<Value> startDevice;
        SmallVector<Value> deviceMcastShape;
        for (uint32_t dim = 0; dim < meshShape.size(); dim++) {
          if (dim == clusterAxis) {
            startDevice.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, 0));
            deviceMcastShape.push_back(rewriter.create<arith::ConstantIndexOp>(
                loc, meshShape[clusterAxis]));
          } else {
            startDevice.push_back(
                rewriter.create<d2m::MeshPositionOp>(loc, dim));
            deviceMcastShape.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, 1));
          }
        }

        SmallVector<Value> coreIndices;
        for (uint32_t i = 0; i < inputRank; i++) {
          coreIndices.push_back(rewriter.create<d2m::CoreIndexOp>(loc, i));
        }

        // Synchronize: fabric semaphore increment mcast to all devices in
        // cluster axis then wait till value is (num devices - 1)
        rewriter.create<d2m::DeviceSynchronizeOp>(loc, startSemaphore,
                                                  startDevice, deviceMcastShape,
                                                  num_devices - 1, coreIndices);

        SmallVector<Value> inputIndices;
        for (uint32_t i = 0; i < inputRank; i++) {
          if (i == workerCoreSplitDim) {
            inputIndices.push_back(rewriter.create<d2m::CoreIndexOp>(loc, i));
          } else {
            inputIndices.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, 0));
          }
        }

        SmallVector<SmallVector<int64_t>> mcastGridDims(1);
        auto blockArgsVec = createBlockArguments(
            rewriter, block, loc, TypeRange(inputStreamResult),
            TypeRange(outputStreamResult), generic, {inputIndices},
            mcastGridDims);
        Value loadResult = blockArgsVec[0];

        // Create affine map to translate input indices to output indices
        mlir::SmallVector<mlir::AffineExpr> results;
        for (uint32_t i = 0; i < inputRank; i++) {
          auto currentIndex =
              mlir::getAffineDimExpr(i + 1, rewriter.getContext());
          if (i == static_cast<uint32_t>(op.getAllGatherDim())) {
            // device index * device shard size + shard index
            auto deviceIndex = mlir::getAffineDimExpr(0, rewriter.getContext());
            auto deviceShardOffset = mlir::getAffineConstantExpr(
                outputViewGrid[i] / num_devices, rewriter.getContext());
            results.push_back(deviceIndex * deviceShardOffset + currentIndex);
          } else {
            results.push_back(currentIndex);
          }
        }
        auto outputIndexingMap = mlir::AffineMap::get(inputRank + 1, 0, results,
                                                      rewriter.getContext());
        auto meshPosition =
            rewriter.create<d2m::MeshPositionOp>(loc, clusterAxis);
        SmallVector<Value> inputIndicesWithMeshPosition = inputIndices;
        inputIndicesWithMeshPosition.insert(
            inputIndicesWithMeshPosition.begin(), meshPosition);
        SmallVector<Value> outputIndices = ttmlir::utils::fullyApplyAffineMap(
            rewriter, loc, outputIndexingMap, inputIndicesWithMeshPosition);

        SmallVector<Value> storeResults;
        auto remoteStoreOp = rewriter.create<d2m::RemoteStoreOp>(
            loc, outputStreamResult.getType(), outputStreamResult,
            outputIndices, loadResult, startDevice, deviceMcastShape,
            endSemaphore, inputIndices);
        Value storeResult = remoteStoreOp.getResult();
        storeResults.push_back(storeResult);

        rewriter.create<d2m::SemaphoreWaitOp>(
            loc, endSemaphore,
            rewriter.create<arith::ConstantIndexOp>(loc, num_devices - 1));

        rewriter.create<d2m::YieldOp>(loc, storeResults);
      }
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);

    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op->getResult(0).getType()));
    return llvm::success();
  }
};

struct TensorManipulationInfo {
  AffineMap map;
  bool canBeTilized;
};

namespace {
static unsigned getUnitDevicePhysicalRank(unsigned logicalRank) {
  return std::max(logicalRank, 2u);
}

static SmallVector<Value> buildZeroIndices(OpBuilder &builder, Location loc,
                                           int64_t rank) {
  SmallVector<Value> indices;
  indices.reserve(rank);
  for (int64_t i = 0; i < rank; ++i) {
    indices.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
  }
  return indices;
}

static bool isScalarUnitVolumeReshape(RankedTensorType inputType,
                                      RankedTensorType outputType) {
  if (inputType.getRank() != 0 && outputType.getRank() != 0) {
    return false;
  }

  return inputType.hasStaticShape() && outputType.hasStaticShape() &&
         ttmlir::utils::volume<int64_t>(inputType.getShape()) == 1 &&
         ttmlir::utils::volume<int64_t>(outputType.getShape()) == 1;
}

static LogicalResult rewriteScalarReshape(ttir::ReshapeOp op,
                                          ttir::ReshapeOp::Adaptor adaptor,
                                          ConversionPatternRewriter &rewriter) {
  auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
  auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
  if (!isScalarUnitVolumeReshape(inputType, outputType)) {
    return rewriter.notifyMatchFailure(
        op, "requires scalar input or output with static unit volume");
  }

  Location loc = op.getLoc();
  SmallVector<Value> inputIndices =
      buildZeroIndices(rewriter, loc, inputType.getRank());
  Value scalar =
      rewriter.create<tensor::ExtractOp>(loc, adaptor.getInput(), inputIndices);

  if (outputType.getRank() == 0) {
    rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(op, outputType, scalar);
    return success();
  }

  auto empty = rewriter.create<tensor::EmptyOp>(loc, outputType, ValueRange{});
  SmallVector<Value> outputIndices =
      buildZeroIndices(rewriter, loc, outputType.getRank());
  rewriter.replaceOpWithNewOp<tensor::InsertOp>(op, scalar, empty,
                                                outputIndices);
  return success();
}

template <typename TensorManipulationOp,
          TensorManipulationInfo (*LogicalInfoFn)(TensorManipulationOp)>
class D2MTensorManipulationOpRewriter
    : public OpConversionPattern<TensorManipulationOp>,
      D2MNamedRewriterCommon {
public:
  D2MTensorManipulationOpRewriter(const TypeConverter &typeConverter,
                                  mlir::MLIRContext *ctx,
                                  ttcore::MemorySpace defaultInputMemSpace,
                                  ttcore::MemorySpace defaultOutputMemSpace,
                                  bool ttnnMode, bool /*collapseTensors*/,
                                  bool enableMulticastInference)
      : OpConversionPattern<TensorManipulationOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, /*collapse*/ false,
                               enableMulticastInference) {}

  LogicalResult
  matchAndRewrite(TensorManipulationOp op,
                  typename TensorManipulationOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if constexpr (std::is_same_v<TensorManipulationOp, ttir::ReshapeOp>) {
      auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
      auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
      if (isScalarUnitVolumeReshape(inputType, outputType)) {
        return rewriteScalarReshape(op, adaptor, rewriter);
      }
    }

    TensorManipulationInfo info = LogicalInfoFn(op);
    AffineMap deviceMap =
        projectLogicalMapToUnitDeviceSpace(rewriter, info.map);

    auto origInputs = adaptor.getOperands();
    auto origOutputs =
        createDpsOutputs(op.getLoc(), rewriter, {op.getResult().getType()});

    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {origInputs, origOutputs},
                                   /*tiled*/ info.canBeTilized);
    assert(outputs.size() == 1);

    auto outTy = mlir::cast<RankedTensorType>(outputs[0].getType());
    auto layout = mlir::cast<ttcore::MetalLayoutAttr>(outTy.getEncoding());
    auto newLayout = ttcore::MetalLayoutAttr::get(
        layout.getContext(), layout.getLogicalShape(), layout.getMemorySpace(),
        layout.getMemoryLayout(), layout.getCollapsedIntervals(),
        layout.getDimAlignments());
    auto newOutTy = RankedTensorType::get(outTy.getShape(),
                                          outTy.getElementType(), newLayout);

    // Express the data rearrangement as a view. The allocator will later
    // decide whether to insert a CB allocation for any GenericOp that
    // consumes this view.
    auto view = rewriter.create<d2m::ViewLayoutOp>(op.getLoc(), newOutTy,
                                                   inputs[0], deviceMap,
                                                   /*reinterpretLayout=*/false);

    rewriter.replaceOp(op, unLayoutResult(rewriter, view->getResult(0),
                                          op->getResult(0).getType()));

    return success();
  }

  static AffineMap projectLogicalMapToUnitDeviceSpace(Builder &builder,
                                                      AffineMap logicalMap) {
    unsigned outputLogicalRank = logicalMap.getNumDims();
    unsigned inputLogicalRank = logicalMap.getNumResults();
    unsigned outputPhysicalRank = getUnitDevicePhysicalRank(outputLogicalRank);
    unsigned inputPhysicalRank = getUnitDevicePhysicalRank(inputLogicalRank);
    unsigned outputSyntheticRank = outputPhysicalRank - outputLogicalRank;
    unsigned inputSyntheticRank = inputPhysicalRank - inputLogicalRank;
    unsigned outputDeviceRank = outputPhysicalRank * 2;
    unsigned inputDeviceRank = inputPhysicalRank * 2;

    // Compose the logical map with the unit-grid device shard coordinates.
    // Rank-1 logical tensors have a synthetic leading physical row dimension,
    // so logical dim 0 maps to shard dim 1 instead of shard dim 0.
    SmallVector<AffineExpr> outputLogicalExprs;
    outputLogicalExprs.reserve(outputLogicalRank);
    for (unsigned i = 0; i < outputLogicalRank; ++i) {
      outputLogicalExprs.push_back(builder.getAffineDimExpr(
          outputPhysicalRank + outputSyntheticRank + i));
    }
    AffineMap outputDeviceToLogical = AffineMap::get(
        outputDeviceRank, 0, outputLogicalExprs, builder.getContext());
    AffineMap outputDeviceToInputLogical =
        logicalMap.compose(outputDeviceToLogical);

    SmallVector<AffineExpr> deviceExprs;
    deviceExprs.reserve(inputDeviceRank);

    // Grid coordinate mapping (first inputPhysicalRank results).
    for (unsigned i = 0; i < inputPhysicalRank; ++i) {
      // The indexing is all zeros on the unit grid.
      deviceExprs.push_back(builder.getAffineConstantExpr(0));
    }

    // Shard coordinate mapping. Synthetic rank-1 row dims are fixed to zero;
    // real logical dims come from the projected logical map.
    for (unsigned i = 0; i < inputPhysicalRank; ++i) {
      if (i < inputSyntheticRank) {
        deviceExprs.push_back(builder.getAffineConstantExpr(0));
        continue;
      }
      deviceExprs.push_back(
          outputDeviceToInputLogical.getResult(i - inputSyntheticRank));
    }

    return AffineMap::get(outputDeviceRank, 0, deviceExprs,
                          builder.getContext());
  }
};
} // namespace

static TensorManipulationInfo rearrangeLogicalInfo(ttir::RearrangeOp op) {
  mlir::FailureOr<AffineMap> maybeMap = op.getInvPatternMap();
  assert(succeeded(maybeMap));
  AffineMap invMap = *maybeMap;
  bool canBeTilized = false;
  unsigned inputRank = invMap.getNumResults();
  unsigned outputRank = invMap.getNumDims();
  if (inputRank >= 2 && outputRank >= 2) {
    AffineExpr expectedInner2 =
        getAffineDimExpr(outputRank - 2, op.getContext());
    AffineExpr expectedInner1 =
        getAffineDimExpr(outputRank - 1, op.getContext());
    canBeTilized = invMap.getResult(inputRank - 2) == expectedInner2 &&
                   invMap.getResult(inputRank - 1) == expectedInner1;
  }
  return {invMap, canBeTilized};
}

static TensorManipulationInfo sliceLogicalInfo(ttir::SliceStaticOp op) {
  MLIRContext *ctx = op.getContext();
  SmallVector<int32_t> begins =
      extractFromIntegerArrayAttr<int32_t>(op.getBegins());
  SmallVector<int32_t> ends =
      extractFromIntegerArrayAttr<int32_t>(op.getEnds());
  SmallVector<int32_t> step =
      extractFromIntegerArrayAttr<int32_t>(op.getStep());
  assert(begins.size() == ends.size());
  assert(begins.size() == step.size());
  assert(begins.size() ==
         static_cast<size_t>(op.getInput().getType().getRank()));
  assert(begins.size() ==
         static_cast<size_t>(op.getResult().getType().getRank()));

  SmallVector<AffineExpr> exprs;
  for (size_t d = 0; d < begins.size(); d++) {
    exprs.push_back(getAffineDimExpr(d, ctx) * step[d] + begins[d]);
  }
  AffineMap map = AffineMap::get(exprs.size(), 0, exprs, ctx);
  bool canBeTilized = false;
  size_t rank = begins.size();
  if (rank >= 2) {
    ArrayRef<int64_t> inputShape = op.getInput().getType().getShape();
    canBeTilized =
        begins[rank - 2] == 0 &&
        ends[rank - 2] == static_cast<int32_t>(inputShape[rank - 2]) &&
        step[rank - 2] == 1 && begins[rank - 1] == 0 &&
        ends[rank - 1] == static_cast<int32_t>(inputShape[rank - 1]) &&
        step[rank - 1] == 1;
  }
  return {map, canBeTilized};
}

static TensorManipulationInfo permuteLogicalInfo(ttir::PermuteOp op) {
  auto *ctx = op.getContext();
  ArrayRef<int64_t> permutation = op.getPermutation();
  unsigned logicalRank = permutation.size();
  assert(logicalRank >= 2 && "Permute must have at least 2 dimensions");
  // Verify last dimension is not identity for outer permute handling.
  const bool noInnerPermute =
      !(permutation[logicalRank - 2] == static_cast<int64_t>(logicalRank - 1) &&
        permutation[logicalRank - 1] == static_cast<int64_t>(logicalRank - 2));
  assert(noInnerPermute && "Complex permutes (both inner and outer "
                           "permutations) are not supported.");
  // Check if innermost two dimensions are identity-mapped (preserved).
  bool canBeTilized =
      permutation[logicalRank - 2] == static_cast<int64_t>(logicalRank - 2) &&
      permutation[logicalRank - 1] == static_cast<int64_t>(logicalRank - 1);
  SmallVector<AffineExpr> results(logicalRank);
  for (auto [dstIdx, srcIdx] : llvm::enumerate(permutation)) {
    results[dstIdx] = mlir::getAffineDimExpr(srcIdx, ctx);
  }
  AffineMap map = AffineMap::get(logicalRank, /*numSymbols=*/0, results, ctx);
  return {map, canBeTilized};
}

// Compute logical map for ReshapeOp: linearize output coords, delinearize to
// input coords. This handles rank changes (e.g., 2D -> 3D).
// Returns a map from output logical coords to input logical coords.
static TensorManipulationInfo reshapeLogicalInfo(ttir::ReshapeOp op) {
  auto inputTensorType = mlir::cast<RankedTensorType>(op.getInput().getType());
  auto outputTensorType =
      mlir::cast<RankedTensorType>(op.getResult().getType());

  ArrayRef<int64_t> inputShape = inputTensorType.getShape();
  ArrayRef<int64_t> outputShape = outputTensorType.getShape();

  int32_t inputLogicalRank = static_cast<int32_t>(inputShape.size());
  int32_t outputLogicalRank = static_cast<int32_t>(outputShape.size());

  bool canBeTilized = false;
  if (inputLogicalRank >= 2 && outputLogicalRank >= 2) {
    canBeTilized =
        inputShape[inputLogicalRank - 2] ==
            outputShape[outputLogicalRank - 2] &&
        inputShape[inputLogicalRank - 1] == outputShape[outputLogicalRank - 1];
  }

  MLIRContext *ctx = op.getContext();
  Builder builder(ctx);

  SmallVector<int64_t> outputStrides;
  int64_t stride = 1;
  for (int64_t i = outputShape.size() - 1; i >= 0; --i) {
    outputStrides.insert(outputStrides.begin(), stride);
    stride *= outputShape[i];
  }

  SmallVector<int64_t> inputStrides;
  stride = 1;
  for (int64_t i = inputShape.size() - 1; i >= 0; --i) {
    inputStrides.insert(inputStrides.begin(), stride);
    stride *= inputShape[i];
  }

  AffineExpr linearIdx = builder.getAffineConstantExpr(0);
  for (int32_t i = 0; i < outputLogicalRank; ++i) {
    AffineExpr dim = builder.getAffineDimExpr(i);
    AffineExpr strideExpr = builder.getAffineConstantExpr(outputStrides[i]);
    linearIdx = linearIdx + dim * strideExpr;
  }

  SmallVector<AffineExpr> reshapeExprs;
  AffineExpr remainingIdx = linearIdx;
  for (int32_t i = 0; i < inputLogicalRank; ++i) {
    if (i == inputLogicalRank - 1) {
      reshapeExprs.push_back(remainingIdx);
    } else {
      AffineExpr strideExpr = builder.getAffineConstantExpr(inputStrides[i]);
      reshapeExprs.push_back(remainingIdx.floorDiv(strideExpr));
      remainingIdx = remainingIdx % strideExpr;
    }
  }

  AffineMap map = AffineMap::get(outputLogicalRank, 0, reshapeExprs, ctx);
  return {map, canBeTilized};
}

// Compute logical map for ConcatenateHeadsOp:
// Input: [batch, num_heads, seq_len, head_dim]
// Output: [batch, seq_len, num_heads * head_dim]
// This is equivalent to: permute [0, 2, 1, 3] then reshape to merge last 2
// dims. Returns a map from output logical coords to input logical coords.
static TensorManipulationInfo
concatenateHeadsLogicalInfo(ttir::ConcatenateHeadsOp op) {
  auto inputTensorType = mlir::cast<RankedTensorType>(op.getInput().getType());
  auto outputTensorType =
      mlir::cast<RankedTensorType>(op.getResult().getType());

  ArrayRef<int64_t> inputShape = inputTensorType.getShape();
  ArrayRef<int64_t> outputShape = outputTensorType.getShape();

  assert(inputShape.size() == 4 &&
         "Input must be 4D: [batch, num_heads, seq_len, head_dim]");
  assert(outputShape.size() == 3 &&
         "Output must be 3D: [batch, seq_len, hidden_dim]");

  int64_t numHeads = inputShape[1];
  int64_t headDim = inputShape[3];
  int64_t hiddenDim = outputShape[2];

  assert(numHeads * headDim == hiddenDim &&
         "Output hidden_dim must equal num_heads * head_dim");

  // Just reshuffle tiles when head_dim is a multiple of the tile width.
  constexpr int64_t tileWidth = ttcore::TileType::getDefaultShape()[1];
  bool canBeTilized = (headDim % tileWidth == 0);
  // Scale the constants by tile width when tilized.
  int64_t headDimDivisor = canBeTilized ? (headDim / tileWidth) : headDim;

  MLIRContext *ctx = op.getContext();
  Builder builder(ctx);

  SmallVector<AffineExpr> exprs;
  exprs.push_back(builder.getAffineDimExpr(0)); // batch
  exprs.push_back(
      builder.getAffineDimExpr(2).floorDiv(headDimDivisor));     // num_heads
  exprs.push_back(builder.getAffineDimExpr(1));                  // seq_len
  exprs.push_back(builder.getAffineDimExpr(2) % headDimDivisor); // head_dim

  AffineMap map = AffineMap::get(/*dimCount=*/3, /*symbolCount=*/0, exprs, ctx);
  return {map, canBeTilized};
}

class D2MSliceStaticOpNoCConstraintsRewriter
    : public OpConversionPattern<ttir::SliceStaticOp> {
public:
  D2MSliceStaticOpNoCConstraintsRewriter(const TypeConverter &typeConverter,
                                         mlir::MLIRContext *ctx)
      : OpConversionPattern<ttir::SliceStaticOp>(typeConverter, ctx,
                                                 /*benefit=*/10) {}

  LogicalResult
  matchAndRewrite(ttir::SliceStaticOp op, ttir::SliceStaticOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto begins = extractFromIntegerArrayAttr<int32_t>(op.getBegins());
    auto ends = extractFromIntegerArrayAttr<int32_t>(op.getEnds());
    auto step = extractFromIntegerArrayAttr<int32_t>(op.getStep());

    auto inType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto inShape = inType.getShape();
    auto outType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto outShape = outType.getShape();

    const int32_t rank = static_cast<int32_t>(inType.getRank());
    if (rank < 2) {
      return rewriter.notifyMatchFailure(
          op, "NoC-constrained slice rewrite requires rank >= 2");
    }

    const int32_t alignToElements =
        d2m::utils::getNocElementAlignmentL1(op, inType);

    // Assume all shards in L1 already start at aligned addresses.
    const bool isAlignedWidth = begins[rank - 1] % alignToElements == 0;
    const bool isAlignedHeight = begins[rank - 2] % alignToElements == 0;
    const bool notStridedWidth = step[rank - 1] == 1;
    const bool notStridedHeight = step[rank - 2] == 1;

    const bool isNoCFriendlyWidth = isAlignedWidth && notStridedWidth;
    const bool isNoCFriendlyHeight = isAlignedHeight && notStridedHeight;

    if (isNoCFriendlyWidth) {
      return failure();
    }

    // Height <-> Width transpose indices.
    SmallVector<int64_t> hwTransposeIdx(rank);
    std::iota(hwTransposeIdx.begin(), hwTransposeIdx.end(), 0);
    std::swap(hwTransposeIdx[rank - 1], hwTransposeIdx[rank - 2]);

    auto loc = op.getLoc();

    if (isNoCFriendlyHeight) {
      // Transpose - Slice - Transpose.

      auto transposedInShape =
          ttmlir::utils::applyPermutation(inShape, hwTransposeIdx);
      auto transposedInType = RankedTensorType::get(
          transposedInShape, inType.getElementType(), inType.getEncoding());

      auto preTranspose = rewriter.create<ttir::PermuteOp>(
          loc, transposedInType, op.getInput(), hwTransposeIdx);

      // Transpose the slice spec.
      std::swap(begins[rank - 1], begins[rank - 2]);
      std::swap(ends[rank - 1], ends[rank - 2]);
      std::swap(step[rank - 1], step[rank - 2]);

      auto transposedOutShape =
          ttmlir::utils::applyPermutation(outShape, hwTransposeIdx);
      auto transposedOutType = RankedTensorType::get(
          transposedOutShape, outType.getElementType(), outType.getEncoding());

      auto transposedSliceOp = rewriter.create<ttir::SliceStaticOp>(
          loc, transposedOutType, preTranspose.getResult(),
          rewriter.getI32ArrayAttr(begins), rewriter.getI32ArrayAttr(ends),
          rewriter.getI32ArrayAttr(step));

      auto postTranspose = rewriter.create<ttir::PermuteOp>(
          loc, outType, transposedSliceOp.getResult(), hwTransposeIdx);

      rewriter.replaceOp(op, postTranspose.getResult());
    } else {
      // Slice(crop width) - Transpose - Slice(height only) - Transpose.

      // Slice all other dims as instructed, and do a NoC-friendly width crop.
      SmallVector<int32_t> cropWidthBegins(begins);
      SmallVector<int32_t> cropWidthEnds(ends);
      SmallVector<int32_t> cropWidthStep(step);
      cropWidthBegins[rank - 1] =
          ttmlir::utils::alignDown(cropWidthBegins[rank - 1], alignToElements);
      cropWidthEnds[rank - 1] = std::min(
          static_cast<int32_t>(inShape[rank - 1]),
          ttmlir::utils::alignUp(cropWidthEnds[rank - 1], alignToElements));
      cropWidthStep[rank - 1] = 1;

      SmallVector<int64_t> cropWidthOutShape(outShape);
      cropWidthOutShape[rank - 1] =
          cropWidthEnds[rank - 1] - cropWidthBegins[rank - 1];
      auto cropWidthOutType = RankedTensorType::get(
          cropWidthOutShape, outType.getElementType(), outType.getEncoding());

      auto cropWidthSliceOp = rewriter.create<ttir::SliceStaticOp>(
          loc, cropWidthOutType, op.getInput(),
          rewriter.getI32ArrayAttr(cropWidthBegins),
          rewriter.getI32ArrayAttr(cropWidthEnds),
          rewriter.getI32ArrayAttr(cropWidthStep));

      auto transposedCropWidthShape = ttmlir::utils::applyPermutation(
          cropWidthOutType.getShape(), hwTransposeIdx);
      auto transposedCropWidthType = RankedTensorType::get(
          transposedCropWidthShape, outType.getElementType(),
          outType.getEncoding());

      auto preTranspose = rewriter.create<ttir::PermuteOp>(
          loc, transposedCropWidthType, cropWidthSliceOp.getResult(),
          hwTransposeIdx);

      // Construct the height only slice spec.
      SmallVector<int32_t> heightSliceBegins(rank, 0);
      SmallVector<int32_t> heightSliceEnds(outShape);
      SmallVector<int32_t> heightSliceStep(rank, 1);
      // This is the amount we aligned down, trim it.
      heightSliceBegins[rank - 1] =
          begins[rank - 1] - cropWidthBegins[rank - 1];
      // The end is still that far away, just from a new begin.
      heightSliceEnds[rank - 1] =
          heightSliceBegins[rank - 1] + (ends[rank - 1] - begins[rank - 1]);
      // Restore the original step.
      heightSliceStep[rank - 1] = step[rank - 1];

      // Transpose to finish the height slice spec.
      std::swap(heightSliceBegins[rank - 1], heightSliceBegins[rank - 2]);
      std::swap(heightSliceEnds[rank - 1], heightSliceEnds[rank - 2]);
      std::swap(heightSliceStep[rank - 1], heightSliceStep[rank - 2]);

      auto transposedOutShape =
          ttmlir::utils::applyPermutation(outShape, hwTransposeIdx);
      auto transposedOutType = RankedTensorType::get(
          transposedOutShape, outType.getElementType(), outType.getEncoding());

      auto heightSliceOp = rewriter.create<ttir::SliceStaticOp>(
          loc, transposedOutType, preTranspose.getResult(),
          rewriter.getI32ArrayAttr(heightSliceBegins),
          rewriter.getI32ArrayAttr(heightSliceEnds),
          rewriter.getI32ArrayAttr(heightSliceStep));

      auto postTranspose = rewriter.create<ttir::PermuteOp>(
          loc, outType, heightSliceOp.getResult(), hwTransposeIdx);

      rewriter.replaceOp(op, postTranspose.getResult());
    }

    return success();
  }
};

class D2MArgMaxRewriter : public OpConversionPattern<ttir::ArgMaxOp>,
                          D2MNamedRewriterCommon {
public:
  D2MArgMaxRewriter(const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
                    ttcore::MemorySpace defaultInputMemSpace,
                    ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
                    bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ttir::ArgMaxOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

private:
  /// Casts `input` elementwise to `resultType` via `d2m.tile_typecast`.
  /// Emitted inline instead of going through `ttir.typecast` so we don't
  /// rely on the conversion driver revisiting a freshly-created op.
  mlir::Value buildTypecastGeneric(ConversionPatternRewriter &rewriter,
                                   Location loc, mlir::Value input,
                                   RankedTensorType resultType) const {
    SmallVector<Value> origInputs{input};
    SmallVector<Value> origOutputs =
        createDpsOutputs(loc, rewriter, {resultType});
    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {origInputs, origOutputs}, /*tiled*/ true);
    assert(inputs.size() == 1 && outputs.size() == 1);

    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;
    SmallVector<AffineMap> indexingMaps =
        getIdentityAffineMapsArray(rewriter, 2, physicalRank);
    auto parallel = ttcore::IteratorTypeAttr::get(
        rewriter.getContext(), ttcore::IteratorType::Parallel);
    SmallVector<Attribute> iteratorTypes(physicalRank, parallel);

    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    withD2MGenericRegion(
        rewriter, loc, generic, inputs, outputs,
        [&](mlir::ArrayRef<mlir::Value> blockArgs) -> SmallVector<Value> {
          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
              iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

          auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
              loc,
              llvm::to_vector(
                  mlir::ValueRange(blockArgs.take_back(1)).getTypes()),
              /*inputs=*/blockArgs.take_front(1),
              /*outs=*/blockArgs.take_back(1), indexingMaps,
              linalgIteratorTypes,
              [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                  mlir::ValueRange bbArgs) {
                auto outShardTy =
                    mlir::cast<RankedTensorType>(blockArgs.back().getType());
                auto outTileTy =
                    mlir::cast<ttcore::TileType>(outShardTy.getElementType());
                mlir::Value casted = bbBuilder
                                         .create<d2m::TileTypecastOp>(
                                             bbLoc, outTileTy, bbArgs.front())
                                         .getResult();
                bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, casted);
              });

          return {linalgGeneric.getResult(0)};
        });

    return unLayoutResult(rewriter, generic->getResult(0), resultType)
        ->getResult(0);
  }

  d2m::GenericOp buildGenericLinAlg(
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      mlir::ValueRange inputs, mlir::ValueRange outputs,
      mlir::ArrayRef<mlir::AffineMap> indexingMaps,
      mlir::ArrayRef<mlir::Attribute> iteratorTypes,
      llvm::function_ref<mlir::Value(mlir::OpBuilder &, mlir::Location,
                                     mlir::ValueRange)>
          tileBody) const {
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, /*additionalArgs=*/mlir::ValueRange(),
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      auto [inputIndices, mcastGridDims] = createInputIndicesAndMcastGridDims(
          rewriter, loc, generic, enableMulticastInference);
      SmallVector<Value> blockArgsVec = createBlockArguments(
          rewriter, block, loc, mlir::TypeRange(inputs),
          mlir::TypeRange(outputs), generic, inputIndices, mcastGridDims);
      mlir::ArrayRef<Value> blockArgs(blockArgsVec);

      const std::size_t numInputs = inputs.size();
      const std::size_t numOutputs = outputs.size();

      SmallVector<mlir::AffineMap> linalgMaps(indexingMaps.begin(),
                                              indexingMaps.end());
      SmallVector<mlir::Attribute> iteratorTypesVec(iteratorTypes.begin(),
                                                    iteratorTypes.end());
      SmallVector<mlir::utils::IteratorType> linalgIters =
          iteratorTypeTTIRToLinalg(rewriter, iteratorTypesVec);

      auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
          loc,
          llvm::to_vector(
              mlir::ValueRange(blockArgs.take_back(numOutputs)).getTypes()),
          blockArgs.take_front(numInputs), blockArgs.take_back(numOutputs),
          linalgMaps, linalgIters,
          [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
              mlir::ValueRange bbArgs) {
            mlir::Value result = tileBody(bbBuilder, bbLoc, bbArgs);
            bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, result);
          });

      SmallVector<Value> storeResults;
      for (std::size_t i = 0; i < numOutputs; ++i) {
        std::size_t operandIdx = numInputs + i;
        AffineMap storeMap = generic.getIndexingMap(operandIdx);
        SmallVector<Value> indices =
            d2m::utils::buildGridIndices(rewriter, loc, storeMap);
        Value genericOperand = generic->getOperand(operandIdx);
        storeResults.push_back(
            rewriter
                .create<d2m::RemoteStoreOp>(loc, genericOperand.getType(),
                                            genericOperand, indices,
                                            linalgGeneric.getResult(i))
                .getResult());
      }
      rewriter.create<d2m::YieldOp>(loc, storeResults);
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);
    return generic;
  }

  static d2m::ReduceDim dimArgAsReduceDim(ttir::ArgMaxOp op,
                                          std::size_t logicalRank) {
    if (!op.getDimArg()) {
      return d2m::ReduceDim::RC;
    }
    bool reduceC = false, reduceR = false;
    auto dimAttrs = *op.getDimArg();
    for (mlir::Attribute dimAttr : dimAttrs) {
      int64_t d = mlir::cast<mlir::IntegerAttr>(dimAttr).getInt();
      std::size_t nd = normalizeReductionDimIndex(d, logicalRank);
      if (nd == logicalRank - 2) {
        reduceC = true;
      }
      if (nd == logicalRank - 1) {
        reduceR = true;
      }
    }
    if (reduceC && reduceR) {
      return d2m::ReduceDim::RC;
    }
    if (reduceC) {
      return d2m::ReduceDim::C;
    }
    return d2m::ReduceDim::R;
  }

  /// Decompose `ttir.argmax` into a series of D2M ops.
  // dim_arg = 0 => collapse columns, output is a row (max of each column).
  // dim_arg = 1 => collapse rows, output is a column (max of each row).
  // Argmax is decomposed into:
  // Reduce max over target dim -> broadcast back to full shape -> eltwise eq to
  // isolate max elements only -> Arange block to enumerate indices -> broadcast
  // arange row or column to full shape, result is rows or columns of indices
  // repeated over and over -> eltwise multiply with eltwise eq result to
  // isolate indices of max elements -> Reduce max again over target dim, now
  // output is the correct shape -> typecast to i32.
  // Each stage is its own generic for now.
  LogicalResult
  matchAndRewrite(ttir::ArgMaxOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op.getLoc();

    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());

    std::size_t logicalRank = inputType.getRank();
    const bool noCollapse = logicalRank > 2;

    // Setting up reduceMax1.

    // Convert dimArg into a reduceDim enum.
    d2m::ReduceDim reduceDim = dimArgAsReduceDim(op, logicalRank);

    if (inputType.getRank() < 1 || inputType.getRank() > 2) {
      return rewriter.notifyMatchFailure(
          op, "D2M Argmax only works on 1D or 2D tensors at the moment.");
    }
    if (reduceDim == d2m::ReduceDim::RC) {
      return rewriter.notifyMatchFailure(
          op, "D2M Argmax lowering supports only ReduceDim::R and ReduceDim::C "
              "at the moment.");
    }
    if (op.getDimArg()) {
      auto dimAttrs = *op.getDimArg();
      for (mlir::Attribute dimAttr : dimAttrs) {
        int64_t d = mlir::cast<mlir::IntegerAttr>(dimAttr).getInt();
        std::size_t nd = normalizeReductionDimIndex(d, logicalRank);
        if (logicalRank >= 2 && nd < logicalRank - 2) {
          return rewriter.notifyMatchFailure(
              op, "D2M Argmax lowering only supports reducing over the last "
                  "two dims");
        }
      }
    }

    // Full reduction gets turned into a 1D tensor by the TTIRDecomposition
    // pass. Convert into a [1, N] tensor to reuse 2D logic then convert back to
    // 1D at the end.
    Value convertedInput = adaptor.getInput();
    const bool rankNormalized = (logicalRank == 1);
    if (rankNormalized) {
      int64_t n = inputType.getDimSize(0);
      SmallVector<int64_t> rowShape{1, n};
      auto rowInputType = RankedTensorType::get(
          rowShape, inputType.getElementType(), inputType.getEncoding());
      convertedInput = rewriter.create<ttir::ReshapeOp>(
          loc, rowInputType, convertedInput,
          rewriter.getI32ArrayAttr(
              SmallVector<int32_t>(rowShape.begin(), rowShape.end())));
      inputType = rowInputType;
      logicalRank = 2;
      // Reduce the new last dim (row-wise); the original 1D reduction over dim
      // 0 maps to ReduceDim::R on [1, N].
      reduceDim = d2m::ReduceDim::R;
    }

    auto reduceDimAttr = d2m::ReduceDimAttr::get(ctx, reduceDim);

    // Compute the shape of the reduced output (reduced dims become 1).
    SmallVector<int64_t> reducedShape(logicalRank, 1);

    // Determine which dimensions are reduced. After rank normalization the
    // reduction is always the last dim of [1, N]; otherwise use the op's
    // (single, post-ArgMaxPattern) dim_arg.
    SmallVector<bool> isReduced(logicalRank, false);
    if (rankNormalized) {
      isReduced[logicalRank - 1] = true;
    } else if (op.getDimArg()) {
      auto dimAttrs = *op.getDimArg();
      for (auto dimAttr : dimAttrs) {
        int64_t dimension = mlir::cast<IntegerAttr>(dimAttr).getInt();
        isReduced[normalizeReductionDimIndex(dimension, logicalRank)] = true;
      }
    }
    for (std::size_t i = 0; i < logicalRank; ++i) {
      if (!isReduced[i]) {
        // Non-reduced dimensions restore their original size.
        reducedShape[i] = inputType.getDimSize(i);
      }
    }

    auto reducedType = RankedTensorType::get(
        reducedShape, inputType.getElementType(), inputType.getEncoding());

    // Create a scaler (fill value 1.0) and convert inputs/outputs to tiled
    // layout (D2M requires tiled layout).
    auto scaler = createScaler(rewriter, loc, inputType);
    auto maxOrigOutputs = createDpsOutputs(loc, rewriter, {reducedType});
    auto [maskedMaxInput, maxOutputs] = toLayoutOperandsAndResults(
        rewriter, {SmallVector<Value>{convertedInput}, maxOrigOutputs}, true,
        noCollapse, ttcore::OOBVal::NegInf);
    auto [scalerLaidOut, maxouts] = toLayoutOperandsAndResults(
        rewriter, {SmallVector<Value>{scaler}, SmallVector<Value>{}}, true,
        noCollapse);
    SmallVector<Value> maxInputs = {maskedMaxInput[0], scalerLaidOut[0]};

    // Build affine maps for the first reduction.
    const std::size_t physicalRank =
        ttcore::getDeviceLayout(maxOutputs[0]).getRank() / 2;

    mlir::AffineExpr zero = rewriter.getAffineConstantExpr(0);

    AffineMap maxInputMap = rewriter.getMultiDimIdentityMap(physicalRank);
    AffineMap maxScalerMap = AffineMap::get(physicalRank, 0, {zero, zero}, ctx);
    mlir::MutableAffineMap outputAccum(
        rewriter.getMultiDimIdentityMap(physicalRank));
    if (reduceDim == d2m::ReduceDim::R) {
      outputAccum.setResult(physicalRank - 1, zero);
    }
    if (reduceDim == d2m::ReduceDim::C) {
      outputAccum.setResult(physicalRank - 2, zero);
    }
    AffineMap maxOutputMap = outputAccum.getAffineMap();

    SmallVector<AffineMap> indexingMaps = {maxInputMap, maxScalerMap,
                                           maxOutputMap};

    // Build iterator types for reduce max: start with all parallel, then mark
    // reduced dimensions.
    SmallVector<mlir::Attribute> iteratorTypes(
        physicalRank,
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));
    if (reduceDim == d2m::ReduceDim::R) {
      iteratorTypes[physicalRank - 1] =
          ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Reduction);
    }
    if (reduceDim == d2m::ReduceDim::C) {
      iteratorTypes[physicalRank - 2] =
          ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Reduction);
    }

    d2m::GenericOp reduceMax1 = buildGenericLinAlg(
        rewriter, loc, maxInputs, maxOutputs, indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &bb, mlir::Location l, mlir::ValueRange bbArgs) {
          // bbArgs[0] = one input tile (a).
          // bbArgs[1] = one scaler tile (b).
          // bbArgs[2] = one output tile (c, the accumulator).
          return bb
              .create<d2m::TileReduceMaxOp>(l, bbArgs[2].getType(), bbArgs[0],
                                            bbArgs[1], bbArgs[2], reduceDimAttr)
              .getResult();
        });

    // Setting up the broadcast.
    // Broadcast the reduced result back to full tile shape.
    d2m::TileBcastType tileBcastType;
    switch (reduceDim) {
    case d2m::ReduceDim::R:
      // ReduceDim::R reduces the last dim (W) -> result is one value per row
      // (filled 0-column) -> broadcast as Col (replicate across columns).
      tileBcastType = d2m::TileBcastType::Col;
      break;
    case d2m::ReduceDim::C:
      // ReduceDim::C reduces the second-to-last dim (H) -> result is one value
      // per column (filled 0-row) -> broadcast as Row (replicate down).
      tileBcastType = d2m::TileBcastType::Row;
      break;
    case d2m::ReduceDim::RC:
      // Should never hit. RC reduction is converted into an R reduction
      // earlier.
      tileBcastType = d2m::TileBcastType::Scalar;
      break;
    }

    // Use the outputs from reduceMax1 as inputs to the broadcast.
    SmallVector<Value> bcastInputs(reduceMax1.getResults().begin(),
                                   reduceMax1.getResults().end());
    // Output type same as input type to argmax.
    auto bcastOrigOutputs = createDpsOutputs(loc, rewriter, {inputType});
    auto [bcastin, bcastOutputs] = toLayoutOperandsAndResults(
        rewriter, {SmallVector<Value>{}, bcastOrigOutputs}, true, noCollapse);

    // Build affine maps for the broadcast.
    AffineMap bcastInputMap = outputAccum.getAffineMap();
    AffineMap bcastOutputMap = rewriter.getMultiDimIdentityMap(physicalRank);
    SmallVector<AffineMap> bcastMaps = {bcastInputMap, bcastOutputMap};

    // All iterators are parallel for bcast op.
    SmallVector<mlir::Attribute> bcastIters(
        physicalRank,
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

    // Convert tileBcastType enum to attribute.
    auto tileBcastTypeAttr = d2m::TileBcastTypeAttr::get(ctx, tileBcastType);

    d2m::GenericOp bcast1 = buildGenericLinAlg(
        rewriter, loc, bcastInputs, bcastOutputs, bcastMaps, bcastIters,
        [&](mlir::OpBuilder &bb, mlir::Location l, mlir::ValueRange bbArgs) {
          // bbArgs[0] = one reduced tile (row-tile / col-tile / scalar-tile).
          // bbArgs[1] = one full output tile.
          return bb
              .create<d2m::TileBcastOp>(
                  l,
                  bbArgs[1].getType(), // result type = full tile type.
                  bbArgs[0],           // input = the reduced tile.
                  tileBcastTypeAttr    // how to expand it.
                  )
              .getResult();
        });

    // Setting up the tile_eq op.

    // Build affine maps for the equality comparison.
    AffineMap eqInputMap = rewriter.getMultiDimIdentityMap(physicalRank);
    AffineMap eqOutputMap = rewriter.getMultiDimIdentityMap(physicalRank);
    SmallVector<AffineMap> eqMaps = {eqInputMap, eqInputMap, eqOutputMap};

    // All iterators are parallel for tile_eq.
    SmallVector<mlir::Attribute> eqIters(
        physicalRank,
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

    auto eqOrigOutputs = createDpsOutputs(loc, rewriter, {inputType});
    // GridSelection requires each laid-out ToLayout result to feed exactly one
    // GenericOp, so create a separate layout for the input in eq1 rather than
    // reusing reduceMax1's maxInputs[0]. Mask padding with NegInf so padded
    // lanes don't spuriously compare equal to the broadcast max.
    auto [eqLaidOutInput, eqInputUnused] = toLayoutOperandsAndResults(
        rewriter, {SmallVector<Value>{convertedInput}, SmallVector<Value>{}},
        true, noCollapse, ttcore::OOBVal::NegInf);
    SmallVector<Value> eqInputs = {eqLaidOutInput[0], bcast1->getResult(0)};
    auto [eqin, eqOutputs] = toLayoutOperandsAndResults(
        rewriter, {SmallVector<Value>{}, eqOrigOutputs}, true, noCollapse);

    // Building the generic op for tile_eq.
    d2m::GenericOp eq1 = buildGenericLinAlg(
        rewriter, loc, eqInputs, eqOutputs, eqMaps, eqIters,
        [&](mlir::OpBuilder &bb, mlir::Location l, mlir::ValueRange bbArgs) {
          return bb
              .create<d2m::TileEqOp>(
                  l,
                  bbArgs[0].getType(), // result type = same tile type, values
                                       // are 0.0/1.0.
                  bbArgs[0],           // lhs = original input tile.
                  bbArgs[1]            // rhs = broadcasted max tile.
                  )
              .getResult();
        });

    // Setting up the arange op.

    auto arangeOrigOutputs = createDpsOutputs(loc, rewriter, {inputType});
    auto [arangeins, arangeOutputs] = toLayoutOperandsAndResults(
        rewriter, {SmallVector<Value>{}, arangeOrigOutputs}, true, noCollapse);
    Value arangeOutput = arangeOutputs[0];

    // Create a scratch tile (copied from D2MArangeOpRewriter).
    auto arangeTensorType =
        mlir::cast<RankedTensorType>(arangeOutput.getType());
    auto arangeLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(arangeTensorType.getEncoding());
    auto arangeTileType =
        mlir::cast<ttcore::TileType>(arangeTensorType.getElementType());
    Type arangeElemType = arangeTileType.getElementType();
    ArrayRef<int64_t> arangeGridShape =
        arangeLayout.getGridShape(arangeTensorType);
    SmallVector<int64_t> scratchShape(arangeGridShape.begin(),
                                      arangeGridShape.end());
    scratchShape.append({1, 1});
    auto scratchTileType = ttcore::TileType::get(arangeElemType);
    auto scratchLayout = ttcore::MetalLayoutAttr::get(
        ctx, SmallVector<int64_t>{1, 1}, ttcore::MemorySpace::DeviceL1,
        ttcore::TensorMemoryLayout::Sharded);
    Value indexTileTensor =
        rewriter
            .create<d2m::EmptyOp>(loc, scratchShape, scratchTileType,
                                  scratchLayout)
            .getResult();

    // Compute numElements: for C (column reduction), we need all output
    // columns; for R (row reduction), we need all output rows.
    int64_t numElements = (reduceDim == d2m::ReduceDim::C)
                              ? inputType.getDimSize(logicalRank - 2)
                              : inputType.getDimSize(logicalRank - 1);

    // Build affine maps: scratch input always at (0,0), output is identity.
    AffineMap arangeConstMap =
        AffineMap::get(physicalRank, 0, {zero, zero}, ctx);
    AffineMap arangeIdentMap = rewriter.getMultiDimIdentityMap(physicalRank);
    SmallVector<AffineMap> arangeMaps = {arangeConstMap, arangeIdentMap};

    // All iterators are parallel for arange.
    SmallVector<Attribute> arangeIters(
        physicalRank,
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

    SmallVector<Value> arangeGenericInputs = {indexTileTensor};
    auto arange = rewriter.create<d2m::GenericOp>(
        loc, arangeGenericInputs, arangeOutputs,
        /*additionalArgs=*/ValueRange(),
        rewriter.getAffineMapArrayAttr(arangeMaps),
        rewriter.getArrayAttr(arangeIters));

    // Fill the arange with descending indices (start=numElements, step=-1).
    // Should fill column major if reduceDim is C (i.e. argmax over columns,
    // output is a row).
    // Descending indices are used to ensure that the lowest index is returned
    // in case of a tie.
    withD2MGenericRegion(
        rewriter, loc, arange, arangeGenericInputs, arangeOutputs,
        [&](ArrayRef<Value> blockArgs) -> SmallVector<Value> {
          Value idxTile = blockArgs[0];
          Value outTile = blockArgs[1];
          Value result =
              rewriter
                  .create<d2m::ArangeBlockOp>(
                      loc, idxTile, outTile, numElements,
                      /*start=*/numElements,
                      /*step=*/-1,
                      (reduceDim == d2m::ReduceDim::C ? true : false))
                  .getResult();
          return {result};
        });

    // Setting up the broadcast after arange.

    // Broadcast first row or column to full shape.
    d2m::TileBcastType postArangeBcastType;
    switch (reduceDim) {
    case d2m::ReduceDim::R:
      postArangeBcastType = d2m::TileBcastType::Row;
      break;
    case d2m::ReduceDim::C:
      postArangeBcastType = d2m::TileBcastType::Col;
      break;
    case d2m::ReduceDim::RC:
      // Should never hit. RC reduction is converted into an R reduction
      // earlier.
      postArangeBcastType = d2m::TileBcastType::Scalar;
      break;
    }
    // Convert postArangeBcastType enum to attribute.
    d2m::TileBcastTypeAttr postArangeBcastTypeAttr =
        d2m::TileBcastTypeAttr::get(ctx, postArangeBcastType);
    SmallVector<Value> postArangeBcastInputs(arange.getResults().begin(),
                                             arange.getResults().end());
    auto postArangeBcastOrigOutputs =
        createDpsOutputs(loc, rewriter, {inputType});
    auto [postArangeBcastins, postArangeBcastOutputs] =
        toLayoutOperandsAndResults(
            rewriter, {SmallVector<Value>{}, postArangeBcastOrigOutputs}, true,
            noCollapse);

    // Build affine maps: input is reduced, output is identity.
    mlir::MutableAffineMap postArangeInMap(
        rewriter.getMultiDimIdentityMap(physicalRank));
    switch (postArangeBcastType) {
    case d2m::TileBcastType::Row:
      postArangeInMap.setResult(physicalRank - 2, zero);
      break;
    case d2m::TileBcastType::Col:
      postArangeInMap.setResult(physicalRank - 1, zero);
      break;
    default:
      break;
    }
    AffineMap postArangeBcastInputMap = postArangeInMap.getAffineMap();
    AffineMap postArangeBcastOutputMap =
        rewriter.getMultiDimIdentityMap(physicalRank);
    SmallVector<AffineMap> postArangeBcastMaps = {postArangeBcastInputMap,
                                                  postArangeBcastOutputMap};

    SmallVector<Attribute> postArangeBcastIters(
        physicalRank,
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

    d2m::GenericOp postArangeBcast = buildGenericLinAlg(
        rewriter, loc, postArangeBcastInputs, postArangeBcastOutputs,
        postArangeBcastMaps, postArangeBcastIters,
        [&](mlir::OpBuilder &bb, mlir::Location l, mlir::ValueRange bbArgs) {
          // bbArgs[0] = arange output [[32...1],...].
          // bbArgs[1] = output tile with row 0 copied across.
          return bb
              .create<d2m::TileBcastOp>(
                  l,
                  bbArgs[1].getType(),    // result type = full tile type.
                  bbArgs[0],              // input = arange tile.
                  postArangeBcastTypeAttr // how to expand it.
                  )
              .getResult();
        });

    Value indexOperand = postArangeBcast->getResult(0);

    // Setting up the tile_mul op.
    AffineMap mulInputMap = rewriter.getMultiDimIdentityMap(physicalRank);
    AffineMap mulOutputMap = rewriter.getMultiDimIdentityMap(physicalRank);
    SmallVector<AffineMap> mulMaps = {mulInputMap, mulInputMap, mulOutputMap};

    SmallVector<Attribute> mulIters(
        physicalRank,
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));

    auto mulOrigOutputs = createDpsOutputs(loc, rewriter, {inputType});
    SmallVector<Value> mulInputs{indexOperand, eq1->getResult(0)};
    auto [mulins, mulOutputs] = toLayoutOperandsAndResults(
        rewriter, {SmallVector<Value>{}, mulOrigOutputs}, true, noCollapse);

    d2m::GenericOp mul = buildGenericLinAlg(
        rewriter, loc, mulInputs, mulOutputs, mulMaps, mulIters,
        [&](mlir::OpBuilder &bb, mlir::Location l, mlir::ValueRange bbArgs) {
          return bb
              .create<d2m::TileMulOp>(
                  l,
                  bbArgs[0].getType(), // result type = same tile type.
                  bbArgs[0],           // lhs = result of bcast.
                  bbArgs[1]            // rhs = result of eq.
                  )
              .getResult();
        });

    // Setting up second reduce max.
    // Create a separate scaler for reduceMax2 rather than reusing reduceMax1's
    // maxInputs[1]: GridSelection requires each ToLayout result to feed exactly
    // one GenericOp.
    auto scaler2 = createScaler(rewriter, loc, inputType);
    auto [scaler2LaidOut, scaler2Unused] = toLayoutOperandsAndResults(
        rewriter, {SmallVector<Value>{scaler2}, SmallVector<Value>{}}, true,
        noCollapse);
    SmallVector<Value> max2Inputs = {mul->getResult(0), scaler2LaidOut[0]};
    auto max2OrigOutputs = createDpsOutputs(loc, rewriter, {reducedType});
    auto [max2ins, max2Outputs] = toLayoutOperandsAndResults(
        rewriter, {SmallVector<Value>{}, max2OrigOutputs}, true, noCollapse);

    // Building the generic op for the second reduce max.
    d2m::GenericOp reduceMax2 = buildGenericLinAlg(
        rewriter, loc, max2Inputs, max2Outputs, indexingMaps, iteratorTypes,
        [&](mlir::OpBuilder &bb, mlir::Location l, mlir::ValueRange bbArgs) {
          // bbArgs[0] = one mul(eq1, arange) tile (a).
          // bbArgs[1] = one scaler tile (b).
          // bbArgs[2] = one output tile (c, the accumulator).
          return bb
              .create<d2m::TileReduceMaxOp>(l, bbArgs[2].getType(), bbArgs[0],
                                            bbArgs[1], bbArgs[2], reduceDimAttr)
              .getResult();
        });

    // The arange uses descending indices (start=N, step=-1), so reduceMax2
    // returns `N - smallestMatchingIndex` (ties resolve to smallest index,
    // matching torch's argmax). Recover the actual index with `result = N -
    // result` via elementwise `tile_sub(fill(N), reduced)`.
    SmallVector<AffineMap> reflectMaps = {
        rewriter.getMultiDimIdentityMap(physicalRank),
        rewriter.getMultiDimIdentityMap(physicalRank)};
    SmallVector<mlir::Attribute> reflectIters(
        physicalRank,
        ttcore::IteratorTypeAttr::get(ctx, ttcore::IteratorType::Parallel));
    auto reflectOrigOutputs = createDpsOutputs(loc, rewriter, {reducedType});
    auto [reflectIns, reflectOutputs] = toLayoutOperandsAndResults(
        rewriter, {SmallVector<Value>{}, reflectOrigOutputs}, true, noCollapse);
    SmallVector<Value> reflectInputs = {reduceMax2->getResult(0)};
    d2m::GenericOp reflect = buildGenericLinAlg(
        rewriter, loc, reflectInputs, reflectOutputs, reflectMaps, reflectIters,
        [&](mlir::OpBuilder &bb, mlir::Location l, mlir::ValueRange bbArgs) {
          auto tileTy = bbArgs[0].getType();
          auto elemTy = mlir::cast<ttcore::TileType>(tileTy).getElementType();
          auto nAttr =
              mlir::FloatAttr::get(elemTy, static_cast<double>(numElements));
          Value nScalar = bb.create<mlir::arith::ConstantOp>(l, elemTy, nAttr);
          Value nTile =
              bb.create<d2m::TileFillOp>(l, tileTy, nScalar).getResult();
          return bb.create<d2m::TileSubOp>(l, tileTy, nTile, bbArgs[0])
              .getResult();
        });

    // Convert the reduced bf16 index back to plain (untiled) host layout first,
    // then typecast to si32.
    auto reducedHostType = RankedTensorType::get(reducedType.getShape(),
                                                 inputType.getElementType());
    Value reducedHost =
        unLayoutResult(rewriter, reflect->getResult(0), reducedHostType)
            ->getResult(0);

    // Cast the reduced bf16 index to si32. For a rank-normalized argmax the
    // reduced result is rank-2 [1, 1]; typecast at that rank and reshape back
    // to the original (possibly rank-0/1) output shape afterwards.
    auto typecastResultType =
        rankNormalized ? RankedTensorType::get(reducedType.getShape(),
                                               outputType.getElementType())
                       : outputType;
    Value result =
        buildTypecastGeneric(rewriter, loc, reducedHost, typecastResultType);

    if (rankNormalized) {
      result = rewriter.create<ttir::ReshapeOp>(
          loc, outputType, result,
          rewriter.getI32ArrayAttr(SmallVector<int32_t>(
              outputType.getShape().begin(), outputType.getShape().end())));
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace mlir::tt

namespace mlir::tt {
void populateTTIRToD2MPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                               TypeConverter &typeConverter,
                               ttcore::MemorySpace defaultInputMemSpace,
                               ttcore::MemorySpace defaultOutputMemSpace,
                               bool ttnnMode, bool collapseTensors,
                               bool enableMulticastInference) {
  // clang-format off
  patterns.add<
    // Elementwise.
    D2MNamedElementwiseRewriter<ttir::AbsOp,             d2m::TileAbsOp>,
    D2MNamedElementwiseRewriter<ttir::AcosOp,            d2m::TileAcosOp>,
    D2MNamedElementwiseRewriter<ttir::AddOp,             d2m::TileAddOp>,
    D2MNamedElementwiseRewriter<ttir::AsinOp,            d2m::TileAsinOp>,
    D2MNamedElementwiseRewriter<ttir::AtanOp,            d2m::TileAtanOp>,
    D2MNamedElementwiseRewriter<ttir::Atan2Op,           d2m::TileAtan2Op>,
    D2MNamedElementwiseRewriter<ttir::BitwiseAndOp,      d2m::TileBitwiseAndOp>,
    D2MNamedElementwiseRewriter<ttir::BitwiseNotOp,      d2m::TileBitwiseNotOp>,
    D2MNamedElementwiseRewriter<ttir::BitwiseOrOp,       d2m::TileBitwiseOrOp>,
    D2MNamedElementwiseRewriter<ttir::BitwiseXorOp,      d2m::TileBitwiseXorOp>,
    D2MNamedElementwiseRewriter<ttir::CeilOp,            d2m::TileCeilOp>,
    D2MNamedElementwiseRewriter<ttir::ClampScalarOp,     d2m::TileClampScalarOp>,
    D2MNamedElementwiseRewriter<ttir::ClampTensorOp,     d2m::TileMaximumOp>,
    D2MNamedElementwiseRewriter<ttir::CosOp,             d2m::TileCosOp>,
    D2MNamedElementwiseRewriter<ttir::DivOp,             d2m::TileDivOp>,
    D2MNamedElementwiseRewriter<ttir::ErfOp,             d2m::TileErfOp>,
    D2MNamedElementwiseRewriter<ttir::ErfcOp,            d2m::TileErfcOp>,
    D2MNamedElementwiseRewriter<ttir::ExpOp,             d2m::TileExpOp>,
    D2MNamedElementwiseRewriter<ttir::Exp2Op,            d2m::TileExp2Op>,
    D2MNamedElementwiseRewriter<ttir::Expm1Op,          d2m::TileExpm1Op>,
    D2MNamedElementwiseRewriter<ttir::FloorOp,           d2m::TileFloorOp>,
    D2MNamedElementwiseRewriter<ttir::FracOp,           d2m::TileFracOp>,
    D2MNamedElementwiseRewriter<ttir::GeluOp,            d2m::TileGeluOp>,
    D2MNamedElementwiseRewriter<ttir::HardsigmoidOp,     d2m::TileHardsigmoidOp>,
    D2MNamedElementwiseRewriter<ttir::LogOp,             d2m::TileLogOp>,
    D2MNamedElementwiseRewriter<ttir::Log1pOp,          d2m::TileLog1pOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalAndOp,        d2m::TileMulOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalLeftShiftOp,  d2m::TileLogicalLeftShiftOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalNotOp,        d2m::TileLogicalNotOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalOrOp,         d2m::TileAddOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalRightShiftOp, d2m::TileLogicalRightShiftOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalXorOp,        d2m::TileSubOp>,
    D2MNamedElementwiseRewriter<ttir::RightShiftOp,        d2m::TileRightShiftOp>,
    D2MNamedElementwiseRewriter<ttir::MultiplyOp,        d2m::TileMulOp>,
    D2MNamedElementwiseRewriter<ttir::MaximumOp,         d2m::TileMaximumOp>,
    D2MNamedElementwiseRewriter<ttir::MinimumOp,         d2m::TileMinimumOp>,
    D2MNamedElementwiseRewriter<ttir::NegOp,             d2m::TileNegativeOp>,
    D2MNamedElementwiseRewriter<ttir::PowOp,             d2m::TilePowOp>,
    D2MNamedElementwiseRewriter<ttir::ReciprocalOp,      d2m::TileRecipOp>,
    D2MNamedElementwiseRewriter<ttir::ReluOp,            d2m::TileReluOp>,
    D2MNamedElementwiseRewriter<ttir::RsqrtOp,           d2m::TileRsqrtOp>,
    D2MNamedElementwiseRewriter<ttir::SigmoidOp,         d2m::TileSigmoidOp>,
    D2MNamedElementwiseRewriter<ttir::SignOp,            d2m::TileSignOp>,
    D2MNamedElementwiseRewriter<ttir::SignbitOp,        d2m::TileSignbitOp>,
    D2MNamedElementwiseRewriter<ttir::SeluOp,           d2m::TileSeluOp>,
    D2MNamedElementwiseRewriter<ttir::SiluOp,            d2m::TileSiluOp>,
    D2MNamedElementwiseRewriter<ttir::SoftsignOp,       d2m::TileSoftsignOp>,
    D2MNamedElementwiseRewriter<ttir::SinOp,             d2m::TileSinOp>,
    D2MNamedElementwiseRewriter<ttir::SqrtOp,            d2m::TileSqrtOp>,
    D2MNamedElementwiseRewriter<ttir::SquareOp,          d2m::TileSquareOp>,
    D2MNamedElementwiseRewriter<ttir::SubtractOp,        d2m::TileSubOp>,
    D2MNamedElementwiseRewriter<ttir::TanOp,             d2m::TileTanOp>,
    D2MNamedElementwiseRewriter<ttir::TanhOp,            d2m::TileTanhOp>,
    D2MNamedElementwiseRewriter<ttir::TruncOp,          d2m::TileTruncOp>,
    D2MNamedElementwiseRewriter<ttir::WhereOp,           d2m::TileWhereOp>,
    // Comparison.
    D2MNamedElementwiseRewriter<ttir::EqualOp,           d2m::TileEqOp>,
    D2MNamedElementwiseRewriter<ttir::NotEqualOp,        d2m::TileNeOp>,
    D2MNamedElementwiseRewriter<ttir::GreaterThanOp,     d2m::TileGtOp>,
    D2MNamedElementwiseRewriter<ttir::GreaterEqualOp,    d2m::TileGeOp>,
    D2MNamedElementwiseRewriter<ttir::LessThanOp,        d2m::TileLtOp>,
    D2MNamedElementwiseRewriter<ttir::LessEqualOp,       d2m::TileLeOp>,
    // Outer-dim (and integer) reductions: accumulate full-tile binary ops.
    D2MNamedAccumReductionRewriter<ttir::SumOp,  d2m::TileAddOp>,
    D2MNamedAccumReductionRewriter<ttir::MaxOp,  d2m::TileMaximumOp>,
    D2MNamedAccumReductionRewriter<ttir::MinOp,  d2m::TileMinimumOp>,
    D2MNamedAccumReductionRewriter<ttir::MeanOp, d2m::TileAddOp>,
    // Inner (tile C/R) reductions: tile_reduce_* (float) /
    // tile_sfpu_reduce_* (integer). Mean has no integer variant.
    D2MNamedTileReduceRewriter<ttir::MaxOp,  d2m::TileReduceMaxOp, d2m::TileSFPUReduceMaxOp>,
    D2MNamedTileReduceRewriter<ttir::MeanOp, d2m::TileReduceMeanOp>,
    D2MNamedTileReduceRewriter<ttir::SumOp,  d2m::TileReduceSumOp, d2m::TileSFPUReduceSumOp>,
    // Data movement.
    D2MNamedElementwiseRewriter<ttir::TypecastOp,        d2m::TileTypecastOp>,
    D2MBroadcastRewriter,
    // Argmax
    D2MArgMaxRewriter,
    // Tensor manipulation/View ops.
    D2MConcatRewriter,
    D2MTensorManipulationOpRewriter<ttir::RearrangeOp,        rearrangeLogicalInfo>,
    D2MTensorManipulationOpRewriter<ttir::ReshapeOp,          reshapeLogicalInfo>,
    D2MTensorManipulationOpRewriter<ttir::SliceStaticOp,      sliceLogicalInfo>,
    D2MTensorManipulationOpRewriter<ttir::ConcatenateHeadsOp, concatenateHeadsLogicalInfo>,
    // Permute (handles transpose ops, since they're canonicalized into permutes).
    D2MPermuteRewriter,
    D2MMatmulBlockToLinalgGeneric,
    D2MTensorManipulationOpRewriter<ttir::PermuteOp, permuteLogicalInfo>,
    // Full / zeros / ones (constant fill via tile_fill).
    D2MConstantFillOpRewriter<ttir::FullOp>,
    D2MConstantFillOpRewriter<ttir::ZerosOp>,
    D2MConstantFillOpRewriter<ttir::OnesOp>,
    // CCL
    D2MAllGatherRewriter
  >(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, ttnnMode, collapseTensors, enableMulticastInference);

  // Handle SliceStatic cases that need a transpose-based rewrite to satisfy
  // NoC alignment before the generic SliceStatic lowering consumes them.
  patterns.add<D2MSliceStaticOpNoCConstraintsRewriter>(typeConverter, ctx);

  // Decompose inner-dim min reductions to neg(max(neg)); runs during
  // TTIRToD2M instead of as a separate pre-pass.
  patterns.add<D2MInnerMinDecompositionRewriter>(typeConverter, ctx);

  // ToLayout 1:1 conversion.
  patterns.add<D2MToLayoutOpRewriter>(typeConverter, ctx, ttnnMode);

  // Creation ops 1:1 conversion.
  patterns.add<D2MEmptyOpRewriter>(typeConverter, ctx);

  // Mesh ops 1:1 conversion.
  patterns.add<D2MMeshShardOpRewriter>(typeConverter, ctx);

  // Matmul.
  patterns.add<D2MMatmulRewriter<d2m::TileMatmulOp>>(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace,  ttnnMode, collapseTensors, enableMulticastInference);

  // Arange.
  patterns.add<D2MArangeOpRewriter>(typeConverter, ctx, defaultInputMemSpace,
                                    defaultOutputMemSpace, ttnnMode,
                                    collapseTensors, enableMulticastInference);

  // TopK.
  patterns.add<D2MTopKRewriter>(typeConverter, ctx, defaultInputMemSpace,
                                defaultOutputMemSpace, ttnnMode,
                                collapseTensors, enableMulticastInference);

  // Embedding.
  patterns.add<D2MEmbeddingOpRewriter>(
    typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, ttnnMode,
    collapseTensors, enableMulticastInference);

  // Rand.
  patterns.add<D2MRandOpRewriter>(typeConverter, ctx, defaultInputMemSpace,
    defaultOutputMemSpace, ttnnMode,
    collapseTensors, enableMulticastInference);

  // clang-format on
}

#define GEN_PASS_DEF_TTIRTOD2M
#include "ttmlir/Conversion/Passes.h.inc"

namespace {
class TTIRToD2MPass final
    : public mlir::tt::impl::TTIRToD2MBase<TTIRToD2MPass> {
public:
  using Base = mlir::tt::impl::TTIRToD2MBase<TTIRToD2MPass>;

  TTIRToD2MPass() = default;

  TTIRToD2MPass(const TTIRToD2MOptions &options) : Base() {
    this->defaultInputMemSpace = options.defaultInputMemSpace;
    this->defaultOutputMemSpace = options.defaultOutputMemSpace;
    this->ttnnMode = options.ttnnMode;
    this->collapseTensorsTo2D = options.collapseTensorsTo2D;
    this->enableMulticastInference = options.enableMulticastInference;
  }

  TTIRToD2MPass(const TTIRToD2MPass &rhs) : Base(rhs) {
    // Workaround: Passes are required to be copy-constructible but autogen'ed
    // base class copy constructors ignore Pass option fields.
    this->defaultInputMemSpace = rhs.defaultInputMemSpace;
    this->defaultOutputMemSpace = rhs.defaultOutputMemSpace;
    this->ttnnMode = rhs.ttnnMode;
    this->collapseTensorsTo2D = rhs.collapseTensorsTo2D;
    this->enableMulticastInference = rhs.enableMulticastInference;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });

    RewritePatternSet patterns(ctx);
    populateTTIRToD2MPatterns(ctx, patterns, typeConverter,
                              defaultInputMemSpace, defaultOutputMemSpace,
                              ttnnMode, collapseTensorsTo2D,
                              enableMulticastInference);

    ConversionTarget target(*ctx);
    target.addIllegalDialect<mlir::tt::ttir::TTIRDialect>();
    target.addLegalDialect<::mlir::BuiltinDialect>();
    target.addLegalDialect<::mlir::func::FuncDialect>();
    target.addLegalDialect<::mlir::linalg::LinalgDialect>();
    target.addLegalDialect<::mlir::arith::ArithDialect>();
    target.addLegalDialect<::mlir::affine::AffineDialect>();
    target.addLegalDialect<mlir::tt::d2m::D2MDialect>();
    target.addLegalDialect<mlir::tt::ttcore::TTCoreDialect>();

    // Keep some TTIR ops legal if they don't have D2M equivalents.
    target.addLegalOp<ttir::TTNNMetalLayoutCastOp>();

    target.addIllegalOp<mlir::tt::d2m::TileMatmulBlockOp>();

    // Tensor ops are used for local scratch buffers and scalar-only reshapes
    // that should stay in host tensor space instead of creating rank-0 layouts.
    target
        .addLegalOp<::mlir::tensor::EmptyOp, ::mlir::tensor::ExtractOp,
                    ::mlir::tensor::FromElementsOp, ::mlir::tensor::InsertOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToD2MPass() {
  return std::make_unique<TTIRToD2MPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createTTIRToD2MPass(const TTIRToD2MOptions &options) {
  return std::make_unique<TTIRToD2MPass>(options);
}

} // namespace mlir::tt
