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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <array>

namespace mlir::tt {

namespace {
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
  /// `D2MNamedOuterReductionRewriter` and `D2MNamedInnerReductionRewriter`).
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
    impliedGrid[impliedGrid.size() - 1] /=
        ttcore::TileType::getDefaultShape()[0];
    impliedGrid[impliedGrid.size() - 2] /=
        ttcore::TileType::getDefaultShape()[1];

    return impliedGrid;
  }

  llvm::SmallVector<int64_t>
  getLegacyGrid(ttnn::TTNNLayoutAttr ttnnLayout) const {
    llvm::SmallVector<int64_t> ttnnGridShape(ttnnLayout.getGrid().getShape());

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
    dimAlignments[dimAlignments.size() - 1] =
        ttcore::TileType::getDefaultShape()[0];
    dimAlignments[dimAlignments.size() - 2] =
        ttcore::TileType::getDefaultShape()[1];

    ttcore::MetalLayoutAttr metalLayout;
    if (mlir::isa<ttnn::TTNNNDLayoutAttr>(ttnnLayout)) {
      // There is no dim collapsing for ND layouts.
      auto emptyIntervalType = RankedTensorType::get(
          {0, 2}, IntegerType::get(rewriter.getContext(), 64));
      DenseIntElementsAttr emptyCollapseIntervals =
          DenseIntElementsAttr::get(emptyIntervalType, ArrayRef<int64_t>{});
      metalLayout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), tensorType.getShape(), ttcore::OOBVal::Undef,
          memSpace, memLayout, emptyCollapseIntervals, dimAlignments);
    } else {
      metalLayout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), tensorType.getShape(), ttcore::OOBVal::Undef,
          memSpace, memLayout,
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

  // Create a ToLayout operation for a value using the provided layout
  // information with a simple 1x1 grid; actual grid optimization and proper
  // dimension alignments are computed later in the D2MGridSelection pass.
  Value createOptimalLayoutOp(Value value, ttcore::MemorySpace memSpace,
                              bool tiled, bool noCollapse,
                              mlir::ConversionPatternRewriter &rewriter,
                              ttcore::OOBVal oobVal) const {
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
          llvm::SmallVector<int64_t> ttnnGridShape(
              ttnnLayout.getGrid().getShape());
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
    ArrayRef<int64_t> logicalShape = tensorType.getShape();

    Type elementType = tensorType.getElementType();
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
          rewriter.getContext(), logicalShape, ttcore::OOBVal::Undef, memSpace,
          ttcore::TensorMemoryLayout::Sharded, emptyCollapseIntervals);

    } else {
      layout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), logicalShape, oobVal, memSpace,
          ttcore::TensorMemoryLayout::Sharded);
    }

    // Get raw, unsharded physical shape.
    llvm::SmallVector<int64_t> unshardedShape =
        layout.getPhysicalShape(tileShape);

    // Use a placeholder, 1-filled grid for this pass.
    llvm::SmallVector<int64_t> simpleGrid(unshardedShape.size(), 1);

    llvm::SmallVector<int64_t> shardedShape =
        layout.getDeviceShape(simpleGrid, tileShape);

    auto emptyOp = rewriter.create<d2m::EmptyOp>(value.getLoc(), shardedShape,
                                                 elementType, layout);

    // For ND tensors (logicalShape.size() > 2), set placeholder virtual grid
    // mappings on the EmptyOp.  These will be replaced when GridSelection
    // optimizes the grid.
    if (logicalShape.size() > 2) {
      auto [forwardMap, inverseMap] =
          ttmlir::d2m::utils::grids::createCoreVirtMaps(rewriter.getContext(),
                                                        simpleGrid, {1, 1});
      emptyOp.setVirtualGridInverseMappingAttr(AffineMapAttr::get(inverseMap));
      emptyOp.setVirtualGridForwardMappingAttr(AffineMapAttr::get(forwardMap));
    }

    return rewriter.create<d2m::ToLayoutOp>(value.getLoc(), value, emptyOp)
        ->getResult(0);
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
      generic->getOperand(i);

      // Build grid indices from the indexing map
      SmallVector<Value> indices = inputIndices[i];

      // Create a buffer for the load result
      auto bufferOp = builder.create<tensor::EmptyOp>(
          loc, shardType.getShape(), shardType.getElementType());
      Value buffer = bufferOp.getResult();

      operands.push_back(buffer);
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

    const std::size_t physicalRank =
        ttcore::getDeviceLayout(output).getRank() / 2;

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
  static constexpr bool isComparisonOp =
      std::is_same_v<ConcreteOp, ttir::EqualOp> ||
      std::is_same_v<ConcreteOp, ttir::NotEqualOp> ||
      std::is_same_v<ConcreteOp, ttir::GreaterThanOp> ||
      std::is_same_v<ConcreteOp, ttir::GreaterEqualOp> ||
      std::is_same_v<ConcreteOp, ttir::LessThanOp> ||
      std::is_same_v<ConcreteOp, ttir::LessEqualOp>;

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
    TT_assert(outRank >= 2);
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
    if constexpr (isComparisonOp) {
      // For comparison ops, first subtract then compare with zero.
      yield = bbBuilder.create<d2m::TileSubOp>(loc, resultTypes, operands);
      yield = bbBuilder.create<TileOp>(loc, resultTypes, yield);
    } else if constexpr (std::is_same_v<ConcreteOp, ttir::ClampTensorOp>) {
      // Decompose into maximum(input, min) then minimum(result, max).
      yield = bbBuilder.create<d2m::TileMaximumOp>(
          loc, resultTypes, ValueRange{operands[0], operands[1]});
      yield = bbBuilder.create<d2m::TileMinimumOp>(
          loc, resultTypes, ValueRange{yield, operands[2]});
    } else if constexpr (std::is_same_v<ConcreteOp, ttir::ClampScalarOp>) {
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

    SmallVector<mlir::AffineMap> bcastIndexingMaps;
    SmallVector<d2m::TileBcastType> tileBcastTypes;
    std::tie(bcastIndexingMaps, tileBcastTypes) =
        getImplicitBcastInfo(rewriter, origInputs, origOutputs);

    // Implicit bcast if tile-level bcast exists or any input indexing map is
    // not identity.
    const bool isImplicitBcast =
        !bcastIndexingMaps.empty() &&
        llvm::any_of(ArrayRef<mlir::AffineMap>(bcastIndexingMaps)
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

    auto indexingMaps =
        isImplicitBcast
            ? bcastIndexingMaps
            : getAffineMapsArray(rewriter, numOperands, physicalRank);

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

        // Create 'linalg.generic' accepting 'blockArgs'.
        auto linalgIndexingMaps =
            isImplicitBcast
                ? bcastIndexingMaps
                : getAffineMapsArray(rewriter, numOperands, physicalRank);

        SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
            iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

        // Collect attributes to forward to tile ops (e.g., min/max for clamp).
        SmallVector<NamedAttribute> opAttrs;
        if constexpr (std::is_same_v<ConcreteOp, ttir::ClampScalarOp>) {
          opAttrs.push_back(rewriter.getNamedAttr("min", op.getMinAttr()));
          opAttrs.push_back(rewriter.getNamedAttr("max", op.getMaxAttr()));
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

// ----------------------------------------------------------------------------
//
// D2MNamedOuterReductionRewriter: outer logical-dimension reductions require
// explicit accumulation into the output buffer via d2m.generic with reduction
// iterators and remote load/store.
namespace {
template <typename ConcreteOp, typename TileAccumulateOp>
class D2MNamedOuterReductionRewriter final
    : public mlir::OpConversionPattern<ConcreteOp>,
      D2MNamedRewriterCommon {
public:
  D2MNamedOuterReductionRewriter(const TypeConverter &typeConverter,
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

  static mlir::Attribute zeroFillAttrForOutput(mlir::OpBuilder &builder,
                                               mlir::Type elemType) {
    if (mlir::isa<mlir::FloatType>(elemType)) {
      return mlir::FloatAttr::get(builder.getF32Type(), 0.0);
    }
    return mlir::IntegerAttr::get(builder.getI32Type(), 0);
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

    mlir::FailureOr<mlir::Value> filledOutput = lowerRankedTensorFillViaGeneric(
        rewriter, loc, outputType,
        zeroFillAttrForOutput(rewriter, outputType.getElementType()));
    if (mlir::failed(filledOutput)) {
      return failure();
    }

    mlir::SmallVector<mlir::Value> origInputs = adaptor.getOperands();
    mlir::SmallVector<mlir::Value> origOutputs = {*filledOutput};
    bool noCollapse = inputType.getRank() > 2;
    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {origInputs, origOutputs},
        /*tiled=*/true, noCollapse, getReductionOOBVal());

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = numInputs + numOutputs;
    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;
    assertPhysicalIteratorRankForReduction(physicalRank);

    mlir::SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, op, numOperands, physicalRank);
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

    mlir::AffineMap outputIndexingMap = generic.getIndexingMap(numInputs);
    mlir::SmallVector<mlir::Value> outputIndices =
        d2m::utils::buildGridIndices(rewriter, loc, outputIndexingMap);
    mlir::Value outputLoadBuffer = rewriter.create<mlir::tensor::EmptyOp>(
        loc, outputShardType.getShape(), outputShardType.getElementType());
    mlir::Value accumulatorSlice = createRemoteTensorSlice(
        rewriter, loc, outputShardType, outputLoadBuffer,
        generic->getOperand(numInputs), outputIndices, {});

    std::size_t shardRank = outputShardType.getRank();
    auto [accumulatorMap, linalgIteratorTypes] =
        shardLinalgAccumulationSignature(rewriter, dimArg, shardRank);
    mlir::SmallVector<mlir::AffineMap> linalgIndexingMaps = {
        rewriter.getMultiDimIdentityMap(shardRank), accumulatorMap};

    auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
        loc, mlir::TypeRange{accumulatorSlice.getType()},
        mlir::ValueRange{inputSlice}, mlir::ValueRange{accumulatorSlice},
        linalgIndexingMaps, linalgIteratorTypes,
        [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
            mlir::ValueRange bbArgs) {
          mlir::Value reduced = bbBuilder.create<TileAccumulateOp>(
              bbLoc, mlir::TypeRange{bbArgs[1].getType()},
              mlir::ValueRange{bbArgs[0], bbArgs[1]});
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
    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op->getResult(0).getType()));
    return mlir::success();
  }

  static constexpr ttcore::OOBVal getReductionOOBVal() {
    if constexpr (std::is_same_v<TileAccumulateOp, d2m::TileMaximumOp>) {
      return ttcore::OOBVal::NegInf;
    }
    return ttcore::OOBVal::Zero;
  }

  static bool isOuterReduction(ConcreteOp op) {
    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    int64_t rank = inputType.getRank();
    std::optional<ArrayAttr> maybeDimArg = op.getDimArg();
    if (rank < 2 || !maybeDimArg.has_value()) {
      return false;
    }
    for (auto dimAttr : *maybeDimArg) {
      int64_t dim = mlir::cast<IntegerAttr>(dimAttr).getInt();
      std::size_t n =
          normalizeReductionDimIndex(dim, static_cast<std::size_t>(rank));
      if (static_cast<int64_t>(n) < rank - 2) {
        return true;
      }
    }
    return false;
  }

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, ConcreteOp op, std::size_t arity,
                     std::size_t rank) {
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
    SmallVector<mlir::AffineMap> maps(arity - 1,
                                      builder.getMultiDimIdentityMap(rank));
    maps.emplace_back(accumulator.getAffineMap());
    return maps;
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

// D2MNamedInnerReductionRewriter: tile C/R reductions (last two physical
// iterator dims), similar to the elementwise group except ops whose tiled
// counterparts need a scaler (e.g. 1/N for mean)—a single filled tile
// broadcast across the lhs indexing space.
namespace {
template <typename ConcreteOp, typename TileOp>
class D2MNamedInnerReductionRewriter final
    : public mlir::OpConversionPattern<ConcreteOp>,
      D2MNamedRewriterCommon {

public:
  D2MNamedInnerReductionRewriter<ConcreteOp, TileOp>(
      const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
      ttcore::MemorySpace defaultInputMemSpace,
      ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
      bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

private:
  // Return the identity OOB fill value for this reduction's tile op.
  // Padded elements must not affect the reduction result.
  static constexpr ttcore::OOBVal getReductionOOBVal() {
    if constexpr (std::is_same_v<TileOp, d2m::TileReduceMaxOp>) {
      return ttcore::OOBVal::NegInf;
    } else if constexpr (std::is_same_v<TileOp, d2m::TileReduceSumOp> ||
                         std::is_same_v<TileOp, d2m::TileReduceMeanOp>) {
      return ttcore::OOBVal::Zero;
    } else {
      static_assert(ttmlir::utils::always_false<TileOp>(),
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
    SmallVector<mlir::Value> newInputs(origInputs.begin(), origInputs.end());
    newInputs.emplace_back(this->createScaler(
        rewriter, loc,
        mlir::cast<mlir::RankedTensorType>(origInputs.front().getType()),
        getScaleValue(op)));
    auto inputTensorType =
        mlir::cast<RankedTensorType>(origInputs.front().getType());
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
        getAffineMapsArray(rewriter, op, numOperands, physicalRank);
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

        SmallVector<mlir::AffineMap> linalgIndexingMaps =
            getAffineMapsArray(rewriter, op, numOperands, physicalRank);
        SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
            iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

        // Propagate attributes.

        SmallVector<mlir::NamedAttribute> attributes;
        {
          // Propagate 'dim_arg' as 'ReduceDim'.
          attributes.emplace_back(
              d2m::ReduceDimAttr::getMnemonic(),
              d2m::ReduceDimAttr::get(ctx,
                                      dimArgAsReduceDim(op, physicalRank)));
        }

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
              mlir::Value yield = bbBuilder.create<TileOp>(
                  loc,
                  /* resultTypes */ bbArgs.take_back(numOutputs).getTypes(),
                  /* operands */ bbArgs, attributes);
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
                     std::size_t rank) {
    mlir::ArrayAttr dimArg = getDimArg(op);

    mlir::AffineExpr zero =
        mlir::getAffineConstantExpr(0, builder.getContext());

    mlir::MutableAffineMap accumulator(builder.getMultiDimIdentityMap(rank));
    forAllDims(rank, dimArg, [&](std::size_t index, bool dropped) {
      if (dropped) {
        accumulator.setResult(index, zero);
      }
    });
    SmallVector<mlir::AffineMap> maps(arity - 2,
                                      builder.getMultiDimIdentityMap(rank));
    std::array<mlir::AffineExpr, 2> zeros{zero, zero};
    maps.emplace_back(mlir::AffineMap::get(/* dimCount */ rank,
                                           /* symbolCount */ 0, zeros,
                                           builder.getContext()));
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

  // Create a reduction scaler value for a given type of tensor operand
  // (tile_fill inside d2m.generic, same path as ttir.full lowering).
  mlir::Value createScaler(mlir::ConversionPatternRewriter &rewriter,
                           mlir::Location loc, mlir::RankedTensorType inputType,
                           double fillValue = 1.0) const {

    Type elementType = inputType.getElementType();
    Attribute encoding = nullptr;

    // If input has TTNNLayoutAttr, create scaler with matching layout.
    // This ensures the scaler goes through TTNNMetalLayoutCastOp path.
    if (auto ttnnLayout = mlir::dyn_cast_if_present<ttnn::TTNNLayoutAttr>(
            inputType.getEncoding())) {
      // Create a 1x1 grid TTNNLayoutAttr for the scaler.
      auto grid = ttcore::GridAttr::get(rewriter.getContext(), {1, 1});
      auto tileType = ttcore::TileType::get(
          elementType, ttcore::TileType::getDefaultShape());
      auto memref =
          MemRefType::get({1, 1}, tileType, MemRefLayoutAttrInterface{},
                          ttnnLayout.getMemref().getMemorySpace());
      encoding = ttnn::TTNNLayoutAttr::get(
          rewriter.getContext(), rewriter.getMultiDimIdentityMap(2), grid,
          memref, ttnnLayout.getMemLayout(),
          /*tensorMesh=*/nullptr, /*ignorePhysicalLayout=*/false,
          /*exactGrid=*/true);
    }

    mlir::RankedTensorType scalerType = RankedTensorType::get(
        ttcore::TileType::getDefaultShape(), elementType, encoding);

    mlir::Attribute fillAttr;
    if (mlir::isa<mlir::FloatType>(elementType)) {
      fillAttr = mlir::FloatAttr::get(rewriter.getF32Type(), fillValue);
    } else if (mlir::isa<mlir::IntegerType>(elementType)) {
      fillAttr = mlir::IntegerAttr::get(rewriter.getI32Type(),
                                        static_cast<int64_t>(fillValue));
    } else {
      llvm_unreachable("unexpected input element type");
    }

    mlir::FailureOr<mlir::Value> filled = this->lowerRankedTensorFillViaGeneric(
        rewriter, loc, scalerType, fillAttr);
    if (mlir::failed(filled)) {
      llvm_unreachable("scaler fill lowering failed");
    }
    return *filled;
  }

  /// Map `dim_arg` into `ReduceDim` for the **physical** iterator rank of the
  /// d2m.generic (tile C = index rank-2, R = rank-1). If neither is reduced,
  /// the reduction only hits "outer" iterator dims: `ttir.sum` should have
  /// matched `D2MNamedOuterReductionRewriter` first; other TTIR reduction ops
  /// do not yet have an outer-dimension lowering here.
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
        "D2MNamedInnerReductionRewriter: dim_arg does not reduce tile C or R "
        "(last "
        "two physical iterator dims). For ttir.sum over outer logical dims, "
        "D2MNamedOuterReductionRewriter should match first; max/mean (and "
        "other "
        "reductions) are not lowered for that case yet.");
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
    checkPreconditions(op);

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

    // TODO(#2591) handle 'transpose_{a,b}' attributes.

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, numOperands, physicalRank);
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
          rewriter.create<TileOp>(loc,
                                  /* resultTypes */ mlir::TypeRange(),
                                  /* operands */ blockArgs);

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

          static constexpr std::size_t tileOpNumInputs = 3;
          static constexpr std::size_t tileOpNumOutputs = 1;

          SmallVector<mlir::AffineMap> linalgIndexingMaps =
              getAffineMapsArray(rewriter, numOperands, physicalRank);
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
                mlir::Value yield = bbBuilder.create<TileOp>(
                    loc, /* resultTypes */
                    bbArgs.take_back(tileOpNumOutputs).getTypes(),
                    /* operands */ bbArgs.take_front(tileOpNumInputs));

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
    assert((!op.getTransposeA() && !op.getTransposeB()) &&
           "TODO(#2591) expected no transpose attributes");

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
  /// \param builder OpBuilder for creating affine expressions
  /// \param arity Number of operands (must be 3: LHS, RHS, OUT)
  /// \param rank Physical rank of the matmul operation (logical tensor rank)
  /// \return Vector of affine maps for [LHS, RHS, OUT]
  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, std::size_t arity,
                     std::size_t rank) {
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

    // RHS last two dimensions: [..., K, N]
    rhsExprs.push_back(builder.getAffineDimExpr(rank));     // K (contraction)
    rhsExprs.push_back(builder.getAffineDimExpr(rank - 1)); // N (columns)

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
        ctx, permuted.logicalShape, inputLayout.getOobVal(),
        inputLayout.getMemorySpace(), inputLayout.getMemoryLayout(),
        inputLayout.getCollapsedIntervals(), permuted.dimAlignments);

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

  PermutationResult
  computePermutation(mlir::ConversionPatternRewriter &rewriter,
                     ArrayRef<int64_t> permutation,
                     ArrayRef<int64_t> inputPhysicalShape, unsigned deviceRank,
                     ArrayRef<int64_t> inputLogicalShape,
                     ArrayRef<int64_t> inputDimAlignments) const {

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
  using ConcreteOp = ttir::ConcatOp;

public:
  D2MConcatRewriter(const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
                    ttcore::MemorySpace defaultInputMemSpace,
                    ttcore::MemorySpace defaultOutputMemSpace,
                    bool /*ttnnMode*/, bool /*collapseTensors*/,
                    bool enableMulticastInference)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               /*ttnnMode=*/false, /*collapseTensors=*/false,
                               enableMulticastInference) {}

  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto origInputs = adaptor.getOperands();
    assert(origInputs.size() > 1);

    auto origOutputs =
        createDpsOutputs(op.getLoc(), rewriter, {op.getResult().getType()});

    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {origInputs, origOutputs},
                                   /*tiled=*/true, /*noCollapse=*/true);

    int32_t dim = op.getDim();
    if (dim < 0) {
      dim += mlir::cast<RankedTensorType>(origInputs[0].getType()).getRank();
    }

    auto compositeView = rewriter.create<d2m::CompositeViewOp>(
        op.getLoc(), outputs[0].getType(), inputs, dim);

    rewriter.replaceOp(op, unLayoutResult(rewriter, compositeView->getResult(0),
                                          op.getResult().getType()));
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

    llvm::SmallVector<int64_t> ttnnGridShape(ttnnLayout.getGrid().getShape());
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

class D2MFullOpRewriter : public OpConversionPattern<ttir::FullOp>,
                          D2MNamedRewriterCommon {
public:
  D2MFullOpRewriter(const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
                    ttcore::MemorySpace defaultInputMemSpace,
                    ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
                    bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ttir::FullOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}

  LogicalResult
  matchAndRewrite(ttir::FullOp op, ttir::FullOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType resultType = op.getResult().getType();
    Attribute fillAttr = op.getFillValueAttr();
    mlir::FailureOr<mlir::Value> filled =
        lowerRankedTensorFillViaGeneric(rewriter, loc, resultType, fillAttr);
    if (mlir::failed(filled)) {
      return rewriter.notifyMatchFailure(
          op, "full lowering expects float or integer fill attribute and "
              "matching tensor element type");
    }
    rewriter.replaceOp(op, *filled);
    return success();
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

    if (resultType.getRank() != 2) {
      return rewriter.notifyMatchFailure(
          op, "D2M arange requires 2D tensor; decomposition pass should "
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
    Type f32Type = rewriter.getF32Type();
    llvm::ArrayRef<int64_t> gridShape =
        outputLayout.getGridShape(outputTensorType);
    SmallVector<int64_t> scratchShape(gridShape.begin(), gridShape.end());
    scratchShape.append({1, 1}); // One tile
    auto tileType = ttcore::TileType::get(f32Type);
    SmallVector<int64_t> scratchLogicalShape = {1, 1};
    auto scratchLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), scratchLogicalShape, ttcore::OOBVal::Undef,
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

    // Mark index tile (index 0) as scratch - it doesn't need streaming.
    generic.setScratchInputsAttr(rewriter.getDenseI64ArrayAttr({0}));

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
    auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
        op.getLoc(), adaptor.getOutput().getType(),
        SmallVector<Value>{adaptor.getA(), adaptor.getB()}, adaptor.getOutput(),
        getAffineMapsArray(rewriter, adaptor.getOperands().size(),
                           tensorA.getRank()),
        getIteratorTypesArray(rewriter, tensorA.getRank()),
        [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
            mlir::ValueRange bbArgs) {
          mlir::Value mm = bbBuilder.create<d2m::TileMatmulOp>(
              bbLoc, bbArgs.take_back(1).getTypes(), bbArgs);
          bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, mm);
        });

    rewriter.replaceAllUsesExcept(adaptor.getOutput(),
                                  linalgGeneric.getResult(0), linalgGeneric);
    rewriter.eraseOp(op);

    return llvm::success();
  }

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, std::size_t arity,
                     std::size_t rank) {
    assert(arity == 3 && "expected 3 operands");
    // TODO(#2592) for handling higher ranks if it's needed.
    assert(rank == 2 && "expected a rank 2 operation");
    mlir::MLIRContext *ctx = builder.getContext();

    return SmallVector<mlir::AffineMap>{makeAffineMap(ctx, {0, 2}),
                                        makeAffineMap(ctx, {2, 1}),
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



//forwarding AG
//    step 1:
// load my chunk of input
// write out to correct place on next device and increment semaphore
//    step 2:
// for loop (ring - 1 steps)
// wait for semaphore inc
// write out chunk to next device in the correct place in the output directly and increment semaphore

// concerns: local input forwarding could intheory be parallelized with forwarding
// but that requires actually having addition connections/BW to benefit
// actually it's not just local input that would benefit from parallelized forwarding
// each chunk in above approach would have to be completely received wherease ideally we 
// would like to the following in parallel

// thread 1: start sending local input to next device
// thread 2: take what we previously received and forward it to next device
//  aside form above conversation, this also requires a CB for forwrading to happen correctly in parallel
//    a) or at least a semaphore incrementing to tell us we need to forward another bit of data
//    and in that case we would be forwarding directly to the output
//    b) or we could have a CB for forwarding and then we could forward to and do correct remote cb ops for streaming remotely
//    but then we also have to locally store (with a noc remote store) the chunk we recive before we forward
// thread 3 (if we do 2b): take incoming data that came from previous device and store in output tensor

// 2b is the best option imo since we can forward large chunks regardless of how we have sharded our output
// and noc remote store can just handle the coalesced write
// and we also get a lot of parallelization so each chunk of data will only take as much time as needed to
// complexity is we need 3 threads, two fabric ones, one local one

// getting two fabric threads is not possible right now, bt we should still have the remote streaming mechanism
// and also have a separate thread for doing local store? so we can move large chunks of data through fabric?

// method 2b also wold have a very similar to ideal RS implementation perhaps, except we have even one more thread
// that is loading locally the device local input chunk to add with
// and then compute wait on both chunks, adds them, and writes out data to next device (remote forwrad streaming)




// forwarding reduce scatter
//    step 1:
// load a chunk of input
// write out to some scratch buffer on next device? and increment semaphore
//    step 2:
// for loop (ring - 1 steps)
// 
// wait for semaphore inc
// load correct chunk in local input and add with chunk in scratch buffer and store output in a a cb or directly in output
// store result back in scratch or in output, then send it to next device and increment semaphore

// scratch buffer needs to be double buffered?
// or same size as the whole input?


// write out chunk to next device in the correct place in the output directly and increment semaphore


// write out tensore to next device
// wait for chunk from previous device

class D2MReduceScatterRingForwardingAlgorithmRewriter : public OpConversionPattern<ttir::ReduceScatterOp>,
                             D2MNamedRewriterCommon {
public:
  D2MReduceScatterRingForwardingAlgorithmRewriter(const TypeConverter &typeConverter,
                       mlir::MLIRContext *ctx,
                       ttcore::MemorySpace defaultInputMemSpace,
                       ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
                       bool collapseTensors, bool enableMulticastInference)
      : OpConversionPattern<ttir::ReduceScatterOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors,
                               enableMulticastInference) {}
  Value createEmptyOp(mlir::ConversionPatternRewriter &rewriter,
                         mlir::Location loc,
                         Type originalInputType,
                         llvm::SmallVector<int64_t> workerGridShape) const {
    auto tensorType =
        mlir::cast<mlir::RankedTensorType>(originalInputType);
    ArrayRef<int64_t> logicalShape = tensorType.getShape();
    Type tiledElementType = ttcore::TileType::get(
        tensorType.getElementType(), ttcore::TileType::getDefaultShape());
    ttcore::MetalLayoutAttr layout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), logicalShape, ttcore::OOBVal::Undef,
        memorySpaces[0], ttcore::TensorMemoryLayout::Sharded);
    auto optimalGrid = d2m::utils::computeOptimalBlockShardedGrid(
        layout.getPhysicalShape(ttcore::TileType::getDefaultShape()),
        workerGridShape);
    llvm::SmallVector<int64_t> deviceShape =
        layout.getDeviceShape(optimalGrid, ttcore::TileType::getDefaultShape());
    auto emptyOp = rewriter.create<d2m::EmptyOp>(
        loc, deviceShape, tiledElementType, layout);
    return emptyOp.getResult();
  }

  // fix!!! make this a common function
  Value createGlobalSemaphore(mlir::Location loc,
                              mlir::ConversionPatternRewriter &rewriter,
                              llvm::SmallVector<int64_t> workerGridShape,
                              uint32_t initialValue) const {
    Type elementType = mlir::IntegerType::get(rewriter.getContext(), 32,
                                              mlir::IntegerType::Unsigned);
    ttcore::MetalLayoutAttr layout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), workerGridShape, ttcore::OOBVal::Undef,
        memorySpaces[0], ttcore::TensorMemoryLayout::Sharded,
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

  // rewriter, inputRank, op.getScatterDim(), inputPhysicalShape[op.getScatterDim()] / num_devices
  // get affine map to translate unit grid to unitgrid+grid dim for device/shard idx 
  // need a view where first dim is device idx, and later dims are normal worker core grid
  // have a normal operand grid
  // fix!!!: not needed??? is below function createDeviceShardedViewLayoutOp enough???
  Value extendedRankView(mlir::ConversionPatternRewriter &rewriter,
                                          mlir::Location loc,
                                          Value tensor) const {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(tensor.getType());
    uint32_t tensorRank = tensorType.getShape().size() / 2;
    auto mapping = rewriter.getMultiDimIdentityMap(tensorRank+1).shiftDims(1);
    
    
    auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
    llvm::SmallVector<int64_t> physicalShape = layout.getPhysicalShape(ttcore::TileType::getDefaultShape());
    llvm::SmallVector<int64_t> newGrid(tensorRank + 1, 1);
    auto newTensorShape = layout.getDeviceShape(newGrid, ttcore::TileType::getDefaultShape());

    auto newTensorType = RankedTensorType::get(
        newTensorShape, tensorType.getElementType(),
        tensorType.getEncoding());

    auto unitReblockingView = rewriter.create<d2m::ViewLayoutOp>(
        loc, newTensorType, tensor, mapping,
        /*reinterpretLayout=*/false);

    return unitReblockingView.getResult();
  }

  // TODO: see if this can be simplified/commonized with projectLogicalMapToUnitDeviceSpace
  Value createDeviceShardedViewLayoutOp(mlir::ConversionPatternRewriter &rewriter,
                                          mlir::Location loc,
                                          Value tensor,
                                          uint32_t dimToShard,
                                          uint32_t num_devices) const {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(tensor.getType());
    auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
    llvm::SmallVector<int64_t> physicalShape = layout.getPhysicalShape(ttcore::TileType::getDefaultShape());
    uint32_t tensorRank = physicalShape.size();
    auto deviceShardOffset = physicalShape[dimToShard] / num_devices;
    
    // Create affine map to translate input indices to output indices in the following manner:
    // (num_devices, 1, 1, 1, y, x) -> (1, 1, Y, X)
    mlir::SmallVector<mlir::AffineExpr> results;
    for (uint32_t i = 0; i < tensorRank; i++) {
      results.push_back(mlir::getAffineConstantExpr(0, rewriter.getContext()));
    }
    for (uint32_t i = 0; i < tensorRank; i++) {
      auto currentIndex =
          mlir::getAffineDimExpr((tensorRank + 1) + (i + 1), rewriter.getContext());
      if (i == dimToShard) {
        // device index * device shard size + shard index
        auto deviceIndex = mlir::getAffineDimExpr(0, rewriter.getContext());
        auto deviceShardOffsetExpr = mlir::getAffineConstantExpr(
            deviceShardOffset, rewriter.getContext());
        results.push_back(deviceIndex * deviceShardOffsetExpr + currentIndex);
      } else {
        results.push_back(currentIndex);
      }
    }

    auto deviceShardIndexingMap = mlir::AffineMap::get(2 * (tensorRank + 1), 0, results,
                                                  rewriter.getContext());
    
    // create a view that projects onto unit grid, then expands to higher rank to include device idx
    llvm::SmallVector<int64_t> unitGrid(tensorRank, 1);
    auto [_, unitGridReblockMap] =
        ttmlir::utils::calculateReblockMapForGrid(
          layout.getDeviceShape(unitGrid, ttcore::TileType::getDefaultShape()), layout.getGridShape(tensorType),
            tensorType.getContext());
    auto reblockMap = unitGridReblockMap.compose(deviceShardIndexingMap);
    
    // new device shape is same as old shape with an extra dim for both grid and shard
    // grid shape[0] = num_devices
    // shard shape[0] = 1
    // shard shape[dimToShard] = original shard shape[dimToShard] / num_devices
    // get device shape on unit grid
    auto newTensorShape = layout.getDeviceShape(llvm::SmallVector<int64_t>(tensorRank, 1), ttcore::TileType::getDefaultShape());
    // then adjust shard idx[dimToShard] to be divided by num_devices
    newTensorShape[tensorRank + dimToShard] /= num_devices;
    // then add a new dim for the device idx
    newTensorShape.insert(newTensorShape.begin() + tensorRank, 1);
    newTensorShape.insert(newTensorShape.begin(), num_devices);

    auto newTensorType = RankedTensorType::get(
        newTensorShape, tensorType.getElementType(),
        tensorType.getEncoding());
    auto unitReblockingView = rewriter.create<d2m::ViewLayoutOp>(
        loc, newTensorType, tensor, reblockMap,
        /*reinterpretLayout=*/false);
    return unitReblockingView.getResult();
  }

  // TODO: it would be good to always add a reverse view on the output actually (i.e. 
  // port this logic to grid selection if it 
  // doesn't already exist)
  Value createInverseDeviceShardedViewLayoutOp(mlir::ConversionPatternRewriter &rewriter,
                                          mlir::Location loc,
                                          Value tensor,
                                          uint32_t dimToShard,
                                          uint32_t num_devices,
                                          ArrayRef<int64_t> targetDeviceShape) const {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(tensor.getType());
    auto layout = mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
    llvm::SmallVector<int64_t> physicalShape = layout.getPhysicalShape(ttcore::TileType::getDefaultShape());
    uint32_t logicalRank = 2; //physicalShape.size() - 1;

    llvm::errs() << "physicalShape.size(): " << physicalShape.size() << "\n";

    auto deviceShardOffset = physicalShape[dimToShard] / num_devices;
    
    // Create affine map to translate input indices to output indices in the following manner:
    // (1, 1, Y, X) -> (num_devices, 1, 1, 1, y, x)
    mlir::SmallVector<mlir::AffineExpr> results;
    // device index * device shard size + shard index
    auto deviceShardOffsetExpr = mlir::getAffineConstantExpr(
        deviceShardOffset, rewriter.getContext());
    auto deviceIndex = mlir::getAffineDimExpr(logicalRank + dimToShard, rewriter.getContext()).floorDiv(deviceShardOffsetExpr);
    results.push_back(deviceIndex);
    for (uint32_t i = 0; i < logicalRank; i++) {
      results.push_back(mlir::getAffineConstantExpr(0, rewriter.getContext()));
    }
    // add first shard idx
    results.push_back(mlir::getAffineConstantExpr(0, rewriter.getContext()));
    for (uint32_t i = 0; i < logicalRank; i++) {
      auto currentIndex =
          mlir::getAffineDimExpr(logicalRank + i, rewriter.getContext());
      if (i == dimToShard) {
        auto deviceShardOffsetExpr = mlir::getAffineConstantExpr(
            deviceShardOffset, rewriter.getContext());
        results.push_back(currentIndex % deviceShardOffsetExpr);
      } else {
        results.push_back(currentIndex);
      }
    }

    auto inverseDeviceShardIndexingMap = mlir::AffineMap::get(2 * logicalRank, 0, results,
                                                  rewriter.getContext());
    
    // create a view that projects onto unit grid, then expands to higher rank to include device idx
    llvm::SmallVector<int64_t> unitGrid(logicalRank, 1);
    auto [_, gridReblockMap] =
        ttmlir::utils::calculateReblockMapForGrid(
            targetDeviceShape, unitGrid,
            tensorType.getContext());
    auto reblockMap = inverseDeviceShardIndexingMap.compose(gridReblockMap);

    auto newTensorType = RankedTensorType::get(
      targetDeviceShape, tensorType.getElementType(),
        tensorType.getEncoding());
    auto unitReblockingView = rewriter.create<d2m::ViewLayoutOp>(
        loc, newTensorType, tensor, reblockMap,
        /*reinterpretLayout=*/false);
    return unitReblockingView.getResult();
  }

  // fix!!!: commonize with all gather
  uint32_t getWorkerCoreSplitDim(Value input, uint32_t num_cores, uint32_t excludeDim = 0) const {
    auto inputTensorType = mlir::cast<mlir::RankedTensorType>(input.getType());
    auto inputLayout = mlir::cast<ttcore::MetalLayoutAttr>(inputTensorType.getEncoding());
    auto inputPhysicalShape = inputLayout.getPhysicalShape(ttcore::TileType::getDefaultShape());
    uint32_t inputRank = inputPhysicalShape.size();

    uint32_t workerCoreSplitDim = inputRank;
    for (uint32_t i = 0; i < inputRank; i++) {
      if (i != excludeDim && inputPhysicalShape[i] % num_cores == 0) {
        workerCoreSplitDim = i;
        break;
      }
    }
    assert(workerCoreSplitDim != inputRank &&
           "No dim found for worker core split");
    return workerCoreSplitDim;
  }

  // fix!!!: commonize with GridSelection::deriveGridAttrForOutput
  static ttcore::GridAttr deriveGridAttrForOutput(Value output,
                                                ArrayRef<int64_t> gridShape,
                                                OpBuilder &builder) {
    auto layout = ttcore::getDeviceLayout(cast<ShapedType>(output.getType()));
    auto metalLayout = mlir::dyn_cast<ttcore::MetalLayoutAttr>(layout);
    if (!metalLayout) {
      return builder.getAttr<ttcore::GridAttr>(gridShape);
    }

    if (auto invMap = d2m::utils::getVirtualGridInverseMapping(output)) {
      // Get virtual to physical map from output as well.
      auto fwdMap = *d2m::utils::getVirtualGridForwardMapping(output);
      size_t rank = gridShape.size();
      fwdMap = ttmlir::utils::affineMapDropBackResults(fwdMap, rank);
      for (int i = rank - 1; i >= 0; i--) {
        fwdMap = ttmlir::utils::dropDim(fwdMap, rank + i);
      }
      fwdMap =
          fwdMap.insertResult(getAffineConstantExpr(0, builder.getContext()), 0);

      return builder.getAttr<ttcore::GridAttr>(gridShape, fwdMap, *invMap);
    }

    auto existingRemapping = d2m::utils::getAssociatedRemapping(output);
    if (!existingRemapping.has_value() || existingRemapping->isEmpty() ||
        existingRemapping->isIdentity()) {
      return builder.getAttr<ttcore::GridAttr>(gridShape);
    }

    auto indexMap = *existingRemapping;
    constexpr size_t kExpectedDimsFor2DDeviceShape = 2 * 2;
    bool is2DPermutation =
        indexMap.isPermutation() &&
        indexMap.getNumResults() == kExpectedDimsFor2DDeviceShape &&
        indexMap.getNumInputs() == kExpectedDimsFor2DDeviceShape;
    if (!is2DPermutation) {
      return builder.getAttr<ttcore::GridAttr>(gridShape);
    }

    auto [forwardMap, inverseMap] =
        ttmlir::utils::createGridForwardAndInverseMapFor2DPermutation(
            indexMap, gridShape.size(), builder.getContext());
    return builder.getAttr<ttcore::GridAttr>(gridShape, forwardMap, inverseMap);
  }

  LogicalResult
  matchAndRewrite(ttir::ReduceScatterOp op, ttir::ReduceScatterOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // ring forwarding algorithm for reduce scatter
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
    if (topology != ttcore::Topology::Ring) {
      return rewriter.notifyMatchFailure(op, "Only ring topology is supported");
    }
    // TODO: get num links from DeviceAttr
    uint32_t num_links = 1;
    // We use unidir routing mode for ring topology so we get 2 cores for each
    // link
    int num_cores = 1; // fix once added logic for ring direction
        //topology == ttcore::Topology::Ring ? num_links * 2 : num_links * 1;
    int num_devices = meshShape[clusterAxis];

    // Create global semaphores for synchronization
    Value startSemaphore =
        createGlobalSemaphore(loc, rewriter, workerGridShape, 0);
    // better name???
    Value iterationSemaphore =
        createGlobalSemaphore(loc, rewriter, workerGridShape, 0);

    // create input and output
    auto origOutputs =
        createDpsOutputs(loc, rewriter, {op.getResult().getType()});
    SmallVector<Value> origInputs = adaptor.getOperands();
    auto input = rewriter.create<d2m::ToLayoutOp>(origInputs[0].getLoc(),
      origInputs[0], createEmptyOp(rewriter, loc, origInputs[0].getType(), workerGridShape))
        ->getResult(0);
    auto scratchInput = createEmptyOp(rewriter, loc, origInputs[0].getType(), workerGridShape);
    auto output = rewriter.create<d2m::ToLayoutOp>(origOutputs[0].getLoc(),
      origOutputs[0], createEmptyOp(rewriter, loc, origOutputs[0].getType(), workerGridShape))
        ->getResult(0);
    auto inputDeviceShardedView = createDeviceShardedViewLayoutOp(rewriter, loc, input, op.getScatterDim(), num_devices);
    auto scratchInputDeviceShardedView = createDeviceShardedViewLayoutOp(rewriter, loc, scratchInput, op.getScatterDim(), num_devices);
    auto outputDeviceShardedView = createDeviceShardedViewLayoutOp(rewriter, loc, output, op.getScatterDim(), 1);

    // Find dim to split work across cores and use it to calc view grids for
    // input and output Go through all dims in order and find one that where
    // num_cores divides the dim size
    uint32_t workerCoreSplitDim = getWorkerCoreSplitDim(inputDeviceShardedView, num_cores, 0);
    // inputRank is the logical rank of the original tensor
    uint32_t inputRank =
      mlir::cast<mlir::RankedTensorType>(origInputs[0].getType())
          .getShape()
          .size();
    // gridRank is the grid rank from the device-sharded view (half of the device shape rank)
    auto inputDeviceShardedViewType = mlir::cast<mlir::RankedTensorType>(inputDeviceShardedView.getType());
    uint32_t gridRank = inputDeviceShardedViewType.getRank() / 2;
    llvm::SmallVector<int64_t> inputViewGrid(gridRank, 1);
    llvm::SmallVector<int64_t> outputViewGrid(gridRank, 1);
    inputViewGrid[0] = num_devices;
    inputViewGrid[workerCoreSplitDim] *= num_cores;
    outputViewGrid[workerCoreSplitDim] *= num_cores;

    auto inputViewTensorType = mlir::cast<mlir::RankedTensorType>(inputDeviceShardedView.getType());
    auto inputStreamTensorType =
        d2m::utils::reblockShapedType(inputViewTensorType, inputViewGrid);
    auto inputStreamReblockMap = ttmlir::utils::calculateReblockMap(
      inputViewTensorType.getShape(), inputStreamTensorType.getShape(),
        rewriter.getContext());
    Value inputStreamResult =
        rewriter
            .create<d2m::ViewLayoutOp>(op.getLoc(), inputStreamTensorType,
                                       inputDeviceShardedView, inputStreamReblockMap)
            ->getResult(0);
    Value scratchInputStreamResult =
        rewriter
            .create<d2m::ViewLayoutOp>(op.getLoc(), inputStreamTensorType,
                                       scratchInputDeviceShardedView, inputStreamReblockMap)
            ->getResult(0);

    auto outputTensorType =
        mlir::cast<mlir::RankedTensorType>(outputDeviceShardedView.getType());
    auto outputStreamTensorType =
        d2m::utils::reblockShapedType(outputTensorType, outputViewGrid);
    auto outputStreamReblockMap = ttmlir::utils::calculateReblockMap(
        outputTensorType.getShape(), outputStreamTensorType.getShape(),
        rewriter.getContext());
    Value outputStreamResult =
        rewriter
            .create<d2m::ViewLayoutOp>(op.getLoc(), outputStreamTensorType,
                                       outputDeviceShardedView, outputStreamReblockMap)
            ->getResult(0);

    // Create generic in explicit form: block factors, indexing maps, and
    // iterator types are empty
    SmallVector<mlir::AffineMap> emptyIndexingMaps;
    SmallVector<mlir::Attribute> emptyIteratorTypes;
    ArrayRef<int64_t> emptyBlockFactors = {};
    auto genericGrid=
      deriveGridAttrForOutput(outputStreamResult, outputViewGrid, rewriter);
    // fix!! make this less hacky!!!
    // remove the first dim (device idx)
    auto fabricConnectionConfig = ttcore::FabricConnectionConfigAttr::get(
        rewriter.getContext(), ttcore::NocIndex::Noc0, topology, clusterAxis,
        ttcore::RoutingMode::UnidirRingTorus, num_links);
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, TypeRange(outputStreamResult), ValueRange({inputStreamResult, scratchInputStreamResult}),
        ValueRange({outputStreamResult}), ValueRange({startSemaphore, iterationSemaphore}),
        genericGrid,
        rewriter.getI64ArrayAttr(emptyBlockFactors),
        rewriter.getAffineMapArrayAttr(emptyIndexingMaps),
        rewriter.getArrayAttr(emptyIteratorTypes),
        rewriter.getArrayAttr(
            rewriter.getAttr<d2m::ThreadAttr>(d2m::ThreadType::Unified)),
        /*scratch_inputs=*/nullptr, fabricConnectionConfig, /*numRegions=*/1);

    // Create one bb in 'generic''s region and set its arguments.
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block =rewriter.createBlock(&region);

      // Populate 'block'.
      {
        // get the start device on the cluster axis and shape to multicast along
        // that axis
        SmallVector<Value> nextDevice;
        SmallVector<Value> previousDevice;
        for (uint32_t dim = 0; dim < meshShape.size(); dim++) {
          if (dim == clusterAxis) {
            // add 1 to mesh position for next device and mod by num_devices
            nextDevice.push_back(
                rewriter.create<arith::RemSIOp>(loc, rewriter.create<arith::AddIOp>(loc, rewriter.create<d2m::MeshPositionOp>(loc, dim), rewriter.create<arith::ConstantIndexOp>(loc, 1)), rewriter.create<arith::ConstantIndexOp>(loc, num_devices)));
            // subtract 1 from mesh position for previous device and mod by num_devices
            previousDevice.push_back(
                rewriter.create<arith::RemSIOp>(loc, 
                  rewriter.create<arith::SubIOp>(loc, 
                    rewriter.create<arith::AddIOp>(loc, 
                      rewriter.create<d2m::MeshPositionOp>(loc, dim), 
                      rewriter.create<arith::ConstantIndexOp>(loc, num_devices)
                    ),
                    rewriter.create<arith::ConstantIndexOp>(loc, 1)
                  ), 
                  rewriter.create<arith::ConstantIndexOp>(loc, num_devices)
                ));
            // fix once added logic for ring direction!!!
          } else {
            nextDevice.push_back(
                rewriter.create<d2m::MeshPositionOp>(loc, dim));
            previousDevice.push_back(
                rewriter.create<d2m::MeshPositionOp>(loc, dim));
          }
        }

        SmallVector<Value> coreIndices;
        for (uint32_t i = 0; i < inputRank; i++) {
          coreIndices.push_back(rewriter.create<d2m::CoreIndexOp>(loc, i));
        }

        // Synchronize with previous device since that is our only sender in ring forwarding algorithm
        rewriter.create<d2m::DeviceSynchronizeOp>(loc, startSemaphore,
                                                  previousDevice, ValueRange(),
                                                  1, coreIndices);
              

        auto meshPosition =
            rewriter.create<d2m::MeshPositionOp>(loc, clusterAxis);
        SmallVector<Value> beginIterationIndices = coreIndices;
        beginIterationIndices.insert(
          beginIterationIndices.begin(), meshPosition);

        SmallVector<SmallVector<int64_t>> mcastGridDims(1);
        // fix!!!
        auto blockArgsVec = createBlockArguments(
            rewriter, block, loc, TypeRange({inputStreamResult}),
            TypeRange({}), generic, {beginIterationIndices},
            mcastGridDims);
        Value inputLocalBuffer = blockArgsVec[0];

        // initial load/store from input to scratch tensor
        auto inputLocalBufferLoadResult = rewriter
          .create<d2m::RemoteLoadOp>(loc, inputLocalBuffer.getType(), inputLocalBuffer,
                                    inputStreamResult, beginIterationIndices)
          .getResult();
        rewriter.create<d2m::RemoteStoreOp>(
          loc, scratchInputStreamResult.getType(), scratchInputStreamResult,
          beginIterationIndices, inputLocalBufferLoadResult, nextDevice, ValueRange(),
          iterationSemaphore, coreIndices);

        // for loop: ////////////////////////////////////////////////////////////////////////////
        // create scratch input
        // remote_load local_buf input[indices]
        // for int i = 0; i < ring size - 1; i++
        //   remote_store from buf on next device into scratch input with sem inc
        //   remote_load local_buf input[indices] (order probably doesn’t matter here if scheduled on separate threads)
        //   wait for semaphore
        //   remote_load local_buf scratch_input[indices]
        //   add tensor 0, tensor 1 to get tensor 2 and store in buf?
        // endfor
        auto forLoop = rewriter.create<mlir::affine::AffineForOp>(loc, 1, num_devices, 1);
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(forLoop.getBody());
          
          // iterate over the shards starting from the one corresposing to our mesh position
          // and going backwards (partially reducing our local shard with the shard coming from previous device in each iteration)
          // until we have iterated over all of them
          SmallVector<Value> iterationIndices = coreIndices;
          Value iterationIndex = rewriter.create<arith::RemSIOp>(loc, 
            rewriter.create<arith::SubIOp>(loc, 
              rewriter.create<arith::AddIOp>(loc, 
                meshPosition, 
                rewriter.create<arith::ConstantIndexOp>(loc, num_devices)
              ),
              forLoop.getInductionVar()
            ), 
            rewriter.create<arith::ConstantIndexOp>(loc, num_devices));
          iterationIndices.insert(iterationIndices.begin(), iterationIndex);
          Value iteration_count = forLoop.getInductionVar();

          // fix!!! do we need separate cbs?
          auto blockArgsVec = createBlockArguments(
            rewriter, block, loc, TypeRange({inputStreamResult, scratchInputStreamResult}),
            TypeRange({outputStreamResult}), generic, {beginIterationIndices, beginIterationIndices},
            mcastGridDims);
          Value inputLocalBuffer = blockArgsVec[0];
          Value scratchLocalBuffer = blockArgsVec[1];
          Value outputLocalBuffer = blockArgsVec[2];


          auto inputLocalBufferLoadResult = rewriter
              .create<d2m::RemoteLoadOp>(loc, inputLocalBuffer.getType(), inputLocalBuffer,
                                        inputStreamResult, iterationIndices)
              .getResult();
          rewriter.create<d2m::SemaphoreWaitOp>(
              loc, iterationSemaphore, iteration_count);
          auto scratchLocalBufferLoadResult = rewriter
              .create<d2m::RemoteLoadOp>(loc, scratchLocalBuffer.getType(), scratchLocalBuffer,
                                        scratchInputStreamResult, iterationIndices)
              .getResult();

          // add the two together
          uint32_t numInputs = 2;
          uint32_t numOutputs = 1;
          SmallVector<mlir::AffineMap> linalgIndexingMaps =
            getIdentityAffineMapsArray(rewriter, numInputs + numOutputs, gridRank);
          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes(gridRank, mlir::utils::IteratorType::parallel);
          auto linalgGeneric = rewriter.create<mlir::linalg::GenericOp>(
            loc,
            /* result tensor types */
            outputLocalBuffer.getType(),
            /* inputs */ ValueRange{inputLocalBufferLoadResult, scratchLocalBufferLoadResult},
            /* outputs */ outputLocalBuffer, linalgIndexingMaps,
            linalgIteratorTypes,
            [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                mlir::ValueRange bbArgs) {
                mlir::Value yield = bbBuilder.create<d2m::TileAddOp>(
                    loc,
                    /* resultTypes */ bbArgs.take_back(numOutputs).getTypes(),
                    /* operands */ bbArgs.take_front(numInputs));
              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
            });

          // send to next device
          rewriter.create<d2m::RemoteStoreOp>(
            loc, scratchInputStreamResult.getType(), scratchInputStreamResult,
            iterationIndices, linalgGeneric->getResult(0), nextDevice, ValueRange(),
            iterationSemaphore, coreIndices);
        } // end for loop

        // final load/store from scratch tensor to output
        auto blockArgsVec2 = createBlockArguments(
          rewriter, block, loc, TypeRange({scratchInputStreamResult}),
          TypeRange({}), generic, {beginIterationIndices},
          mcastGridDims);
        Value scratchLocalBuffer = blockArgsVec2[0];

        rewriter.create<d2m::SemaphoreWaitOp>(
          loc, iterationSemaphore, rewriter.create<arith::ConstantIndexOp>(loc, num_devices));
        auto scratchLocalBufferLoadResult = rewriter
          .create<d2m::RemoteLoadOp>(loc, scratchLocalBuffer.getType(), scratchLocalBuffer,
                                    scratchInputStreamResult, beginIterationIndices)
          .getResult();

        SmallVector<Value> outputIndices = coreIndices;
        outputIndices.insert(
          outputIndices.begin(), rewriter.create<arith::ConstantIndexOp>(loc, 0));
        auto finalRemoteStoreOp = rewriter.create<d2m::RemoteStoreOp>(
          loc, outputStreamResult.getType(), outputStreamResult,
          outputIndices, scratchLocalBufferLoadResult);

        SmallVector<Value> storeResults;
        storeResults.push_back(finalRemoteStoreOp.getResult());
        rewriter.create<d2m::YieldOp>(loc, storeResults);
      }
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);

    auto inverseDeviceShardView = createInverseDeviceShardedViewLayoutOp(rewriter, loc, generic->getResult(0), op.getScatterDim(), num_devices, mlir::cast<mlir::RankedTensorType>(output.getType()).getShape());

    rewriter.replaceOp(op, unLayoutResult(rewriter, inverseDeviceShardView,
                                          op->getResult(0).getType()));
    return llvm::success();
  }
};

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
        rewriter.getContext(), logicalShape, ttcore::OOBVal::Undef,
        memorySpaces[0], ttcore::TensorMemoryLayout::Sharded);
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
        rewriter.getContext(), workerGridShape, ttcore::OOBVal::Undef,
        memorySpaces[0], ttcore::TensorMemoryLayout::Sharded,
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
        ttcore::OOBVal::Undef, memorySpaces[0],
        ttcore::TensorMemoryLayout::Sharded);
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
        /*scratch_inputs=*/nullptr, fabricConnectionConfig, /*numRegions=*/1);

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
        layout.getContext(), layout.getLogicalShape(), layout.getOobVal(),
        layout.getMemorySpace(), layout.getMemoryLayout(),
        layout.getCollapsedIntervals(), layout.getDimAlignments());
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
    unsigned outputDeviceRank = outputLogicalRank * 2;

    // Shift the logical map's dim references to shard dimensions.
    // Logical dims d0, d1, d2... become device shard dims
    // d(outputLogicalRank), d(outputLogicalRank+1), d(outputLogicalRank+2)...
    SmallVector<AffineExpr> shardExprs;
    for (auto expr : logicalMap.getResults()) {
      shardExprs.push_back(
          expr.shiftDims(outputLogicalRank, outputLogicalRank));
    }

    SmallVector<AffineExpr> deviceExprs;

    // Grid coordinate mapping (first inputLogicalRank results).
    for (unsigned i = 0; i < inputLogicalRank; ++i) {
      if (inputLogicalRank == outputLogicalRank) {
        // Same rank: identity mapping for grid (matches original behavior)
        deviceExprs.push_back(builder.getAffineDimExpr(i));
      } else if (inputLogicalRank < outputLogicalRank) {
        // Expanding (e.g., 2D -> 3D): map input grid dims to output's last
        // inputLogicalRank grid dims.
        unsigned outputGridIdx = outputLogicalRank - inputLogicalRank + i;
        deviceExprs.push_back(builder.getAffineDimExpr(outputGridIdx));
      } else {
        // Contracting (e.g., 3D -> 2D): map last outputLogicalRank input grid
        // dims to output grid, pad the rest with 0.
        if (i < inputLogicalRank - outputLogicalRank) {
          deviceExprs.push_back(builder.getAffineConstantExpr(0));
        } else {
          unsigned outputGridIdx = i - (inputLogicalRank - outputLogicalRank);
          deviceExprs.push_back(builder.getAffineDimExpr(outputGridIdx));
        }
      }
    }

    for (auto expr : shardExprs) {
      deviceExprs.push_back(expr);
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
    // This is also the minimum NoC transfer size in bytes.
    const int32_t nocAlignmentL1 =
        ttcore::getOpChipDescAttr(op).getNocL1AddressAlignBytes();

    auto begins = extractFromIntegerArrayAttr<int32_t>(op.getBegins());
    auto ends = extractFromIntegerArrayAttr<int32_t>(op.getEnds());
    auto step = extractFromIntegerArrayAttr<int32_t>(op.getStep());

    auto inType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto inShape = inType.getShape();
    auto outType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto outShape = outType.getShape();

    const int32_t rank = static_cast<int32_t>(inType.getRank());
    assert(rank >= 2);

    const int32_t elementBytes =
        std::max(1, static_cast<int32_t>(inType.getElementTypeBitWidth()) / 8);

    assert((nocAlignmentL1 != 0) && (nocAlignmentL1 % elementBytes == 0));
    const int32_t alignToElements = nocAlignmentL1 / elementBytes;

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
    D2MNamedElementwiseRewriter<ttir::FloorOp,           d2m::TileFloorOp>,
    D2MNamedElementwiseRewriter<ttir::GeluOp,            d2m::TileGeluOp>,
    D2MNamedElementwiseRewriter<ttir::HardsigmoidOp,     d2m::TileHardsigmoidOp>,
    D2MNamedElementwiseRewriter<ttir::LogOp,             d2m::TileLogOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalAndOp,      d2m::TileMulOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalNotOp,      d2m::TileLogicalNotOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalOrOp,       d2m::TileAddOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalXorOp,      d2m::TileSubOp>,
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
    D2MNamedElementwiseRewriter<ttir::SiluOp,            d2m::TileSiluOp>,
    D2MNamedElementwiseRewriter<ttir::SinOp,             d2m::TileSinOp>,
    D2MNamedElementwiseRewriter<ttir::SqrtOp,            d2m::TileSqrtOp>,
    D2MNamedElementwiseRewriter<ttir::SubtractOp,        d2m::TileSubOp>,
    D2MNamedElementwiseRewriter<ttir::TanOp,             d2m::TileTanOp>,
    D2MNamedElementwiseRewriter<ttir::TanhOp,            d2m::TileTanhOp>,
    D2MNamedElementwiseRewriter<ttir::WhereOp,           d2m::TileWhereOp>,
    // Comparison.
    D2MNamedElementwiseRewriter<ttir::EqualOp,           d2m::TileEqzOp>,
    D2MNamedElementwiseRewriter<ttir::NotEqualOp,        d2m::TileNezOp>,
    D2MNamedElementwiseRewriter<ttir::GreaterThanOp,     d2m::TileGtzOp>,
    D2MNamedElementwiseRewriter<ttir::GreaterEqualOp,    d2m::TileGezOp>,
    D2MNamedElementwiseRewriter<ttir::LessThanOp,        d2m::TileLtzOp>,
    D2MNamedElementwiseRewriter<ttir::LessEqualOp,       d2m::TileLezOp>,
    // Named outer-dimension reductions.
    D2MNamedOuterReductionRewriter<ttir::SumOp,               d2m::TileAddOp>,
    // Named inner (tile C/R) reductions.
    D2MNamedInnerReductionRewriter<ttir::MaxOp,               d2m::TileReduceMaxOp>,
    D2MNamedInnerReductionRewriter<ttir::MeanOp,              d2m::TileReduceMeanOp>,
    D2MNamedInnerReductionRewriter<ttir::SumOp,               d2m::TileReduceSumOp>,
    // Data movement.
    D2MNamedElementwiseRewriter<ttir::TypecastOp,        d2m::TileTypecastOp>,
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
    // CCL
    D2MAllGatherRewriter,
    //D2MAllGatherRingForwardingAlgorithmRewriter,
    D2MReduceScatterRingForwardingAlgorithmRewriter
  >(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, ttnnMode, collapseTensors, enableMulticastInference);

  // High-priority rewriter for SliceStatic ops that violate NoC constraints.
  patterns.add<D2MSliceStaticOpNoCConstraintsRewriter>(typeConverter, ctx);

  // ToLayout 1:1 conversion.
  patterns.add<D2MToLayoutOpRewriter>(typeConverter, ctx, ttnnMode);

  // Creation ops 1:1 conversion.
  patterns.add<D2MEmptyOpRewriter>(typeConverter, ctx);
  patterns.add<D2MFullOpRewriter>(
      typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, ttnnMode,
      collapseTensors, enableMulticastInference);

  // Mesh ops 1:1 conversion.
  patterns.add<D2MMeshShardOpRewriter>(typeConverter, ctx);

  // Matmul.
  patterns.add<D2MMatmulRewriter<d2m::TileMatmulOp>>(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace,  ttnnMode, collapseTensors, enableMulticastInference);

  // Arange.
  patterns.add<D2MArangeOpRewriter>(typeConverter, ctx, defaultInputMemSpace,
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

    // Tensor empty is used within GenericOp regions to create local scratch
    // buffers for remote_load and remote_store ops.
    target.addLegalOp<::mlir::tensor::EmptyOp>();

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
