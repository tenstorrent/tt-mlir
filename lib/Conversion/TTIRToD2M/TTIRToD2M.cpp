// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToD2M/TTIRToD2M.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
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
                         bool ttnnMode, bool collapseTensors)
      : memorySpaces{defaultInputMemSpace, defaultOutputMemSpace},
        ttnnMode(ttnnMode), collapseTensors(collapseTensors) {}

  static bool isTTNNTensor(Type type) {
    auto tensor = mlir::dyn_cast<RankedTensorType>(type);
    return tensor &&
           mlir::isa_and_nonnull<ttnn::TTNNLayoutAttr>(tensor.getEncoding());
  }

  void assertTTNNLayoutSupported(ttnn::TTNNLayoutAttr ttnnLayout) const {
    assert(ttnnLayout.isDeviceBufferType() && "Must be a device tensor");

    // With these assumptions we can use the default alignment and dim
    // collapsing behavior in the MetalLayoutAttr.
    assert(ttnnLayout.isTiled() &&
           "Row major TTNN layouts are not supported yet");
    assert(
        mlir::cast<ttcore::TileType>(ttnnLayout.getElementType()).getHeight() ==
            ttcore::TileType::getDefaultShape()[0] &&
        "Only default tile shape is supported");
  }

  RankedTensorType
  getMetalTensorFromTTNNTensor(mlir::ConversionPatternRewriter &rewriter,
                               Value value) const {
    auto tensorType = mlir::cast<mlir::RankedTensorType>(value.getType());
    auto ttnnLayout =
        mlir::cast<ttnn::TTNNLayoutAttr>(tensorType.getEncoding());

    assertTTNNLayoutSupported(ttnnLayout);

    ttcore::MemorySpace memSpace =
        ttnnLayout.getBufferType() == ttnn::BufferType::DRAM
            ? ttcore::MemorySpace::DeviceDRAM
            : ttcore::MemorySpace::DeviceL1;

    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);
    auto intervalTy = RankedTensorType::get({1, 2}, i64Ty);
    DenseIntElementsAttr collapsedIntervals =
        DenseIntElementsAttr::get(intervalTy, llvm::ArrayRef<int64_t>({0, -1}));

    ttcore::TensorMemoryLayout memLayout =
        (ttnnLayout.getMemLayout().getValue() ==
         ttnn::TensorMemoryLayout::Interleaved)
            ? ttcore::TensorMemoryLayout::Interleaved
            : ttcore::TensorMemoryLayout::Sharded;

    llvm::SmallVector<int64_t> dimAlignments(tensorType.getShape().size(), 1);
    dimAlignments[dimAlignments.size() - 1] = 32;
    dimAlignments[dimAlignments.size() - 2] = 32;

    bool needVirtualGrid = ttnnLayout.getMemLayout().getValue() ==
                               ttnn::TensorMemoryLayout::HeightSharded ||
                           ttnnLayout.getMemLayout().getValue() ==
                               ttnn::TensorMemoryLayout::WidthSharded;
    AffineMap indexAffineMap = AffineMap::get(rewriter.getContext());
    llvm::SmallVector<int64_t> ttnnGridShape(ttnnLayout.getGrid().getShape());
    llvm::SmallVector<int64_t> optimalGrid = ttnnGridShape;
    if (needVirtualGrid) {
      if (ttnnLayout.getMemLayout().getValue() ==
          ttnn::TensorMemoryLayout::HeightSharded) {
        optimalGrid = {ttnnGridShape[0] * ttnnGridShape[1], 1};
      } else if (ttnnLayout.getMemLayout().getValue() ==
                 ttnn::TensorMemoryLayout::WidthSharded) {
        optimalGrid = {1, ttnnGridShape[0] * ttnnGridShape[1]};
      }
      auto [fwdMap, _] = ttmlir::d2m::utils::grids::createCoreVirtMaps(
          rewriter.getContext(), optimalGrid, ttnnGridShape);
      indexAffineMap = fwdMap;
    }

    auto metalLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), tensorType.getShape(), ttcore::OOBVal::Undef,
        memSpace, memLayout, collapsedIntervals, dimAlignments, indexAffineMap);

    llvm::SmallVector<int64_t> unshardedShape =
        metalLayout.getPhysicalShape(ttcore::TileType::getDefaultShape());

    llvm::SmallVector<int64_t> shardedShape = metalLayout.getDeviceShape(
        optimalGrid, ttcore::TileType::getDefaultShape());

    Type elementType = ttnnLayout.getElementType();
    return mlir::RankedTensorType::get(shardedShape, elementType, metalLayout);
  }

  // Create a ToLayout operation for a value using the provided layout
  // information with a simple 1x1 grid; actual grid optimization and proper
  // dimension alignments are computed later in the D2MGridSelection pass.
  Value createOptimalLayoutOp(Value value, ttcore::MemorySpace memSpace,
                              bool tiled, bool noCollapse,
                              mlir::ConversionPatternRewriter &rewriter) const {
    if (isTTNNTensor(value.getType())) {
      assert(ttnnMode && "Unexpected TTNN tensor as op operand");
      auto metalTensorType = getMetalTensorFromTTNNTensor(rewriter, value);
      auto metalCastOp = rewriter.create<ttir::TTNNMetalLayoutCastOp>(
          value.getLoc(), metalTensorType, value);

      ttcore::MemorySpace metalTensorMemSpace =
          mlir::cast<ttcore::MetalLayoutAttr>(metalTensorType.getEncoding())
              .getMemorySpace();
      if (metalTensorMemSpace == ttcore::MemorySpace::DeviceL1) {
        // Reblock L1 operand to unit grid to align with other operands while
        // preserving original TTNN tensor shape. These views will be removed in
        // GridSelection by insertTTNNDRAMStreams().
        auto unitReblockingView = rewriter.create<d2m::ViewLayoutOp>(
            value.getLoc(), d2m::utils::reblockTensor(metalTensorType, {1, 1}),
            metalCastOp->getResult(0));
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

      // For ND uncollapsed shapes, instantiate a core virtual grid map that
      // collapses the default ND unit grid to a 2D unit grid. These mappings
      // will be replaced if the layout is optimized in GridSelection.
      AffineMap coreVirtMap = AffineMap::get(rewriter.getContext());
      if (logicalShape.size() > 2) {
        llvm::SmallVector<int64_t> unitGrid(logicalShape.size(), 1);
        coreVirtMap = ttmlir::d2m::utils::grids::createCoreVirtMaps(
                          rewriter.getContext(), unitGrid, {1, 1})
                          .first;
      }

      layout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), logicalShape, ttcore::OOBVal::Undef, memSpace,
          ttcore::TensorMemoryLayout::Sharded, emptyCollapseIntervals,
          coreVirtMap);

    } else {
      layout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), logicalShape, ttcore::OOBVal::Undef, memSpace,
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
    return rewriter.create<d2m::ToLayoutOp>(value.getLoc(), value, emptyOp)
        ->getResult(0);
  }

  // Insert ToLayout operations for a genericOp's operands and results,
  // including sharding and tilizing, with simple 1x1 grids; grid optimization
  // happens later in the D2MGridSelection pass.
  std::array<mlir::SmallVector<Value>, 2> toLayoutOperandsAndResults(
      mlir::ConversionPatternRewriter &rewriter,
      std::array<mlir::SmallVector<Value>, 2> operandsAndResults, bool tiled,
      bool noCollapse = false) const {
    std::array<mlir::SmallVector<Value>, 2> result;

    for (Value operand : operandsAndResults[0]) {
      result[0].push_back(createOptimalLayoutOp(operand, memorySpaces[0], tiled,
                                                noCollapse, rewriter));
    }
    for (Value operand : operandsAndResults[1]) {
      result[1].push_back(createOptimalLayoutOp(operand, memorySpaces[1], tiled,
                                                noCollapse, rewriter));
    }

    return result;
  }

  Operation *unLayoutResult(mlir::ConversionPatternRewriter &rewriter,
                            Value fromValue, Type toResultType) const {
    if (isTTNNTensor(toResultType)) {
      assert(ttnnMode && "Unexpected TTNN tensor as op result");
      return rewriter.create<ttir::TTNNMetalLayoutCastOp>(
          fromValue.getLoc(), toResultType, fromValue);
    }
    auto output =
        rewriter.create<d2m::EmptyOp>(fromValue.getLoc(), toResultType);
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

  static SmallVector<Value> createBlockArguments(mlir::OpBuilder &builder,
                                                 mlir::Block *block,
                                                 mlir::Location loc,
                                                 mlir::TypeRange inputs,
                                                 mlir::TypeRange outputs) {
    auto fn = [&](Type t) {
      mlir::RankedTensorType tensorType = mlir::cast<mlir::RankedTensorType>(t);
      ttcore::MetalLayoutAttr layout =
          mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
      auto shardShape = layout.getShardShape(tensorType);
      block->addArgument(d2m::CBType::get(mlir::RankedTensorType::get(
                             shardShape, tensorType.getElementType())),
                         loc);
    };

    llvm::for_each(mlir::TypeRange(inputs), fn);
    llvm::for_each(mlir::TypeRange(outputs), fn);

    SmallVector<Value> operands;
    for (auto arg : block->getArguments()) {
      Value acquire =
          (arg.getArgNumber() < inputs.size())
              ? builder.create<d2m::WaitOp>(loc, arg).getResult()
              : builder.create<d2m::ReserveOp>(loc, arg).getResult();
      operands.push_back(acquire);
    }
    return operands;
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
      bool collapseTensors)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors) {}

private:
  static constexpr bool isComparisonOp =
      std::is_same_v<TileOp, d2m::TileEqzOp> ||
      std::is_same_v<TileOp, d2m::TileNezOp> ||
      std::is_same_v<TileOp, d2m::TileGtzOp> ||
      std::is_same_v<TileOp, d2m::TileGezOp> ||
      std::is_same_v<TileOp, d2m::TileLtzOp> ||
      std::is_same_v<TileOp, d2m::TileLezOp>;

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
                           ArrayRef<d2m::TileBcastType> tileBcastTypes) const {
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

    const bool isImplicitBcast = llvm::any_of(tileBcastTypes, [](auto type) {
      return type != d2m::TileBcastType::None;
    });

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
        loc, inputs, outputs, rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    // Create one bb in 'generic''s region and set its arguments.
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block'.
      {
        auto blockArgsVec = createBlockArguments(
            rewriter, block, loc, TypeRange(inputs), TypeRange(outputs));
        ArrayRef<Value> blockArgs(blockArgsVec);

        // Create 'linalg.generic' accepting 'blockArgs'.
        auto linalgIndexingMaps =
            isImplicitBcast
                ? bcastIndexingMaps
                : getAffineMapsArray(rewriter, numOperands, physicalRank);

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
              createComputeRegion(bbBuilder, bbLoc, bbArgs, rewriter, loc,
                                  numInputs, numOutputs, tileBcastTypes);
            });

        rewriter.create<d2m::YieldOp>(loc, linalgGeneric->getResults());
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
// Rewriting reduction ops is similar to the elementwise group except for
// ops whose tiled counterparts require a scaler operand ('weights', etc).
// This rewriter will emit a single tile scaler operand that will be
// broadcast across the lhs indexing space.
namespace {
template <typename ConcreteOp, typename TileOp>
class D2MNamedReductionRewriter final
    : public mlir::OpConversionPattern<ConcreteOp>,
      D2MNamedRewriterCommon {

public:
  D2MNamedReductionRewriter<ConcreteOp, TileOp>(
      const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
      ttcore::MemorySpace defaultInputMemSpace,
      ttcore::MemorySpace defaultOutputMemSpace, bool ttnnMode,
      bool collapseTensors)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors) {}

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto origInputs = adaptor.getOperands();
    auto origOutputs =
        createDpsOutputs(loc, rewriter, {op.getResult().getType()});
    SmallVector<mlir::Value> newInputs(origInputs.begin(), origInputs.end());
    newInputs.emplace_back(createScaler(
        rewriter, loc,
        mlir::cast<mlir::RankedTensorType>(origInputs.front().getType())));
    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {newInputs, origOutputs},
                                   /*tiled*/ true);

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, op, numOperands, physicalRank);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, op, physicalRank);

    // Create 'd2m.generic' accepting extended operands.
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    // Create one bb in 'generic''s region and set its arguments.
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block'.
      {
        auto blockArgsVec = createBlockArguments(
            rewriter, block, loc, TypeRange(inputs), TypeRange(outputs));
        ArrayRef<Value> blockArgs(blockArgsVec);
        assert(blockArgs.size() == numOperands);

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

        rewriter.create<d2m::YieldOp>(loc, linalgGeneric->getResults());
      }
    }
    rewriter.finalizeOpModification(generic);
    rewriter.restoreInsertionPoint(insertPoint);

    rewriter.replaceOp(op, unLayoutResult(rewriter, generic->getResult(0),
                                          op->getResult(0).getType()));
    return llvm::success();
  }

  static void checkPreconditions(ConcreteOp op) {
    // For reductions, require 'dim_arg' and 'keep_dim'=true for now.
    assert(op.getDimArg() && "expected dim_arg attribute to be set");
    assert(op.getKeepDimAttr().getValue() && "expected default keep_dim=true");
  }

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, ConcreteOp op, std::size_t arity,
                     std::size_t rank) {
    assert(rank > 0);
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

  // Create a reduction scaler value for a given type of tensor operand
  // (at the current 'builder' insertion point).
  static mlir::Value createScaler(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::RankedTensorType inputType) {

    Type elementType = inputType.getElementType();
    Attribute encoding = nullptr;

    // If input has TTNNLayoutAttr, create scaler with matching layout.
    // This ensures the scaler goes through TTNNMetalLayoutCastOp path.
    if (auto ttnnLayout = mlir::dyn_cast_if_present<ttnn::TTNNLayoutAttr>(
            inputType.getEncoding())) {
      // Create a 1x1 grid TTNNLayoutAttr for the scaler.
      auto grid = ttcore::GridAttr::get(builder.getContext(), {1, 1});
      auto tileType = ttcore::TileType::get(
          elementType, ttcore::TileType::getDefaultShape());
      auto memref =
          MemRefType::get({1, 1}, tileType, MemRefLayoutAttrInterface{},
                          ttnnLayout.getMemref().getMemorySpace());
      encoding = ttnn::TTNNLayoutAttr::get(
          builder.getContext(), builder.getMultiDimIdentityMap(2), grid, memref,
          ttnnLayout.getMemLayout(),
          /*tensorMesh=*/nullptr, /*ignorePhysicalLayout=*/false,
          /*exactGrid=*/true);
    }

    mlir::RankedTensorType scalerType = RankedTensorType::get(
        ttcore::TileType::getDefaultShape(), elementType, encoding);

    // d2m.full requires fill_value to be 32-bit float or 32-bit integer.
    mlir::Attribute one;
    if (mlir::isa<mlir::FloatType>(elementType)) {
      one = mlir::FloatAttr::get(builder.getF32Type(), 1.0);
    } else if (mlir::isa<mlir::IntegerType>(elementType)) {
      one = mlir::IntegerAttr::get(builder.getI32Type(), 1);
    } else {
      llvm_unreachable("unexpected input element type");
    }

    return builder.create<d2m::FullOp>(
        loc, scalerType, llvm::to_vector_of<int32_t>(scalerType.getShape()),
        one);
  }

  static d2m::ReduceDim dimArgAsReduceDim(ConcreteOp op, std::size_t rank) {
    // TODO(#2613) This implements a very simple case; more work is required
    // to decompose more than 2 right-most dims being reduced over.
    assert(rank <= 64 && "rank value too large for a 64-bit set");
    std::uint64_t bits = 0;
    forAllDims(rank, getDimArg(op), [&](std::size_t index, bool dropped) {
      if (dropped) {
        bits |= (1L << index);
      }
    });

    switch (bits) {
    case 1:
      return d2m::ReduceDim::C;
    case 2:
      return d2m::ReduceDim::R;
    case 3:
      return d2m::ReduceDim::RC;
    }
    llvm_unreachable("unexpected dimArg bit pattern");
  }

  static mlir::ArrayAttr getDimArg(ConcreteOp op) {
    std::optional<::mlir::ArrayAttr> attr = op.getDimArg();
    assert(attr.has_value() && "expected 'dim_arg' attribute to be present");
    return *attr;
  }

  template <typename F>
  static void forAllDims(std::size_t rank, mlir::ArrayAttr dimArg, F &&fn) {
    SmallVector<bool> dims(rank, false);
    for (auto reduceDim : dimArg) {
      int64_t dim = mlir::cast<IntegerAttr>(reduceDim).getInt();
      dim = (dim + rank) % rank;
      assert(0 <= dim && dim < static_cast<std::int64_t>(rank));
      dims[dim] = true;
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
                    bool collapseTensors)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors) {}

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::Location loc = op->getLoc();

    auto origOutputs =
        createDpsOutputs(loc, rewriter, {op.getResult().getType()});
    auto origInputs = adaptor.getOperands();
    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {origInputs, origOutputs}, /*tiled*/ true);

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;

    // TODO(#2591) handle 'transpose_{a,b}' attributes.

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, numOperands, physicalRank);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, physicalRank);

    // Create 'd2m.generic' accepting 'op's operands.
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs, rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes));

    // Create one bb in 'generic''s region and set its arguments.
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block'.
      {
        auto blockArgsVec = createBlockArguments(
            rewriter, block, loc, TypeRange(inputs), TypeRange(outputs));
        ArrayRef<Value> blockArgs(blockArgsVec);

        // Delegate next level of nesting to a "block" op.

        if constexpr (std::is_same_v<d2m::TileMatmulBlockOp, TileOp>) {
          rewriter.create<TileOp>(loc,
                                  /* resultTypes */ mlir::TypeRange(),
                                  /* operands */ blockArgs);
          // In pure tensor semantics, explicitly yield the output shard.
          rewriter.create<d2m::YieldOp>(loc, blockArgs.take_back(numOutputs));

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

          rewriter.create<d2m::YieldOp>(loc, linalgGeneric->getResults());
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
  }

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, std::size_t arity,
                     std::size_t rank) {
    assert(arity == 3 && "expected 3 operands");
    // TODO(#2592) handle higher ranks, if needed in this pass.
    assert(rank == 2 && "expected a rank 2 operation");
    mlir::MLIRContext *ctx = builder.getContext();

    return SmallVector<mlir::AffineMap>{makeAffineMap(ctx, {0, 2}),
                                        makeAffineMap(ctx, {2, 1}),
                                        makeAffineMap(ctx, {0, 1})};
  }

  static SmallVector<mlir::Attribute>
  getIteratorTypesArray(mlir::OpBuilder &builder, std::size_t rank) {
    assert(rank == 2 && "expected a rank 2 operation");
    auto parallel = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Parallel);
    auto reduction = ttcore::IteratorTypeAttr::get(
        builder.getContext(), ttcore::IteratorType::Reduction);
    return SmallVector<mlir::Attribute>{parallel, parallel, reduction};
  }

  static mlir::AffineMap makeAffineMap(mlir::MLIRContext *ctx,
                                       std::array<unsigned, 2> targets) {
    return mlir::AffineMap::getMultiDimMapWithTargets(3, targets, ctx);
  }
};
} // namespace

// ----------------------------------------------------------------------------
//
// Lower PermuteOp into a D2M StreamLayoutOp (to reblock into new tile-level
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
                     bool /*collapseTensors*/)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, /*collapseTensors*/ false) {}

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
        inputLayout.getCollapsedIntervals(), permuted.dimAlignments,
        permuted.transposeMap);

    auto viewType = mlir::RankedTensorType::get(
        permuted.physicalShape, inputTensorType.getElementType(), resultLayout);

    // For inner permute, we need a streamLayout to do reblocking.
    auto storage = rewriter.create<d2m::EmptyOp>(
        loc, permuted.physicalShape, inputTensorType.getElementType(),
        resultLayout);
    auto stream =
        rewriter.create<d2m::StreamLayoutOp>(loc, viewType, inputs[0], storage);
    inputs[0] = stream.getResult();
    unsigned logicalRank = deviceRank / 2;
    // For inner permute, we alse need a GenericOp to transpose each individual
    // tile.
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange blockArgs) {
          assert(blockArgs.size() == 2);
          auto identityMap = builder.getMultiDimIdentityMap(logicalRank);
          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes(
              logicalRank, mlir::utils::IteratorType::parallel);

          auto input =
              builder.create<d2m::WaitOp>(bodyLoc, blockArgs[0]).getResult();
          auto output =
              builder.create<d2m::ReserveOp>(bodyLoc, blockArgs[1]).getResult();

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

          builder.create<d2m::YieldOp>(bodyLoc, linalgGeneric->getResults());
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

// Simple conversion for ttir.to_layout -> d2m.to_layout.
class D2MToLayoutOpRewriter : public OpConversionPattern<ttir::ToLayoutOp> {
  using OpConversionPattern<ttir::ToLayoutOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ttir::ToLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto outType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    Value empty = rewriter.create<d2m::EmptyOp>(op.getLoc(), outType.getShape(),
                                                outType.getElementType(),
                                                outType.getEncoding());
    auto newOp = rewriter.create<d2m::ToLayoutOp>(op.getLoc(),
                                                  adaptor.getInput(), empty);
    rewriter.replaceOp(op, newOp.getResult(0));
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

    // Create d2m.empty with same shape and element type.
    auto d2mEmpty = rewriter.create<d2m::EmptyOp>(
        op.getLoc(), tensorType.getShape(), tensorType.getElementType(),
        tensorType.getEncoding());

    rewriter.replaceOp(op, d2mEmpty.getResult());
    return success();
  }
};

class D2MFullOpRewriter : public OpConversionPattern<ttir::FullOp> {
  using OpConversionPattern<ttir::FullOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttir::FullOp op, ttir::FullOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<d2m::FullOp>(
        op, op.getResult().getType(), op.getShape(), op.getFillValueAttr());
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
template <typename TensorManipulationOp,
          AffineMap (*LogicalAffineMapFn)(TensorManipulationOp)>
class D2MTensorManipulationOpRewriter
    : public OpConversionPattern<TensorManipulationOp>,
      D2MNamedRewriterCommon {
public:
  D2MTensorManipulationOpRewriter(const TypeConverter &typeConverter,
                                  mlir::MLIRContext *ctx,
                                  ttcore::MemorySpace defaultInputMemSpace,
                                  ttcore::MemorySpace defaultOutputMemSpace,
                                  bool ttnnMode, bool /*collapseTensors*/)
      : OpConversionPattern<TensorManipulationOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, /*collapse*/ false) {}

  LogicalResult
  matchAndRewrite(TensorManipulationOp op,
                  typename TensorManipulationOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AffineMap deviceMap =
        projectLogicalMapToUnitDeviceSpace(rewriter, LogicalAffineMapFn(op));

    auto origInputs = adaptor.getOperands();
    auto origOutputs =
        createDpsOutputs(op.getLoc(), rewriter, {op.getResult().getType()});

    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {origInputs, origOutputs},
                                   /*tiled*/ false);
    assert(outputs.size() == 1);

    auto outTy = mlir::cast<RankedTensorType>(outputs[0].getType());
    auto layout = mlir::cast<ttcore::MetalLayoutAttr>(outTy.getEncoding());
    auto newLayout = ttcore::MetalLayoutAttr::get(
        layout.getContext(), layout.getLogicalShape(), layout.getOobVal(),
        layout.getMemorySpace(), layout.getMemoryLayout(),
        layout.getCollapsedIntervals(), layout.getDimAlignments(), deviceMap);
    auto newOutTy = RankedTensorType::get(outTy.getShape(),
                                          outTy.getElementType(), newLayout);

    auto storage =
        rewriter.create<d2m::EmptyOp>(op.getLoc(), outputs[0].getType());
    auto view = rewriter.create<d2m::StreamLayoutOp>(
        op.getLoc(), newOutTy, inputs[0], storage.getResult());

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

static AffineMap rearrangeLogicalMap(ttir::RearrangeOp op) {
  mlir::FailureOr<AffineMap> maybeMap = op.getInvPatternMap();
  assert(succeeded(maybeMap));
  return *maybeMap;
}

static AffineMap sliceLogicalMap(ttir::SliceStaticOp op) {
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
  return AffineMap::get(exprs.size(), 0, exprs, ctx);
}

static AffineMap permuteLogicalMap(ttir::PermuteOp op) {
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
  SmallVector<AffineExpr> results(logicalRank);
  for (auto [dstIdx, srcIdx] : llvm::enumerate(permutation)) {
    results[dstIdx] = mlir::getAffineDimExpr(srcIdx, ctx);
  }
  return AffineMap::get(logicalRank, /*numSymbols=*/0, results, ctx);
}

// Compute logical map for ReshapeOp: linearize output coords, delinearize to
// input coords. This handles rank changes (e.g., 2D -> 3D).
// Returns a map from output logical coords to input logical coords.
static AffineMap reshapeLogicalMap(ttir::ReshapeOp op) {
  auto inputTensorType = mlir::cast<RankedTensorType>(op.getInput().getType());
  auto outputTensorType =
      mlir::cast<RankedTensorType>(op.getResult().getType());

  ArrayRef<int64_t> inputShape = inputTensorType.getShape();
  ArrayRef<int64_t> outputShape = outputTensorType.getShape();

  int32_t inputLogicalRank = static_cast<int32_t>(inputShape.size());
  int32_t outputLogicalRank = static_cast<int32_t>(outputShape.size());

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

  return AffineMap::get(outputLogicalRank, 0, reshapeExprs, ctx);
}

// Gather op conversion.
// The ttir gather op conversion has been adapted from
// https://github.com/openxla/stablehlo/blob/4c0d4841519aed22e3689c30b72a0e4228051249/stablehlo/conversions/linalg/transforms/StablehloLegalizeToLinalg.cpp#L1766.
// and https://openxla.org/stablehlo/spec#gather

/*
Example test case
ttmlir-opt --mlir-print-ir-after-all --ttir-to-ttmetal-pipeline

module {
  func.func public @test_gather_0(%arg0: tensor<32000x1024xf32>, %arg1:
tensor<1x32xi32>) -> tensor<1x32x1024xf32> { %0 = ttir.empty() :
tensor<1x32x1024xf32> %1 = "ttir.gather"(%arg0, %arg1, %0)
<{collapsed_slice_dims = array<i64: 0>, index_vector_dim = 2 : si64,
indices_are_sorted = false, offset_dims = array<i64: 2>, operand_batching_dims =
array<i64>, slice_sizes = array<i64: 1, 1024>, start_index_map = array<i64: 0>,
start_indices_batching_dims = array<i64>}> : (tensor<32000x1024xf32>,
tensor<1x32xi32>, tensor<1x32x1024xf32>) -> tensor<1x32x1024xf32> return %1 :
tensor<1x32x1024xf32>
  }
}

func.func public @test_gather_0(%arg0: tensor<32000x1024xf32>, %arg1:
tensor<1x32xi32>) -> tensor<1x32x1024xf32> { %0 = d2m.empty() :
tensor<1x32x1024xf32> %1 = d2m.empty() : tensor<1x1x32000x1024xf32,
#ttcore.metal_layout<logical_shape = 32000x1024, dim_alignments = 32x32,
collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram>>
  %2 = d2m.to_layout %arg0, %1 : tensor<32000x1024xf32> into
tensor<1x1x32000x1024xf32, #ttcore.metal_layout<logical_shape = 32000x1024,
dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> :
tensor<2x2xi64>, undef, dram>> -> tensor<1x1x32000x1024xf32,
#ttcore.metal_layout<logical_shape = 32000x1024, dim_alignments = 32x32,
collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram>>
  %3 = d2m.empty() : tensor<1x1x1x32xi32, #ttcore.metal_layout<logical_shape =
1x32, dim_alignments = 32x1, collapsed_intervals = dense<[[0, 1], [1, 2]]> :
tensor<2x2xi64>, undef, dram>> %4 = d2m.to_layout %arg1, %3 : tensor<1x32xi32>
into tensor<1x1x1x32xi32, #ttcore.metal_layout<logical_shape = 1x32,
dim_alignments = 32x1, collapsed_intervals = dense<[[0, 1], [1, 2]]> :
tensor<2x2xi64>, undef, dram>> -> tensor<1x1x1x32xi32,
#ttcore.metal_layout<logical_shape = 1x32, dim_alignments = 32x1,
collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram>>
  %5 = d2m.empty() : tensor<1x32x1024xf32>
  %6 = d2m.empty() : tensor<1x1x32x1024xf32, #ttcore.metal_layout<logical_shape
= 1x32x1024, dim_alignments = 1x32x1, collapsed_intervals = dense<[[0, 2], [2,
3]]> : tensor<2x2xi64>, undef, dram>> %7 = d2m.to_layout %5, %6 :
tensor<1x32x1024xf32> into tensor<1x1x32x1024xf32,
#ttcore.metal_layout<logical_shape = 1x32x1024, dim_alignments = 1x32x1,
collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, dram>>
-> tensor<1x1x32x1024xf32, #ttcore.metal_layout<logical_shape = 1x32x1024,
dim_alignments = 1x32x1, collapsed_intervals = dense<[[0, 2], [2, 3]]> :
tensor<2x2xi64>, undef, dram>> %8 = d2m.generic {block_factors = [1, 1, 1], grid
= #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1, d2) -> (0, 0)>,
affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d1, d2)>],
iterator_types = [#ttcore.iterator_type<parallel>,
#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads =
[#d2m.thread<datamovement>]} ins(%2, %4 : tensor<1x1x32000x1024xf32,
#ttcore.metal_layout<logical_shape = 32000x1024, dim_alignments = 32x32,
collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram>>,
tensor<1x1x1x32xi32, #ttcore.metal_layout<logical_shape = 1x32, dim_alignments =
32x1, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef,
dram>>) outs(%7 : tensor<1x1x32x1024xf32, #ttcore.metal_layout<logical_shape =
1x32x1024, dim_alignments = 1x32x1, collapsed_intervals = dense<[[0, 2], [2,
3]]> : tensor<2x2xi64>, undef, dram>>)  { ^datamovement0(%cb0:
!d2m.cb<tensor<32000x1024xf32>, riscv_l1>, %cb1: !d2m.cb<tensor<1x32xi32>,
riscv_l1>, %cb2: !d2m.cb<tensor<32x1024xf32>, riscv_l1>): %11 = d2m.wait %cb0 :
<tensor<32000x1024xf32>, riscv_l1> -> tensor<32000x1024xf32> %12 = d2m.wait %cb1
: <tensor<1x32xi32>, riscv_l1> -> tensor<1x32xi32> %13 = d2m.reserve %cb2 :
<tensor<32x1024xf32>, riscv_l1> -> tensor<32x1024xf32> affine.for %arg2 = 0 to 1
{ affine.for %arg3 = 0 to 32 { affine.for %arg4 = 0 to 1024 { %c0 =
arith.constant 0 : index %c1 = arith.constant 1 : index %c2 = arith.constant 2 :
index %extracted = tensor.extract %12[%arg2, %arg3] : tensor<1x32xi32> %14 =
arith.index_cast %extracted : i32 to index %c32000 = arith.constant 32000 :
index %15 = arith.subi %c32000, %c1 : index %16 = arith.maxsi %c0, %14 : index
          %17 = arith.minsi %16, %15 : index
          %c2_0 = arith.constant 2 : index
          %dim = tensor.dim %13, %c2_0 : tensor<32x1024xf32>
          %18 = arith.addi %17, %c0 : index
          %19 = arith.addi %18, %c0 : index
          %20 = arith.addi %c0, %c0 : index
          %21 = arith.addi %20, %arg4 : index
          %22 = arith.muli %arg2, %arg3 : index
          %tx = d2m.dma %11 [%19, %21], %13 [%22, %arg4] :
(tensor<32000x1024xf32>, tensor<32x1024xf32>) -> !d2m.mem_tx d2m.dma_wait %tx
        }
      }
    }
    d2m.yield %13 : (tensor<32x1024xf32>)
  } : tensor<1x1x32x1024xf32, #ttcore.metal_layout<logical_shape = 1x32x1024,
dim_alignments = 1x32x1, collapsed_intervals = dense<[[0, 2], [2, 3]]> :
tensor<2x2xi64>, undef, dram>> %9 = d2m.empty() : tensor<1x32x1024xf32> %10 =
d2m.to_layout %8, %9 : tensor<1x1x32x1024xf32,
#ttcore.metal_layout<logical_shape = 1x32x1024, dim_alignments = 1x32x1,
collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, dram>>
into tensor<1x32x1024xf32> -> tensor<1x32x1024xf32> return %10 :
tensor<1x32x1024xf32>
*/

/*
The general algorithm is as follows:
The goal is to lower ttir.gather into d2m.generic op.

We first insert necessary layout conversions for the input operand,
start_indices and output tensor. All these will be collapsed into 2D tensors and
live in dram memory space.

Then we create the d2m.generic op
This will be 1x1 grid
Operand, start_indices and the output tensor will all be inputs into the
d2m.generic op. However, operand iteration space does not overlap with any parts
of the start_indices or output tensor iteration space. Thus, we will "broadcast"
the operand across the start_indices and output tensor iteration space.

Next, we want to fill the body of the d2m.generic op.
We will use the algorithm found on stablehlo operations page:
https://openxla.org/stablehlo/spec#gather The general idea is to iterate over
the output tensor space, and for each index in the output tensor, we map the
output index to some input index. Then we copy the value found at that input
index to the output index.

Finally, we need to insert layout conversions for the output of the d2m.generic
op to match the original output type. We insert a dma op at this location to
bring in the data from dram into the core. A future optimization is to check the
"slice_sizes" (which will correspond to the inner most loop of the affine loops)
and do a coleasced read using the dma.
*/

Value extractIndexFromTensor(OpBuilder &builder, Location loc, Value tensor,
                             ShapedType originalType,
                             ArrayRef<Value> tensorIndex = {}) {
  Value extracted =
      builder.create<mlir::tensor::ExtractOp>(loc, tensor, tensorIndex);
  if (extracted.getType().isIndex()) {
    return extracted;
  }
  return originalType.getElementType().isUnsignedInteger()
             ? builder.createOrFold<arith::IndexCastUIOp>(
                   loc, builder.getIndexType(), extracted)
             : builder.createOrFold<arith::IndexCastOp>(
                   loc, builder.getIndexType(), extracted);
}

Value getEmptySparseTensor(OpBuilder &builder, Location loc, ShapedType type,
                           ArrayRef<Value> dynSizes) {
  auto allocTensor = builder.create<bufferization::AllocTensorOp>(
      loc, llvm::cast<TensorType>(type), dynSizes,
      /*copy=*/Value(),
      /*memory_space=*/IntegerAttr());
  return allocTensor;
}

Value getEmptyTensor(OpBuilder &builder, Location loc, ShapedType type,
                     ArrayRef<Value> dynSizes) {
  auto empty =
      builder.create<d2m::EmptyOp>(loc, type.getShape(), type.getElementType());
  return empty;
}

Value getEmptyTensorFor(OpBuilder &builder, Location loc, ShapedType resultType,
                        Operation *op, ValueRange operands) {
  bool isSparse =
      mlir::sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;
  // Collect the sizes for a ranked tensor to be passed as parameter to a
  // new tensor initialization operation. This operation only needs the
  // dynamic sizes.
  SmallVector<Value> sizes;
  if (!resultType.hasStaticShape()) {
    // Ask the op for its output shape.
    auto shapeSource = cast<InferShapedTypeOpInterface>(op);
    SmallVector<Value, 1> reifiedShapes;
    if (failed(shapeSource.reifyReturnTypeShapes(builder, operands,
                                                 reifiedShapes))) {
      llvm::report_fatal_error("could not reify");
    }
    assert(reifiedShapes.size() == 1 && "Expected one reified result");
    // Construct sizes for the required dimensions.
    for (const auto &en : llvm::enumerate(resultType.getShape())) {
      if (en.value() != ShapedType::kDynamic) {
        continue;
      }
      Value idx = builder.create<arith::ConstantIndexOp>(loc, en.index());
      Value extracted = builder.create<tensor::ExtractOp>(loc, reifiedShapes[0],
                                                          ValueRange{idx});
      sizes.push_back(extracted);
    }
  }
  return isSparse ? getEmptySparseTensor(builder, loc, resultType, sizes)
                  : getEmptyTensor(builder, loc, resultType, sizes);
}

class D2MGatherOpRewriter : public OpConversionPattern<ttir::GatherOp> {
  using OpConversionPattern<ttir::GatherOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ttir::GatherOp gatherOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::OpBuilder builder(getContext());
    builder.setInsertionPointAfter(gatherOp);
    Location loc = gatherOp.getLoc();

    // 1) Get all the necessary values/types/attributes
    Value startIndices = adaptor.getStartIndices();
    Value operand = adaptor.getInput();

    RankedTensorType operandType =
        getTypeConverter()->convertType<RankedTensorType>(operand.getType());
    RankedTensorType startIndicesType =
        getTypeConverter()->convertType<RankedTensorType>(
            startIndices.getType());
    RankedTensorType resultType =
        getTypeConverter()->convertType<RankedTensorType>(gatherOp.getType());

    int64_t resultRank = resultType.getRank();
    // slice_sizes has to have the same size as operand.rank, and doing it this
    // way permits an unranked operand.
    int64_t operandRank = gatherOp.getSliceSizes().size();
    int64_t indexVectorDim = gatherOp.getIndexVectorDim();
    ArrayRef<int64_t> offsetDims = gatherOp.getOffsetDims();
    ArrayRef<int64_t> collapsedSliceDims = gatherOp.getCollapsedSliceDims();
    ArrayRef<int64_t> operandBatchingDims = gatherOp.getOperandBatchingDims();
    ArrayRef<int64_t> startIndicesBatchingDims =
        gatherOp.getStartIndicesBatchingDims();
    ArrayRef<int64_t> startIndexMap = gatherOp.getStartIndexMap();

    // 2) Insert to metal layout conversions for operand/indices/output

    // Commonly used parameters
    auto i64Ty = builder.getI64Type();
    auto intervalTy = RankedTensorType::get({2, 2}, i64Ty);

    // Insert to metal layout conversion for the operand.
    llvm::SmallVector<int64_t> dimAlignments(operandType.getShape().size(), 1);
    dimAlignments[dimAlignments.size() - 1] = 32;
    dimAlignments[dimAlignments.size() - 2] = 32;

    auto dataOperand = llvm::SmallVector<int64_t, 4>{0, 1, 1, 2};
    DenseIntElementsAttr collapsedIntervalsOperand = DenseIntElementsAttr::get(
        intervalTy, llvm::ArrayRef<int64_t>(dataOperand));

    auto metalLayout = ttcore::MetalLayoutAttr::get(
        getContext(), operandType.getShape(), ttcore::OOBVal::Undef,
        ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Sharded,
        collapsedIntervalsOperand, dimAlignments);

    llvm::SmallVector<int64_t> operandShape = {1, 1};
    operandShape.append(operandType.getShape().begin(),
                        operandType.getShape().end());

    auto empty = builder.create<d2m::EmptyOp>(
        loc, operandShape, operandType.getElementType(), metalLayout);

    auto operandLayoutOp =
        builder.create<d2m::ToLayoutOp>(loc, operand, empty, nullptr);

    // Insert to metal layout conversion for the indices.
    llvm::SmallVector<int64_t> indexDimAlignments(
        startIndicesType.getShape().size(), 1);
    indexDimAlignments[indexDimAlignments.size() - 1] = 1;
    indexDimAlignments[indexDimAlignments.size() - 2] = 32;

    auto dataIndices = llvm::SmallVector<int64_t, 4>{0, 1, 1, 2};
    DenseIntElementsAttr collapsedIntervalsIndices = DenseIntElementsAttr::get(
        intervalTy, llvm::ArrayRef<int64_t>(dataIndices));

    auto indexMetalLayout = ttcore::MetalLayoutAttr::get(
        getContext(), startIndicesType.getShape(), ttcore::OOBVal::Undef,
        ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Sharded,
        collapsedIntervalsIndices, indexDimAlignments);

    llvm::SmallVector<int64_t> startIndicesShape = {1, 1};
    startIndicesShape.append(startIndicesType.getShape().begin(),
                             startIndicesType.getShape().end());

    auto indexEmpty = builder.create<d2m::EmptyOp>(
        loc, startIndicesShape, startIndicesType.getElementType(),
        indexMetalLayout);

    auto startIndicesLayoutOp =
        builder.create<d2m::ToLayoutOp>(loc, startIndices, indexEmpty, nullptr);

    // Create the output tensor
    llvm::SmallVector<int64_t> resultDimAlignments(resultType.getShape().size(),
                                                   1);
    resultDimAlignments[resultDimAlignments.size() - 1] = 1;
    resultDimAlignments[resultDimAlignments.size() - 2] = 32;
    resultDimAlignments[resultDimAlignments.size() - 2] = 32;

    auto dataResult = llvm::SmallVector<int64_t, 4>{0, 2, 2, 3};
    DenseIntElementsAttr collapsedIntervalsResult = DenseIntElementsAttr::get(
        intervalTy, llvm::ArrayRef<int64_t>(dataResult));

    auto resultMetalLayout = ttcore::MetalLayoutAttr::get(
        getContext(), resultType.getShape(), ttcore::OOBVal::Undef,
        ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Sharded,
        collapsedIntervalsResult, resultDimAlignments);

    Value emptyOp = getEmptyTensorFor(builder, loc, resultType, gatherOp,
                                      adaptor.getOperands());
    llvm::SmallVector<int64_t> resultShape = {1};
    resultShape.append(resultType.getShape().begin(),
                       resultType.getShape().end());

    auto resultEmpty = builder.create<d2m::EmptyOp>(
        loc, resultShape, resultType.getElementType(), resultMetalLayout);

    auto resultIndicesLayoutOp =
        builder.create<d2m::ToLayoutOp>(loc, emptyOp, resultEmpty, nullptr);

    // 3) Create the d2m.generic op

    // Define a 1x1 grid
    tt::ttcore::GridAttr grid = ttcore::GridAttr::get(
        getContext(), llvm::SmallVector<int64_t, 4>{1, 1});
    unsigned numDims = 3;
    unsigned numSymbols = 0;

    // We define 3 iterators, one for operand, one for start_indices and one for
    // result

    // Operand iteration space does not participate in iteration space of
    // d2m.generic op, so we "broadcast" it affine_map<(d0, d1, d2) -> (0, 0)>
    auto map1 = AffineMap::get(
        numDims, numSymbols,
        {builder.getAffineConstantExpr(0), builder.getAffineConstantExpr(0)},
        builder.getContext());

    // Start indices iteration space does participate in iteration space of
    // d2m.generic op affine_map<(d0, d1, d2) -> (d0, d1)>
    auto map2 = AffineMap::get(
        numDims, numSymbols,
        {builder.getAffineDimExpr(0), builder.getAffineDimExpr(1)},
        builder.getContext());

    // Result indices iteration space does participate in iteration space of
    // d2m.generic op Since we collapsed dimensions to 2D, we use the last 2
    // iterators affine_map<(d0, d1, d2) -> (d1, d2)>
    auto map3 = AffineMap::get(
        numDims, numSymbols,
        {builder.getAffineDimExpr(1), builder.getAffineDimExpr(2)},
        builder.getContext());

    SmallVector<AffineMap, 3> maps = {map1, map2, map3};
    auto mapArrayAttr = builder.getAffineMapArrayAttr(maps);

    // Define parallel iterator types, since no reduction is involved here
    mlir::tt::ttcore::IteratorTypeAttr attr0 =
        mlir::tt::ttcore::IteratorTypeAttr::get(
            getContext(), mlir::tt::ttcore::IteratorType::Parallel);
    mlir::tt::ttcore::IteratorTypeAttr attr1 =
        mlir::tt::ttcore::IteratorTypeAttr::get(
            getContext(), mlir::tt::ttcore::IteratorType::Parallel);
    mlir::tt::ttcore::IteratorTypeAttr attr2 =
        mlir::tt::ttcore::IteratorTypeAttr::get(
            getContext(), mlir::tt::ttcore::IteratorType::Parallel);
    llvm::SmallVector<mlir::tt::ttcore::IteratorTypeAttr, 3> iteratorArrayAttr =
        {attr0, attr1, attr2};
    llvm::SmallVector<mlir::Attribute, 3> iteratorAttrs(
        iteratorArrayAttr.begin(), iteratorArrayAttr.end());
    mlir::ArrayAttr iteratorArrayAttrList =
        mlir::ArrayAttr::get(getContext(), iteratorAttrs);
    llvm::SmallVector<int64_t, 4> blockFactors = {1, 1, 1};

    llvm::SmallVector<Value> genericOpInputs = {
        operandLayoutOp.getResult(0), startIndicesLayoutOp.getResult(0)};
    llvm::SmallVector<Value> genericOpOutputs = {
        resultIndicesLayoutOp.getResult(0)};

    // All cb's live in riscv l1 memory
    // ex: d2m.cb<tensor<32000x1024xf32>, riscv_l1>
    llvm::SmallVector<ttcore::MemorySpace> blockArgumentMemorySpaces = {
        ttcore::MemorySpace::DeviceRiscvL1, ttcore::MemorySpace::DeviceRiscvL1,
        ttcore::MemorySpace::DeviceRiscvL1};

    // Create the d2m.generic op
    auto genericOp = builder.create<d2m::GenericOp>(
        loc, genericOpInputs, genericOpOutputs, mapArrayAttr,
        iteratorArrayAttrList,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange blockArgs) {},
        blockArgumentMemorySpaces, d2m::ThreadType::Datamovement, grid,
        blockFactors);

    // Create the output to_layout op to convert back to ttir layout
    auto finalOutput = builder.create<d2m::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto toLayoutOp = builder.create<d2m::ToLayoutOp>(
        loc, genericOp.getResult(0), finalOutput, nullptr);

    // 4) Fill in the body of d2m.generic op
    // During this stage, we will develop the affine loops and index
    // calculations to get the input index value which maps to an output index
    // value We will then update the output tensor at the output index with the
    // value from input tensor at the calculated input index

    // Get inside the d2m generic op body
    Region &region = genericOp.getRegion(0);
    Block &block = region.front();
    OpBuilder::InsertionGuard guard(rewriter);
    builder.setInsertionPointToEnd(&block);

    Value blockOperand = block.getArgument(0);
    Value blockIndex = block.getArgument(1);
    Value blockOutput = block.getArgument(2);

    auto blockOperandWaitOp = builder.create<d2m::WaitOp>(loc, blockOperand);
    auto blockIndexWaitOp = builder.create<d2m::WaitOp>(loc, blockIndex);
    auto blockOutputReserveOp =
        builder.create<d2m::ReserveOp>(loc, blockOutput);

    // Create affine loops for each result dimension
    SmallVector<Value> ivs;
    // Verify all static before using affine.for
    for (int64_t i = 0; i < resultRank; ++i) {
      int64_t dim = resultType.getDimSize(i);
      auto loop = builder.create<mlir::affine::AffineForOp>(loc, 0, dim);
      builder.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
    }

    /*
    batch_dims = [d for d in axes(result) and d not in offset_dims]
    */
    // Dimensions in the result that aren't offset dimensions are called batch.
    SmallVector<int64_t> batchDims;
    for (int64_t dim = 0; dim < resultRank; ++dim) {
      if (!llvm::is_contained(offsetDims, dim)) {
        batchDims.push_back(dim);
      }
    }

    // We'll need these later and creating them on demand we end up with
    // duplicates.
    SmallVector<Value> constants;
    for (int64_t i = 0, e = std::max({resultRank, operandRank, int64_t{2}});
         i < e; ++i) {
      auto constOp = builder.create<mlir::arith::ConstantOp>(
          loc, rewriter.getIndexAttr(i));
      constants.push_back(constOp);
    }

    // Now the complicated part. For a given output dimension we build up an
    // index into the input. It's composed of two parts: the index coming from
    // start_indices, and the offset from that index along the offset
    // dimensions. Everything includes dimension shuffling and remapping as well
    // because of the way gather is defined to allow for any-layout input by
    // adding more attributes.

    /*
    batch_index = result_index[batch_dims...]
    */
    // The base gather index (`G` in the documentation) points to a place in
    // start_indices along the batch dimensions.
    SmallVector<Value> gatherIndex;
    for (int64_t dim : batchDims) {
      gatherIndex.push_back(ivs[dim]);
    }

    /*
    start_index is defined as:
    start_indices[bi0, ..., :, ..., biN] where bi are individual elements in
    batch_index and : is inserted at the index_vector_dim index, if
    index_vector_dim < rank(start_indices). [start_indices[batch_index]]
    otherwise.
    */
    SmallVector<Value> indexFromStartIndices;
    for (size_t i = 0, e = startIndexMap.size(); i != e; ++i) {
      // The index along the index_vector dimension of start_indices varies.
      // Basically indexFromStartIndices indexes into a "row" along
      // index_vector_dim, where the row is selected by the current output
      // index.
      // But if index_vector_dim is equal to start_indices.rank, then
      // start_indices gets a trailing 1 dimension added. So the row we're
      // extracting always has length 1 and the index into it is always 0, so we
      // just use the gather index directly
      SmallVector<Value> gCombine(gatherIndex);
      if (indexVectorDim != startIndicesType.getRank()) {
        assert(indexVectorDim <= static_cast<int64_t>(gCombine.size()));
        gCombine.insert(gCombine.begin() + indexVectorDim, constants[i]);
      }

      indexFromStartIndices.push_back(extractIndexFromTensor(
          builder, loc, blockIndexWaitOp.getResult(),
          gatherOp.getStartIndices().getType(), gCombine));
    }

    /*
    For d_operand in axes(operand),
    full_start_index[d_operand] = clamp(start_index[d_start], 0, dim(operand,
    d_operand) - slice_sizes[d_operand]) if d_operand =
    start_index_map[d_start]. full_start_index[d_operand] = 0 otherwise.
    */
    // But then start indices are shuffled by the start index map. To make a
    // full index into the operand, all missing indices are zeroes.
    SmallVector<Value> remappedIndexFromIndices(operandRank, constants[0]);
    for (auto [idx, value] : llvm::enumerate(startIndexMap)) {
      remappedIndexFromIndices[value] = indexFromStartIndices[idx];
    }

    /*
    For d_operand in axes(operand),
    full_batching_index[d_operand] = batch_index[d_start - (d_start <
    index_vector_dim ? 0 : 1)] if d_operand = operand_batching_dims[i_batching]
    and d_start = start_indices_batching_dims[i_batching].
    full_batching_index[d_operand] = 0 otherwise.
    */
    // Now we construct the index based on the operand/start_indices batching
    // dimensions.
    SmallVector<Value> indexFromBatching(operandRank, constants[0]);
    for (auto [operandDim, indicesDim] :
         llvm::zip_equal(operandBatchingDims, startIndicesBatchingDims)) {
      indexFromBatching[operandDim] =
          gatherIndex[indicesDim + (indicesDim < indexVectorDim ? 0 : 1)];
    }

    /*
    offset_index = result_index[offset_dims...]
    */
    auto isCollapsedOrBatching = [&](int64_t dim) {
      return llvm::is_contained(collapsedSliceDims, dim) ||
             llvm::is_contained(operandBatchingDims, dim);
    };

    // Now we construct the index based on the offset. First we need to remap
    // the offset dimensions by dropping the collapsed/batching indices.
    SmallVector<unsigned> remappedOffsetDims;
    for (int64_t i = 0; i < operandRank; ++i) {
      if (!isCollapsedOrBatching(i)) {
        remappedOffsetDims.push_back(static_cast<unsigned>(i));
      }
    }
    assert(remappedOffsetDims.size() == offsetDims.size());

    // Clamp out of bounds indices.
    for (int i = 0, operandIndexDim = 0; i < operandRank; ++i) {
      // Compute the size of the output shape dimension corresponding to this
      // index dimension. If it's collapsed set it to 1.
      Value outputDimSize = constants[1];
      if (!isCollapsedOrBatching(i)) {
        outputDimSize = builder.create<mlir::tensor::DimOp>(
            loc, blockOutputReserveOp.getResult(),
            offsetDims[operandIndexDim++]);
      }

      // If this is a skipped dimension, we're done and don't have to clamp.
      if (remappedIndexFromIndices[i] == constants[0]) {
        continue;
      }
      d2m::CBType thisType = mlir::cast<d2m::CBType>(blockOperand.getType());

      Value operandDimSize = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getIndexAttr(thisType.getShape()[i]));

      Value largestValidIndex = builder.create<mlir::arith::SubIOp>(
          loc, operandDimSize, outputDimSize);

      // Clamp indices to [0, i, operand_dim-output_dim].
      Value clamp = builder.create<mlir::arith::MinSIOp>(
          loc,
          builder.create<mlir::arith::MaxSIOp>(loc, constants[0],
                                               remappedIndexFromIndices[i]),
          largestValidIndex);
      remappedIndexFromIndices[i] = clamp;
    }

    /*
    full_offset_index = [oi0, ..., 0, ..., oiN] where oi are individual elements
    in offset_index, and 0 is inserted at indices from collapsed_slice_dims and
    operand_batching_dims.
    */
    // For the (remapped) offset dimensions, the index is the current index in
    // the output. As before this is expanded to a full index into the operand
    // by using zeros for the missing indices.
    SmallVector<Value> indexFromOffset(operandRank, constants[0]);
    for (auto [remappedOffsetDim, offsetDim] :
         llvm::zip_equal(remappedOffsetDims, offsetDims)) {
      indexFromOffset[remappedOffsetDim] = ivs[offsetDim];
    }

    /*
    operand_index = full_start_index + full_batching_index + full_offset_index
    */
    // Now we add together our three indices to get the final index into the
    // operand.
    SmallVector<Value> combinedIndex;
    for (int64_t i = 0; i < operandRank; ++i) {
      combinedIndex.push_back(builder.create<mlir::arith::AddIOp>(
          loc, rewriter.getIndexType(),
          builder.create<mlir::arith::AddIOp>(loc, rewriter.getIndexType(),
                                              remappedIndexFromIndices[i],
                                              indexFromBatching[i]),
          indexFromOffset[i]));
    }

    // 5) Create the DMA ops to read from operand and write to output
    Value extractOperand;
    if (isa<RankedTensorType>(operand.getType())) {
      extractOperand = blockOperandWaitOp.getResult();
    } else {
      // Cannot extract from unranked tensors, cast to ranked first.
      SmallVector<int64_t> dims(operandRank, ShapedType::kDynamic);
      auto type = RankedTensorType::get(
          dims, cast<TensorType>(operand.getType()).getElementType());
      extractOperand = builder.create<mlir::tensor::CastOp>(
          loc, type, blockOperandWaitOp.getResult());
    }

    SmallVector<Value> finalIVs;
    if (ivs.size() > 2) {
      Value collapsedLeading = ivs.front();
      for (size_t i = 1; i < ivs.size() - 1; ++i) {
        collapsedLeading =
            builder.create<arith::MulIOp>(loc, collapsedLeading, ivs[i]);
      }

      Value lastDim = ivs.back();
      finalIVs.push_back(collapsedLeading);
      finalIVs.push_back(lastDim);
    }

    auto dmaOp =
        builder.create<d2m::DMAOp>(loc, extractOperand, combinedIndex,
                                   blockOutputReserveOp.getResult(), finalIVs);
    builder.create<d2m::DMAWaitOp>(loc, dmaOp);

    builder.setInsertionPointToEnd(&block);
    builder.create<d2m::YieldOp>(loc, blockOutputReserveOp.getResult());

    rewriter.replaceOp(gatherOp, toLayoutOp.getResult(0));
    return success();
  }
};

// Scatter op conversion.
// The scatter op conversion is currently a work in progress.
// The algorithm is adapted from https://openxla.org/stablehlo/spec#scatter.

/*
Example test case
ttmlir-opt --mlir-print-ir-after-all --ttir-to-ttmetal-pipeline

module @jit_scatter {
  func.func public @test_scatter(%arg0: tensor<1x3x320x320xf32>, %arg1:
tensor<1x1xi64>, %arg2: tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32> { %0 =
ttir.empty() : tensor<1x3x320x320xf32> %1 = "ttir.scatter"(%arg0, %arg1, %arg2,
%0) <{index_vector_dim = 1 : i32, indices_are_sorted = false,
input_batching_dims = array<i32>, inserted_window_dims = array<i32: 0>,
scatter_dims_to_operand_dims = array<i32: 0>, scatter_indices_batching_dims =
array<i32>, unique_indices = false, update_window_dims = array<i32: 1, 2, 3>}> :
(tensor<1x3x320x320xf32>, tensor<1x1xi64>, tensor<1x3x32x32xf32>,
tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32> return %1 :
tensor<1x3x320x320xf32>
  }
}

"func.func"() <{function_type = (tensor<1x3x320x320xf32>, tensor<1x1xi64>,
tensor<1x3x32x32xf32>) -> tensor<1x3x320x320xf32>, sym_name = "test_scatter",
sym_visibility = "public"}> ({ ^bb0(%arg0: tensor<1x3x320x320xf32>, %arg1:
tensor<1x1xi64>, %arg2: tensor<1x3x32x32xf32>): %0 = "d2m.empty"() : () ->
tensor<1x3x320x320xf32> %1 = "d2m.empty"() : () -> tensor<1x1x960x320xf32,
#ttcore.metal_layout<logical_shape = 1x3x320x320, dim_alignments = 1x1x32x32,
collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, dram>>
  %2 = "d2m.to_layout"(%arg0, %1) : (tensor<1x3x320x320xf32>,
tensor<1x1x960x320xf32, #ttcore.metal_layout<logical_shape = 1x3x320x320,
dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> :
tensor<2x2xi64>, undef, dram>>) -> tensor<1x1x960x320xf32,
#ttcore.metal_layout<logical_shape = 1x3x320x320, dim_alignments = 1x1x32x32,
collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, dram>>
  %3 = "d2m.empty"() : () -> tensor<1x1x1x1xi64,
#ttcore.metal_layout<logical_shape = 1x1, dim_alignments = 1x1,
collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram>>
  %4 = "d2m.to_layout"(%arg1, %3) : (tensor<1x1xi64>, tensor<1x1x1x1xi64,
#ttcore.metal_layout<logical_shape = 1x1, dim_alignments = 1x1,
collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram>>)
-> tensor<1x1x1x1xi64, #ttcore.metal_layout<logical_shape = 1x1, dim_alignments
= 1x1, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef,
dram>> %5 = "d2m.empty"() : () -> tensor<1x1x96x32xf32,
#ttcore.metal_layout<logical_shape = 1x3x32x32, dim_alignments = 1x1x32x32,
collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, dram>>
  %6 = "d2m.to_layout"(%arg2, %5) : (tensor<1x3x32x32xf32>,
tensor<1x1x96x32xf32, #ttcore.metal_layout<logical_shape = 1x3x32x32,
dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> :
tensor<2x2xi64>, undef, dram>>) -> tensor<1x1x96x32xf32,
#ttcore.metal_layout<logical_shape = 1x3x32x32, dim_alignments = 1x1x32x32,
collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, dram>>
  %7 = "d2m.empty"() : () -> tensor<1x3x320x320xf32>
  %8 = "d2m.to_layout"(%arg0, %7) : (tensor<1x3x320x320xf32>,
tensor<1x3x320x320xf32>) -> tensor<1x3x320x320xf32> %9 = "d2m.empty"() : () ->
tensor<1x1x960x320xf32, #ttcore.metal_layout<logical_shape = 1x3x320x320,
dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> :
tensor<2x2xi64>, undef, dram>> %10 = "d2m.to_layout"(%8, %9) :
(tensor<1x3x320x320xf32>, tensor<1x1x960x320xf32,
#ttcore.metal_layout<logical_shape = 1x3x320x320, dim_alignments = 1x1x32x32,
collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, dram>>)
-> tensor<1x1x960x320xf32, #ttcore.metal_layout<logical_shape = 1x3x320x320,
dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> :
tensor<2x2xi64>, undef, dram>> %11 = "d2m.generic"(%2, %4, %6, %10)
<{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps =
[affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0, d1) -> (0, 0)>, affine_map<(d0,
d1) -> (d0, d1)>, affine_map<(d0, d1) -> (0, 0)>], iterator_types =
[#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
operandSegmentSizes = array<i32: 3, 1>, threads = [#d2m.thread<datamovement>]}>
({ ^bb0(%arg3: !d2m.cb<tensor<960x320xf32>, riscv_l1>, %arg4:
!d2m.cb<tensor<1x1xi64>, riscv_l1>, %arg5: !d2m.cb<tensor<96x32xf32>, riscv_l1>,
%arg6: !d2m.cb<tensor<960x320xf32>, riscv_l1>): %14 = "d2m.wait"(%arg3) :
(!d2m.cb<tensor<960x320xf32>, riscv_l1>) -> tensor<960x320xf32> %15 =
"d2m.wait"(%arg4) : (!d2m.cb<tensor<1x1xi64>, riscv_l1>) -> tensor<1x1xi64> %16
= "d2m.wait"(%arg5) : (!d2m.cb<tensor<96x32xf32>, riscv_l1>) ->
tensor<96x32xf32> %17 = "d2m.reserve"(%arg6) : (!d2m.cb<tensor<960x320xf32>,
riscv_l1>) -> tensor<960x320xf32> "affine.for"() <{lowerBoundMap = affine_map<()
-> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index,
upperBoundMap = affine_map<() -> (1)>}> ({ ^bb0(%arg7: index): "affine.for"()
<{lowerBoundMap = affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0,
0>, step = 1 : index, upperBoundMap = affine_map<() -> (3)>}> ({ ^bb0(%arg8:
index): "affine.for"() <{lowerBoundMap = affine_map<() -> (0)>,
operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 : index, upperBoundMap =
affine_map<() -> (32)>}> ({ ^bb0(%arg9: index): "affine.for"() <{lowerBoundMap =
affine_map<() -> (0)>, operandSegmentSizes = array<i32: 0, 0, 0>, step = 1 :
index, upperBoundMap = affine_map<() -> (32)>}> ({ ^bb0(%arg10: index): %18 =
"arith.constant"() <{value = 1 : index}> : () -> index %19 = "tensor.dim"(%15,
%18) : (tensor<1x1xi64>, index) -> index %20 = "tensor.extract_slice"(%15,
%arg7, %19) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, static_offsets =
array<i64: -9223372036854775808, 0>, static_sizes = array<i64: 1,
-9223372036854775808>, static_strides = array<i64: 1, 1>}> : (tensor<1x1xi64>,
index, index) -> tensor<1x?xi64> %21 = "arith.constant"() <{value = 0 : index}>
: () -> index %22 = "arith.constant"() <{value = 0 : index}> : () -> index %23 =
"arith.constant"() <{value = 0 : index}> : () -> index %24 = "arith.constant"()
<{value = 0 : index}> : () -> index %25 = "arith.constant"() <{value = 0 :
index}> : () -> index %26 = "tensor.extract"(%20, %25) : (tensor<1x?xi64>,
index) -> i64 %27 = "arith.constant"() <{value = 0 : index}> : () -> index %28 =
"arith.constant"() <{value = 0 : index}> : () -> index %29 = "arith.constant"()
<{value = 0 : index}> : () -> index %30 = "arith.constant"() <{value = 0 :
index}> : () -> index %31 = "arith.constant"() <{value = 0 : index}> : () ->
index %32 = "arith.constant"() <{value = 0 : index}> : () -> index %33 =
"arith.constant"() <{value = 0 : index}> : () -> index %34 = "arith.constant"()
<{value = 0 : index}> : () -> index %35 = "arith.constant"() <{value = 0 :
index}> : () -> index %36 = "arith.addi"(%26, %31) <{overflowFlags =
#arith.overflow<none>}> : (i64, index) -> index %37 = "arith.addi"(%36, %35)
<{overflowFlags = #arith.overflow<none>}> : (index, index) -> index %38 =
"arith.addi"(%22, %32) <{overflowFlags = #arith.overflow<none>}> : (index,
index) -> index %39 = "arith.addi"(%38, %arg8) <{overflowFlags =
#arith.overflow<none>}> : (index, index) -> index %40 = "arith.addi"(%23, %33)
<{overflowFlags = #arith.overflow<none>}> : (index, index) -> index %41 =
"arith.addi"(%40, %arg9) <{overflowFlags = #arith.overflow<none>}> : (index,
index) -> index %42 = "arith.addi"(%24, %34) <{overflowFlags =
#arith.overflow<none>}> : (index, index) -> index %43 = "arith.addi"(%42,
%arg10) <{overflowFlags = #arith.overflow<none>}> : (index, index) -> index %44
= "d2m.dma"(%16, %arg7, %arg8, %arg9, %arg10, %17, %37, %39, %41, %43)
<{operandSegmentSizes = array<i32: 1, 4, 1, 4, 0, 0>}> : (tensor<96x32xf32>,
index, index, index, index, tensor<960x320xf32>, index, index, index, index) ->
!d2m.mem_tx "d2m.dma_wait"(%44) : (!d2m.mem_tx) -> () "affine.yield"() : () ->
()
          }) : () -> ()
          "affine.yield"() : () -> ()
        }) : () -> ()
        "affine.yield"() : () -> ()
      }) : () -> ()
      "affine.yield"() : () -> ()
    }) : () -> ()
    "d2m.yield"(%17) : (tensor<960x320xf32>) -> ()
  }) : (tensor<1x1x960x320xf32, #ttcore.metal_layout<logical_shape =
1x3x320x320, dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 2],
[2, 3]]> : tensor<2x2xi64>, undef, dram>>, tensor<1x1x1x1xi64,
#ttcore.metal_layout<logical_shape = 1x1, dim_alignments = 1x1,
collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, dram>>,
tensor<1x1x96x32xf32, #ttcore.metal_layout<logical_shape = 1x3x32x32,
dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> :
tensor<2x2xi64>, undef, dram>>, tensor<1x1x960x320xf32,
#ttcore.metal_layout<logical_shape = 1x3x320x320, dim_alignments = 1x1x32x32,
collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>, undef, dram>>)
-> tensor<1x1x960x320xf32, #ttcore.metal_layout<logical_shape = 1x3x320x320,
dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> :
tensor<2x2xi64>, undef, dram>> %12 = "d2m.empty"() : () ->
tensor<1x3x320x320xf32> %13 = "d2m.to_layout"(%11, %12) :
(tensor<1x1x960x320xf32, #ttcore.metal_layout<logical_shape = 1x3x320x320,
dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 2], [2, 3]]> :
tensor<2x2xi64>, undef, dram>>, tensor<1x3x320x320xf32>) ->
tensor<1x3x320x320xf32> "func.return"(%13) : (tensor<1x3x320x320xf32>) -> ()
*/

/*
The general algorithm is as follows:
The goal is to lower ttir.scatter into d2m.generic op.

We first insert necessary layout conversions for the input operand,
scatter_indices, updates and output tensor. All these will be collapsed into 2D
tensors and live in dram memory space.

Then we create the d2m.generic op
This will be 1x1 grid
Operand, scatter_indices, updates and the output tensor will all be inputs into
the d2m.generic op. However, the only input that participates in the iteration
space is the update tensor. Thus, we will "broadcast" the operand,
scatter_indices and output across the update tensor iteration space.

Next, we want to fill the body of the d2m.generic op.
We will use the algorithm found on stablehlo operations page:
https://openxla.org/stablehlo/spec#scatter The general idea is to iterate over
the update tensor space, and for each index in the update tensor, we map the
update index to some output index. Then we copy the value found at that update
index to the output index.

Finally, we need to insert layout conversions for the output of the d2m.generic
op to match the original output type. We insert a dma op at this location to
bring in the data from dram into the core.
*/

class D2MScatterOpRewriter : public OpConversionPattern<ttir::ScatterOp> {
  using OpConversionPattern<ttir::ScatterOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ttir::ScatterOp scatterOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    mlir::OpBuilder builder(getContext());
    builder.setInsertionPointAfter(scatterOp);
    Location loc = scatterOp.getLoc();

    // 1) Get all the necessary values/types/attributes
    Value input = adaptor.getInput();
    Value scatterIndices = adaptor.getScatterIndices();
    Value updates = adaptor.getUpdate();
    Value output = adaptor.getOutput();

    auto inputType =
        getTypeConverter()->convertType<RankedTensorType>(input.getType());
    auto scatterIndicesType = getTypeConverter()->convertType<RankedTensorType>(
        scatterIndices.getType());
    auto updateType =
        getTypeConverter()->convertType<RankedTensorType>(updates.getType());
    auto outputType =
        getTypeConverter()->convertType<RankedTensorType>(output.getType());

    int64_t inputRank = inputType.getRank();
    int64_t scatterIndicesRank = scatterIndicesType.getRank();
    int64_t updateRank = updateType.getRank();
    int64_t outputRank = outputType.getRank();

    ArrayRef<int32_t> updateWindowDims = scatterOp.getUpdateWindowDims();
    ArrayRef<int32_t> insertedWindowDims = scatterOp.getInsertedWindowDims();
    ArrayRef<int32_t> inputBatchingDims = scatterOp.getInputBatchingDims();
    ArrayRef<int32_t> scatterIndicesBatchingDims =
        scatterOp.getScatterIndicesBatchingDims();
    ArrayRef<int32_t> scatterDimsToOperandDims =
        scatterOp.getScatterDimsToOperandDims();
    int64_t indexVectorDim = scatterOp.getIndexVectorDim();

    // 2) Insert to metal layout conversions for operand/indices/updates/output

    // Commonly used parameters
    auto i64Ty = builder.getI64Type();
    auto intervalTy = RankedTensorType::get({2, 2}, i64Ty);

    // Insert to metal layout conversion for the input.
    llvm::SmallVector<int64_t> inputAlignments(inputType.getShape().size(), 1);
    inputAlignments[inputAlignments.size() - 1] = 32;
    inputAlignments[inputAlignments.size() - 2] = 32;
    inputAlignments[inputAlignments.size() - 3] = 1;
    inputAlignments[inputAlignments.size() - 4] = 1;

    auto inputCollapsedDims = llvm::SmallVector<int64_t, 4>{0, 2, 2, 3};
    DenseIntElementsAttr inputCollapsedDimsAttr = DenseIntElementsAttr::get(
        intervalTy, llvm::ArrayRef<int64_t>(inputCollapsedDims));

    auto inputMetalLayout = ttcore::MetalLayoutAttr::get(
        getContext(), inputType.getShape(), ttcore::OOBVal::Undef,
        ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Sharded,
        inputCollapsedDimsAttr, inputAlignments);

    // Create new shape that collapses the leading dimensions in inputType into
    // 2D
    llvm::SmallVector<int64_t> newInputShape(inputType.getShape().begin(),
                                             inputType.getShape().end());

    if (inputRank > 2) {
      int64_t collapsedLeading = inputType.getDimSize(0);
      for (int64_t i = 1; i < inputRank - 1; ++i) {
        collapsedLeading *= inputType.getDimSize(i);
      }
      newInputShape = {collapsedLeading, inputType.getDimSize(inputRank - 1)};
    }

    // Add mesh shape information
    llvm::SmallVector<int64_t> inputWithMeshShape = {1, 1};
    inputWithMeshShape.append(newInputShape.begin(), newInputShape.end());

    // Create d2m::ToLayoutOp for input
    auto inputEmpty = builder.create<d2m::EmptyOp>(
        loc, inputWithMeshShape, inputType.getElementType(), inputMetalLayout);

    auto inputLayoutOp =
        builder.create<d2m::ToLayoutOp>(loc, input, inputEmpty, nullptr);

    // Insert to metal layout conversion for the scatterIndices.
    llvm::SmallVector<int64_t> scatterIndicesAlignments(
        scatterIndicesType.getShape().size(), 1);
    scatterIndicesAlignments[scatterIndicesAlignments.size() - 1] = 1;
    scatterIndicesAlignments[scatterIndicesAlignments.size() - 2] = 1;

    auto scatterIndicesCollapsedDims =
        llvm::SmallVector<int64_t, 4>{0, 1, 1, 2};
    DenseIntElementsAttr scatterIndicesCollapsedDimsAttr =
        DenseIntElementsAttr::get(
            intervalTy, llvm::ArrayRef<int64_t>(scatterIndicesCollapsedDims));

    auto scatterIndicesMetalLayout = ttcore::MetalLayoutAttr::get(
        getContext(), scatterIndicesType.getShape(), ttcore::OOBVal::Undef,
        ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Sharded,
        scatterIndicesCollapsedDimsAttr, scatterIndicesAlignments);

    // Create new shape that collapses the leading dimensions in
    // scatterIndicesType into 2D
    llvm::SmallVector<int64_t> newScatterIndicesShape(
        scatterIndicesType.getShape().begin(),
        scatterIndicesType.getShape().end());

    if (scatterIndicesRank > 2) {
      int64_t collapsedLeading = scatterIndicesType.getDimSize(0);
      for (int64_t i = 1; i < scatterIndicesRank - 1; ++i) {
        collapsedLeading *= scatterIndicesType.getDimSize(i);
      }
      newScatterIndicesShape = {collapsedLeading, scatterIndicesType.getDimSize(
                                                      scatterIndicesRank - 1)};
    }

    // Add mesh shape information
    llvm::SmallVector<int64_t> scatterIndicesWithMeshShape = {1, 1};
    scatterIndicesWithMeshShape.append(newScatterIndicesShape.begin(),
                                       newScatterIndicesShape.end());

    // Create d2m::ToLayoutOp for scatterIndices
    auto scatterIndicesEmpty = builder.create<d2m::EmptyOp>(
        loc, scatterIndicesWithMeshShape, scatterIndicesType.getElementType(),
        scatterIndicesMetalLayout);

    auto scatterIndicesLayoutOp = builder.create<d2m::ToLayoutOp>(
        loc, scatterIndices, scatterIndicesEmpty, nullptr);

    // Insert to metal layout conversion for the updates.
    llvm::SmallVector<int64_t> updatesAlignments(updateType.getShape().size(),
                                                 1);
    updatesAlignments[updatesAlignments.size() - 1] = 32;
    updatesAlignments[updatesAlignments.size() - 2] = 32;
    updatesAlignments[updatesAlignments.size() - 3] = 1;
    updatesAlignments[updatesAlignments.size() - 4] = 1;

    auto updatesCollapsedDims = llvm::SmallVector<int64_t, 4>{0, 2, 2, 3};
    DenseIntElementsAttr updatesCollapsedDimsAttr = DenseIntElementsAttr::get(
        intervalTy, llvm::ArrayRef<int64_t>(updatesCollapsedDims));

    auto updatesMetalLayout = ttcore::MetalLayoutAttr::get(
        getContext(), updateType.getShape(), ttcore::OOBVal::Undef,
        ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Sharded,
        updatesCollapsedDimsAttr, updatesAlignments);

    // Create new shape that collapses the leading dimensions in updateType into
    // 2D
    llvm::SmallVector<int64_t> newUpdatesShape(updateType.getShape().begin(),
                                               updateType.getShape().end());

    if (updateRank > 2) {
      int64_t collapsedLeading = updateType.getDimSize(0);
      for (int64_t i = 1; i < updateRank - 1; ++i) {
        collapsedLeading *= updateType.getDimSize(i);
      }
      newUpdatesShape = {collapsedLeading,
                         updateType.getDimSize(updateRank - 1)};
    }

    // Add mesh shape information
    llvm::SmallVector<int64_t> updatesWithMeshShape = {1, 1};
    updatesWithMeshShape.append(newUpdatesShape.begin(), newUpdatesShape.end());

    // Create d2m::ToLayoutOp for updates
    auto updatesEmpty = builder.create<d2m::EmptyOp>(
        loc, updatesWithMeshShape, updateType.getElementType(),
        updatesMetalLayout);

    auto updatesLayoutOp =
        builder.create<d2m::ToLayoutOp>(loc, updates, updatesEmpty, nullptr);

    // Create d2m::EmptyOp for output

    // We need to create a copy of the input tensor to use as the output base.
    // Currently a hack to use d2m::ToLayoutOp to perform the copy where the
    // pre/post metal layouts are the same.
    auto copiedInputEmpty = builder.create<d2m::EmptyOp>(
        loc, inputType.getShape(), inputType.getElementType());

    auto copiedInputLayoutOp =
        builder.create<d2m::ToLayoutOp>(loc, input, copiedInputEmpty, nullptr);

    // Insert to metal layout conversion for the output.
    llvm::SmallVector<int64_t> outputAlignments(outputType.getShape().size(),
                                                1);
    outputAlignments[outputAlignments.size() - 1] = 32;
    outputAlignments[outputAlignments.size() - 2] = 32;
    outputAlignments[outputAlignments.size() - 3] = 1;
    outputAlignments[outputAlignments.size() - 4] = 1;

    auto outputCollapsedDims = llvm::SmallVector<int64_t, 4>{0, 2, 2, 3};
    DenseIntElementsAttr outputCollapsedDimsAttr = DenseIntElementsAttr::get(
        intervalTy, llvm::ArrayRef<int64_t>(outputCollapsedDims));

    auto outputMetalLayout = ttcore::MetalLayoutAttr::get(
        getContext(), outputType.getShape(), ttcore::OOBVal::Undef,
        ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Sharded,
        outputCollapsedDimsAttr, outputAlignments);

    // Create new shape that collapses the leading dimensions in updateType into
    // 2D
    llvm::SmallVector<int64_t> newOutputShape(outputType.getShape().begin(),
                                              outputType.getShape().end());

    if (outputRank > 2) {
      int64_t collapsedLeading = outputType.getDimSize(0);
      for (int64_t i = 1; i < outputRank - 1; ++i) {
        collapsedLeading *= outputType.getDimSize(i);
      }
      newOutputShape = {collapsedLeading,
                        outputType.getDimSize(outputRank - 1)};
    }

    // Add mesh shape information
    llvm::SmallVector<int64_t> outputWithMeshShape = {1, 1};
    outputWithMeshShape.append(newOutputShape.begin(), newOutputShape.end());

    auto outputEmpty = builder.create<d2m::EmptyOp>(loc, outputWithMeshShape,
                                                    outputType.getElementType(),
                                                    outputMetalLayout);

    auto outputLayoutOp = builder.create<d2m::ToLayoutOp>(
        loc, copiedInputLayoutOp.getResult(0), outputEmpty, nullptr);

    // 3) Create the d2m.generic op

    // Create the affine maps
    unsigned numDims = 2;
    unsigned numSymbols = 0;

    // Create affine map for operand
    // Operand does not participate in iteration space of generic op, so we
    // "broadcast" it affine_map<(d0, d1) -> (0, 0)>
    auto inputAffineMap = AffineMap::get(
        numDims, numSymbols,
        {builder.getAffineConstantExpr(0), builder.getAffineConstantExpr(0)},
        builder.getContext());

    // Create affine map for scatterIndices
    // Scatter indices does not participate in iteration space of generic op, so
    // we "broadcast" it affine_map<(d0, d1) -> (0, 0)>
    auto scatterIndicesAffineMap = AffineMap::get(
        numDims, numSymbols,
        {builder.getAffineConstantExpr(0), builder.getAffineConstantExpr(0)},
        builder.getContext());

    // Create affine map for update
    // Update participates in iteration space of generic op, and its the only
    // one, so it receives all the iterators affine_map<(d0, d1) -> (d0, d1)>
    auto updateAffineMap = AffineMap::get(
        numDims, numSymbols,
        {builder.getAffineDimExpr(0), builder.getAffineDimExpr(1)},
        builder.getContext());

    // Create affine map for output
    // Output does not participate in iteration space of generic op, so we
    // "broadcast" it affine_map<(d0, d1) -> (0, 0)>
    auto outputAffineMap = AffineMap::get(
        numDims, numSymbols,
        {builder.getAffineConstantExpr(0), builder.getAffineConstantExpr(0)},
        builder.getContext());

    SmallVector<AffineMap, 4> affineMapsList = {
        inputAffineMap, scatterIndicesAffineMap, updateAffineMap,
        outputAffineMap};
    auto affineMapsListAttr = builder.getAffineMapArrayAttr(affineMapsList);

    // Create iterator types
    mlir::tt::ttcore::IteratorTypeAttr iteratorOneAttr =
        mlir::tt::ttcore::IteratorTypeAttr::get(
            getContext(), mlir::tt::ttcore::IteratorType::Parallel);
    mlir::tt::ttcore::IteratorTypeAttr iteratorTwoAttr =
        mlir::tt::ttcore::IteratorTypeAttr::get(
            getContext(), mlir::tt::ttcore::IteratorType::Parallel);

    llvm::SmallVector<mlir::tt::ttcore::IteratorTypeAttr, 2>
        iteratorArrayAttrList = {iteratorOneAttr, iteratorTwoAttr};
    llvm::SmallVector<mlir::Attribute, 2> iteratorAttr(
        iteratorArrayAttrList.begin(), iteratorArrayAttrList.end());
    mlir::ArrayAttr iteratorArrayAttr =
        mlir::ArrayAttr::get(getContext(), iteratorAttr);

    // Create block factors
    llvm::SmallVector<int64_t, 2> blockFactors = {1, 1};

    // Create 1x1 grid
    tt::ttcore::GridAttr grid = ttcore::GridAttr::get(
        getContext(), llvm::SmallVector<int64_t, 4>{1, 1});

    // Create input and output lists for the generic op
    llvm::SmallVector<Value> genericOpInputs = {
        inputLayoutOp.getResult(0), scatterIndicesLayoutOp.getResult(0),
        updatesLayoutOp.getResult(0)};
    llvm::SmallVector<Value> genericOpOutputs = {outputLayoutOp.getResult(0)};

    // Create block argument memory spaces
    llvm::SmallVector<ttcore::MemorySpace> blockArgumentMemorySpaces = {
        ttcore::MemorySpace::DeviceRiscvL1, ttcore::MemorySpace::DeviceRiscvL1,
        ttcore::MemorySpace::DeviceRiscvL1, ttcore::MemorySpace::DeviceRiscvL1};

    // Create the d2m.generic op
    auto genericOp = builder.create<d2m::GenericOp>(
        loc, genericOpInputs, genericOpOutputs, affineMapsListAttr,
        iteratorArrayAttr,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange blockArgs) {},
        blockArgumentMemorySpaces, d2m::ThreadType::Datamovement, grid,
        blockFactors);

    // Create the output to_layout op to convert back to ttir layout
    auto finalOutput = builder.create<d2m::EmptyOp>(
        loc, outputType.getShape(), outputType.getElementType());

    auto finalOutputToLayoutOp = builder.create<d2m::ToLayoutOp>(
        loc, genericOp.getResult(0), finalOutput, nullptr);

    // 4) Fill in the body of d2m.generic op
    // During this stage, we will develop the affine loops to iterate across the
    // update tensor For each element in the update tensor, we will calculate
    // the corresponding index in the output tensor We will then update the
    // output tensor at that index with the value from the update tensor
    Region &region = genericOp.getRegion(0);
    Block &block = region.front();
    OpBuilder::InsertionGuard guard(rewriter);
    builder.setInsertionPointToEnd(&block);

    Value inputBlockArgument = block.getArgument(0);
    Value scatterIndicesBlockArgument = block.getArgument(1);
    Value updateBlockArgument = block.getArgument(2);
    Value outputBlockArgument = block.getArgument(3);

    [[maybe_unused]] auto inputWaitOp =
        builder.create<d2m::WaitOp>(loc, inputBlockArgument);
    auto scatterIndicesWaitOp =
        builder.create<d2m::WaitOp>(loc, scatterIndicesBlockArgument);
    auto updateWaitOp = builder.create<d2m::WaitOp>(loc, updateBlockArgument);
    auto outputReserveOp =
        builder.create<d2m::ReserveOp>(loc, outputBlockArgument);

    // Create affine loops to iterate over the update tensor
    llvm::SmallVector<int64_t> axes = {0, 1, 2, 3};
    SmallVector<Value> update_index;
    for (int64_t i = 0; i < updateRank; ++i) {
      int64_t dim = updateType.getDimSize(i);
      auto loop = builder.create<mlir::affine::AffineForOp>(loc, 0, dim);
      builder.setInsertionPointToStart(loop.getBody());
      update_index.push_back(loop.getInductionVar());
    }

    /*
    update_scatter_dims = [d for d in axes(updates[0]) and d not in
    update_window_dims]
    */
    SmallVector<int64_t> updateScatterDims;
    for (int64_t axis : axes) {
      if (!llvm::is_contained(updateWindowDims, axis)) {
        updateScatterDims.push_back(axis);
      }
    }

    /*
    update_scatter_index = update_index[update_scatter_dims...]
    */
    SmallVector<Value> updateScatterIndex;
    for (int64_t dim : updateScatterDims) {
      updateScatterIndex.push_back(update_index[dim]);
    }

    /*
    start_index is defined as:
    scatter_indices[si0, ..., :, ..., siN] where si are individual elements in
    update_scatter_index and : is inserted at the index_vector_dim index, if
    index_vector_dim < rank(scatter_indices).
    [scatter_indices[update_scatter_index]] otherwise.
    */
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    if (indexVectorDim < scatterIndicesRank) {
      size_t scatterLen = updateScatterIndex.size() + 1;
      size_t updateIndexCopyPos = 0;

      for (size_t dim = 0; dim < scatterLen; ++dim) {
        if (static_cast<int64_t>(dim) == indexVectorDim) {
          offsets.push_back(builder.getIndexAttr(0));
          Value dimVal = builder.create<mlir::tensor::DimOp>(
              loc, scatterIndicesWaitOp.getResult(), dim);
          sizes.push_back(dimVal);
          strides.push_back(builder.getIndexAttr(1));
        } else {
          Value indexVal = updateScatterIndex[updateIndexCopyPos++];
          offsets.push_back(indexVal);
          sizes.push_back(builder.getIndexAttr(1));
          strides.push_back(builder.getIndexAttr(1));
        }
      }
    } else {
      size_t scatterLen = updateScatterIndex.size();
      size_t updateIndexCopyPos = 0;

      for (size_t dim = 0; dim < scatterLen; ++dim) {
        Value indexVal = updateScatterIndex[updateIndexCopyPos++];
        offsets.push_back(indexVal);
        sizes.push_back(builder.getIndexAttr(1));
        strides.push_back(builder.getIndexAttr(1));
      }
    }

    auto startIndex = builder.create<mlir::tensor::ExtractSliceOp>(
        loc, scatterIndicesWaitOp.getResult(), offsets, sizes, strides);

    /*
    For d_input in axes(inputs[0]),
    full_start_index[d_input] = start_index[d_start] if d_input =
    scatter_dims_to_operand_dims[d_start]. full_start_index[d_input] = 0
    otherwise.
    */
    SmallVector<Value> fullStartIndex;

    for (int64_t i = 0; i < inputRank; ++i) {
      fullStartIndex.push_back(builder.create<mlir::arith::ConstantOp>(
          loc, builder.getIndexAttr(0)));
    }

    for (size_t dim = 0; dim < scatterDimsToOperandDims.size(); ++dim) {
      Value scatterDims = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getIndexAttr(dim));
      fullStartIndex[scatterDimsToOperandDims[dim]] =
          builder.create<mlir::tensor::ExtractOp>(loc, startIndex.getResult(),
                                                  ValueRange{scatterDims});
    }

    /*
    For d_input in axes(inputs[0]),
    full_batching_index[d_input] = update_scatter_index[d_start - (d_start <
    index_vector_dim ? 0 : 1)] if d_input = input_batching_dims[i_batching] and
    d_start = scatter_indices_batching_dims[i_batching].
    full_batching_index[d_input] = 0 otherwise.
    */
    SmallVector<Value> fullBatchingIndex;

    for (int64_t i = 0; i < inputRank; ++i) {
      fullBatchingIndex.push_back(builder.create<mlir::arith::ConstantOp>(
          loc, builder.getIndexAttr(0)));
    }

    for (int64_t dim = 0; dim < inputRank; ++dim) {
      if (llvm::is_contained(inputBatchingDims, dim)) {
        auto it = llvm::find(inputBatchingDims, dim);
        int64_t batchingIdx = std::distance(inputBatchingDims.begin(), it);
        int64_t startDim = scatterIndicesBatchingDims[batchingIdx];

        if (startDim < indexVectorDim) {
          fullBatchingIndex[dim] = updateScatterIndex[startDim];
        } else {
          fullBatchingIndex[dim] = updateScatterIndex[startDim - 1];
        }
      } else {
        fullBatchingIndex[dim] = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(0));
      }
    }

    /*
    update_window_index = update_index[update_window_dims...]
    */
    SmallVector<Value> updateWindowIndex;
    for (int64_t dim : updateWindowDims) {
      updateWindowIndex.push_back(update_index[dim]);
    }

    /*
    full_window_index = [wi0, ..., 0, ..., wiN] where wi are individual elements
    in update_window_index, and 0 is inserted at indices from
    inserted_window_dims and input_batching_dims
    */
    SmallVector<Value> fullWindowIndex;
    size_t updateWindowIndexPos = 0;
    size_t totalLen = updateWindowDims.size() + insertedWindowDims.size();

    for (size_t i = 0; i < totalLen; ++i) {
      if (llvm::is_contained(insertedWindowDims, static_cast<int32_t>(i)) ||
          llvm::is_contained(inputBatchingDims, static_cast<int32_t>(i))) {
        fullWindowIndex.push_back(builder.create<mlir::arith::ConstantOp>(
            loc, builder.getIndexAttr(0)));
      } else {
        fullWindowIndex.push_back(updateWindowIndex[updateWindowIndexPos++]);
      }
    }

    /*
    result_index = full_start_index + full_batching_index + full_window_index
    */
    SmallVector<Value> resultIndex;
    for (int64_t i = 0; i < inputRank; ++i) {
      Value sum1 = builder.create<mlir::arith::AddIOp>(
          loc, rewriter.getIndexType(), fullStartIndex[i],
          fullBatchingIndex[i]);
      Value sum2 = builder.create<mlir::arith::AddIOp>(
          loc, rewriter.getIndexType(), sum1, fullWindowIndex[i]);
      resultIndex.push_back(sum2);
    }

    /*
    We need to check the bounds of result_index before performing the scatter
    update, this is currently todo if is_index_in_bounds(output, result_index):
        output[tuple(result_index)] = updates[i, j, k, a, b] +
    output[tuple(result_index)]
    */

    // Create the DMA op to perform the scatter update
    auto dmaOp =
        builder.create<d2m::DMAOp>(loc, updateWaitOp.getResult(), update_index,
                                   outputReserveOp.getResult(), resultIndex);
    builder.create<d2m::DMAWaitOp>(loc, dmaOp);

    builder.setInsertionPointToEnd(&block);

    // This is crashing, I don't know why
    builder.create<d2m::YieldOp>(loc, outputReserveOp.getResult());

    rewriter.replaceOp(scatterOp, finalOutputToLayoutOp.getResult(0));
    return success();
  }
};

} // namespace mlir::tt

namespace mlir::tt {
void populateTTIRToD2MPatterns(MLIRContext *ctx, RewritePatternSet &patterns,
                               TypeConverter &typeConverter,
                               ttcore::MemorySpace defaultInputMemSpace,
                               ttcore::MemorySpace defaultOutputMemSpace,
                               bool ttnnMode, bool collapseTensors) {
  // clang-format off
  patterns.add<
    // Elementwise.
    D2MNamedElementwiseRewriter<ttir::AbsOp,         d2m::TileAbsOp>,
    D2MNamedElementwiseRewriter<ttir::AddOp,         d2m::TileAddOp>,
    D2MNamedElementwiseRewriter<ttir::BitwiseAndOp,  d2m::TileBitwiseAndOp>,
    D2MNamedElementwiseRewriter<ttir::BitwiseNotOp,  d2m::TileBitwiseNotOp>,
    D2MNamedElementwiseRewriter<ttir::BitwiseOrOp,   d2m::TileBitwiseOrOp>,
    D2MNamedElementwiseRewriter<ttir::BitwiseXorOp,  d2m::TileBitwiseXorOp>,
    D2MNamedElementwiseRewriter<ttir::CeilOp,        d2m::TileCeilOp>,
    D2MNamedElementwiseRewriter<ttir::CosOp,         d2m::TileCosOp>,
    D2MNamedElementwiseRewriter<ttir::DivOp,         d2m::TileDivOp>,
    D2MNamedElementwiseRewriter<ttir::ErfOp,         d2m::TileErfOp>,
    D2MNamedElementwiseRewriter<ttir::ErfcOp,        d2m::TileErfcOp>,
    D2MNamedElementwiseRewriter<ttir::ExpOp,         d2m::TileExpOp>,
    D2MNamedElementwiseRewriter<ttir::FloorOp,       d2m::TileFloorOp>,
    D2MNamedElementwiseRewriter<ttir::GeluOp,        d2m::TileGeluOp>,
    D2MNamedElementwiseRewriter<ttir::HardsigmoidOp, d2m::TileHardsigmoidOp>,
    D2MNamedElementwiseRewriter<ttir::LogOp,         d2m::TileLogOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalNotOp,  d2m::TileLogicalNotOp>,
    D2MNamedElementwiseRewriter<ttir::MultiplyOp,    d2m::TileMulOp>,
    D2MNamedElementwiseRewriter<ttir::MaximumOp,     d2m::TileMaximumOp>,
    D2MNamedElementwiseRewriter<ttir::MinimumOp,     d2m::TileMinimumOp>,
    D2MNamedElementwiseRewriter<ttir::NegOp,         d2m::TileNegativeOp>,
    D2MNamedElementwiseRewriter<ttir::PowOp,         d2m::TilePowOp>,
    D2MNamedElementwiseRewriter<ttir::ReciprocalOp,  d2m::TileRecipOp>,
    D2MNamedElementwiseRewriter<ttir::ReluOp,        d2m::TileReluOp>,
    D2MNamedElementwiseRewriter<ttir::RsqrtOp,       d2m::TileRsqrtOp>,
    D2MNamedElementwiseRewriter<ttir::SigmoidOp,     d2m::TileSigmoidOp>,
    D2MNamedElementwiseRewriter<ttir::SignOp,        d2m::TileSignOp>,
    D2MNamedElementwiseRewriter<ttir::SiluOp,        d2m::TileSiluOp>,
    D2MNamedElementwiseRewriter<ttir::SinOp,         d2m::TileSinOp>,
    D2MNamedElementwiseRewriter<ttir::SqrtOp,        d2m::TileSqrtOp>,
    D2MNamedElementwiseRewriter<ttir::SubtractOp,    d2m::TileSubOp>,
    D2MNamedElementwiseRewriter<ttir::TanOp,         d2m::TileTanOp>,
    D2MNamedElementwiseRewriter<ttir::TanhOp,        d2m::TileTanhOp>,
    D2MNamedElementwiseRewriter<ttir::WhereOp,       d2m::TileWhereOp>,

    // Comparison.
    D2MNamedElementwiseRewriter<ttir::EqualOp,        d2m::TileEqzOp>,
    D2MNamedElementwiseRewriter<ttir::NotEqualOp,     d2m::TileNezOp>,
    D2MNamedElementwiseRewriter<ttir::GreaterThanOp,  d2m::TileGtzOp>,
    D2MNamedElementwiseRewriter<ttir::GreaterEqualOp, d2m::TileGezOp>,
    D2MNamedElementwiseRewriter<ttir::LessThanOp,     d2m::TileLtzOp>,
    D2MNamedElementwiseRewriter<ttir::LessEqualOp,    d2m::TileLezOp>,

    // Reduction.
    D2MNamedReductionRewriter<ttir::MaxOp,          d2m::TileReduceMaxOp>,
    D2MNamedReductionRewriter<ttir::SumOp,          d2m::TileReduceSumOp>,
    // Data movement.
    D2MNamedElementwiseRewriter<ttir::TypecastOp,     d2m::TileTypecastOp>,
    // Tensor manipulation/View ops.
    D2MTensorManipulationOpRewriter<ttir::RearrangeOp, rearrangeLogicalMap>,
    D2MTensorManipulationOpRewriter<ttir::ReshapeOp, reshapeLogicalMap>,
    D2MTensorManipulationOpRewriter<ttir::SliceStaticOp, sliceLogicalMap>,
    // Permute (handles tranpose ops, since they're canonicalized into permutes).
    D2MTensorManipulationOpRewriter<ttir::PermuteOp, permuteLogicalMap>,
    D2MPermuteRewriter
  >(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, ttnnMode, collapseTensors);


  // ToLayout 1:1 conversion.
  patterns.add<D2MToLayoutOpRewriter>(typeConverter, ctx);

  // Creation ops 1:1 conversion.
  patterns.add<D2MEmptyOpRewriter, D2MFullOpRewriter>(typeConverter, ctx);

  // Mesh ops 1:1 conversion.
  patterns.add<D2MMeshShardOpRewriter>(typeConverter, ctx);

  // Matmul.
  patterns.add<D2MMatmulRewriter<d2m::TileMatmulOp>>(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace,  ttnnMode, collapseTensors);

  // Gather
  patterns.add<D2MGatherOpRewriter>(typeConverter, ctx);

  // Scatter
  patterns.add<D2MScatterOpRewriter>(typeConverter, ctx);

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
  }

  TTIRToD2MPass(const TTIRToD2MPass &rhs) : Base(rhs) {
    // Workaround: Passes are required to be copy-constructible but autogen'ed
    // base class copy constructors ignore Pass option fields.
    this->defaultInputMemSpace = rhs.defaultInputMemSpace;
    this->defaultOutputMemSpace = rhs.defaultOutputMemSpace;
    this->ttnnMode = rhs.ttnnMode;
    this->collapseTensorsTo2D = rhs.collapseTensorsTo2D;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });

    RewritePatternSet patterns(ctx);
    populateTTIRToD2MPatterns(ctx, patterns, typeConverter,
                              defaultInputMemSpace, defaultOutputMemSpace,
                              ttnnMode, collapseTensorsTo2D);

    ConversionTarget target(*ctx);
    target.addIllegalDialect<mlir::tt::ttir::TTIRDialect>();
    target.addLegalDialect<::mlir::BuiltinDialect>();
    target.addLegalDialect<::mlir::func::FuncDialect>();
    target.addLegalDialect<::mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::tt::d2m::D2MDialect>();
    target.addLegalDialect<mlir::tt::ttcore::TTCoreDialect>();

    // Keep some TTIR ops legal if they don't have D2M equivalents.
    target.addLegalOp<ttir::TTNNMetalLayoutCastOp>();

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
