// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToD2M/TTIRToD2M.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
// D2M generic/region ops
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/Utils.h"

#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
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
                         const llvm::SmallVector<int64_t> &targetGridShape,
                         bool ttnnMode, bool collapseTensors)
      : memorySpaces{defaultInputMemSpace, defaultOutputMemSpace},
        targetGridShape(targetGridShape),
        targetSquareGridShape(d2m::utils::getSquareTargetGrid(targetGridShape)),
        ttnnMode(ttnnMode), collapseTensors(collapseTensors) {
    assert(!targetGridShape.empty());
  }

  // Compute optimal grid shape that works for all provided layout infos.
  llvm::SmallVector<int64_t>
  computeOptimalGrid(ArrayRef<int64_t> physicalShape) const {
    llvm::SmallVector<int64_t> grid;
    grid.reserve(physicalShape.size());

    assert(physicalShape.size() >= targetSquareGridShape.size());

    const size_t gridRankDiff =
        physicalShape.size() - targetSquareGridShape.size();
    grid.assign(gridRankDiff, 1);

    for (size_t i = gridRankDiff; i < physicalShape.size(); ++i) {
      const int64_t dim = physicalShape[i];
      assert(dim > 0);
      // Find largest grid dimension that divides evenly.
      for (int64_t g = targetSquareGridShape[i - gridRankDiff]; g > 0; g--) {
        if (dim % g == 0) {
          grid.push_back(g);
          break;
        }
      }
    }

    assert(grid.size() == physicalShape.size());

    return grid;
  }

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
    bool isBlockSharded = ttnnLayout.hasL1BufferType() &&
                          ttnnLayout.getMemLayout().getValue() ==
                              ttnn::TensorMemoryLayout::BlockSharded;
    bool isInterleaved = ttnnLayout.hasInterleavedDRAMTensorMemoryLayout();
    assert((isBlockSharded || isInterleaved) &&
           "Only block sharded L1 or interleaved DRAM tensor memory layouts "
           "are supported");
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

    // Hardcode collapse intervals to [[0, -1)] to match ttnn.
    auto i64Ty = IntegerType::get(rewriter.getContext(), 64);
    auto intervalTy = RankedTensorType::get({1, 2}, i64Ty);
    DenseIntElementsAttr collapsedIntervals =
        DenseIntElementsAttr::get(intervalTy, llvm::ArrayRef<int64_t>({0, -1}));

    ttcore::TensorMemoryLayout memLayout =
        (ttnnLayout.getMemLayout().getValue() ==
         ttnn::TensorMemoryLayout::Interleaved)
            ? ttcore::TensorMemoryLayout::Interleaved
            : ttcore::TensorMemoryLayout::Sharded;

    // For tiled tensors the tile dims need to be 32 aligned.
    llvm::SmallVector<int64_t> dimAlignments(tensorType.getShape().size(), 1);
    dimAlignments[dimAlignments.size() - 1] = 32;
    dimAlignments[dimAlignments.size() - 2] = 32;

    // The index map in TTNNLayoutAttr is for collapsing an N-D tensor on to
    // the grid. It has no relevance to the index map in MetalLayoutAttr.
    // MetalLayoutAttr takes the grid shape of the device, not the grid on which
    // the tensor is sharded.
    auto metalLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), tensorType.getShape(), targetSquareGridShape,
        ttcore::OOBVal::Undef, memSpace, memLayout, collapsedIntervals,
        dimAlignments);

    llvm::SmallVector<int64_t> unshardedShape =
        metalLayout.getPhysicalShape(ttcore::TileType::getDefaultShape());

    llvm::SmallVector<int64_t> shardedShape = metalLayout.getDeviceShape(
        ttnnLayout.getGrid().getShape(), ttcore::TileType::getDefaultShape());

    Type elementType = ttnnLayout.getElementType();
    return mlir::RankedTensorType::get(shardedShape, elementType, metalLayout);
  }

  // Create a ToLayout op for a value using the provided layout info and grid.
  Value createOptimalLayoutOp(Value value, ttcore::MemorySpace memSpace,
                              bool tiled,
                              mlir::ConversionPatternRewriter &rewriter) const {
    if (isTTNNTensor(value.getType())) {
      assert(ttnnMode && "Unexpected TTNN tensor as op operand");
      return rewriter.create<ttir::TTNNMetalLayoutCastOp>(
          value.getLoc(), getMetalTensorFromTTNNTensor(rewriter, value), value);
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
    if (!collapseTensors) {
      auto emptyIntervalType = RankedTensorType::get(
          {0, 2}, IntegerType::get(rewriter.getContext(), 64));

      DenseIntElementsAttr emptyCollapseIntervals =
          DenseIntElementsAttr::get(emptyIntervalType, ArrayRef<int64_t>{});

      layout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), logicalShape, targetSquareGridShape,
          ttcore::OOBVal::Undef, memSpace, ttcore::TensorMemoryLayout::Sharded,
          emptyCollapseIntervals);

    } else {
      // Default-constructed collapse intervals will collapse to 2D.
      layout = ttcore::MetalLayoutAttr::get(
          rewriter.getContext(), logicalShape, targetSquareGridShape,
          ttcore::OOBVal::Undef, memSpace, ttcore::TensorMemoryLayout::Sharded);
    }

    // Get raw, unsharded physical shape.
    llvm::SmallVector<int64_t> unshardedShape =
        layout.getPhysicalShape(tileShape);

    // Calculate optimal grid for given physical shape.
    llvm::SmallVector<int64_t> optimalGrid = computeOptimalGrid(unshardedShape);

    // Get optimal sharded, on-device shape.
    llvm::SmallVector<int64_t> shardedShape =
        layout.getDeviceShape(optimalGrid, tileShape);

    auto emptyOp = rewriter.create<d2m::EmptyOp>(value.getLoc(), shardedShape,
                                                 elementType, layout);
    return rewriter.create<d2m::ToLayoutOp>(value.getLoc(), value, emptyOp)
        ->getResult(0);
  }

  // Insert toLayout ops for a genericOp's operands and results; this includes
  // sharding, tilizing, etc. This func computes appropriate optimal grid shape
  // as well.
  std::array<mlir::SmallVector<Value>, 2> toLayoutOperandsAndResults(
      mlir::ConversionPatternRewriter &rewriter,
      std::array<mlir::SmallVector<Value>, 2> operandsAndResults,
      bool tiled) const {
    std::array<mlir::SmallVector<Value>, 2> result;

    for (Value operand : operandsAndResults[0]) {
      result[0].push_back(
          createOptimalLayoutOp(operand, memorySpaces[0], tiled, rewriter));
    }
    for (Value operand : operandsAndResults[1]) {
      result[1].push_back(
          createOptimalLayoutOp(operand, memorySpaces[1], tiled, rewriter));
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

  // Common need to navigate DPS (<inputs>;<inits>) operand split:
  // note that this requires only 'getDpsInits()' to be available.
  template <typename Adaptor>
  static std::array<mlir::SmallVector<Value>, 2>
  splitDpsSignature(Adaptor adaptor, size_t numDPSInits) {
    auto numOperands = adaptor.getOperands().size();
    assert(numDPSInits <= numOperands && "expected numDPSInits <= numOperands");
    auto numInputs = numOperands - numDPSInits;
    mlir::ValueRange inputs = adaptor.getOperands().take_front(numInputs);
    mlir::ValueRange outputs = adaptor.getOperands().drop_front(numInputs);
    return {inputs, outputs};
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

  static Block::BlockArgListType createBlockArguments(mlir::Block *block,
                                                      mlir::Location loc,
                                                      mlir::TypeRange inputs,
                                                      mlir::TypeRange outputs) {
    auto fn = [&](Type t) {
      mlir::RankedTensorType tensorType = mlir::cast<mlir::RankedTensorType>(t);
      ttcore::MetalLayoutAttr layout =
          mlir::cast<ttcore::MetalLayoutAttr>(tensorType.getEncoding());
      auto shardShape = layout.getShardShape(tensorType);
      block->addArgument(
          mlir::RankedTensorType::get(shardShape, tensorType.getElementType()),
          loc);
    };

    llvm::for_each(mlir::TypeRange(inputs), fn);
    llvm::for_each(mlir::TypeRange(outputs), fn);
    return block->getArguments();
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

  // Helper to access a canonicalized form of input grid.  This will ensure two
  // things:
  // 1. We square-ify grids, so that transpose etc. will work. e.g. 13x10 ->
  // 10x10.
  // 2. If we wish to have uncollapsed tensors of rank greater than 2, we will
  // 1-pad the leading grid dims.  E.g. a 3d grid will be 1xXxY.
  const llvm::SmallVector<int64_t>
  paddedAndSquaredInputGridShape(size_t rank) const {
    assert(rank >= targetSquareGridShape.size());
    llvm::SmallVector<int64_t> grid(rank, 1);
    const size_t diff = rank - targetSquareGridShape.size();
    for (size_t i = 0; i < targetSquareGridShape.size(); ++i) {
      grid[i + diff] = targetSquareGridShape[i];
    }
    return grid;
  }

  // Helper to get output grid shape--this will be the canonical grid shape,
  // padded with 1s in leading dimensions as needed to match output grid rank.
  ttcore::GridAttr getOutputGrid(MLIRContext *ctx,
                                 ShapedType outputType) const {
    const size_t outputGridRank = outputType.getRank() / 2;
    return ttcore::GridAttr::get(
        ctx, paddedAndSquaredInputGridShape(outputGridRank));
  }

protected:
  // Default memory spaces for {inputs, outputs}.
  std::array<ttcore::MemorySpace, 2> memorySpaces;

private:
  // Actual HW grid shape.
  llvm::SmallVector<int64_t> targetGridShape;

  // Workaround variable to represent maximum square grid actual target grid can
  // hold. We need this to make Blackhole's nonsquare grid work properly for
  // tranpose.  This will treat e.g. 13x10 grid as 10x10 (take minimum element
  // in targetGridShape, and extend it to all indexes).
  llvm::SmallVector<int64_t> targetSquareGridShape;

protected:
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
      ttcore::MemorySpace defaultOutputMemSpace,
      const llvm::SmallVector<int64_t> &targetGridShape, bool ttnnMode,
      bool collapseTensors)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               targetGridShape, ttnnMode, collapseTensors) {}

private:
  static constexpr bool isComparisonOp =
      std::is_same_v<TileOp, d2m::TileEqzOp> ||
      std::is_same_v<TileOp, d2m::TileNezOp> ||
      std::is_same_v<TileOp, d2m::TileGtzOp> ||
      std::is_same_v<TileOp, d2m::TileGezOp> ||
      std::is_same_v<TileOp, d2m::TileLtzOp> ||
      std::is_same_v<TileOp, d2m::TileLezOp>;

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [origInputs, origOutputs] =
        splitDpsSignature(adaptor, op.getDpsInits().size());
    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {origInputs, origOutputs},
                                   /*tiled*/ true);

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    assert(numOperands == op->getNumOperands());

    ttcore::GridAttr grid =
        getOutputGrid(ctx, mlir::cast<ShapedType>(outputs[0].getType()));

    const std::size_t rank = grid.getShape().size();

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, numOperands, rank);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, rank);

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
        auto blockArgs = createBlockArguments(block, loc, TypeRange(inputs),
                                              TypeRange(outputs));

        // Create 'linalg.generic' accepting 'blockArgs'.

        SmallVector<mlir::AffineMap> linalgIndexingMaps =
            getAffineMapsArray(rewriter, numOperands, rank);
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
              mlir::Value yield;

              if constexpr (isComparisonOp) {
                // For comparison ops, first subtract then compare with zero.
                mlir::Value subResult = bbBuilder.create<d2m::TileSubOp>(
                    loc, /*resultTypes=*/bbArgs.take_back(numOutputs),
                    /*operands=*/bbArgs.take_front(numInputs));
                yield = bbBuilder.create<TileOp>(
                    loc, /*resultTypes=*/bbArgs.take_back(numOutputs),
                    /*operands=*/subResult);
              } else {
                // For regular elementwise ops, create TileOp directly.
                yield = bbBuilder.create<TileOp>(
                    loc,
                    /* resultTypes */ bbArgs.take_back(numOutputs).getTypes(),
                    /* operands */ bbArgs.take_front(numInputs));
              }

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
      ttcore::MemorySpace defaultOutputMemSpace,
      const llvm::SmallVector<int64_t> &targetGridShape, bool ttnnMode,
      bool collapseTensors)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               targetGridShape, ttnnMode, collapseTensors) {}

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [origInputs, origOutputs] =
        splitDpsSignature(adaptor, op.getDpsInits().size());
    SmallVector<mlir::Value> newInputs(origInputs.begin(), origInputs.end());
    newInputs.emplace_back(createScaler(
        rewriter, loc,
        mlir::cast<mlir::RankedTensorType>(origInputs.front().getType())
            .getElementType()));
    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {newInputs, origOutputs},
                                   /*tiled*/ true);

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    // Minus 1 for the scaler operand.
    assert((numOperands - 1) == op->getNumOperands());

    ttcore::GridAttr grid =
        getOutputGrid(ctx, mlir::cast<ShapedType>(outputs[0].getType()));

    const std::size_t rank = grid.getShape().size();

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, op, numOperands, rank);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, op, rank);

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
        auto blockArgs = createBlockArguments(block, loc, TypeRange(inputs),
                                              TypeRange(outputs));
        assert(blockArgs.size() == numOperands);

        // Create 'linalg.generic' accepting 'blockArgs'.

        SmallVector<mlir::AffineMap> linalgIndexingMaps =
            getAffineMapsArray(rewriter, op, numOperands, rank);
        SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
            iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

        // Propagate attributes.

        SmallVector<mlir::NamedAttribute> attributes;
        {
          // Propagate 'dim_arg' as 'ReduceDim'.
          attributes.emplace_back(
              d2m::ReduceDimAttr::getMnemonic(),
              d2m::ReduceDimAttr::get(ctx, dimArgAsReduceDim(op, rank)));
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
                                  mlir::Type elementType) {
    mlir::RankedTensorType scalerType =
        RankedTensorType::get(ttcore::TileType::getDefaultShape(), elementType);

    mlir::Attribute one;
    if (mlir::isa<mlir::FloatType>(elementType)) {
      one = mlir::FloatAttr::get(elementType, 1.0);
    } else if (mlir::isa<mlir::IntegerType>(elementType)) {
      one = mlir::IntegerAttr::get(elementType, 1);
    } else {
      llvm_unreachable("unexpected input element type");
    }

    mlir::DenseElementsAttr scalerValue =
        mlir::SplatElementsAttr::get(scalerType, one);

    return builder.create<ttir::ConstantOp>(loc, scalerType, scalerValue);
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
                    ttcore::MemorySpace defaultOutputMemSpace,
                    const llvm::SmallVector<int64_t> &targetGridShape,
                    bool ttnnMode, bool collapseTensors)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               targetGridShape, ttnnMode, collapseTensors) {}

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [origInputs, origOutputs] =
        splitDpsSignature(adaptor, op.getDpsInits().size());
    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {origInputs, origOutputs}, /*tiled*/ true);

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    assert(numOperands == op->getNumOperands());

    ttcore::GridAttr grid =
        getOutputGrid(ctx, mlir::cast<ShapedType>(outputs[0].getType()));

    const std::size_t rank = grid.getShape().size();

    // TODO(#2591) handle 'transpose_{a,b}' attributes.

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, numOperands, rank);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, rank);

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
        auto blockArgs = createBlockArguments(block, loc, TypeRange(inputs),
                                              TypeRange(outputs));

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
              getAffineMapsArray(rewriter, numOperands, rank);
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
                     ttcore::MemorySpace defaultOutputMemSpace,
                     const llvm::SmallVector<int64_t> &targetGridShape,
                     bool ttnnMode, bool collapseTensors)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               targetGridShape, ttnnMode, collapseTensors) {}

  LogicalResult
  matchAndRewrite(ttir::PermuteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto permutation = op.getPermutation();

    const int64_t permuteSize = static_cast<int64_t>(permutation.size());
    // Transpose pattern on inner dims.
    if (permuteSize == 2 || permutation[permuteSize - 2] == permuteSize - 1 ||
        permutation[permuteSize - 1] == permuteSize - 2) {
      return permuteInnerDims(op, adaptor, rewriter);
    }
    // Unhandled conversion case.
    return failure();
  }

  // Handler for permutation of inner dims (i.e. transpose).
  LogicalResult
  permuteInnerDims(ttir::PermuteOp op, typename ConcreteOp::Adaptor adaptor,
                   mlir::ConversionPatternRewriter &rewriter) const {
    auto permutation = op.getPermutation();
    assert(permutation.size() == 2 && permutation[0] == 1 &&
           permutation[1] == 0 && "Only 2D transpose supported");

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [origInputs, origOutputs] =
        splitDpsSignature(adaptor, op.getDpsInits().size());

    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {origInputs, origOutputs},
                                   /*tiled*/ true);

    auto inputTensorType =
        mlir::cast<mlir::RankedTensorType>(inputs[0].getType());
    auto inputShape = inputTensorType.getShape();
    const unsigned deviceRank = static_cast<unsigned>(inputShape.size());

    // Compute permutation map and permuted shape.
    auto [transposeMap, resultShape] = computePermutationMapAndShape(
        rewriter, permutation, inputShape, deviceRank);

    // Create the result layout by composing with input layout.
    auto inputLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(inputTensorType.getEncoding());
    AffineMap composedMap = transposeMap.compose(
        inputLayout.getIndexAffineMapOrIdentity(deviceRank));

    auto resultLayout = ttcore::MetalLayoutAttr::get(
        ctx, inputLayout.getLogicalShape(), inputLayout.getDimAlignments(),
        inputLayout.getCollapsedIntervals(), inputLayout.getOobVal(),
        inputLayout.getMemorySpace(), inputLayout.getMemoryLayout(),
        composedMap);

    auto viewType = mlir::RankedTensorType::get(
        resultShape, inputTensorType.getElementType(), resultLayout);

    // For inner permute, we need as streamLayout to do reblocking.
    auto storage = rewriter.create<d2m::EmptyOp>(
        loc, resultShape, inputTensorType.getElementType(), resultLayout);
    auto stream =
        rewriter.create<d2m::StreamLayoutOp>(loc, viewType, inputs[0], storage);
    inputs[0] = stream.getResult();

    // For inner permute, we alse need a GenericOp to transpose each individual
    // tile.
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange blockArgs) {
          auto identityMap = builder.getMultiDimIdentityMap(2);
          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes(
              2, mlir::utils::IteratorType::parallel);

          auto linalgGeneric = builder.create<mlir::linalg::GenericOp>(
              bodyLoc,
              llvm::to_vector(
                  mlir::ValueRange(blockArgs.take_back(1)).getTypes()),
              blockArgs.take_front(1), blockArgs.take_back(1),
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
  // Apply permutation mapping to affine map and input shape to get permuted map
  // and shape.
  std::pair<AffineMap, SmallVector<int64_t>> computePermutationMapAndShape(
      mlir::ConversionPatternRewriter &rewriter, ArrayRef<int64_t> permutation,
      ArrayRef<int64_t> inputShape, unsigned deviceRank) const {

    unsigned logicalRank = deviceRank / 2;
    assert(logicalRank == permutation.size());
    SmallVector<AffineExpr> results(deviceRank);
    SmallVector<int64_t> resultShape(deviceRank);

    for (auto [dstIdx, srcIdx] : llvm::enumerate(permutation)) {
      // Permute grid mapping.
      results[dstIdx] = rewriter.getAffineDimExpr(srcIdx);
      // Permute shard mapping.
      results[logicalRank + dstIdx] =
          rewriter.getAffineDimExpr(logicalRank + srcIdx);

      // Permute grid shape.
      resultShape[dstIdx] = inputShape[srcIdx];
      // Permute shard shape.
      resultShape[dstIdx + logicalRank] = inputShape[srcIdx + logicalRank];
    }

    AffineMap transposeMap =
        AffineMap::get(deviceRank, 0, results, rewriter.getContext());
    return {transposeMap, resultShape};
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
    auto newOp = rewriter.create<d2m::ToLayoutOp>(
        op.getLoc(), adaptor.getInput(), empty, nullptr);
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

} // namespace mlir::tt

namespace mlir::tt {
void populateTTIRToD2MPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    ttcore::MemorySpace defaultInputMemSpace,
    ttcore::MemorySpace defaultOutputMemSpace,
    const llvm::SmallVector<int64_t> &targetGridShape, bool ttnnMode,
    bool collapseTensors) {
  // clang-format off
  patterns.add<
    // Elementwise.
    D2MNamedElementwiseRewriter<ttir::AbsOp,        d2m::TileAbsOp>,
    D2MNamedElementwiseRewriter<ttir::AddOp,        d2m::TileAddOp>,
    D2MNamedElementwiseRewriter<ttir::CeilOp,       d2m::TileCeilOp>,
    D2MNamedElementwiseRewriter<ttir::CosOp,        d2m::TileCosOp>,
    D2MNamedElementwiseRewriter<ttir::DivOp,        d2m::TileDivOp>,
    D2MNamedElementwiseRewriter<ttir::ExpOp,        d2m::TileExpOp>,
    D2MNamedElementwiseRewriter<ttir::FloorOp,      d2m::TileFloorOp>,
    D2MNamedElementwiseRewriter<ttir::GeluOp,       d2m::TileGeluOp>,
    D2MNamedElementwiseRewriter<ttir::LogOp,        d2m::TileLogOp>,
    D2MNamedElementwiseRewriter<ttir::LogicalNotOp, d2m::TileLogicalNotOp>,
    D2MNamedElementwiseRewriter<ttir::MultiplyOp,   d2m::TileMulOp>,
    D2MNamedElementwiseRewriter<ttir::MaximumOp,    d2m::TileMaximumOp>,
    D2MNamedElementwiseRewriter<ttir::NegOp,        d2m::TileNegativeOp>,
    D2MNamedElementwiseRewriter<ttir::PowTensorOp,  d2m::TilePowOp>,
    D2MNamedElementwiseRewriter<ttir::ReciprocalOp, d2m::TileRecipOp>,
    D2MNamedElementwiseRewriter<ttir::RsqrtOp,      d2m::TileRsqrtOp>,
    D2MNamedElementwiseRewriter<ttir::SigmoidOp,    d2m::TileSigmoidOp>,
    D2MNamedElementwiseRewriter<ttir::SinOp,        d2m::TileSinOp>,
    D2MNamedElementwiseRewriter<ttir::SqrtOp,       d2m::TileSqrtOp>,
    D2MNamedElementwiseRewriter<ttir::SubtractOp,   d2m::TileSubOp>,
    D2MNamedElementwiseRewriter<ttir::TanOp,        d2m::TileTanOp>,

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
    // Permute (handles tranpose ops, since they're canonicalized into permutes).
    D2MPermuteRewriter
  >(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, targetGridShape, ttnnMode, collapseTensors);

  // ToLayout 1:1 conversion.
  patterns.add<D2MToLayoutOpRewriter>(typeConverter, ctx);

  // Creation ops 1:1 conversion.
  patterns.add<D2MEmptyOpRewriter>(typeConverter, ctx);

  // Matmul.
  patterns.add<D2MMatmulRewriter<d2m::TileMatmulOp>>(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, targetGridShape, ttnnMode, collapseTensors);
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
    this->overrideDeviceShape = options.overrideDeviceShape;
    this->ttnnMode = options.ttnnMode;
    this->collapseTensorsTo2D = options.collapseTensorsTo2D;
  }

  TTIRToD2MPass(const TTIRToD2MPass &rhs) : Base(rhs) {
    // Workaround: Passes are required to be copy-constructible but autogen'ed
    // base class copy constructors ignore Pass option fields.
    this->defaultInputMemSpace = rhs.defaultInputMemSpace;
    this->defaultOutputMemSpace = rhs.defaultOutputMemSpace;
    this->overrideDeviceShape = rhs.overrideDeviceShape;
    this->ttnnMode = rhs.ttnnMode;
    this->collapseTensorsTo2D = rhs.collapseTensorsTo2D;
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();

    // Get target grid shape from device or override.
    llvm::SmallVector<int64_t> gridShape = getTargetGridShape();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });

    RewritePatternSet patterns(ctx);
    populateTTIRToD2MPatterns(ctx, patterns, typeConverter,
                              defaultInputMemSpace, defaultOutputMemSpace,
                              gridShape, ttnnMode, collapseTensorsTo2D);

    ConversionTarget target(*ctx);
    target.addIllegalDialect<mlir::tt::ttir::TTIRDialect>();
    target.addLegalDialect<::mlir::BuiltinDialect>();
    target.addLegalDialect<::mlir::func::FuncDialect>();
    target.addLegalDialect<::mlir::linalg::LinalgDialect>();
    target.addLegalDialect<mlir::tt::d2m::D2MDialect>();
    target.addLegalDialect<mlir::tt::ttcore::TTCoreDialect>();

    // Keep some TTIR ops legal if they don't have D2M equivalents.
    target.addLegalOp<mlir::tt::ttir::ConstantOp>();
    target.addLegalOp<mlir::tt::ttir::FullOp>();
    target.addLegalOp<ttir::MeshShardOp>();
    target.addLegalOp<ttir::TTNNMetalLayoutCastOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

private:
  // Helper to get defined device shape if an override is not provided.
  llvm::SmallVector<int64_t> getTargetGridShape() {
    if (!overrideDeviceShape.empty()) {
      return llvm::to_vector(overrideDeviceShape);
    }

    // Get from device if no override given.
    ::mlir::ModuleOp moduleOp = getOperation();
    mlir::tt::ttcore::DeviceAttr device =
        mlir::tt::ttcore::lookupDevice(moduleOp);
    assert(device && "Device not found");
    return llvm::to_vector(device.getWorkerGrid().getShape());
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
