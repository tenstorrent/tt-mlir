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

    auto metalLayout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), tensorType.getShape(), ttcore::OOBVal::Undef,
        memSpace, memLayout, collapsedIntervals, dimAlignments);

    llvm::SmallVector<int64_t> unshardedShape =
        metalLayout.getPhysicalShape(ttcore::TileType::getDefaultShape());

    llvm::SmallVector<int64_t> shardedShape = metalLayout.getDeviceShape(
        ttnnLayout.getGrid().getShape(), ttcore::TileType::getDefaultShape());

    Type elementType = ttnnLayout.getElementType();
    return mlir::RankedTensorType::get(shardedShape, elementType, metalLayout);
  }

  // Create a ToLayout operation for a value using the provided layout
  // information with a simple 1x1 grid; actual grid optimization and proper
  // dimension alignments are computed later in the D2MGridSelection pass.
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
          rewriter.getContext(), logicalShape, ttcore::OOBVal::Undef, memSpace,
          ttcore::TensorMemoryLayout::Sharded, emptyCollapseIntervals);

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

  void createComputeRegion(mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                           mlir::ValueRange bbArgs,
                           mlir::ConversionPatternRewriter &rewriter,
                           mlir::Location loc, const size_t numInputs,
                           const size_t numOutputs) const {
    mlir::ValueRange operands = bbArgs.take_front(numInputs);
    mlir::TypeRange resultTypes = bbArgs.take_back(numOutputs);

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

    auto [origInputs, origOutputs] =
        splitDpsSignature(adaptor, op.getDpsInits().size());
    auto [inputs, outputs] =
        toLayoutOperandsAndResults(rewriter, {origInputs, origOutputs},
                                   /*tiled*/ true);
    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);
    assert(numOperands == op->getNumOperands());

    const std::size_t physicalRank =
        ttcore::getDeviceLayout(outputs[0]).getRank() / 2;

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

        // Create 'linalg.generic' accepting 'blockArgs'.

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
              createComputeRegion(bbBuilder, bbLoc, bbArgs, rewriter, loc,
                                  numInputs, numOutputs);
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

    auto [origInputs, origOutputs] =
        splitDpsSignature(adaptor, op.getDpsInits().size());
    auto [inputs, outputs] = toLayoutOperandsAndResults(
        rewriter, {origInputs, origOutputs}, /*tiled*/ true);

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    assert(numOperands == op->getNumOperands());

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
                     bool collapseTensors)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        D2MNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                               ttnnMode, collapseTensors) {}

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
    auto inputLayout =
        mlir::cast<ttcore::MetalLayoutAttr>(inputTensorType.getEncoding());

    // Compute permutation for all relevant attributes.
    auto permuted = computePermutation(
        rewriter, permutation, inputShape, deviceRank,
        inputLayout.getLogicalShape(), inputLayout.getDimAlignments());

    // Create the result layout by composing with input layout.
    AffineMap composedMap = permuted.transposeMap.compose(
        inputLayout.getIndexAffineMapOrIdentity(deviceRank));

    auto resultLayout = ttcore::MetalLayoutAttr::get(
        ctx, permuted.logicalShape, permuted.dimAlignments,
        inputLayout.getCollapsedIntervals(), inputLayout.getOobVal(),
        inputLayout.getMemorySpace(), inputLayout.getMemoryLayout(),
        composedMap);

    auto viewType = mlir::RankedTensorType::get(
        permuted.physicalShape, inputTensorType.getElementType(), resultLayout);

    // For inner permute, we need as streamLayout to do reblocking.
    auto storage = rewriter.create<d2m::EmptyOp>(
        loc, permuted.physicalShape, inputTensorType.getElementType(),
        resultLayout);
    auto stream =
        rewriter.create<d2m::StreamLayoutOp>(loc, viewType, inputs[0], storage);
    inputs[0] = stream.getResult();

    // For inner permute, we alse need a GenericOp to transpose each individual
    // tile.
    auto generic = rewriter.create<d2m::GenericOp>(
        loc, inputs, outputs,
        [&](OpBuilder &builder, Location bodyLoc, ValueRange blockArgs) {
          assert(blockArgs.size() == 2);
          auto identityMap = builder.getMultiDimIdentityMap(2);
          SmallVector<mlir::utils::IteratorType> linalgIteratorTypes(
              2, mlir::utils::IteratorType::parallel);

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

// Gather op conversion.
// The ttir gather op conversion has been adapted from
// https://github.com/openxla/stablehlo/blob/4c0d4841519aed22e3689c30b72a0e4228051249/stablehlo/conversions/linalg/transforms/StablehloLegalizeToLinalg.cpp#L1766.
Value extractIndexFromTensor(OpBuilder& builder, Location loc, Value tensor,
                             ShapedType originalType,
                             ArrayRef<Value> tensorIndex = {}) {
  Value extracted = builder.create<mlir::tensor::ExtractOp>(loc, tensor, tensorIndex);
  if (extracted.getType().isIndex()) return extracted;
  return originalType.getElementType().isUnsignedInteger()
             ? builder.createOrFold<arith::IndexCastUIOp>(
                   loc, builder.getIndexType(), extracted)
             : builder.createOrFold<arith::IndexCastOp>(
                   loc, builder.getIndexType(), extracted);
}

SmallVector<utils::IteratorType, 3> getParallelAndReductionIterators(
    unsigned nLoops, unsigned nReduction) {
  SmallVector<utils::IteratorType, 3> res(nLoops - nReduction,
                                          utils::IteratorType::parallel);
  res.append(nReduction, utils::IteratorType::reduction);
  return res;
}

SmallVector<utils::IteratorType, 3> getNParallelLoopsAttrs(
    unsigned nParallelLoops) {
  return getParallelAndReductionIterators(nParallelLoops, 0);
}

Value getEmptySparseTensor(OpBuilder& builder, Location loc, ShapedType type,
                           ArrayRef<Value> dynSizes) {
  auto allocTensor = builder.create<bufferization::AllocTensorOp>(
    loc,
    llvm::cast<TensorType>(type),
    dynSizes,
    /*copy=*/Value(),
    /*memory_space=*/IntegerAttr());
  return allocTensor;
}

Value getEmptyTensor(OpBuilder& builder, Location loc, ShapedType type,
                     ArrayRef<Value> dynSizes) {
    auto empty = builder.create<d2m::EmptyOp>(
      loc,
      type.getShape(),
      type.getElementType());
  return empty;
}

Value getEmptyTensorFor(OpBuilder& builder, Location loc, ShapedType resultType,
                        Operation* op, ValueRange operands) {
  bool isSparse = mlir::sparse_tensor::getSparseTensorEncoding(resultType) != nullptr;
  // Collect the sizes for a ranked tensor to be passed as parameter to a
  // new tensor initialization operation. This operation only needs the
  // dynamic sizes.
  SmallVector<Value> sizes;
  if (!resultType.hasStaticShape()) {
    // Ask the op for its output shape.
    auto shapeSource = cast<InferShapedTypeOpInterface>(op);
    SmallVector<Value, 1> reifiedShapes;
    if (failed(shapeSource.reifyReturnTypeShapes(builder, operands, reifiedShapes))) {
      llvm::report_fatal_error("could not reify");
    }
    assert(reifiedShapes.size() == 1 && "Expected one reified result");
    // Construct sizes for the required dimensions.
    for (const auto& en : llvm::enumerate(resultType.getShape())) {
      if (en.value() != ShapedType::kDynamic) continue;
      Value idx = builder.create<arith::ConstantIndexOp>(loc, en.index());
      Value extracted = builder.create<tensor::ExtractOp>(loc, reifiedShapes[0], ValueRange{idx});
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

    Value startIndices = adaptor.getStartIndices();
    Value operand = adaptor.getInput();

    RankedTensorType operandType = getTypeConverter()->convertType<RankedTensorType>(operand.getType());
    RankedTensorType startIndicesType = getTypeConverter()->convertType<RankedTensorType>(startIndices.getType());
    RankedTensorType resultType = getTypeConverter()->convertType<RankedTensorType>(gatherOp.getType());

    int64_t resultRank = resultType.getRank();
    // slice_sizes has to have the same size as operand.rank, and doing it this
    // way permits an unranked operand.
    int64_t operandRank = gatherOp.getSliceSizes().size();
    int64_t indexVectorDim = gatherOp.getIndexVectorDim();
    ArrayRef<int64_t> offsetDims =
        gatherOp.getOffsetDims();
    ArrayRef<int64_t> collapsedSliceDims =
        gatherOp.getCollapsedSliceDims();
    ArrayRef<int64_t> operandBatchingDims =
        gatherOp.getOperandBatchingDims();
    ArrayRef<int64_t> startIndicesBatchingDims =
        gatherOp.getStartIndicesBatchingDims();
    ArrayRef<int64_t> startIndexMap =
        gatherOp.getStartIndexMap();

    // Insert to metal layout conversion for the operand.
    auto i64Ty = builder.getI64Type();
    auto intervalTy = RankedTensorType::get({2, 2}, i64Ty);
    llvm::SmallVector<int64_t> dimAlignments(operandType.getShape().size(), 1);
    dimAlignments[dimAlignments.size() - 1] = 32;
    dimAlignments[dimAlignments.size() - 2] = 32;

    auto dataOperand = llvm::SmallVector<int64_t, 4>{0, 1, 1, 2};
    DenseIntElementsAttr collapsedIntervalsOperand = DenseIntElementsAttr::get(intervalTy, llvm::ArrayRef<int64_t>(dataOperand));
    
    auto metalLayout = ttcore::MetalLayoutAttr::get(
      getContext(), operandType.getShape(), ttcore::OOBVal::Undef,
      ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Sharded, collapsedIntervalsOperand, dimAlignments);

    llvm::SmallVector<int64_t> operandShape = {1, 1};
    operandShape.append(operandType.getShape().begin(),
                        operandType.getShape().end());
    auto empty = builder.create<d2m::EmptyOp>(
      loc,
      operandShape,
      operandType.getElementType(),
      metalLayout);

    [[maybe_unused]] auto operandLayoutOp = builder.create<d2m::ToLayoutOp>(loc, operand, empty, nullptr);

    // Insert to metal layout conversion for the indices.
    llvm::SmallVector<int64_t> indexDimAlignments(startIndicesType.getShape().size(), 1);
    indexDimAlignments[indexDimAlignments.size() - 1] = 1;
    indexDimAlignments[indexDimAlignments.size() - 2] = 32;

    auto dataIndices = llvm::SmallVector<int64_t, 4>{0, 1, 1, 2};
    DenseIntElementsAttr collapsedIntervalsIndices = DenseIntElementsAttr::get(intervalTy, llvm::ArrayRef<int64_t>(dataIndices));

    auto indexMetalLayout = ttcore::MetalLayoutAttr::get(
      getContext(), startIndicesType.getShape(), ttcore::OOBVal::Undef,
      ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Sharded, collapsedIntervalsIndices, indexDimAlignments);

      llvm::SmallVector<int64_t> startIndicesShape = {1, 1};
      startIndicesShape.append(startIndicesType.getShape().begin(),
                          startIndicesType.getShape().end());
    auto indexEmpty = builder.create<d2m::EmptyOp>(
      loc,
      startIndicesShape,
      startIndicesType.getElementType(),
      indexMetalLayout);

    [[maybe_unused]] auto startIndicesLayoutOp = builder.create<d2m::ToLayoutOp>(loc, startIndices, indexEmpty, nullptr);

    // Create the output tensor.
    llvm::SmallVector<int64_t> resultDimAlignments(resultType.getShape().size(), 1);
    resultDimAlignments[resultDimAlignments.size() - 1] = 1;
    resultDimAlignments[resultDimAlignments.size() - 2] = 32;
    resultDimAlignments[resultDimAlignments.size() - 2] = 32;

    auto dataResult = llvm::SmallVector<int64_t, 4>{0, 2, 2, 3};
    DenseIntElementsAttr collapsedIntervalsResult = DenseIntElementsAttr::get(intervalTy, llvm::ArrayRef<int64_t>(dataResult));

    auto resultMetalLayout = ttcore::MetalLayoutAttr::get(
      getContext(), resultType.getShape(), ttcore::OOBVal::Undef,
      ttcore::MemorySpace::DeviceDRAM, ttcore::TensorMemoryLayout::Sharded, collapsedIntervalsResult, resultDimAlignments);

    Value emptyOp = getEmptyTensorFor(builder, loc, resultType, gatherOp,adaptor.getOperands());
    llvm::SmallVector<int64_t> resultShape = {1};
    resultShape.append(resultType.getShape().begin(),
                          resultType.getShape().end());
    auto resultEmpty = builder.create<d2m::EmptyOp>(
      loc,
      resultShape,
      resultType.getElementType(),
      resultMetalLayout);

    [[maybe_unused]] auto resultIndicesLayoutOp = builder.create<d2m::ToLayoutOp>(loc, emptyOp, resultEmpty, nullptr);

    // Create the D2M generic op.
    // void GenericOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange results, ::mlir::ValueRange inputs, ::mlir::ValueRange outputs, ::mlir::tt::ttcore::GridAttr grid, ::mlir::ArrayAttr block_factors, ::mlir::ArrayAttr indexing_maps, ::mlir::ArrayAttr iterator_types, ::mlir::ArrayAttr threads, unsigned regionsCount) {
  
    tt::ttcore::GridAttr grid = ttcore::GridAttr::get(getContext(), llvm::SmallVector<int64_t, 4>{1, 1});
    unsigned numDims = 3;
    unsigned numSymbols = 0;

    // affine_map<(d0, d1, d2) -> (0, 0)>
    auto map1 = AffineMap::get(numDims, numSymbols,
      {builder.getAffineConstantExpr(0), builder.getAffineConstantExpr(0)},
      builder.getContext());

    // affine_map<(d0, d1, d2) -> (d0, d1)>
    auto map2 = AffineMap::get(numDims, numSymbols,
      {builder.getAffineDimExpr(0), builder.getAffineDimExpr(1)},
      builder.getContext());

    // affine_map<(d0, d1, d2) -> (d1, d2)>
    auto map3 = AffineMap::get(numDims, numSymbols,
      {builder.getAffineDimExpr(1), builder.getAffineDimExpr(2)},
      builder.getContext());

      SmallVector<AffineMap, 3> maps = {map1, map2, map3};
    auto mapArrayAttr = builder.getAffineMapArrayAttr(maps);

    mlir::tt::ttcore::IteratorTypeAttr attr0 = mlir::tt::ttcore::IteratorTypeAttr::get(getContext(), mlir::tt::ttcore::IteratorType::Parallel);
    mlir::tt::ttcore::IteratorTypeAttr attr1 = mlir::tt::ttcore::IteratorTypeAttr::get(getContext(), mlir::tt::ttcore::IteratorType::Parallel);
    mlir::tt::ttcore::IteratorTypeAttr attr2 = mlir::tt::ttcore::IteratorTypeAttr::get(getContext(), mlir::tt::ttcore::IteratorType::Parallel);
    llvm::SmallVector<mlir::tt::ttcore::IteratorTypeAttr, 3> iteratorArrayAttr = {attr0, attr1, attr2};
    llvm::SmallVector<mlir::Attribute, 3> iteratorAttrs(iteratorArrayAttr.begin(), iteratorArrayAttr.end());
    mlir::ArrayAttr iteratorArrayAttrTaps = mlir::ArrayAttr::get(getContext(), iteratorAttrs);
    llvm::SmallVector<int64_t, 4> blockFactors = {1, 1, 1};

    llvm::SmallVector<Value> tapsInputs = {operandLayoutOp.getResult(0), startIndicesLayoutOp.getResult(0)};
    llvm::SmallVector<Value> tapsOutputs = {resultIndicesLayoutOp.getResult(0)};


    llvm::SmallVector<ttcore::MemorySpace> blockArgumentMemorySpaces = {
      ttcore::MemorySpace::DeviceRiscvL1,
      ttcore::MemorySpace::DeviceRiscvL1,
      ttcore::MemorySpace::DeviceRiscvL1
    };
    
    // create generic op
    [[maybe_unused]] auto genericOp = builder.create<d2m::GenericOp>(loc, tapsInputs, tapsOutputs, mapArrayAttr, iteratorArrayAttrTaps, 
      [&](OpBuilder &builder, Location bodyLoc, ValueRange blockArgs) {
      },
      blockArgumentMemorySpaces,
    d2m::ThreadType::Datamovement,
    grid, blockFactors
    );

    // create to layout for generic op
    auto finalOutput = builder.create<d2m::EmptyOp>(
      loc,
      resultType.getShape(),
      resultType.getElementType());
    [[maybe_unused]] auto toLayoutOp = builder.create<d2m::ToLayoutOp>(loc, genericOp.getResult(0), finalOutput, nullptr);

    Region &region = genericOp.getRegion(0);
    Block &block = region.front();
    OpBuilder::InsertionGuard guard(rewriter); 
    builder.setInsertionPointToEnd(&block);

    Value blockOperand = block.getArgument(0);
    [[maybe_unused]] Value blockIndex = block.getArgument(1);
    [[maybe_unused]] Value blockOutput = block.getArgument(2);

    auto blockOperandWaitOp = builder.create<d2m::WaitOp>(loc, blockOperand);
    auto blockIndexWaitOp = builder.create<d2m::WaitOp>(loc, blockIndex);
    auto blockOutputReserveOp = builder.create<d2m::ReserveOp>(loc, blockOutput);
   
    SmallVector<Value> ivs;
    // Verify all static before using affine.for
    for (int64_t i = 0; i < resultRank; ++i) {
      int64_t dim = resultType.getDimSize(i);
      auto loop = builder.create<mlir::affine::AffineForOp>(loc, 0, dim);
      builder.setInsertionPointToStart(loop.getBody());
      ivs.push_back(loop.getInductionVar());
    }

    // Dimensions in the result that aren't offset dimensions are called batch.
    SmallVector<int64_t> batchDims;
    for (int64_t dim = 0; dim < resultRank; ++dim) {
      if (!llvm::is_contained(offsetDims, dim)) {
        batchDims.push_back(dim);
      }
    }

    // We'll need these later and creating them on demand we end up with
    // duplicates, which also makes lit tests really hard to write.
    SmallVector<Value> constants;
    for (int64_t i = 0, e = std::max({resultRank, operandRank, int64_t{2}}); i < e; ++i) {
      auto constOp = builder.create<mlir::arith::ConstantOp>(loc, rewriter.getIndexAttr(i));
      constants.push_back(constOp);
    }

    // Now the complicated part. For a given output dimension we build up an
    // index into the input. It's composed of two parts: the index coming from
    // start_indices, and the offset from that index along the offset
    // dimensions. Everything includes dimension shuffling and remapping as well
    // because of the way gather is defined to allow for any-layout input by
    // adding more attributes.

    // The base gather index (`G` in the documentation) points to a place in
    // start_indices along the batch dimensions.
    SmallVector<Value> gatherIndex;
    for (int64_t dim : batchDims) {
      gatherIndex.push_back(ivs[dim]);
    }

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
          builder, loc, blockIndexWaitOp.getResult(), gatherOp.getStartIndices().getType(),
          gCombine));
    }

    // But then start indices are shuffled by the start index map. To make a
    // full index into the operand, all missing indices are zeroes.
    SmallVector<Value> remappedIndexFromIndices(operandRank, constants[0]);
    for (auto [idx, value] : llvm::enumerate(startIndexMap)) {
      remappedIndexFromIndices[value] = indexFromStartIndices[idx];
    }

    // Now we construct the index based on the operand/start_indices batching
    // dimensions.
    SmallVector<Value> indexFromBatching(operandRank, constants[0]);
    for (auto [operandDim, indicesDim] :
          llvm::zip_equal(operandBatchingDims, startIndicesBatchingDims)) {
      indexFromBatching[operandDim] =
          gatherIndex[indicesDim + (indicesDim < indexVectorDim ? 0 : 1)];
    }

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
            loc, blockOutputReserveOp.getResult(), offsetDims[operandIndexDim++]);
      }

      // If this is a skipped dimension, we're done and don't have to clamp.
      if (remappedIndexFromIndices[i] == constants[0]) continue;

      // taps
      // d2m::CBType::get(mlir::RankedTensorType::get(shardShape, tensorType.getElementType()))
      d2m::CBType thisType = mlir::cast<d2m::CBType>(blockOperand.getType());
      
      Value operandDimSize = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getIndexAttr(thisType.getShape()[i]));


      Value largestValidIndex = builder.create<mlir::arith::SubIOp>(
          loc, operandDimSize, outputDimSize);

      // Clamp indices to [0, i, operand_dim-output_dim].
      Value clamp = builder.create<mlir::arith::MinSIOp>(
          loc,
          builder.create<mlir::arith::MaxSIOp>(
              loc, constants[0], remappedIndexFromIndices[i]),
          largestValidIndex);
      remappedIndexFromIndices[i] = clamp;
    }

    // For the (remapped) offset dimensions, the index is the current index in
    // the output. As before this is expanded to a full index into the operand
    // by using zeros for the missing indices.
    SmallVector<Value> indexFromOffset(operandRank, constants[0]);
    for (auto [remappedOffsetDim, offsetDim] :
          llvm::zip_equal(remappedOffsetDims, offsetDims)) {
      indexFromOffset[remappedOffsetDim] = ivs[offsetDim];
    }

    // Now we add together our three indices to get the final index into the
    // operand.
    SmallVector<Value> combinedIndex;
    for (int64_t i = 0; i < operandRank; ++i)
      combinedIndex.push_back(builder.create<mlir::arith::AddIOp>(
          loc, rewriter.getIndexType(),
          builder.create<mlir::arith::AddIOp>(loc, rewriter.getIndexType(),
                                                remappedIndexFromIndices[i],
                                                indexFromBatching[i]),
          indexFromOffset[i]));
    
    Value extractOperand;
    if (isa<RankedTensorType>(operand.getType())) {
      extractOperand = blockOperandWaitOp.getResult();
    } else {
      // Cannot extract from unranked tensors, cast to ranked first.
      SmallVector<int64_t> dims(operandRank, ShapedType::kDynamic);
      auto type = RankedTensorType::get(
          dims, cast<TensorType>(operand.getType()).getElementType());
      extractOperand = builder.create<mlir::tensor::CastOp>(loc, type, blockOperandWaitOp.getResult());
    }

    SmallVector<Value> finalIVs;
    if (ivs.size() > 2) {
      Value collapsedLeading = ivs.front();
      for (size_t i = 1; i < ivs.size() - 1; ++i) {
        collapsedLeading = builder.create<arith::MulIOp>(loc, collapsedLeading, ivs[i]);
      }

      Value lastDim = ivs.back();
      finalIVs.push_back(collapsedLeading);
      finalIVs.push_back(lastDim);
    }

    auto dmaOp = builder.create<d2m::DMAOp>(loc, extractOperand, combinedIndex, blockOutputReserveOp.getResult(), finalIVs);
    builder.create<d2m::DMAWaitOp>(loc, dmaOp);

    builder.setInsertionPointToEnd(&block);
    builder.create<d2m::YieldOp>(loc, blockOutputReserveOp.getResult());
    
    rewriter.replaceOp(gatherOp, toLayoutOp.getResult(0));

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
    D2MNamedElementwiseRewriter<ttir::AbsOp,        d2m::TileAbsOp>,
    D2MNamedElementwiseRewriter<ttir::AddOp,        d2m::TileAddOp>,
    D2MNamedElementwiseRewriter<ttir::BitwiseNotOp, d2m::TileBitwiseNotOp>,
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
    D2MNamedElementwiseRewriter<ttir::PowOp,        d2m::TilePowOp>,
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
