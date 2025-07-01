// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRGeneric/TTIRToTTIRGeneric.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

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

#include <algorithm>
#include <array>

namespace mlir::tt {

namespace {
class TTIRNamedRewriterCommon {
protected:
  using base = TTIRNamedRewriterCommon;

  TTIRNamedRewriterCommon(ttcore::MemorySpace defaultInputMemSpace,
                          ttcore::MemorySpace defaultOutputMemSpace,
                          const llvm::SmallVector<int64_t> &targetGridShape)
      : memorySpaces{defaultInputMemSpace, defaultOutputMemSpace},
        targetGridShape(targetGridShape),
        targetSquareGridShape(
            ttir::utils::getSquareTargetGrid(targetGridShape)) {
    assert(!targetGridShape.empty());
  }

  // Compute optimal grid shape that works for all provided layout infos.
  llvm::SmallVector<int64_t>
  computeOptimalGrid(ArrayRef<int64_t> physicalShape) const {
    llvm::SmallVector<int64_t> grid;

    assert(physicalShape.size() == targetSquareGridShape.size());

    for (size_t i = 0; i < physicalShape.size(); ++i) {
      const int64_t dim = physicalShape[i];
      assert(dim > 0);
      // Find largest grid dimension that divides evenly
      for (int64_t g = targetSquareGridShape[i]; g > 0; g--) {
        if (dim % g == 0) {
          grid.push_back(g);
          break;
        }
      }
    }

    assert(grid.size() == physicalShape.size());

    return grid;
  }

  // Create a ToLayout op for a value using the provided layout info and grid.
  Value createOptimalLayoutOp(Value value, ttcore::MemorySpace memSpace,
                              bool tiled,
                              mlir::ConversionPatternRewriter &rewriter) const {
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

    auto layout = ttcore::MetalLayoutAttr::get(
        rewriter.getContext(), logicalShape, targetSquareGridShape,
        ttcore::OOBVal::Undef, memSpace);

    // Get raw, unsharded physical shape.
    llvm::SmallVector<int64_t> unshardedShape =
        layout.getPhysicalShape(tileShape);

    // Calculate optimal grid for given physical shape.
    llvm::SmallVector<int64_t> optimalGrid = computeOptimalGrid(unshardedShape);

    // Get optimal sharded, on-device shape.
    llvm::SmallVector<int64_t> shardedShape =
        layout.getDeviceShape(optimalGrid, tileShape);

    auto resultType =
        mlir::RankedTensorType::get(shardedShape, elementType, layout);

    auto emptyOp =
        rewriter.create<tt::ttir::EmptyOp>(value.getLoc(), resultType);
    return rewriter
        .create<tt::ttir::ToLayoutOp>(value.getLoc(), value, emptyOp)
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

  static Operation *unLayoutResult(mlir::ConversionPatternRewriter &rewriter,
                                   Value fromValue, Type toResultType) {
    auto output =
        rewriter.create<tt::ttir::EmptyOp>(fromValue.getLoc(), toResultType);
    return rewriter.create<tt::ttir::ToLayoutOp>(fromValue.getLoc(), fromValue,
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

protected:
  // Default memory spaces for {inputs, outputs}.
  std::array<ttcore::MemorySpace, 2> memorySpaces;

private:
  // This should become protected instead once we remove square grid workaround.
  llvm::SmallVector<int64_t> targetGridShape;

protected:
  // Workaround variable to represent maximum square grid actual target grid can
  // hold. We need this to make Blackhole's nonsquare grid work properly for
  // tranpose.  This will treat e.g. 13x10 grid as 10x10 (take minimum element
  // in targetGridShape, and extend it to all indexes).
  llvm::SmallVector<int64_t> targetSquareGridShape;
};
} // namespace

namespace {
// ----------------------------------------------------------------------------
//
// Rewrite elementwise ops by emitting a matching tile version of the op
// into a ttir.generic/linalg.generic nest.
template <typename ConcreteOp, typename TileOp>
class TTIRNamedElementwiseRewriter final
    : public mlir::OpConversionPattern<ConcreteOp>,
      TTIRNamedRewriterCommon {

public:
  TTIRNamedElementwiseRewriter<ConcreteOp, TileOp>(
      const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
      ttcore::MemorySpace defaultInputMemSpace,
      ttcore::MemorySpace defaultOutputMemSpace,
      const llvm::SmallVector<int64_t> &targetGridShape)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        TTIRNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                                targetGridShape) {}

private:
  static constexpr bool isComparisonOp =
      std::is_same_v<TileOp, ttir::TileEqzOp> ||
      std::is_same_v<TileOp, ttir::TileNezOp> ||
      std::is_same_v<TileOp, ttir::TileGtzOp> ||
      std::is_same_v<TileOp, ttir::TileGezOp> ||
      std::is_same_v<TileOp, ttir::TileLtzOp> ||
      std::is_same_v<TileOp, ttir::TileLezOp>;

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

    ttcore::GridAttr grid = ttcore::GridAttr::get(ctx, targetSquareGridShape);

    const std::size_t rank = grid.getShape().size();

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, numOperands, rank);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, rank);

    // Create 'ttir.generic' accepting 'op's operands.
    auto generic = rewriter.create<ttir::GenericOp>(
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
                // For comparison ops, first subtract then compare with zero
                mlir::Value subResult = bbBuilder.create<ttir::TileSubBinaryOp>(
                    loc, /*resultTypes=*/bbArgs.take_back(numOutputs),
                    /*operands=*/bbArgs.take_front(numInputs));
                yield = bbBuilder.create<TileOp>(
                    loc, /*resultTypes=*/bbArgs.take_back(numOutputs),
                    /*operands=*/subResult);
              } else {
                // For regular elementwise ops, create TileOp directly
                yield = bbBuilder.create<TileOp>(
                    loc, /*resultTypes=*/bbArgs.take_back(numOutputs),
                    /*operands=*/bbArgs.take_front(numInputs));
              }

              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
            });

        rewriter.create<ttir::YieldOp>(loc, linalgGeneric->getResults());
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
class TTIRNamedReductionRewriter final
    : public mlir::OpConversionPattern<ConcreteOp>,
      TTIRNamedRewriterCommon {

public:
  TTIRNamedReductionRewriter<ConcreteOp, TileOp>(
      const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
      ttcore::MemorySpace defaultInputMemSpace,
      ttcore::MemorySpace defaultOutputMemSpace,
      const llvm::SmallVector<int64_t> &targetGridShape)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        TTIRNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                                targetGridShape) {}

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

    // minus 1 for the scaler operand
    assert((numOperands - 1) == op->getNumOperands());

    ttcore::GridAttr grid = ttcore::GridAttr::get(ctx, targetSquareGridShape);

    const std::size_t rank = grid.getShape().size();

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, op, numOperands, rank);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, op, rank);

    // Create 'ttir.generic' accepting extended operands.
    auto generic = rewriter.create<ttir::GenericOp>(
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
              tt::ttir::ReduceDimAttr::getMnemonic(),
              tt::ttir::ReduceDimAttr::get(ctx, dimArgAsReduceDim(op, rank)));
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
                  loc, /* resultTypes */ bbArgs.take_back(numOutputs),
                  /* operands */ bbArgs, attributes);
              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
            });

        rewriter.create<ttir::YieldOp>(loc, linalgGeneric->getResults());
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

  static ttir::ReduceDim dimArgAsReduceDim(ConcreteOp op, std::size_t rank) {
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
      return ttir::ReduceDim::C;
    case 2:
      return ttir::ReduceDim::R;
    case 3:
      return ttir::ReduceDim::RC;
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
// Rewrite a MatmulOp into either a TileMatmulOp or TileMatmulBlockOp (selected
// by TileOp template).
namespace {
template <typename TileOp>
class TTIRMatmulRewriter final
    : public mlir::OpConversionPattern<ttir::MatmulOp>,
      TTIRNamedRewriterCommon {

  using ConcreteOp = ttir::MatmulOp;
  static_assert(std::is_same_v<TileOp, ttir::TileMatmulBlockOp> ||
                    std::is_same_v<TileOp, ttir::TileMatmulOp>,
                "Unsupported Matmul TileOp");

public:
  TTIRMatmulRewriter(const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
                     ttcore::MemorySpace defaultInputMemSpace,
                     ttcore::MemorySpace defaultOutputMemSpace,
                     const llvm::SmallVector<int64_t> &targetGridShape)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        TTIRNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                                targetGridShape) {}

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

    ttcore::GridAttr grid = ttcore::GridAttr::get(ctx, targetSquareGridShape);

    const std::size_t rank = grid.getShape().size();

    // TODO(#2591) handle 'transpose_{a,b}' attributes

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, numOperands, rank);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, rank);

    // Create 'ttir.generic' accepting 'op's operands.
    auto generic = rewriter.create<ttir::GenericOp>(
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

        if constexpr (std::is_same_v<ttir::TileMatmulBlockOp, TileOp>) {
          rewriter.create<TileOp>(loc,
                                  /* resultTypes */ mlir::TypeRange(),
                                  /* operands */ blockArgs);
          // In pure tensor semantics, explicitly yield the output shard.
          rewriter.create<ttir::YieldOp>(loc, blockArgs.take_back(numOutputs));

        } else if constexpr (std::is_same_v<ttir::TileMatmulOp, TileOp>) {

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
                    loc, /* resultTypes */ bbArgs.take_back(tileOpNumOutputs),
                    /* operands */ bbArgs.take_front(tileOpNumInputs));

                bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
              });

          rewriter.create<ttir::YieldOp>(loc, linalgGeneric->getResults());
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
    // TODO(#2592) handle higher ranks, if needed in this pass
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
// Lower PermuteOp into a StreamLayoutOp (to reblock into new tile-level
// shape) + GenericOp (to transpose individual tiles).
namespace {
class TTIRPermuteRewriter final
    : public mlir::OpConversionPattern<ttir::PermuteOp>,
      TTIRNamedRewriterCommon {

  using ConcreteOp = ttir::PermuteOp;

public:
  TTIRPermuteRewriter(const TypeConverter &typeConverter,
                      mlir::MLIRContext *ctx,
                      ttcore::MemorySpace defaultInputMemSpace,
                      ttcore::MemorySpace defaultOutputMemSpace,
                      const llvm::SmallVector<int64_t> &targetGridShape)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        TTIRNamedRewriterCommon(defaultInputMemSpace, defaultOutputMemSpace,
                                targetGridShape) {}

  LogicalResult
  matchAndRewrite(ttir::PermuteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto permutation = op.getPermutation();

    const int64_t permuteSize = static_cast<int64_t>(permutation.size());
    // Tranpose pattern on inner dims.
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
        inputLayout.getMemorySpace(), composedMap);

    auto viewType = mlir::RankedTensorType::get(
        resultShape, inputTensorType.getElementType(), resultLayout);

    // For inner permute, we need as streamLayout to do reblocking.
    auto storage = rewriter.create<ttir::EmptyOp>(loc, viewType);
    auto stream = rewriter.create<ttir::StreamLayoutOp>(loc, viewType,
                                                        inputs[0], storage);
    inputs[0] = stream.getResult();

    // For inner permute, we alse need a GenericOp to transpose each individual
    // tile.
    auto generic = rewriter.create<ttir::GenericOp>(
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
                mlir::Value yield = bbBuilder.create<ttir::TileTransposeOp>(
                    bbLoc, bbArgs.take_back(1).getTypes(),
                    bbArgs.take_front(1));
                bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
              });

          builder.create<ttir::YieldOp>(bodyLoc, linalgGeneric->getResults());
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
} // namespace mlir::tt

namespace mlir::tt {
void populateTTIRToTTIRGenericPatterns(
    MLIRContext *ctx, RewritePatternSet &patterns, TypeConverter &typeConverter,
    ttcore::MemorySpace defaultInputMemSpace,
    ttcore::MemorySpace defaultOutputMemSpace,
    const llvm::SmallVector<int64_t> &targetGridShape) {
  // clang-format off
  patterns.add<

    // Elementwise.
    TTIRNamedElementwiseRewriter<ttir::AbsOp,          ttir::TileAbsOp>,
    TTIRNamedElementwiseRewriter<ttir::AddOp,          ttir::TileAddOp>,
    TTIRNamedElementwiseRewriter<ttir::CeilOp,         ttir::TileCeilOp>,
    TTIRNamedElementwiseRewriter<ttir::CosOp,          ttir::TileCosOp>,
    TTIRNamedElementwiseRewriter<ttir::DivOp,          ttir::TileDivOp>,
    TTIRNamedElementwiseRewriter<ttir::ExpOp,          ttir::TileExpOp>,
    TTIRNamedElementwiseRewriter<ttir::FloorOp,        ttir::TileFloorOp>,
    TTIRNamedElementwiseRewriter<ttir::GeluOp,         ttir::TileGeluOp>,
    TTIRNamedElementwiseRewriter<ttir::LogOp,          ttir::TileLogOp>,
    TTIRNamedElementwiseRewriter<ttir::LogicalNotOp,   ttir::TileLogicalNotOp>,
    TTIRNamedElementwiseRewriter<ttir::MultiplyOp,     ttir::TileMulOp>,
    TTIRNamedElementwiseRewriter<ttir::MaximumOp,      ttir::TileMaximumOp>,
    TTIRNamedElementwiseRewriter<ttir::NegOp,          ttir::TileNegativeOp>,
    TTIRNamedElementwiseRewriter<ttir::PowOp,          ttir::TilePowOp>,
    TTIRNamedElementwiseRewriter<ttir::ReciprocalOp,   ttir::TileRecipOp>,
    TTIRNamedElementwiseRewriter<ttir::RsqrtOp,        ttir::TileRsqrtOp>,
    TTIRNamedElementwiseRewriter<ttir::SigmoidOp,      ttir::TileSigmoidOp>,
    TTIRNamedElementwiseRewriter<ttir::SinOp,          ttir::TileSinOp>,
    TTIRNamedElementwiseRewriter<ttir::SqrtOp,         ttir::TileSqrtOp>,
    TTIRNamedElementwiseRewriter<ttir::SubtractOp,     ttir::TileSubOp>,
    TTIRNamedElementwiseRewriter<ttir::TanOp,          ttir::TileTanOp>,

    // Comparison.
    TTIRNamedElementwiseRewriter<ttir::EqualOp,        ttir::TileEqzOp>,
    TTIRNamedElementwiseRewriter<ttir::NotEqualOp,     ttir::TileNezOp>,
    TTIRNamedElementwiseRewriter<ttir::GreaterThanOp,  ttir::TileGtzOp>,
    TTIRNamedElementwiseRewriter<ttir::GreaterEqualOp, ttir::TileGezOp>,
    TTIRNamedElementwiseRewriter<ttir::LessThanOp,     ttir::TileLtzOp>,
    TTIRNamedElementwiseRewriter<ttir::LessEqualOp,    ttir::TileLezOp>,

    // Reduction.
    TTIRNamedReductionRewriter<ttir::MaxOp,            ttir::TileReduceMaxOp>,
    TTIRNamedReductionRewriter<ttir::SumOp,            ttir::TileReduceSumOp>,
    // Data movement.
    TTIRNamedElementwiseRewriter<ttir::TypecastOp,     ttir::TileTypecastOp>,
    // Permute (handles tranpose ops, since they're canonicalized into permutes).
    TTIRPermuteRewriter
  >(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, targetGridShape);


  // Matmul.
  patterns.add<TTIRMatmulRewriter<ttir::TileMatmulOp>>(typeConverter, ctx, defaultInputMemSpace, defaultOutputMemSpace, targetGridShape);
  // clang-format on
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
