// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRGeneric/TTIRToTTIRGeneric.h"

#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <array>

// ----------------------------------------------------------------------------
namespace mlir::tt {

using namespace llvm;

namespace {
class TTIRNamedRewriterCommon {
protected:
  using base = TTIRNamedRewriterCommon;

  // Common need to navigate DPS (<inputs>;<inits>) operand split:
  // note that this requires only 'getDpsInits()' to be available.
  template <typename ConcreteOp>
  static std::array<mlir::ValueRange, 2> splitDpsSignature(ConcreteOp op) {
    // 'DPS inits' (for tensor semantics, tied 1:1 with 'DPS results').
    mlir::ValueRange inits = op.getDpsInits();

    assert(inits.size() <= op->getNumOperands());
    mlir::ValueRange inputs =
        op->getOperands().take_front(op->getNumOperands() - inits.size());

    return {inputs, inits};
  }

  // Input assumptions shared across ops in this pass.
  template <typename ConcreteOp>
  static void checkPreconditions(ConcreteOp op) {
    for (mlir::Type t : op->getOperandTypes()) {
      mlir::RankedTensorType tensorType = mlir::cast<mlir::RankedTensorType>(t);
      MetalLayoutAttr layout =
          mlir::dyn_cast<MetalLayoutAttr>(tensorType.getEncoding());
      assert(layout && "expected tt.metal_layout encoding");
      assert(layout.isTiled() && "expected tiled buffer element type");
      assert((layout.getGrid().getShape() == expectedInputGridShape()) &&
             "unexpected grid shape");
    }
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
                           SmallVector<mlir::Attribute> const &iterators) {
    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    auto reduction = tt::IteratorTypeAttr::get(builder.getContext(),
                                               tt::IteratorType::Reduction);

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

  static constexpr mlir::ArrayRef<int64_t> expectedInputGridShape() {
    return s_expectedInputGridShape;
  }

  static constexpr std::array<int64_t, 2> s_expectedInputGridShape{1, 1};

}; // end of class
} // namespace
// ............................................................................
// Rewrite elementwise ops by emitting a matching tile version of the op
// into a ttir.generic/linang.generic nest.
namespace {
template <typename ConcreteOp, typename TileOp>
class TTIRNamedElementwiseRewriter final
    : public mlir::OpConversionPattern<ConcreteOp>,
      TTIRNamedRewriterCommon {

public:
  using mlir::OpConversionPattern<ConcreteOp>::OpConversionPattern;

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [inputs, outputs] = splitDpsSignature(op);

    std::size_t const numInputs = inputs.size();
    std::size_t const numOutputs = outputs.size();
    std::size_t const numOperands = (numInputs + numOutputs);

    assert(numOperands == op->getNumOperands());

    tt::GridAttr grid = tt::GridAttr::get(ctx, expectedInputGridShape());

    std::size_t const rank = grid.getShape().size();

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, numOperands, rank);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, rank);

    // Create 'ttir.generic' accepting 'op's operands.
    auto generic = rewriter.create<ttir::GenericOp>(
        loc, mlir::TypeRange(outputs), inputs, outputs, grid,
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes), /* regionsCount */ 1);

    // Create one bb in 'generic''s region and set its arguments.
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block'.
      {
        llvm::for_each(op->getOperandTypes(), [&](Type t) {
          mlir::RankedTensorType tensorType = mlir::cast<RankedTensorType>(t);
          tt::MetalLayoutAttr layout =
              mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
          block->addArgument(layout.getMemref(), loc);
        });
        auto blockArgs = block->getArguments();

        // Create 'linalg.generic' accepting 'blockArgs'.

        SmallVector<mlir::AffineMap> linalgIndexingMaps =
            getAffineMapsArray(rewriter, numOperands, rank);
        SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
            iteratorTypeTTIRToLinalg(rewriter, iteratorTypes);

        rewriter.create<mlir::linalg::GenericOp>(
            loc, /* inputs */ blockArgs.take_front(numInputs),
            /* outputs */ blockArgs.take_back(numOutputs), linalgIndexingMaps,
            linalgIteratorTypes,
            [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                mlir::ValueRange bbArgs) {
              mlir::Value yield = bbBuilder.create<TileOp>(
                  loc, /* resultTypes */ bbArgs.take_back(numOutputs),
                  /* operands */ bbArgs.take_front(numInputs));
              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
            });
      }
    }
    rewriter.finalizeOpModification(generic);

    rewriter.replaceOp(op, generic);
    return llvm::success();
  }

  static void checkPreconditions(ConcreteOp op) {
    base::checkPreconditions(op);

    mlir::ArrayRef<int64_t> shape;
    for (mlir::Type t : op->getOperandTypes()) {
      mlir::RankedTensorType tensorType = mlir::cast<mlir::RankedTensorType>(t);
      tt::MetalLayoutAttr layout =
          mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());

      // For elementwise ops, require identical operand shapes for now (no
      // broadcasting, etc).
      if (shape.empty()) {
        shape = layout.getMemref().getShape();
      } else {
        assert((layout.getMemref().getShape() == shape) &&
               "expected identical shard shapes");
      }
    }
  }

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, std::size_t arity,
                     std::size_t rank) {
    return getIdentityAffineMapsArray(builder, arity, rank);
  }

  static SmallVector<mlir::Attribute>
  getIteratorTypesArray(mlir::OpBuilder &builder, std::size_t rank) {
    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    return SmallVector<mlir::Attribute>(rank, parallel);
  }

}; // end of class
} // namespace
// ............................................................................
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
  using mlir::OpConversionPattern<ConcreteOp>::OpConversionPattern;

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [inputs, outputs] = splitDpsSignature(op);

    std::size_t const numOutputs = outputs.size();
    assert(inputs.size() + numOutputs == op->getNumOperands());

    tt::GridAttr grid = tt::GridAttr::get(ctx, expectedInputGridShape());

    std::size_t const rank = grid.getShape().size();

    // Extend the operand block with a single-tile 'weight'/'mask' operand.
    // Our generic signature becomes (<inputs>; scaler; <results>).

    static constexpr bool usingScaler = true;

    std::size_t const numInputs = inputs.size() + usingScaler;
    std::size_t const numOperands = (numInputs + numOutputs);

    SmallVector<mlir::Value> newInputs(inputs.begin(), inputs.end());
    if (usingScaler) {
      newInputs.emplace_back(createScaler(
          rewriter, loc,
          mlir::cast<mlir::RankedTensorType>(inputs.front().getType())));
    }

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, op, numOperands, rank, usingScaler);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, op, rank);

    // Create 'ttir.generic' accepting extended operands.
    auto generic = rewriter.create<ttir::GenericOp>(
        loc, mlir::TypeRange(outputs), newInputs, outputs, grid,
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes), /* regionsCount */ 1);

    // Create one bb in 'generic''s region and set its arguments.
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block'.
      {
        llvm::for_each(mlir::TypeRange(newInputs), [&](Type t) {
          mlir::RankedTensorType tensorType =
              mlir::cast<mlir::RankedTensorType>(t);
          tt::MetalLayoutAttr layout =
              mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
          block->addArgument(layout.getMemref(), loc);
        });
        llvm::for_each(outputs.getTypes(), [&](Type t) {
          mlir::RankedTensorType tensorType =
              mlir::cast<mlir::RankedTensorType>(t);
          tt::MetalLayoutAttr layout =
              mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
          block->addArgument(layout.getMemref(), loc);
        });
        auto blockArgs = block->getArguments();
        assert(blockArgs.size() == numOperands);

        // Create 'linalg.generic' accepting 'blockArgs'.

        SmallVector<mlir::AffineMap> linalgIndexingMaps =
            getAffineMapsArray(rewriter, op, numOperands, rank, usingScaler);
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

        rewriter.create<mlir::linalg::GenericOp>(
            loc, /* inputs */ blockArgs.take_front(numInputs),
            /* outputs */ blockArgs.take_back(numOutputs), linalgIndexingMaps,
            linalgIteratorTypes,
            [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                mlir::ValueRange bbArgs) {
              mlir::Value yield = bbBuilder.create<TileOp>(
                  loc, /* resultTypes */ bbArgs.take_back(numOutputs),
                  /* operands */ bbArgs.take_front(numInputs), attributes);
              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
            });
      }
    }
    rewriter.finalizeOpModification(generic);

    rewriter.replaceOp(op, generic);
    return llvm::success();
  }

  static void checkPreconditions(ConcreteOp op) {
    base::checkPreconditions(op);

    // For reductions, require 'dim_arg' and 'keep_dim'=true for now.
    assert(op.getDimArg() && "expected dim_arg attribute to be set");
    assert(op.getKeepDimAttr().getValue() && "expected default keep_dim=true");
  }

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, ConcreteOp op, std::size_t arity,
                     std::size_t rank, bool usingScaler) {
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
    SmallVector<mlir::AffineMap> maps(arity - 1 - usingScaler,
                                      builder.getMultiDimIdentityMap(rank));
    if (usingScaler) {
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

    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    auto reduction = tt::IteratorTypeAttr::get(builder.getContext(),
                                               tt::IteratorType::Reduction);

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
                                  mlir::RankedTensorType tensorOperandType) {
    mlir::MLIRContext *ctx = builder.getContext();

    mlir::Type elementType = tensorOperandType.getElementType();
    tt::MetalLayoutAttr layout =
        mlir::cast<tt::MetalLayoutAttr>(tensorOperandType.getEncoding());

    tt::TileType tileType = tt::TileType::get(ctx, elementType);
    SmallVector<int64_t> singleTile{1, 1};
    mlir::RankedTensorType scalerType = RankedTensorType::get(
        tileType.getScalarShape(singleTile), elementType,
        layout.withElementType(ctx, tileType).withShardShape(ctx, singleTile));

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
      return ttir::ReduceDim::R;
    case 2:
      return ttir::ReduceDim::C;
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

}; // end of class
} // namespace
// ............................................................................
namespace {
// At this time, matmul ops are rewritten into a ttir.generic without a nested
// linagl.generic because we use metal counterpart op that is already "blocked".
class TTIRMatmulRewriter final
    : public mlir::OpConversionPattern<ttir::MatmulOp>,
      TTIRNamedRewriterCommon {

  using ConcreteOp = ttir::MatmulOp;
  using TileOp = ttir::TileMatmulBlockOp;

public:
  using mlir::OpConversionPattern<ConcreteOp>::OpConversionPattern;

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [inputs, outputs] = splitDpsSignature(op);

    std::size_t const numInputs = inputs.size();
    std::size_t const numOutputs = outputs.size();
    std::size_t const numOperands = (numInputs + numOutputs);

    assert(numOperands == op->getNumOperands());

    tt::GridAttr grid = tt::GridAttr::get(ctx, expectedInputGridShape());

    std::size_t const rank = grid.getShape().size();

    // TODO(#2591) handle 'transpose_{a,b}' attributes

    SmallVector<mlir::AffineMap> indexingMaps =
        getAffineMapsArray(rewriter, numOperands, rank);
    SmallVector<mlir::Attribute> iteratorTypes =
        getIteratorTypesArray(rewriter, rank);

    // Create 'ttir.generic' accepting 'op's operands.
    auto generic = rewriter.create<ttir::GenericOp>(
        loc, mlir::TypeRange(outputs), inputs, outputs, grid,
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorTypes), /* regionsCount */ 1);

    // Create one bb in 'generic''s region and set its arguments.
    rewriter.startOpModification(generic);
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block'.
      {
        llvm::for_each(op->getOperandTypes(), [&](Type t) {
          mlir::RankedTensorType tensorType = mlir::cast<RankedTensorType>(t);
          tt::MetalLayoutAttr layout =
              mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
          block->addArgument(layout.getMemref(), loc);
        });
        auto blockArgs = block->getArguments();

        // Delegate next level of nesting to a "block" op.

        rewriter.create<TileOp>(loc,
                                /* resultTypes */ mlir::TypeRange(),
                                /* operands */ blockArgs);
      }
    }
    rewriter.finalizeOpModification(generic);

    rewriter.replaceOp(op, generic);
    return llvm::success();
  }

  static void checkPreconditions(ConcreteOp op) {
    base::checkPreconditions(op);

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
    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    auto reduction = tt::IteratorTypeAttr::get(builder.getContext(),
                                               tt::IteratorType::Reduction);
    return SmallVector<mlir::Attribute>{parallel, parallel, reduction};
  }

  static mlir::AffineMap makeAffineMap(mlir::MLIRContext *ctx,
                                       std::array<unsigned, 2> targets) {
    return mlir::AffineMap::getMultiDimMapWithTargets(3, targets, ctx);
  }

}; // end of class
} // namespace
// ............................................................................

void populateTTIRToTTIRGenericPatterns(MLIRContext *ctx,
                                       RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  // clang-format off
  patterns.add<
    // Elementwise.
    TTIRNamedElementwiseRewriter<ttir::AddOp,       ttir::TileAddOp>,
    TTIRNamedElementwiseRewriter<ttir::MultiplyOp,  ttir::TileMulOp>,
    TTIRNamedElementwiseRewriter<ttir::ExpOp,       ttir::TileExpOp>,
    TTIRNamedElementwiseRewriter<ttir::LogOp,       ttir::TileLogOp>,
    // Reductions.
    TTIRNamedReductionRewriter<ttir::SumOp,         ttir::TileReduceSumOp>,
    TTIRNamedReductionRewriter<ttir::MaxOp,         ttir::TileReduceMaxOp>,
    // Matmul.
    TTIRMatmulRewriter
  >(typeConverter, ctx);
  // clang-format on
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
