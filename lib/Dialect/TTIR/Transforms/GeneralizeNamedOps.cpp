// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h" // OpTrait::named_op_group
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <array>

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRGENERALIZENAMEDOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

using namespace llvm;

class TTIRNamedRewriterCommon {
protected:
  using base = TTIRNamedRewriterCommon;

  // Common need to navigate DPS (<inputs>;<inits>) operand split:
  // note that this requires only 'getDpsInits()' to be available.
  template <typename ConcreteOp>
  static std::array<mlir::ValueRange, 2> signatureSplit(ConcreteOp op) {
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

  static mlir::ArrayAttr getIdentityAffineMapsAttr(mlir::OpBuilder &builder,
                                                   std::size_t arity,
                                                   std::size_t rank) {
    return builder.getAffineMapArrayAttr(
        getIdentityAffineMapsArray(builder, arity, rank));
  }

  static constexpr mlir::ArrayRef<int64_t> expectedInputGridShape() {
    return s_expectedInputGridShape;
  }

  static constexpr std::array<int64_t, 2> s_expectedInputGridShape{1, 1};

}; // end of class
// ............................................................................
// Rewrite elementwise ops by emitting a matching tile version of the op
// into a ttir.generic/linang.generic nest.
template <typename ConcreteOp, typename TileOp>
class TTIRNamedElementwiseRewriter final
    : public mlir::OpRewritePattern<ConcreteOp>,
      TTIRNamedRewriterCommon {

public:
  using mlir::OpRewritePattern<ConcreteOp>::OpRewritePattern;

private:
  LogicalResult matchAndRewrite(ConcreteOp op,
                                mlir::PatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [inputs, inits] = signatureSplit(op);

    std::size_t const numInputs = inputs.size();
    std::size_t const numInits = inits.size();
    std::size_t const numOperands = (numInputs + numInits);

    assert(numOperands == op->getNumOperands());

    tt::GridAttr grid = tt::GridAttr::get(ctx, expectedInputGridShape());

    std::size_t const rank = grid.getShape().size();

    mlir::ArrayAttr indexingMaps =
        getAffineMapsAttr(rewriter, numOperands, rank);
    mlir::ArrayAttr ttirIteratorTypes =
        getTTIRIteratorTypesAttr(rewriter, rank);

    // Create 'ttir.generic' accepting 'op's operands.
    auto generic = rewriter.create<GenericOp>(
        loc, mlir::TypeRange(inits), inputs, inits, grid, indexingMaps,
        ttirIteratorTypes, /* regionsCount */ 1);

    // Create one bb in 'generic''s region and set its arguments.
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
            getLinalgIteratorTypesArray(rewriter, rank);

        rewriter.create<mlir::linalg::GenericOp>(
            loc, /* inputs */ blockArgs.take_front(numInputs),
            /* outputs */ blockArgs.take_back(numInits), linalgIndexingMaps,
            linalgIteratorTypes,
            [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                mlir::ValueRange bbArgs) {
              mlir::Value yield = bbBuilder.create<TileOp>(
                  loc, /* resultTypes */ bbArgs.take_back(numInits),
                  /* operands */ bbArgs.take_front(numInputs));
              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
            });
      }
    }

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

  // Indexing space navigation.
  //
  //  (1) for ttir.generic:
  //    - getAffineMapsAttr()
  //    - getTTIRIteratorTypesAttr()
  //  (2) for linalg.generic:
  //    - getAffineMapsArray()
  //    - getLinalgIteratorTypesArray()

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, std::size_t arity,
                     std::size_t rank) {
    return getIdentityAffineMapsArray(builder, arity, rank);
  }

  static ArrayAttr getAffineMapsAttr(mlir::OpBuilder &builder,
                                     std::size_t arity, std::size_t rank) {
    return getIdentityAffineMapsAttr(builder, arity, rank);
  }

  static ArrayAttr getTTIRIteratorTypesAttr(mlir::OpBuilder &builder,
                                            std::size_t rank) {
    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    return builder.getArrayAttr(SmallVector<mlir::Attribute>(rank, parallel));
  }

  static SmallVector<mlir::utils::IteratorType>
  getLinalgIteratorTypesArray(mlir::OpBuilder &builder, std::size_t rank) {
    return SmallVector<mlir::utils::IteratorType>(
        rank, mlir::utils::IteratorType::parallel);
  }

}; // end of class
// ............................................................................
// Rewriting reduction ops is similar to the elementwise group except for
// ops whose tiled counterparts require a scaler operand ('weights', etc).
// This rewriter will emit a single tile scaler operand that will be
// broadcast across the lhs indexing space.
template <typename ConcreteOp, typename TileOp>
class TTIRNamedReductionRewriter final
    : public mlir::OpRewritePattern<ConcreteOp>,
      TTIRNamedRewriterCommon {

public:
  using mlir::OpRewritePattern<ConcreteOp>::OpRewritePattern;

private:
  LogicalResult matchAndRewrite(ConcreteOp op,
                                mlir::PatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [inputs, inits] = signatureSplit(op);

    std::size_t const numInits = inits.size();
    assert(inputs.size() + numInits == op->getNumOperands());

    tt::GridAttr grid = tt::GridAttr::get(ctx, expectedInputGridShape());

    std::size_t const rank = grid.getShape().size();

    // Extend the operand block with a single-tile 'weight'/'mask' operand.
    // Our generic signature becomes (<inputs>; scaler; <inits>).

    static constexpr bool usingScaler = true;

    std::size_t const numInputs = inputs.size() + usingScaler;
    std::size_t const numOperands = (numInputs + numInits);

    SmallVector<mlir::Value> inputsWithScaler(inputs.begin(), inputs.end());
    inputsWithScaler.emplace_back(createScaler(
        rewriter, loc,
        mlir::cast<mlir::RankedTensorType>(inputs.front().getType())));

    mlir::ArrayAttr indexingMaps =
        getAffineMapsAttr(rewriter, op, numOperands, rank, usingScaler);
    mlir::ArrayAttr ttirIteratorTypes =
        getTTIRIteratorTypesAttr(rewriter, op, rank);

    // Create 'ttir.generic' accepting extended operands.
    auto generic = rewriter.create<GenericOp>(
        loc, mlir::TypeRange(inits), inputsWithScaler, inits, grid,
        indexingMaps, ttirIteratorTypes, /* regionsCount */ 1);

    // Create one bb in 'generic''s region and set its arguments.
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // Populate 'block.
      {
        llvm::for_each(mlir::TypeRange(inputsWithScaler), [&](Type t) {
          mlir::RankedTensorType tensorType =
              mlir::cast<mlir::RankedTensorType>(t);
          tt::MetalLayoutAttr layout =
              mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
          block->addArgument(layout.getMemref(), loc);
        });
        llvm::for_each(inits.getTypes(), [&](Type t) {
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
            getLinalgIteratorTypesArray(rewriter, op, rank);

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
            /* outputs */ blockArgs.take_back(numInits), linalgIndexingMaps,
            linalgIteratorTypes,
            [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc,
                mlir::ValueRange bbArgs) {
              mlir::Value yield = bbBuilder.create<TileOp>(
                  loc, /* resultTypes */ bbArgs.take_back(numInits),
                  /* operands */ bbArgs.take_front(numInputs), attributes);
              bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
            });
      }
    }

    rewriter.replaceOp(op, generic);
    return llvm::success();
  }

  static void checkPreconditions(ConcreteOp op) {
    base::checkPreconditions(op);

    // For reductions, require 'dim_arg' and 'keep_dim'=true for now.
    assert(op.getDimArg() && "expected dim_arg attribute to be set");
    assert(op.getKeepDimAttr().getValue() && "expected default keep_dim=true");
  }

  // Indexing space navigation.
  //
  //  (1) for ttir.generic:
  //    - getAffineMapsAttr()
  //    - getTTIRIteratorTypesAttr()
  //  (2) for linalg.generic:
  //    - getAffineMapsArray()
  //    - getLinalgIteratorTypesArray()

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
      maps.emplace_back(mlir::AffineMap::get(/* dimCopunt */ rank,
                                             /* symbolCount */ 0, zeros,
                                             builder.getContext()));
    }
    maps.emplace_back(accumulator.getAffineMap());

    return maps;
  }

  static mlir::ArrayAttr getAffineMapsAttr(mlir::OpBuilder &builder,
                                           ConcreteOp op, std::size_t arity,
                                           std::size_t rank, bool usingScaler) {
    return builder.getAffineMapArrayAttr(
        getAffineMapsArray(builder, op, arity, rank, usingScaler));
  }

  static mlir::ArrayAttr getTTIRIteratorTypesAttr(mlir::OpBuilder &builder,
                                                  ConcreteOp op,
                                                  std::size_t rank) {
    SmallVector<mlir::Attribute> iterators(iteratorTypeLinalgToTTIR(
        builder, getLinalgIteratorTypesArray(builder, op, rank)));
    return builder.getArrayAttr(iterators);
  }

  static SmallVector<mlir::utils::IteratorType>
  getLinalgIteratorTypesArray(mlir::OpBuilder &builder, ConcreteOp op,
                              std::size_t rank) {
    mlir::ArrayAttr dimArg = getDimArg(op);

    SmallVector<mlir::utils::IteratorType> iterators(
        rank, mlir::utils::IteratorType::parallel);
    forAllDims(rank, dimArg, [&](std::size_t index, bool dropped) {
      if (dropped) {
        iterators[index] = mlir::utils::IteratorType::reduction;
      }
    });
    return iterators;
  }

  // Convert from linalg enum to equivalent ttir enum.
  static SmallVector<mlir::Attribute> iteratorTypeLinalgToTTIR(
      mlir::OpBuilder &builder,
      SmallVector<mlir::utils::IteratorType> const &iterators) {
    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    auto reduction = tt::IteratorTypeAttr::get(builder.getContext(),
                                               tt::IteratorType::Reduction);
    SmallVector<mlir::Attribute> r;
    for (auto iterator : iterators) {
      switch (iterator) {
      case mlir::utils::IteratorType::parallel: {
        r.emplace_back(parallel);
      } break;
      case mlir::utils::IteratorType::reduction: {
        r.push_back(reduction);
      } break;
      }
    }
    return r;
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

    return builder.create<ConstantOp>(loc, scalerType, scalerValue);
  }

  static ReduceDim dimArgAsReduceDim(ConcreteOp op, std::size_t rank) {
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
      return ReduceDim::R;
    case 2:
      return ReduceDim::C;
    case 3:
      return ReduceDim::RC;
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
// ............................................................................
// At this time, matmul ops are rewritten into a ttir.generic without a nested
// linagl.generic because we use metal counterpart op that is already "blocked".
class TTIRMatmulRewriter final : public mlir::OpRewritePattern<MatmulOp>,
                                 TTIRNamedRewriterCommon {

  using ConcreteOp = MatmulOp;
  using TileOp = TileMatmulBlockOp;

public:
  using mlir::OpRewritePattern<ConcreteOp>::OpRewritePattern;

private:
  LogicalResult matchAndRewrite(ConcreteOp op,
                                mlir::PatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [inputs, inits] = signatureSplit(op);

    std::size_t const numInputs = inputs.size();
    std::size_t const numInits = inits.size();
    std::size_t const numOperands = (numInputs + numInits);

    assert(numOperands == op->getNumOperands());

    tt::GridAttr grid = tt::GridAttr::get(ctx, expectedInputGridShape());

    std::size_t const rank = grid.getShape().size();

    // TODO(#2591) handle 'transpose_{a,b}' attributes

    mlir::ArrayAttr indexingMaps =
        getAffineMapsAttr(rewriter, numOperands, rank);
    mlir::ArrayAttr ttirIteratorTypes =
        getTTIRIteratorTypesAttr(rewriter, rank);

    // Create 'ttir.generic' accepting 'op's operands.
    auto generic = rewriter.create<GenericOp>(
        loc, mlir::TypeRange(inits), inputs, inits, grid, indexingMaps,
        ttirIteratorTypes, /* regionsCount */ 1);

    // Create one bb in 'generic''s region and set its arguments.
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

        // Delegate next level of nesting to a "block", not tile, op.

        rewriter.create<TileOp>(loc,
                                /* resultTypes */ mlir::TypeRange(),
                                /* operands */ blockArgs);
      }
    }

    rewriter.replaceOp(op, generic);
    return llvm::success();
  }

  static void checkPreconditions(ConcreteOp op) {
    base::checkPreconditions(op);

    assert((!op.getTransposeA() && !op.getTransposeB()) &&
           "TODO(#2591) expected no transpose attributes");
  }

  static mlir::ArrayAttr getAffineMapsAttr(mlir::OpBuilder &builder,
                                           std::size_t arity,
                                           std::size_t rank) {
    assert(arity == 3 && "expected 3 operands");
    // TODO(#2592) handle higher ranks, if needed in this pass
    assert(rank == 2 && "expected a rank 2 operation");
    mlir::MLIRContext *ctx = builder.getContext();
    return builder.getAffineMapArrayAttr(SmallVector<mlir::AffineMap>{
        makeAffineMap(ctx, {0, 2}), makeAffineMap(ctx, {2, 1}),
        makeAffineMap(ctx, {0, 1})});
  }

  static mlir::ArrayAttr getTTIRIteratorTypesAttr(mlir::OpBuilder &builder,
                                                  std::size_t rank) {
    assert(rank == 2 && "expected a rank 2 operation");
    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    auto reduction = tt::IteratorTypeAttr::get(builder.getContext(),
                                               tt::IteratorType::Reduction);
    return builder.getArrayAttr(
        SmallVector<mlir::Attribute>{parallel, parallel, reduction});
  }

  static mlir::AffineMap makeAffineMap(mlir::MLIRContext *ctx,
                                       std::array<unsigned, 2> targets) {
    return mlir::AffineMap::getMultiDimMapWithTargets(3, targets, ctx);
  }

}; // end of class

} // namespace
// ............................................................................

struct TTIRGeneralizeNamedOps final
    : impl::TTIRGeneralizeNamedOpsBase<TTIRGeneralizeNamedOps> {

  void runOnOperation() final {
    auto &ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    // clang-format off
    {
      patterns.add<
        TTIRNamedElementwiseRewriter<AddOp,       TileAddOp>,
        TTIRNamedElementwiseRewriter<MultiplyOp,  TileMulOp>,
        TTIRNamedElementwiseRewriter<ExpOp,       TileExpOp>,
        TTIRNamedElementwiseRewriter<LogOp,       TileLogOp>,

        TTIRNamedReductionRewriter<SumOp,         TileReduceSumOp>,
        TTIRNamedReductionRewriter<MaxOp,         TileReduceMaxOp>,

        TTIRMatmulRewriter
      >(&ctx);
    }
    // clang-format on
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }

}; // end of class

#undef GEN_PASS_DEF_TTIRGENERALIZENAMEDOPS

} // namespace mlir::tt::ttir
// ----------------------------------------------------------------------------
