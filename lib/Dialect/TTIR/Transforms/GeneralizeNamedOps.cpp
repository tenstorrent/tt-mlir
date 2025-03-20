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

#include <algorithm>
#include <array>

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

namespace {

using namespace llvm;

class TTIRNamedRewriterCommon {
protected:
  using base = TTIRNamedRewriterCommon;

  // common need to navigate DPS (<inputs>;<inits>) operand split:
  // note that this requires only 'getDpsInits()' to be available
  template <typename ConcreteOp>
  static std::array<mlir::ValueRange, 2> signatureSplit(ConcreteOp op) {
    // 'DPS inits' (for tensor semantics, tied 1:1 with 'DPS results'):
    mlir::ValueRange inits = op.getDpsInits();

    // can now infer 'DPS inputs':
    assert(inits.size() <= op->getNumOperands());
    mlir::ValueRange inputs =
        op->getOperands().take_front(op->getNumOperands() - inits.size());

    return {inputs, inits};
  }

  // input assumptions shared across ops in this pass:

  template <typename ConcreteOp>
  static void checkPreconditions(ConcreteOp op) {
    for (mlir::Type t : op->getOperandTypes()) {
      RankedTensorType tt = mlir::cast<RankedTensorType>(t);
      MetalLayoutAttr layout =
          mlir::dyn_cast<MetalLayoutAttr>(tt.getEncoding());
      assert(layout && "expected tt.metal_layout encoding");
      assert(layout.isTiled() && "expected tiled buffer element type");
      assert((layout.getGrid().getShape() == expectedInputGridShape()) &&
             "unexpected grid shape");
    }
  }

  // convenience getters for identity mappings:

  static SmallVector<mlir::AffineMap>
  getIdentityAffineMapsArray(mlir::OpBuilder &builder, std::size_t arity,
                             std::size_t rank) {
    return {arity, builder.getMultiDimIdentityMap(rank)};
  }

  static ArrayAttr getIdentityAffineMapsAttr(mlir::OpBuilder &builder,
                                             std::size_t arity,
                                             std::size_t rank) {
    return builder.getAffineMapArrayAttr(
        getIdentityAffineMapsArray(builder, arity, rank));
  }

  // for now, assume input is laid out for a 1x1 grid:

  static constexpr ArrayRef<int64_t> expectedInputGridShape() {
    return s_expectedInputGridShape;
  }

  static constexpr std::array<int64_t, 2> s_expectedInputGridShape{1, 1};

}; // end of class
// ............................................................................

template <typename ConcreteOp, typename TileOp>
class TTIRNamedElementwiseRewriter final
    : public mlir::OpRewritePattern<ConcreteOp>,
      TTIRNamedRewriterCommon {

public:
  using mlir::OpRewritePattern<ConcreteOp>::OpRewritePattern; // inherit
                                                              // constructors
private:
  LogicalResult matchAndRewrite(ConcreteOp op,
                                mlir::PatternRewriter &rewriter) const final {
    checkPreconditions(op);

    MLIRContext *ctx = rewriter.getContext();
    auto loc = op->getLoc();

    auto [inputs, inits] = signatureSplit(op);

    std::size_t const numInputs = inputs.size();
    std::size_t const numInits = inits.size();
    std::size_t const numOperands = (numInputs + numInits);

    assert(numOperands == op->getNumOperands());

    // core grid: for now, propagate the input default
    GridAttr grid = GridAttr::get(ctx, expectedInputGridShape());

    auto const rank = grid.getShape().size();

    ArrayAttr indexingMaps = getAffineMapsAttr(rewriter, numOperands, rank);
    ArrayAttr ttirIteratorTypes = getTTIRIteratorTypesAttr(rewriter, rank);

    // create 'ttir.generic' accepting 'op's operands:
    auto generic = rewriter.create<GenericOp>(
        loc, mlir::TypeRange(inits), inputs, inits, grid, indexingMaps,
        ttirIteratorTypes, /* regionsCount */ 1);

    // create one bb in 'generic''s region and set its arguments:
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // populate 'block':
      {
        std::for_each(op->getOperandTypes().begin(),
                      op->getOperandTypes().end(), [&](Type t) {
                        RankedTensorType tt = mlir::cast<RankedTensorType>(t);
                        MetalLayoutAttr layout =
                            mlir::cast<MetalLayoutAttr>(tt.getEncoding());
                        block->addArgument(layout.getMemref(), loc);
                      });
        auto blockArgs = block->getArguments();

        // create 'linalg.generic' accepting 'blockArgs':

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
              Value yield = bbBuilder.create<TileOp>(
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

    ArrayRef<int64_t> shape;
    for (mlir::Type t : op->getOperandTypes()) {
      RankedTensorType tt = mlir::cast<RankedTensorType>(t);
      MetalLayoutAttr layout = mlir::cast<MetalLayoutAttr>(tt.getEncoding());

      // for elementwise ops, require identical operand shapes for now (no
      // broadcasting, etc):
      if (shape.empty()) {
        shape = layout.getMemref().getShape();
      } else {
        assert((layout.getMemref().getShape() == shape) &&
               "expected identical shard shapes");
      }
    }
  }

  // indexing space navigation:
  //
  //  (1) for ttir.generic:
  //    - getAffineMapsAttr()
  //    - getTTIRIteratorTypesAttr()
  //  (2) for linalg.generic:
  //    - getAffineMapsArray()
  //    - getLinalgIteratorTypesArray()
  //
  // [note that *.generics have 'iterator_type's with similar spellings
  // but nevertheless different tablegen/c++ types]

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
    return {rank, mlir::utils::IteratorType::parallel};
  }

}; // end of class
// ............................................................................

template <typename ConcreteOp, typename TileOp>
class TTIRNamedReductionRewriter final
    : public mlir::OpRewritePattern<ConcreteOp>,
      TTIRNamedRewriterCommon {

public:
  using mlir::OpRewritePattern<ConcreteOp>::OpRewritePattern; // inherit
                                                              // constructors
private:
  LogicalResult matchAndRewrite(ConcreteOp op,
                                mlir::PatternRewriter &rewriter) const final {
    checkPreconditions(op);

    MLIRContext *ctx = rewriter.getContext();
    auto loc = op->getLoc();

    auto [inputs, inits] = signatureSplit(op);

    std::size_t const numInputs = inputs.size();
    std::size_t const numInits = inits.size();
    std::size_t const numOperands = (numInputs + numInits);

    assert(numOperands == op->getNumOperands());

    // core grid: for now, propagate the input default
    GridAttr grid = GridAttr::get(ctx, expectedInputGridShape());

    auto const rank = grid.getShape().size();

    ArrayAttr indexingMaps = getAffineMapsAttr(rewriter, op, numOperands, rank);
    ArrayAttr ttirIteratorTypes = getTTIRIteratorTypesAttr(rewriter, op, rank);

    // create 'ttir.generic' accepting 'op's operands:
    auto generic = rewriter.create<GenericOp>(
        loc, mlir::TypeRange(inits), inputs, inits, grid, indexingMaps,
        ttirIteratorTypes, /* regionsCount */ 1);

    // create one bb in 'generic''s region and set its arguments:
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // populate 'block':
      {
        std::for_each(op->getOperandTypes().begin(),
                      op->getOperandTypes().end(), [&](Type t) {
                        RankedTensorType tt = mlir::cast<RankedTensorType>(t);
                        MetalLayoutAttr layout =
                            mlir::cast<MetalLayoutAttr>(tt.getEncoding());
                        block->addArgument(layout.getMemref(), loc);
                      });
        auto blockArgs = block->getArguments();

        // create 'linalg.generic' accepting 'blockArgs':

        SmallVector<mlir::AffineMap> linalgIndexingMaps =
            getAffineMapsArray(rewriter, op, numOperands, rank);
        SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
            getLinalgIteratorTypesArray(rewriter, op, rank);

        // propagate attributes:

        SmallVector<mlir::NamedAttribute> attributes;
        {
          // propagate 'dim_arg' as 'ReduceDim':
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
              Value yield = bbBuilder.create<TileOp>(
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

    // for reductions, require 'dim_arg' and 'keep_dim'=true:
    assert(op.getDimArg() && "expected dim_arg attribute to be set");
    assert(op.getKeepDimAttr().getValue() && "expected default keep_dim=true");
  }

  // indexing space navigation:
  //
  //  (1) for ttir.generic:
  //    - getAffineMapsAttr()
  //    - getTTIRIteratorTypesAttr()
  //  (2) for linalg.generic:
  //    - getAffineMapsArray()
  //    - getLinalgIteratorTypesArray()
  //
  // [note that *.generics have 'iterator_type's with similar spellings
  // but nevertheless different tablegen/c++ types]

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, ConcreteOp op, std::size_t arity,
                     std::size_t rank) {
    assert(rank > 0);
    mlir::ArrayAttr dimArg = getDimArg(op);

    mlir::MutableAffineMap accumulator(builder.getMultiDimIdentityMap(rank));
    forAllDims(rank, dimArg, [&](std::size_t index, bool dropped) {
      if (dropped) {
        accumulator.setResult(
            index, mlir::getAffineConstantExpr(0, builder.getContext()));
      }
    });
    SmallVector<mlir::AffineMap> maps(rank - 1,
                                      builder.getMultiDimIdentityMap(rank));
    maps.emplace_back(accumulator.getAffineMap());
    return maps;
  }

  static ArrayAttr getAffineMapsAttr(mlir::OpBuilder &builder, ConcreteOp op,
                                     std::size_t arity, std::size_t rank) {
    return builder.getAffineMapArrayAttr(
        getAffineMapsArray(builder, op, arity, rank));
  }

  static ArrayAttr getTTIRIteratorTypesAttr(mlir::OpBuilder &builder,
                                            ConcreteOp op, std::size_t rank) {
    mlir::ArrayAttr dimArg = getDimArg(op);

    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    SmallVector<mlir::Attribute> iterators(rank, parallel);
    forAllDims(rank, dimArg, [&](std::size_t index, bool dropped) {
      if (dropped) {
        iterators[index] = tt::IteratorTypeAttr::get(
            builder.getContext(), tt::IteratorType::Reduction);
      }
    });
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

  static ReduceDim dimArgAsReduceDim(ConcreteOp op, std::size_t rank) {
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
    assert(false && "unexpected dimArg bit pattern");
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

class TTIRMatmulRewriter final : public mlir::OpRewritePattern<MatmulOp>,
                                 TTIRNamedRewriterCommon {
public:
  using mlir::OpRewritePattern<MatmulOp>::OpRewritePattern; // inherit
                                                            // constructors
private:
  LogicalResult matchAndRewrite(MatmulOp op,
                                mlir::PatternRewriter &rewriter) const final {
    checkPreconditions(op);

    MLIRContext *ctx = rewriter.getContext();
    auto loc = op->getLoc();

    auto [inputs, inits] = signatureSplit(op);

    std::size_t const numInputs = inputs.size();
    std::size_t const numInits = inits.size();
    std::size_t const numOperands = (numInputs + numInits);

    assert(numOperands == op->getNumOperands());

    // core grid: for now, propagate the input default
    GridAttr grid = GridAttr::get(ctx, expectedInputGridShape());

    auto const rank = grid.getShape().size();

    // TODO(vlad) handle 'transpose_{a,b}' ?

    ArrayAttr indexingMaps = getAffineMapsAttr(rewriter, numOperands, rank);
    ArrayAttr ttirIteratorTypes = getTTIRIteratorTypesAttr(rewriter, rank);

    // create 'ttir.generic' accepting 'op's operands:
    auto generic = rewriter.create<GenericOp>(
        loc, mlir::TypeRange(inits), inputs, inits, grid, indexingMaps,
        ttirIteratorTypes, /* regionsCount */ 1);

    // create one bb in 'generic''s region and set its arguments:
    {
      mlir::Region &region = generic->getRegions().front();
      mlir::Block *block = rewriter.createBlock(&region);

      // populate 'block':
      {
        std::for_each(op->getOperandTypes().begin(),
                      op->getOperandTypes().end(), [&](Type t) {
                        RankedTensorType tt = mlir::cast<RankedTensorType>(t);
                        MetalLayoutAttr layout =
                            mlir::cast<MetalLayoutAttr>(tt.getEncoding());
                        block->addArgument(layout.getMemref(), loc);
                      });
        auto blockArgs = block->getArguments();

        // delegate next level of nesting to a "blocked" 'TileOp':

        rewriter.create<TileMatmulBlockOp>(loc,
                                           /* resultTypes */ mlir::TypeRange(),
                                           /* operands */ blockArgs);
      }
    }

    rewriter.replaceOp(op, generic);
    return llvm::success();
  }

  static void checkPreconditions(MatmulOp op) {
    base::checkPreconditions(op);

    // [add checks specific to matmul here, e.g. 'transpose_*' attrs]
  }

  static ArrayAttr getAffineMapsAttr(mlir::OpBuilder &builder,
                                     std::size_t arity, std::size_t rank) {
    assert(arity == 3 && "expected 3 operands");
    assert(rank == 2 && "expected a rank 2 operation");
    MLIRContext *ctx = builder.getContext();
    return builder.getAffineMapArrayAttr(SmallVector<mlir::AffineMap>{
        makeAffineMap(ctx, {0, 2}), makeAffineMap(ctx, {2, 1}),
        makeAffineMap(ctx, {0, 1})});
  }

  static ArrayAttr getTTIRIteratorTypesAttr(mlir::OpBuilder &builder,
                                            std::size_t rank) {
    assert(rank == 2 && "expected a rank 2 operation");
    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    auto reduction = tt::IteratorTypeAttr::get(builder.getContext(),
                                               tt::IteratorType::Reduction);
    return builder.getArrayAttr(
        SmallVector<mlir::Attribute>{parallel, parallel, reduction});
  }

  static mlir::AffineMap makeAffineMap(MLIRContext *ctx,
                                       std::array<unsigned, 2> targets) {
    return mlir::AffineMap::getMultiDimMapWithTargets(3, targets, ctx);
  }

}; // end of class

} // namespace
// ............................................................................

#define GEN_PASS_DEF_TTIRGENERALIZENAMEDOPS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

struct TTIRGeneralizeNamedOps final
    : impl::TTIRGeneralizeNamedOpsBase<TTIRGeneralizeNamedOps> {

  void runOnOperation() final {
    auto &ctx = getContext();

    mlir::RewritePatternSet patterns(&ctx);
    // clang-format off
    {
      patterns.add<
        // elementwise:
        TTIRNamedElementwiseRewriter<AddOp,       TileAddOp>,
        TTIRNamedElementwiseRewriter<MultiplyOp,  TileMulOp>,
        TTIRNamedElementwiseRewriter<ExpOp,       TileExpOp>,
        TTIRNamedElementwiseRewriter<LogOp,       TileLogOp>,
        // reduction:
        TTIRNamedReductionRewriter<SumOp,         TileReduceSum1Op>,
        TTIRNamedReductionRewriter<MaxOp,         TileReduceMaxOp>,
        // contraction:
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
