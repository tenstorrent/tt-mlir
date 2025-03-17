// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIRTraits.h" // OpTrait::named_op_group
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

#include <algorithm>
#include <array>

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

namespace {

using namespace llvm;

// Traits for performing iteration space-related calculations on ConcreteOps
// that have been marked with TTIRNamed{Elementwise,Reduction,Contraction}
// tablegen traits. These traits are specialized on 'OpTrait::named_op_group::*'
// tags, with the default template used as a common base. It is expected that
// there is sufficient commonality within each trait group such that these trait
// method can be implemented uniformly across each group.
//
template <typename OpGroup = void>
struct iteration_space_traits {

  // navigate DPS (<inputs>;<inits>) operand split: this requires only
  // 'getDpsInits()' to be available
  template <typename ConcreteOp>
  static std::array<mlir::ValueRange, 2> signatureSplit(ConcreteOp op) {
    // 'DPS inits' (for tensor semantics, tied 1:1 with 'DPS results'):
    mlir::ValueRange inits = op.getDpsInits();

    // 'DPS inputs':
    assert(inits.size() <= op->getNumOperands());
    mlir::ValueRange inputs =
        op->getOperands().take_front(op->getNumOperands() - inits.size());

    return {inputs, inits};
  }

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
};

// Iteration space traits specialized for 'TTIRNamedElementwise' ops.
template <>
struct iteration_space_traits<OpTrait::named_op_group::elementwise>
    : iteration_space_traits<> {

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, mlir::Operation *op,
                     std::size_t arity, std::size_t rank) {
    return getIdentityAffineMapsArray(builder, arity, rank);
  }

  static ArrayAttr getAffineMapsAttr(mlir::OpBuilder &builder,
                                     mlir::Operation *op, std::size_t arity,
                                     std::size_t rank) {
    return getIdentityAffineMapsAttr(builder, arity, rank);
  }

  static ArrayAttr getTTIRIteratorTypesAttr(mlir::OpBuilder &builder,
                                            mlir::Operation *op,
                                            std::size_t rank) {
    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    return builder.getArrayAttr(SmallVector<mlir::Attribute>(rank, parallel));
  }

  static SmallVector<mlir::utils::IteratorType>
  getLinalgIteratorTypesArray(mlir::OpBuilder &builder, mlir::Operation *op,
                              std::size_t rank) {
    return {rank, mlir::utils::IteratorType::parallel};
  }
};

// Iteration space traits specialized for 'TTIRNamedReduction' ops.
template <>
struct iteration_space_traits<OpTrait::named_op_group::reduction>
    : iteration_space_traits<> {

  static SmallVector<mlir::AffineMap>
  getAffineMapsArray(mlir::OpBuilder &builder, mlir::Operation *op,
                     std::size_t arity, std::size_t rank) {
    assert(rank > 0);
    mlir::ArrayAttr dimArg = getDimArg(op);

    mlir::MutableAffineMap accumulator(builder.getMultiDimIdentityMap(rank));
    for_all_dims(rank, dimArg, [&](std::size_t index, bool dropped) {
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

  static ArrayAttr getAffineMapsAttr(mlir::OpBuilder &builder,
                                     mlir::Operation *op, std::size_t arity,
                                     std::size_t rank) {
    return builder.getAffineMapArrayAttr(
        getAffineMapsArray(builder, op, arity, rank));
  }

  static ArrayAttr getTTIRIteratorTypesAttr(mlir::OpBuilder &builder,
                                            mlir::Operation *op,
                                            std::size_t rank) {
    mlir::ArrayAttr dimArg = getDimArg(op);

    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    SmallVector<mlir::Attribute> iterators(rank, parallel);
    for_all_dims(rank, dimArg, [&](std::size_t index, bool dropped) {
      if (dropped) {
        iterators[index] = tt::IteratorTypeAttr::get(
            builder.getContext(), tt::IteratorType::Reduction);
      }
    });
    return builder.getArrayAttr(iterators);
  }

  static SmallVector<mlir::utils::IteratorType>
  getLinalgIteratorTypesArray(mlir::OpBuilder &builder, mlir::Operation *op,
                              std::size_t rank) {
    mlir::ArrayAttr dimArg = getDimArg(op);

    SmallVector<mlir::utils::IteratorType> iterators(
        rank, mlir::utils::IteratorType::parallel);
    for_all_dims(rank, dimArg, [&](std::size_t index, bool dropped) {
      if (dropped) {
        iterators[index] = mlir::utils::IteratorType::reduction;
      }
    });
    return iterators;
  }

  static ReduceDim dimArgAsReduceDim(mlir::Operation *op, std::size_t rank) {
    assert(rank <= 64 && "rank value too large for a 64-bit set");

    std::uint64_t bits = 0;
    for_all_dims(rank, getDimArg(op), [&](std::size_t index, bool dropped) {
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

private:
  static mlir::ArrayAttr getDimArg(mlir::Operation *op) {
    auto attr = mlir::dyn_cast<mlir::ArrayAttr>(op->getAttr("dim_arg"));
    assert(attr != nullptr);
    return attr;
  }

  template <typename F>
  static void for_all_dims(std::size_t rank, mlir::ArrayAttr dimArg, F &&fn) {
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

// Iteration space traits specialized for 'TTIRNamedContraction' ops.
template <>
struct iteration_space_traits<OpTrait::named_op_group::contraction>
    : iteration_space_traits<> {

  // [getAffineMapsArray() is not needed at the moment]

  static ArrayAttr getAffineMapsAttr(mlir::OpBuilder &builder,
                                     mlir::Operation *op, std::size_t arity,
                                     std::size_t rank) {
    assert(arity == 3 && "expected 3 operands");
    assert(rank == 2 && "expected a rank 2 operation");
    MLIRContext *ctx = builder.getContext();
    return builder.getAffineMapArrayAttr(SmallVector<mlir::AffineMap>{
        makeAffineMap(ctx, {0, 2}), makeAffineMap(ctx, {2, 1}),
        makeAffineMap(ctx, {0, 1})});
  }

  static ArrayAttr getTTIRIteratorTypesAttr(mlir::OpBuilder &builder,
                                            mlir::Operation *op,
                                            std::size_t rank) {
    assert(rank == 2 && "expected a rank 2 operation");
    auto parallel = tt::IteratorTypeAttr::get(builder.getContext(),
                                              tt::IteratorType::Parallel);
    auto reduction = tt::IteratorTypeAttr::get(builder.getContext(),
                                               tt::IteratorType::Reduction);
    return builder.getArrayAttr(
        SmallVector<mlir::Attribute>{parallel, parallel, reduction});
  }

  // [getLinalgIteratorTypesArray() is not needed at the moment]

private:
  static mlir::AffineMap makeAffineMap(MLIRContext *ctx,
                                       std::array<unsigned, 2> targets) {
    return mlir::AffineMap::getMultiDimMapWithTargets(3, targets, ctx);
  }
};
// ............................................................................
// This type map collects all named ops that are known to have tile_* lowering
// counterpart ops with signatures that are "reasonably similar" to the named
// signatures. To put it differently, these ops don't require decomposition
// or other special case handling to be lowered through ttir.generic.

// clang-format off
using direct_lowerings = std::tuple<
  // elementwise:
  std::pair<AddOp,        TileAddOp>,
  std::pair<MultiplyOp,   TileMulOp>,
  std::pair<ExpOp,        TileExpOp>,
  std::pair<LogOp,        TileLogOp>,
  // reduction:
  std::pair<SumOp,        TileReduceSum1Op>,
  std::pair<MaxOp,        TileReduceMaxOp>,
  // contraction:
  std::pair<MatmulOp,     TileMatmulBlockOp>
>;
// clang-format on

// An OpRewritePattern implementation for all ConcreteOps that are present in
// 'direct_lowerings'. As the dialect grows and/or more ConcreteOps need to
// be handled by the 'TTIRGeneralizeNamedOps' pass, the typemap above
// can be grown and full/partial 'TTIRGeneralizeNamedRewriter' template
// specializations can be added (or both).

template <typename ConcreteOp>
class TTIRGeneralizeNamedRewriter final
    : public mlir::OpRewritePattern<ConcreteOp> {

  using TileOp = ttmlir::utils::map_find_t<ConcreteOp, direct_lowerings>;
  static_assert(!std::is_void_v<TileOp>,
                "this ConcreteOp does not have a direct op mapping");

  using op_group = typename ConcreteOp::named_op_group_type;
  using traits = iteration_space_traits<op_group>;

public:
  using mlir::OpRewritePattern<ConcreteOp>::OpRewritePattern; // inherit
                                                              // constructors

private:
  LogicalResult matchAndRewrite(ConcreteOp op,
                                mlir::PatternRewriter &rewriter) const final {
    checkPreconditions(op);

    MLIRContext *ctx = rewriter.getContext();
    mlir::Operation *baseOp = op.getOperation();
    auto loc = op->getLoc();

    auto [inputs, inits] = traits::signatureSplit(op);

    std::size_t const numInputs = inputs.size();
    std::size_t const numInits = inits.size();
    std::size_t const numOperands = (numInputs + numInits);

    // outs() << "  named op<" << op.getOperationName() << ">:\t"
    //        << op->getNumOperands() << " (" << numInputs << "/" << numInits
    //        << ")  operands (inputs/inits)\n";
    assert(numOperands == op->getNumOperands());

    // core grid: for now, propagate the input default
    GridAttr grid = GridAttr::get(ctx, expectedInputGridShape());

    auto const rank = grid.getShape().size();

    ArrayAttr indexingMaps =
        traits::getAffineMapsAttr(rewriter, baseOp, numOperands, rank);
    ArrayAttr ttirIteratorTypes =
        traits::getTTIRIteratorTypesAttr(rewriter, baseOp, rank);

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
            traits::getAffineMapsArray(rewriter, baseOp, numOperands, rank);
        SmallVector<mlir::utils::IteratorType> linalgIteratorTypes =
            traits::getLinalgIteratorTypesArray(rewriter, baseOp, rank);

        SmallVector<mlir::NamedAttribute> attributes;

        // for reductions, propagate 'dim_arg' as 'ReduceDim':
        if constexpr (std::is_same_v<OpTrait::named_op_group::reduction,
                                     op_group>) {
          attributes.emplace_back(
              tt::ttir::ReduceDimAttr::getMnemonic(),
              tt::ttir::ReduceDimAttr::get(
                  ctx, traits::dimArgAsReduceDim(baseOp, rank)));
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
    ArrayRef<int64_t> shape;
    for (mlir::Type t : op->getOperandTypes()) {
      RankedTensorType tt = mlir::cast<RankedTensorType>(t);
      MetalLayoutAttr layout =
          mlir::dyn_cast<MetalLayoutAttr>(tt.getEncoding());
      assert(layout && "expected tt.metal_layout encoding");
      assert(layout.isTiled() && "expected tiled buffer element type");
      assert((layout.getGrid().getShape() == expectedInputGridShape()) &&
             "unexpected grid shape");

      // for elementwise ops, require identical operand shapes for now (no
      // broadcasting, etc):
      if constexpr (std::is_same_v<OpTrait::named_op_group::elementwise,
                                   op_group>) {
        if (shape.empty()) {
          shape = layout.getMemref().getShape();
        } else {
          assert((layout.getMemref().getShape() == shape) &&
                 "expected identical shard shapes");
        }
      }
    }
    // for reductions, require 'dim_arg' and 'keep_dim'=true:
    if constexpr (std::is_same_v<OpTrait::named_op_group::reduction,
                                 op_group>) {
      assert(op.getDimArg() && "expected dim_arg attribute to be set");
      assert(op.getKeepDimAttr().getValue() &&
             "expected default keep_dim=true");
    }
  }

  static constexpr ArrayRef<int64_t> expectedInputGridShape() {
    return s_expectedInputGridShape;
  }

  // for now, assume input is laid out for a 1x1 grid:
  static constexpr std::array<int64_t, 2> s_expectedInputGridShape{1, 1};

}; // end of class
// ............................................................................
// specialize for ttir.max (TODO(vlad) deal with the 2nd "fake" operand):

template <>
LogicalResult TTIRGeneralizeNamedRewriter<MaxOp>::matchAndRewrite(
    MaxOp op, mlir::PatternRewriter &rewriter) const {
  checkPreconditions(op);

  return op->emitOpError("TODO handle binary op signature");
}
// ............................................................................
// specialize for ttir.matmul (only one level of nesting, 'transpose_{a,b}'
// attributes):

template <>
LogicalResult TTIRGeneralizeNamedRewriter<MatmulOp>::matchAndRewrite(
    MatmulOp op, mlir::PatternRewriter &rewriter) const {
  checkPreconditions(op);

  MLIRContext *ctx = rewriter.getContext();
  mlir::Operation *baseOp = op.getOperation();
  auto loc = op->getLoc();

  auto [inputs, inits] = traits::signatureSplit(op);

  std::size_t const numInputs = inputs.size();
  std::size_t const numInits = inits.size();
  std::size_t const numOperands = (numInputs + numInits);

  // outs() << "  named op<" << op.getOperationName() << ">:\t"
  //        << op->getNumOperands() << " (" << numInputs << "/" << numInits
  //        << ")  operands (inputs/inits)\n";
  assert(numOperands == op->getNumOperands());

  // core grid: for now, propagate the input default
  GridAttr grid = GridAttr::get(ctx, expectedInputGridShape());

  auto const rank = grid.getShape().size();

  // TODO(vlad) handle 'transpose_{a,b}' ?

  ArrayAttr indexingMaps =
      traits::getAffineMapsAttr(rewriter, baseOp, numOperands, rank);
  ArrayAttr ttirIteratorTypes =
      traits::getTTIRIteratorTypesAttr(rewriter, baseOp, rank);

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
      std::for_each(op->getOperandTypes().begin(), op->getOperandTypes().end(),
                    [&](Type t) {
                      RankedTensorType tt = mlir::cast<RankedTensorType>(t);
                      MetalLayoutAttr layout =
                          mlir::cast<MetalLayoutAttr>(tt.getEncoding());
                      block->addArgument(layout.getMemref(), loc);
                    });
      auto blockArgs = block->getArguments();

      // delegate next level of nesting to a "blocked" 'TileOp':

      rewriter.create<TileOp>(loc, /* resultTypes */ mlir::TypeRange(),
                              /* operands */ blockArgs);
    }
  }

  rewriter.replaceOp(op, generic);
  return llvm::success();
}

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
        TTIRGeneralizeNamedRewriter<AddOp>,
        TTIRGeneralizeNamedRewriter<MultiplyOp>,
        TTIRGeneralizeNamedRewriter<ExpOp>,
        TTIRGeneralizeNamedRewriter<LogOp>,
        // reduction:
        TTIRGeneralizeNamedRewriter<SumOp>,
        TTIRGeneralizeNamedRewriter<MaxOp>,
        // contraction:
        TTIRGeneralizeNamedRewriter<MatmulOp>
      >(&ctx);
    }
    // clang-format on
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }

}; // end of class

#undef GEN_PASS_DEF_TTIRGENERALIZENAMEDOPS

} // namespace mlir::tt::ttir
// ----------------------------------------------------------------------------
