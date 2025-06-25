// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRGeneric/TTIRToTTIRGeneric.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
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

  TTIRNamedRewriterCommon(const TTIRToTTIRGenericOptions &options,
                          uint64_t deviceGridRank)
      : memorySpaces{options.defaultInputMemSpace,
                     options.defaultOutputMemSpace},
        deviceGridRank(deviceGridRank) {}

  std::array<mlir::SmallVector<Value>, 2>
  toLayoutOperands(mlir::ConversionPatternRewriter &rewriter,
                   std::array<mlir::SmallVector<Value>, 2> operands,
                   bool tiled) const {
    return {toLayoutOperands(rewriter, operands[0], tiled, memorySpaces[0]),
            toLayoutOperands(rewriter, operands[1], tiled, memorySpaces[1])};
  }

  mlir::SmallVector<Value>
  toLayoutOperands(mlir::ConversionPatternRewriter &rewriter,
                   mlir::SmallVector<Value> operands, bool tiled,
                   MemorySpace memorySpace) const {
    mlir::SmallVector<Value> newOperands;
    for (Value value : operands) {
      mlir::RankedTensorType tensorType =
          mlir::cast<mlir::RankedTensorType>(value.getType());

      // Logical shape is initial tensor shape.
      llvm::SmallVector<int64_t> logicalShape(tensorType.getShape());
      assert(deviceGridRank <= logicalShape.size());

      // Create default grid shape based on deviceGridRank.
      llvm::SmallVector<int64_t> gridShape;
      for (uint64_t i = 0; i < deviceGridRank; ++i) {
        gridShape.push_back(1);
      }

      // Create default tile element type.
      Type elementType = tensorType.getElementType();
      llvm::SmallVector<int64_t> tileShape;
      assert(logicalShape.size() >= 2);
      if (tiled) {
        auto defaultShape = TileType::getDefaultShape();
        tileShape.assign(defaultShape.begin(), defaultShape.end());
        elementType = tt::TileType::get(elementType, tileShape);
      }

      // Create the new MetalLayoutAttr with new element type.
      tt::MetalLayoutAttr layout = tt::MetalLayoutAttr::get(
          rewriter.getContext(), logicalShape, deviceGridRank,
          tt::OOBVal::Undef, memorySpace);

      // Calculate new physical shape based on grid + tiling.
      auto physicalShape = tt::MetalLayoutAttr::derivePhysicalShape(
          logicalShape, gridShape, tileShape, layout.getCollapseIntervals(),
          layout.getDimAlignments());

      mlir::RankedTensorType layoutResultType =
          mlir::RankedTensorType::get(physicalShape, elementType, layout);

      auto output =
          rewriter.create<tt::ttir::EmptyOp>(value.getLoc(), layoutResultType);
      newOperands.emplace_back(
          rewriter.create<tt::ttir::ToLayoutOp>(value.getLoc(), value, output)
              ->getResult(0));
    }
    return newOperands;
  }

  static Operation *unLayoutResult(mlir::ConversionPatternRewriter &rewriter,
                                   Value fromValue, Type toResultType) {
    auto output = rewriter.create<tt::ttir::EmptyOp>(
        fromValue.getLoc(), toResultType); // TODO mem space?
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

  static Block::BlockArgListType createBlockArguments(mlir::Block *block,
                                                      mlir::Location loc,
                                                      mlir::TypeRange inputs,
                                                      mlir::TypeRange outputs) {
    auto fn = [&](Type t) {
      mlir::RankedTensorType tensorType = mlir::cast<mlir::RankedTensorType>(t);
      tt::MetalLayoutAttr layout =
          mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
      block->addArgument(layout.getMemRefType(tensorType), loc);
    };

    llvm::for_each(mlir::TypeRange(inputs), fn);
    llvm::for_each(mlir::TypeRange(outputs), fn);
    return block->getArguments();
  }

  template <typename ConcreteOp>
  static MemorySpace getDefaultMemorySpace(ConcreteOp op, MemorySpace dflt) {
    mlir::ModuleOp parent = op->template getParentOfType<mlir::ModuleOp>();
    if (!parent) {
      return dflt;
    }
    tt::MemorySpaceAttr defaultMemSpaceAttr =
        parent->getAttrOfType<tt::MemorySpaceAttr>(tt::MemorySpaceAttr::name);
    return defaultMemSpaceAttr ? defaultMemSpaceAttr.getValue() : dflt;
  }

  static constexpr mlir::ArrayRef<int64_t> expectedInputGridShape() {
    return s_expectedInputGridShape;
  }

  // Default memory spaces for {inputs, outputs}.
  std::array<MemorySpace, 2> memorySpaces;
  uint64_t deviceGridRank;

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
  TTIRNamedElementwiseRewriter<ConcreteOp, TileOp>(
      const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
      const TTIRToTTIRGenericOptions &options, uint64_t deviceGridRank)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        TTIRNamedRewriterCommon(options, deviceGridRank) {}

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [origInputs, origOutputs] =
        splitDpsSignature(adaptor, op.getDpsInits().size());
    auto [inputs, outputs] =
        toLayoutOperands(rewriter, {origInputs, origOutputs},
                         /*tiled*/ true);

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    assert(numOperands == op->getNumOperands());

    tt::GridAttr grid = tt::GridAttr::get(ctx, expectedInputGridShape());

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
  TTIRNamedReductionRewriter<ConcreteOp, TileOp>(
      const TypeConverter &typeConverter, mlir::MLIRContext *ctx,
      const TTIRToTTIRGenericOptions &options, uint64_t deviceGridRank)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        TTIRNamedRewriterCommon(options, deviceGridRank) {}

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
        toLayoutOperands(rewriter, {newInputs, origOutputs},
                         /*tiled*/ true);

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    // minus 1 for the scaler operand
    assert((numOperands - 1) == op->getNumOperands());

    tt::GridAttr grid = tt::GridAttr::get(ctx, expectedInputGridShape());

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
                                  mlir::Type elementType) {
    mlir::RankedTensorType scalerType =
        RankedTensorType::get(TileType::getDefaultShape(), elementType);

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
                     const TTIRToTTIRGenericOptions &options,
                     uint64_t deviceGridRank)
      : OpConversionPattern<ConcreteOp>(typeConverter, ctx),
        TTIRNamedRewriterCommon(options, deviceGridRank) {}

private:
  LogicalResult
  matchAndRewrite(ConcreteOp op, typename ConcreteOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    checkPreconditions(op);

    mlir::MLIRContext *ctx = rewriter.getContext();
    mlir::Location loc = op->getLoc();

    auto [origInputs, origOutputs] =
        splitDpsSignature(adaptor, op.getDpsInits().size());
    auto [inputs, outputs] =
        toLayoutOperands(rewriter, {origInputs, origOutputs}, /*tiled*/ true);

    const std::size_t numInputs = inputs.size();
    const std::size_t numOutputs = outputs.size();
    const std::size_t numOperands = (numInputs + numOutputs);

    assert(numOperands == op->getNumOperands());

    tt::GridAttr grid = tt::GridAttr::get(ctx, expectedInputGridShape());

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

        } else if constexpr (std::is_same_v<ttir::TileMatmulOp, TileOp>) {

          static constexpr std::size_t tileOpNumInputs = 3;
          static constexpr std::size_t tileOpNumOutputs = 1;

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
                    loc, /* resultTypes */ bbArgs.take_back(tileOpNumOutputs),
                    /* operands */ bbArgs.take_front(tileOpNumInputs));

                bbBuilder.create<mlir::linalg::YieldOp>(bbLoc, yield);
              });
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
                                       TypeConverter &typeConverter,
                                       const TTIRToTTIRGenericOptions &options,
                                       uint64_t deviceGridRank) {
  // clang-format off
  patterns.add<
    // Elementwise.
    TTIRNamedElementwiseRewriter<ttir::AddOp,       ttir::TileAddOp>,
    TTIRNamedElementwiseRewriter<ttir::CeilOp,      ttir::TileCeilOp>,
    TTIRNamedElementwiseRewriter<ttir::CosOp,       ttir::TileCosOp>,
    TTIRNamedElementwiseRewriter<ttir::DivOp,       ttir::TileDivOp>,
    TTIRNamedElementwiseRewriter<ttir::ExpOp,       ttir::TileExpOp>,
    TTIRNamedElementwiseRewriter<ttir::LogOp,       ttir::TileLogOp>,
    TTIRNamedElementwiseRewriter<ttir::MultiplyOp,  ttir::TileMulOp>,
    TTIRNamedElementwiseRewriter<ttir::MaximumOp,   ttir::TileMaximumOp>,
    TTIRNamedElementwiseRewriter<ttir::NegOp,       ttir::TileNegativeOp>,
    TTIRNamedElementwiseRewriter<ttir::RsqrtOp,     ttir::TileRsqrtOp>,
    TTIRNamedElementwiseRewriter<ttir::SigmoidOp,   ttir::TileSigmoidOp>,
    TTIRNamedElementwiseRewriter<ttir::SinOp,       ttir::TileSinOp>,
    TTIRNamedElementwiseRewriter<ttir::SubtractOp,  ttir::TileSubOp>,
    // Reductions.
    TTIRNamedReductionRewriter<ttir::MaxOp,         ttir::TileReduceMaxOp>,
    TTIRNamedReductionRewriter<ttir::SumOp,         ttir::TileReduceSumOp>,
    // Data movement.
    TTIRNamedElementwiseRewriter<ttir::TypecastOp,  ttir::TileTypecastOp>
  >(typeConverter, ctx, options, deviceGridRank);

  // Matmul.
  if (options.useTileMatmul) {
    patterns.add<TTIRMatmulRewriter<ttir::TileMatmulOp>>(typeConverter, ctx, options, deviceGridRank);
  }
  else {
    patterns.add<TTIRMatmulRewriter<ttir::TileMatmulBlockOp>>(typeConverter, ctx, options, deviceGridRank);
  }
  // clang-format on
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
