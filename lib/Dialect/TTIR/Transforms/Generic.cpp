// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICREGION
#define GEN_PASS_DEF_TTIRGENERICREGIONOPERANDSTOMEMREF
#define GEN_PASS_DEF_TTIRGENERICOPCBS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Generic - Region pass
//===----------------------------------------------------------------------===//

class TTIRGenericRegionRewriter
    : public OpInterfaceRewritePattern<GenericRegionOp> {
public:
  TTIRGenericRegionRewriter(MLIRContext *context, bool newLowering)
      : OpInterfaceRewritePattern<GenericRegionOp>(context),
        newLowering(newLowering) {}

  LogicalResult matchAndRewrite(GenericRegionOp op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<GenericOp>(op.getOperation()->getParentOp())) {
      return failure();
    }

    auto dps = cast<DestinationStyleOpInterface>(op.getOperation());

    // Create a generic op.
    auto [indexingMaps, iteratorTypes] = op.getIndexingMaps(rewriter);

    // For testing purposes try getting grid of the resulting tensor and put the
    // op in the grid.
    // TODO(radenko) add a proper debug/test flag.
    auto gridAttr = rewriter.getAttr<GridAttr>();
    auto resEncoding =
        mlir::cast<RankedTensorType>(op->getResult(0).getType()).getEncoding();
    if (resEncoding) {
      auto resLayout = mlir::cast<MetalLayoutAttr>(resEncoding);
      gridAttr = resLayout.getGrid();
    }

    auto genericOp = rewriter.create<ttir::GenericOp>(
        op.getLoc(), op->getResults().getTypes(), dps.getDpsInputs(),
        ValueRange() /* cbs */, dps.getDpsInits(), gridAttr, indexingMaps,
        iteratorTypes);

    // Create a new basic block for the generic op and create block arguments.
    Block *block = rewriter.createBlock(&genericOp.getRegion());
    SmallVector<Location> blockArgumentLocs(genericOp.getOperands().size(),
                                            genericOp.getLoc());
    if (newLowering) {
      SmallVector<Type> blockArgTypes(
          llvm::map_range(genericOp.getOperands().getTypes(), [&](Type type) {
            RankedTensorType tensorType = mlir::cast<RankedTensorType>(type);
            tt::MetalLayoutAttr layout =
                mlir::cast<tt::MetalLayoutAttr>(tensorType.getEncoding());
            return layout.getMemref();
          }));
      block->addArguments(blockArgTypes,
                          blockArgumentLocs);
    } else {
      block->addArguments(TypeRange(genericOp.getOperandTypes()),
                          blockArgumentLocs);
    }

    // Convert the original op into arith/math and into the generic block.
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    op.buildGenericRegion(blockBuilder, block);
    rewriter.replaceOp(op, genericOp);
    return success();
  }

  bool newLowering;
};

class TTIRGenericRegion
    : public impl::TTIRGenericRegionBase<TTIRGenericRegion> {
public:
  using impl::TTIRGenericRegionBase<TTIRGenericRegion>::TTIRGenericRegionBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericRegionRewriter>(&getContext(), newLowering);
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::arith::ArithDialect>();
  }
};

//===----------------------------------------------------------------------===//
// Generic - Operands to memref pass
//===----------------------------------------------------------------------===//

struct TTIRGenericOperandsToMemrefRewriter
    : public OpConversionPattern<GenericOp> {
  using OpConversionPattern<GenericOp>::OpConversionPattern;

  template <typename ValueRange>
  void convertTypes(ValueRange valueRange,
                    DenseMap<Type, Type> const &typeMap) const {
    for (auto operand : valueRange) {
      if (typeMap.count(operand.getType()) == 0) {
        continue;
      }
      operand.setType(typeMap.at(operand.getType()));
    }
  }

  LogicalResult
  matchAndRewrite(GenericOp generic, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Block *entry = &generic.getRegion().front();
    auto firstEntryArgType = entry->getArguments()[0].getType();
    auto encoding =
        mlir::cast<RankedTensorType>(firstEntryArgType).getEncoding();
    if (mlir::isa_and_nonnull<BufferAttr>(encoding)) {
      // Already converted.
      return failure();
    }

    rewriter.modifyOpInPlace(generic, [&]() {
      DenseMap<Type, Type> typeMap;

      for (auto blockArg : entry->getArguments()) {
        uint32_t blockArgNumber = blockArg.getArgNumber();
        auto matchingOperand = generic.getMatchingOperand(blockArgNumber);
        auto operandType = matchingOperand.getType();

        auto bufferLayout = mlir::cast<MetalLayoutAttr>(
            mlir::cast<RankedTensorType>(operandType).getEncoding());
        auto bufferType = operandType;

        int64_t cbIndex = generic.getOperandCbMapping()[blockArgNumber];

        if (cbIndex >= 0) {
          assert(static_cast<size_t>(cbIndex) < generic.getCbs().size());
          auto cb = generic.getCbs()[cbIndex];
          auto cbType = cb.getType();
          auto cbLayout = mlir::cast<MetalLayoutAttr>(
              mlir::cast<RankedTensorType>(cbType).getEncoding());
          bufferLayout = cbLayout;
          bufferType = cbType;
        }

        // TODO(rpavlovic): introduce multiplier for buffer.
        auto buffer = BufferAttr::get(
            getContext(), bufferLayout.getMemref(),
            (cbIndex >= 0 ? BufferAccess::Stream : BufferAccess::Alias));

        auto ty = RankedTensorType::get(
            buffer.getShape(),
            mlir::cast<RankedTensorType>(bufferType).getElementType(), buffer);

        typeMap[blockArg.getType()] = ty;
        blockArg.setType(ty);
      }
      for (Operation &op : generic.getRegion().getOps()) {
        convertTypes(op.getOperands(), typeMap);
        convertTypes(op.getResults(), typeMap);
      }
    });

    return success();
  }
};

class TTIRGenericRegionMemrefTypeConverter : public TypeConverter {
public:
  TTIRGenericRegionMemrefTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](RankedTensorType type) -> Type {
      auto encoding = type.getEncoding();
      assert(encoding);
      if (mlir::isa<BufferAttr>(encoding)) {
        return type;
      }
      auto layout = mlir::cast<MetalLayoutAttr>(type.getEncoding());
      auto buffer =
          BufferAttr::get(ctx, layout.getMemref(), BufferAccess::Alias);
      return RankedTensorType::get(buffer.getShape(), type.getElementType(),
                                   buffer);
    });
  }
};

class TTIRGenericRegionOperandsToMemref
    : public impl::TTIRGenericRegionOperandsToMemrefBase<
          TTIRGenericRegionOperandsToMemref> {
public:
  using impl::TTIRGenericRegionOperandsToMemrefBase<
      TTIRGenericRegionOperandsToMemref>::TTIRGenericRegionOperandsToMemrefBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    TTIRGenericRegionMemrefTypeConverter typeConverter(&getContext());
    patterns.add<TTIRGenericOperandsToMemrefRewriter>(typeConverter,
                                                      &getContext());

    mlir::ConversionTarget target(getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

//===----------------------------------------------------------------------===//
// Generic - CBs pass
//===----------------------------------------------------------------------===//

class TTIRGenericOpCBsRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp generic,
                                PatternRewriter &rewriter) const final {
    if (!generic.getOperandCbMapping().empty()) {
      // Already inserted CBs, therefore skip.
      return failure();
    }

    rewriter.setInsertionPointToStart(generic->getBlock());

    SmallVector<Value> cbValues;
    SmallVector<int64_t> operandCBMapping;

    for (auto operand : generic->getOperands()) {
      auto ty = mlir::cast<RankedTensorType>(operand.getType());

      // Enforcing tiled layout as in kernel we always want to work with tiles.
      auto desiredElementType = rewriter.getType<TileType>(ty.getElementType());
      auto desiredLayout = rewriter.getAttr<MetalLayoutAttr>(
          ty, MemorySpace::DeviceL1, generic.getGrid(), desiredElementType);

      auto operandTy = operand.getType();
      auto operandLayout = mlir::cast<MetalLayoutAttr>(
          mlir::cast<RankedTensorType>(operandTy).getEncoding());

      if (desiredLayout.getGrid() == operandLayout.getGrid()) {
        // TODO(rpavlovic): should we check other layout features such as
        // linear?
        operandCBMapping.push_back(-1);
        continue;
      }

      // Creating a CB for the operand. It takes the same type as the operand,
      // but changes its grid. This may result in overly large CBs at the
      // moment.
      auto emptyOp = rewriter.create<tensor::EmptyOp>(
          generic->getLoc(), ty.getShape(), ty.getElementType(), desiredLayout);
      cbValues.push_back(emptyOp.getResult());
      operandCBMapping.push_back(cbValues.size() - 1);
    }

    rewriter.setInsertionPointAfter(generic);
    auto newGenericOp = rewriter.create<ttir::GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        cbValues, generic.getOutputs(), generic.getGrid(),
        generic.getIndexingMaps(), generic.getIteratorTypes(),
        operandCBMapping);

    auto &oldRegion = generic.getRegion();
    newGenericOp->getRegion(0).takeBody(oldRegion);

    rewriter.replaceOp(generic, newGenericOp);
    return success();
  }
};

class TTIRGenericOpCBs : public impl::TTIRGenericOpCBsBase<TTIRGenericOpCBs> {
public:
  using impl::TTIRGenericOpCBsBase<TTIRGenericOpCBs>::TTIRGenericOpCBsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericOpCBsRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::tt::ttir
