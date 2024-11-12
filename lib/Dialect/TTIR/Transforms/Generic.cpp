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
#define GEN_PASS_DEF_TTIRGENERICKERNEL
#define GEN_PASS_DEF_TTIRGENERICREGION
#define GEN_PASS_DEF_TTIRGENERICREGIONOPERANDSTOMEMREF
#define GEN_PASS_DEF_TTIRGENERICOPCBS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Generic - Kernel pass
//===----------------------------------------------------------------------===//

class TTIRLinalgGenericRewriter : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    return failure();
  }
};

template <typename TTIROpTy>
class TTIRNamedToKernelRewriter : public OpRewritePattern<TTIROpTy> {
public:
  using OpRewritePattern<TTIROpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTIROpTy op,
                                PatternRewriter &rewriter) const final {
    StringRef kernelName;
    StringRef kernelKind;
    if constexpr (std::is_same<TTIROpTy, ttir::MultiplyOp>::value) {
      kernelName = "multiply";
      kernelKind = "eltwise";
    } else if constexpr (std::is_same<TTIROpTy, ttir::AddOp>::value) {
      kernelName = "add";
      kernelKind = "eltwise";
    } else if constexpr (std::is_same<TTIROpTy, ttir::SubtractOp>::value) {
      kernelName = "subtract";
      kernelKind = "eltwise";
    } else if constexpr (std::is_same<TTIROpTy, ttir::GreaterEqualOp>::value) {
      kernelName = "ge";
      kernelKind = "eltwise";
    } else if constexpr (std::is_same<TTIROpTy, ttir::ReluOp>::value) {
      kernelName = "relu";
      kernelKind = "eltwise";
    } else if constexpr (std::is_same<TTIROpTy, ttir::DivOp>::value) {
      kernelName = "div";
      kernelKind = "eltwise";
    } else {
      return rewriter.notifyMatchFailure(op, "Unsupported operation for TTIR");
    }
    assert(kernelName.size() > 0);

    auto kernel = rewriter.create<ttir::KernelOp>(
        op.getLoc(), op.getResultTypes(), kernelName, kernelKind,
        op.getInputs(), op.getOutputs());

    rewriter.replaceOp(op, kernel);

    return success();
  }
};

class TTIRKernelGenericRewriter : public OpRewritePattern<KernelOp> {
public:
  using OpRewritePattern<KernelOp>::OpRewritePattern;

  static bool sameRank(mlir::OperandRange operands) {
    if (operands.empty()) {
      return true;
    }
    auto rank = mlir::cast<RankedTensorType>(operands[0].getType()).getRank();
    for (auto operand : operands) {
      if (mlir::cast<RankedTensorType>(operand.getType()).getRank() != rank) {
        return false;
      }
    }
    return true;
  }

  static std::pair<ArrayAttr, ArrayAttr>
  createEltwiseIndexingMaps(PatternRewriter &rewriter,
                            mlir::OperandRange operands) {
    assert(sameRank(operands) &&
           "For now all operands must have the same rank");
    auto rank = mlir::cast<RankedTensorType>(operands[0].getType()).getRank();
    SmallVector<AffineMap> indexingMaps(operands.size(),
                                        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<Attribute> iteratorTypes(
        rank, rewriter.getAttr<IteratorTypeAttr>(IteratorType::Parallel));
    return {rewriter.getAffineMapArrayAttr(indexingMaps),
            rewriter.getArrayAttr(iteratorTypes)};
  }

  static std::pair<ArrayAttr, ArrayAttr>
  createMatmulIndexingMaps(PatternRewriter &rewriter,
                           mlir::OperandRange operands) {
    assert(sameRank(operands) &&
           "For now all operands must have the same rank");
    auto rank = mlir::cast<RankedTensorType>(operands[0].getType()).getRank();
    assert(rank >= 2 && "Matmul requires rank >= 2");
    auto rank_plus_inner_dim = rank + 1;

    // (d0, d1, d2, d3) -> (d0, d1, d2, d3)
    // lhs (d0, d1, d2, d3) -> (d0, d1, d3) drop d2
    // rhs (d0, d1, d2, d3) -> (d0, d3, d2) drop d1 and swap d2 and d3
    // out (d0, d1, d2, d3) -> (d0, d1, d2) drop d3
    auto id = rewriter.getMultiDimIdentityMap(rank_plus_inner_dim);
    auto lhs = id.dropResult(rank_plus_inner_dim - 2);
    auto rhs = id.dropResult(rank_plus_inner_dim - 3);
    auto rhs_outer = rhs.getResult(rank - 2);
    rhs = rhs.insertResult(rhs_outer, rank);
    rhs = rhs.dropResult(rank - 2);
    auto out = id.dropResult(rank_plus_inner_dim - 1);

    SmallVector<AffineMap> indexingMaps = {lhs, rhs, out};
    SmallVector<Attribute> iteratorTypes(
        rank, rewriter.getAttr<IteratorTypeAttr>(IteratorType::Parallel));
    iteratorTypes.push_back(
        rewriter.getAttr<IteratorTypeAttr>(IteratorType::Systolic));
    return {rewriter.getAffineMapArrayAttr(indexingMaps),
            rewriter.getArrayAttr(iteratorTypes)};
  }

  static std::pair<ArrayAttr, ArrayAttr>
  createIndexingMaps(PatternRewriter &rewriter, StringRef kind,
                     mlir::OperandRange operands) {
    if (kind == "eltwise") {
      return createEltwiseIndexingMaps(rewriter, operands);
    }
    if (kind == "matmul") {
      return createMatmulIndexingMaps(rewriter, operands);
    }
    llvm_unreachable("Unsupported kernel kind");
  }

  static ArrayAttr createOperandConstraints(PatternRewriter &rewriter,
                                            StringRef kind,
                                            mlir::OperandRange operands) {
    auto numOperands = operands.size();
    if (kind == "eltwise") {
      return rewriter.getArrayAttr(SmallVector<Attribute>(
          numOperands, rewriter.getAttr<OperandConstraintAttr>(
                           OperandConstraint::AnyDevice)));
    }
    if (kind == "matmul") {
      return rewriter.getArrayAttr(SmallVector<Attribute>(
          numOperands, rewriter.getAttr<OperandConstraintAttr>(
                           OperandConstraint::AnyDeviceTile)));
    }
    llvm_unreachable("Unsupported kernel kind");
  }

  LogicalResult matchAndRewrite(KernelOp op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<GenericOp>(op.getOperation()->getParentOp())) {
      return failure();
    }

    // Create a dispatch op
    auto [indexingMaps, iteratorTypes] =
        createIndexingMaps(rewriter, op.getKind(), op.getOperands());
    auto constraints =
        createOperandConstraints(rewriter, op.getKind(), op.getOperands());
    auto dispatch = rewriter.create<ttir::GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        ValueRange() /* cbs */, op.getOutputs(), rewriter.getAttr<GridAttr>(),
        indexingMaps, iteratorTypes, constraints);

    // Create a new basic block for the dispatch op and create block arguments
    Block *block = rewriter.createBlock(&dispatch.getRegion());
    SmallVector<Location> blockArgumentLocs(dispatch.getOperands().size(),
                                            dispatch.getLoc());
    block->addArguments(TypeRange(dispatch.getOperandTypes()),
                        blockArgumentLocs);

    // Update the operands of the original op to use the block arguments
    op.getOperation()->setOperands(block->getArguments());

    // Move the original op into the dispatch block
    Operation *operation = op.getOperation()->clone();
    block->push_back(operation);
    rewriter.setInsertionPoint(block, block->end());
    rewriter.create<ttir::YieldOp>(dispatch.getLoc(),
                                   ValueRange({operation->getResult(0)}));
    rewriter.replaceOp(op, dispatch);
    return success();
  }
};

class TTIRGenericKernel
    : public impl::TTIRGenericKernelBase<TTIRGenericKernel> {
public:
  using impl::TTIRGenericKernelBase<TTIRGenericKernel>::TTIRGenericKernelBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<
        TTIRLinalgGenericRewriter, TTIRKernelGenericRewriter,
        TTIRNamedToKernelRewriter<AddOp>, TTIRNamedToKernelRewriter<MultiplyOp>,
        TTIRNamedToKernelRewriter<SubtractOp>,
        TTIRNamedToKernelRewriter<GreaterEqualOp>,
        TTIRNamedToKernelRewriter<DivOp>, TTIRNamedToKernelRewriter<ReluOp>>(
        &getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

//===----------------------------------------------------------------------===//
// Generic - Region pass
//===----------------------------------------------------------------------===//

class TTIRGenericRegionRewriter
    : public OpInterfaceRewritePattern<GenericRegionOp> {
public:
  using OpInterfaceRewritePattern<GenericRegionOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(GenericRegionOp op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<GenericOp>(op.getOperation()->getParentOp())) {
      return failure();
    }

    auto dps = cast<DestinationStyleOpInterface>(op.getOperation());

    // Create a generic op.
    auto [indexingMaps, iteratorTypes] = op.getIndexingMaps(rewriter);
    auto constraints = rewriter.getArrayAttr(SmallVector<Attribute>(
        op->getNumOperands(), rewriter.getAttr<OperandConstraintAttr>(
                                  OperandConstraint::AnyDeviceTile)));

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
        iteratorTypes, constraints);

    // Create a new basic block for the generic op and create block arguments.
    Block *block = rewriter.createBlock(&genericOp.getRegion());
    SmallVector<Location> blockArgumentLocs(genericOp.getOperands().size(),
                                            genericOp.getLoc());
    block->addArguments(TypeRange(genericOp.getOperandTypes()),
                        blockArgumentLocs);

    // Convert the original op into arith/math and into the generic block.
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    op.buildGenericRegion(blockBuilder, block);
    rewriter.replaceOp(op, genericOp);
    return success();
  }
};

class TTIRGenericRegion
    : public impl::TTIRGenericRegionBase<TTIRGenericRegion> {
public:
  using impl::TTIRGenericRegionBase<TTIRGenericRegion>::TTIRGenericRegionBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericRegionRewriter>(&getContext());
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
    SmallVector<Attribute> oldConstraints;
    SmallVector<Attribute> cbConstraints;
    size_t i = 0;

    for (auto operand : generic->getOperands()) {
      size_t operandIdx = i++;
      oldConstraints.push_back(generic.getOperandConstraints()[operandIdx]);

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

      // Inheriting constraints from the original operand.
      // OperandConstraint inherittedConstraint =
      //     mlir::cast<OperandConstraintAttr>(
      //         generic.getOperandConstraints()[operandIdx])
      //         .getValue();
      // inherittedConstraint =
      //     bitEnumSet(inherittedConstraint, OperandConstraint::L1);
      // inherittedConstraint =
      //     bitEnumClear(inherittedConstraint, OperandConstraint::DRAM);
      // inherittedConstraint =
      //     bitEnumClear(inherittedConstraint, OperandConstraint::System);

      // Fixing constraint to L1 for the CB operand.
      // TODO(rpavlovic) remove or use code above when we decide on the operand
      // constraints model.
      cbConstraints.push_back(
          rewriter.getAttr<OperandConstraintAttr>(OperandConstraint::L1));
    }

    SmallVector<Attribute> combinedConstraints;
    combinedConstraints.append(oldConstraints.begin(),
                               oldConstraints.begin() +
                                   generic.getInputs().size());
    combinedConstraints.append(cbConstraints.begin(), cbConstraints.end());
    combinedConstraints.append(oldConstraints.begin() +
                                   generic.getInputs().size(),
                               oldConstraints.end());
    auto newConstraintsArray = rewriter.getArrayAttr(combinedConstraints);

    rewriter.setInsertionPointAfter(generic);
    auto newGenericOp = rewriter.create<ttir::GenericOp>(
        generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
        cbValues, generic.getOutputs(), generic.getGrid(),
        generic.getIndexingMaps(), generic.getIteratorTypes(),
        newConstraintsArray, operandCBMapping);

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
