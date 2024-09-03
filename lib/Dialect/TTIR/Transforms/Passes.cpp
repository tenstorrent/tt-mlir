// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "ttmlir/Dialect/TTIR/Analysis/LegalGridAnalysis.h"
#include "ttmlir/Dialect/TTIR/Analysis/OptimalTargetGridAnalysis.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICKERNEL
#define GEN_PASS_DEF_TTIRGENERICREGION
#define GEN_PASS_DEF_TTIRGENERICREGIONOPERANDSTOMEMREF
#define GEN_PASS_DEF_TTIRLAYOUT
#define GEN_PASS_DEF_TTIRSPLITCOMPOUNDLAYOUT
#define GEN_PASS_DEF_TTIRALLOCATE
#define GEN_PASS_DEF_TTIRGRIDSET
#define GEN_PASS_DEF_TTIRIMPLICITDEVICE
#define GEN_PASS_DEF_TTIRLOADSYSTEMDESC
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRImplicitDevice
    : public impl::TTIRImplicitDeviceBase<TTIRImplicitDevice> {
public:
  using impl::TTIRImplicitDeviceBase<
      TTIRImplicitDevice>::TTIRImplicitDeviceBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();

    if (not module->hasAttr(tt::DeviceAttr::name)) {
      assert(module->hasAttr(tt::SystemDescAttr::name));
      auto systemDesc = module->getAttr(tt::SystemDescAttr::name);
      module->setAttr(
          tt::DeviceAttr::name,
          tt::DeviceAttr::get(&getContext(),
                              mlir::cast<tt::SystemDescAttr>(systemDesc)));
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

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
      kernelName = "mulitply";
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
        op.getInputs(), op.getOutputs(), op.getOperandConstraints());

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
        op.getOutputs(), rewriter.getAttr<GridAttr>(), indexingMaps,
        iteratorTypes, constraints);

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

    // Create a dispatch op
    auto [indexingMaps, iteratorTypes] = op.getIndexingMaps(rewriter);
    auto constraints = rewriter.getArrayAttr(SmallVector<Attribute>(
        op->getNumOperands(), rewriter.getAttr<OperandConstraintAttr>(
                                  OperandConstraint::AnyDeviceTile)));
    auto dispatch = rewriter.create<ttir::GenericOp>(
        op.getLoc(), op->getResults().getTypes(), dps.getDpsInputs(),
        dps.getDpsInits(), rewriter.getAttr<GridAttr>(), indexingMaps,
        iteratorTypes, constraints);

    // Create a new basic block for the dispatch op and create block arguments
    Block *block = rewriter.createBlock(&dispatch.getRegion());
    SmallVector<Location> blockArgumentLocs(dispatch.getOperands().size(),
                                            dispatch.getLoc());
    block->addArguments(TypeRange(dispatch.getOperandTypes()),
                        blockArgumentLocs);

    // Convert the original op into arith/math and into the dispatch block
    OpBuilder blockBuilder = OpBuilder::atBlockEnd(block);
    op.buildGenericRegion(blockBuilder, block);
    rewriter.replaceOp(op, dispatch);
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
    bool isConverted = static_cast<bool>(
        mlir::cast<RankedTensorType>(firstEntryArgType).getEncoding());
    if (isConverted) {
      // Already converted
      return failure();
    }

    rewriter.modifyOpInPlace(generic, [&]() {
      DenseMap<Type, Type> typeMap;
      for (auto [operand, blockArg] :
           llvm::zip(generic->getOperands(), entry->getArguments())) {
        auto ty = getTypeConverter()->convertType(operand.getType());
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
      auto layout = mlir::cast<LayoutAttr>(type.getEncoding());
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

inline MemorySpace getMemorySpace(MemRefType memref) {
  return mlir::cast<MemorySpaceAttr>(memref.getMemorySpace()).getValue();
}

inline MemorySpace getMemorySpace(LayoutAttr layout) {
  return getMemorySpace(layout.getMemref());
}

inline MemorySpace getMemorySpace(RankedTensorType ty) {
  assert(ty.getEncoding());
  auto layout = mlir::cast<LayoutAttr>(ty.getEncoding());
  return getMemorySpace(layout);
}

inline OperandConstraint
memorySpaceAsOperandConstraint(MemorySpace memorySpace) {
  switch (memorySpace) {
  case MemorySpace::System:
  case MemorySpace::SystemMMIO:
    return OperandConstraint::System;
  case MemorySpace::DeviceDRAM:
    return OperandConstraint::DRAM;
  case MemorySpace::DeviceL1:
    return OperandConstraint::L1;
  }
}

inline OperandConstraint
memoryLayoutAsOperandConstraint(TensorMemoryLayout memoryLayout) {
  switch (memoryLayout) {
  case TensorMemoryLayout::NoneLayout:
    return OperandConstraint::NoneLayout;
  case TensorMemoryLayout::Interleaved:
    return OperandConstraint::Interleaved;
  case TensorMemoryLayout::SingleBank:
    return OperandConstraint::SingleBank;
  case TensorMemoryLayout::HeightSharded:
    return OperandConstraint::HeightSharded;
  case TensorMemoryLayout::WidthSharded:
    return OperandConstraint::WidthSharded;
  case TensorMemoryLayout::BlockSharded:
    return OperandConstraint::BlockSharded;
  }
}

inline MemorySpace getLegalMemorySpace(OperandConstraint operandConstraint,
                                       MemorySpace defaultMemorySpace) {
  if (bitEnumContainsAny(operandConstraint,
                         memorySpaceAsOperandConstraint(defaultMemorySpace))) {
    return defaultMemorySpace;
  }
  if (bitEnumContainsAny(operandConstraint, OperandConstraint::DRAM)) {
    return MemorySpace::DeviceDRAM;
  }
  if (bitEnumContainsAny(operandConstraint, OperandConstraint::L1)) {
    return MemorySpace::DeviceL1;
  }
  return MemorySpace::System;
}

inline TensorMemoryLayout
getLegalTensorMemoryLayout(OperandConstraint operandConstraint,
                           MemorySpace targetMemorySpace,
                           TensorMemoryLayout defaultDeviceMemLayout) {
  if (defaultDeviceMemLayout == TensorMemoryLayout::NoneLayout) {
    return TensorMemoryLayout::NoneLayout;
  }
  if (isSystemMemorySpace(targetMemorySpace)) {
    return TensorMemoryLayout::NoneLayout;
  } else {
    assert(isDeviceMemorySpace(targetMemorySpace));
    if (bitEnumContainsAny(operandConstraint, memoryLayoutAsOperandConstraint(
                                                  defaultDeviceMemLayout))) {
      return defaultDeviceMemLayout;
    }
  }

  std::map<OperandConstraint, TensorMemoryLayout> validLayoutsMap = {
      {OperandConstraint::Interleaved, TensorMemoryLayout::Interleaved},
      {OperandConstraint::SingleBank, TensorMemoryLayout::SingleBank},
      {OperandConstraint::HeightSharded, TensorMemoryLayout::HeightSharded},
      {OperandConstraint::WidthSharded, TensorMemoryLayout::WidthSharded},
      {OperandConstraint::BlockSharded, TensorMemoryLayout::BlockSharded}};

  for (const auto &[constraintLayout, memLayout] : validLayoutsMap) {
    if (bitEnumContainsAny(operandConstraint, constraintLayout)) {
      return memLayout;
    }
  }

  return TensorMemoryLayout::NoneLayout;
}

class TTIRLayoutTensorTypeConverter : public TypeConverter {
public:
  TTIRLayoutTensorTypeConverter(MLIRContext *ctx, MemorySpace initMemorySpace,
                                GridAttr deviceGrid) {
    addConversion([](Type type) { return type; });
    addConversion([ctx, initMemorySpace,
                   deviceGrid](RankedTensorType type) -> Type {
      auto layout = type.getEncoding();
      if (layout) {
        return type;
      }
      std::int64_t deviceGridRank = deviceGrid.getShape().size();
      // Default to single core grid
      auto tensorGrid = GridAttr::get(ctx, deviceGridRank);
      // Default to initMemorySpace, the optimizer might decide otherwise
      auto newLayout = LayoutAttr::get(ctx, type, initMemorySpace, tensorGrid);
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newLayout);
    });
  }
};

class TTIRLayoutTensorTypeRewriter : public RewritePattern {
public:
  TTIRLayoutTensorTypeRewriter(const TypeConverter &converter, MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        converter(&converter) {}

  template <typename ValueRange>
  bool convertTypes(ValueRange valueRange, SmallVector<Type> &newTypes) const {
    bool updated = false;
    auto result = converter->convertTypes(valueRange.getTypes(), newTypes);
    if (result.failed()) {
      return false;
    }
    for (auto [operand, newType] : llvm::zip(valueRange, newTypes)) {
      if (operand.getType() == newType) {
        continue;
      }
      operand.setType(newType);
      updated = true;
    }
    return updated;
  }

  bool convertFuncType(Operation *op, PatternRewriter &rewriter) const {
    auto funcOp = dyn_cast<func::FuncOp>(op);
    if (not funcOp) {
      return false;
    }
    SmallVector<Type> inputTypes(funcOp.getArgumentTypes());
    SmallVector<Type> outputTypes(funcOp.getResultTypes());
    for (Type &ty : inputTypes) {
      ty = converter->convertType(ty);
    }
    for (Type &ty : outputTypes) {
      ty = converter->convertType(ty);
    }
    auto newType = rewriter.getType<FunctionType>(inputTypes, outputTypes);
    if (funcOp.getFunctionType() == newType) {
      return false;
    }
    funcOp.setFunctionType(newType);
    return true;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Skip if we're inside a GenericOp
    if (mlir::isa<GenericOp>(op->getParentOp())) {
      return failure();
    }
    bool updated = false;
    SmallVector<Type> operands;
    SmallVector<Type> results;
    updated |= convertTypes(op->getOperands(), operands);
    updated |= convertTypes(op->getResults(), results);
    updated |= convertFuncType(op, rewriter);
    return updated ? success() : failure();
  }

  const TypeConverter *converter;
};

static std::optional<Value>
createToLayoutOp(PatternRewriter &rewriter, Location loc, Value input,
                 MemorySpace desiredMemorySpace,
                 TensorMemoryLayout desiredMemLayout, bool tiled) {

  auto ty = mlir::cast<RankedTensorType>(input.getType());
  auto currLayout = mlir::cast<LayoutAttr>(ty.getEncoding());
  auto currMemorySpace = currLayout.getMemorySpace();
  auto currElementType = currLayout.getElementType();
  auto currMemLayout = currLayout.getMemLayout();
  auto desiredElementType =
      tiled ? rewriter.getType<TileType>(ty.getElementType())
            : ty.getElementType();
  if (currMemorySpace == desiredMemorySpace &&
      currElementType == desiredElementType &&
      currMemLayout == desiredMemLayout) {
    return std::nullopt;
  }

  auto desiredLayout =
      rewriter.getAttr<LayoutAttr>(ty, desiredMemorySpace, currLayout.getGrid(),
                                   desiredElementType, desiredMemLayout);
  auto output = rewriter.create<tensor::EmptyOp>(
      loc, ty.getShape(), ty.getElementType(), desiredLayout);

  tensor::EmptyOp exising_empty = input.getDefiningOp<tensor::EmptyOp>();
  if (exising_empty) {
    rewriter.replaceOp(exising_empty, output);
    return output.getResult();
  }
  return rewriter
      .create<ttir::ToLayoutOp>(loc, output.getType(), input, output)
      ->getResult(0);
}

static std::optional<Value>
createToLayoutOp(PatternRewriter &rewriter, Location loc, Value input,
                 OperandConstraint operandConstraint,
                 MemorySpace defaultMemorySpace,
                 TensorMemoryLayout defaultDeviceMemoryLayout) {
  auto desiredMemorySpace =
      getLegalMemorySpace(operandConstraint, defaultMemorySpace);

  auto desiredMemoryLayout = getLegalTensorMemoryLayout(
      operandConstraint, desiredMemorySpace, defaultDeviceMemoryLayout);

  bool tiled =
      !bitEnumContainsAny(operandConstraint, OperandConstraint::Scalar);
  return createToLayoutOp(rewriter, loc, input, desiredMemorySpace,
                          desiredMemoryLayout, tiled);
}

class TTIRLayoutDPSOperandsRewriter
    : public OpInterfaceRewritePattern<DestinationStyleOpInterface> {
public:
  TTIRLayoutDPSOperandsRewriter(MLIRContext *ctx,
                                MemorySpace defaultMemorySpace,
                                TensorMemoryLayout defaultDeviceMemoryLayout)
      : OpInterfaceRewritePattern<DestinationStyleOpInterface>(ctx),
        defaultMemorySpace(defaultMemorySpace),
        defaultDeviceMemoryLayout(defaultDeviceMemoryLayout) {}

  LogicalResult matchAndRewrite(DestinationStyleOpInterface op,
                                PatternRewriter &rewriter) const final {
    if (mlir::isa<ToLayoutOp>(op.getOperation())) {
      // Skip the ToLayoutOp itself
      return failure();
    }

    assert(op->template hasTrait<TTIROp::Trait>());
    bool modified = false;
    for (auto &operand : op->getOpOperands()) {
      bool isResult = op.isDpsInit(&operand);

      // TTNN Conv2d moves input, weight, and bias from host to device
      // itself. Inserting the ToLayoutOp on these operands is thus problematic.
      if (mlir::isa<Conv2dOp>(op.getOperation()) && !isResult) {
        continue;
      }
      auto operandConstraint =
          mlir::cast<OperandConstraintAttr>(
              mlir::cast<TTIROp>(op.getOperation())
                  .getOperandConstraints()[operand.getOperandNumber()])
              .getValue();
      auto desiredLayout = createToLayoutOp(
          rewriter, op.getLoc(), operand.get(), operandConstraint,
          defaultMemorySpace, defaultDeviceMemoryLayout);

      if (desiredLayout) {
        rewriter.modifyOpInPlace(op, [&]() {
          modified = true;
          op->setOperand(operand.getOperandNumber(), *desiredLayout);
          if (isResult) {
            // If this is the output operand, update the result type
            op->getResult(0).setType(desiredLayout->getType());
          }
        });
      }
    }

    return modified ? success() : failure();
  }

private:
  MemorySpace defaultMemorySpace;
  TensorMemoryLayout defaultDeviceMemoryLayout;
};

class TTIRLayoutFuncReturnRewriter
    : public OpRewritePattern<mlir::func::ReturnOp> {
public:
  TTIRLayoutFuncReturnRewriter(MLIRContext *ctx, MemorySpace initMemorySpace,
                               TensorMemoryLayout defaultDeviceMemoryLayout)
      : OpRewritePattern<mlir::func::ReturnOp>(ctx),
        initMemorySpace(initMemorySpace),
        defaultDeviceMemoryLayout(defaultDeviceMemoryLayout) {}

  LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (auto &operand : op->getOpOperands()) {
      // Leave the return values in initMemorySpace, optimizer might decide
      // otherwise
      bool tiled = false;
      TensorMemoryLayout initMemoryLayout = TensorMemoryLayout::NoneLayout;
      if (isDeviceMemorySpace(initMemorySpace)) {
        initMemoryLayout = defaultDeviceMemoryLayout;
      }
      if (auto layout =
              createToLayoutOp(rewriter, op.getLoc(), operand.get(),
                               initMemorySpace, initMemoryLayout, tiled);
          layout) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.setOperand(operand.getOperandNumber(), *layout); });
        modified = true;
      }
    }
    return modified ? success() : failure();
  }

private:
  MemorySpace initMemorySpace;
  TensorMemoryLayout defaultDeviceMemoryLayout;
};

class TTIRLayout : public impl::TTIRLayoutBase<TTIRLayout> {
public:
  using impl::TTIRLayoutBase<TTIRLayout>::TTIRLayoutBase;

  void runOnOperation() final {
    {
      auto device = getCurrentScopeDevice(getOperation());
      assert(device && "Device not found");
      TTIRLayoutTensorTypeConverter typeConverter(
          &getContext(), initMemorySpace, device.getGrid());
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRLayoutTensorTypeRewriter>(typeConverter, &getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }
    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTIRLayoutDPSOperandsRewriter>(
          &getContext(), defaultMemorySpace, defaultDeviceMemoryLayout);
      patterns.add<TTIRLayoutFuncReturnRewriter>(&getContext(), initMemorySpace,
                                                 defaultDeviceMemoryLayout);
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }
};

class TTIRSplitCompoundLayoutRewriter : public OpRewritePattern<ToLayoutOp> {
public:
  using OpRewritePattern<ToLayoutOp>::OpRewritePattern;

  Value createToLayoutOp(PatternRewriter &rewriter, Location loc, Value input,
                         LayoutAttr desiredLayout) const {
    auto ty = mlir::cast<RankedTensorType>(input.getType());
    auto output = rewriter.create<tensor::EmptyOp>(
        loc, ty.getShape(), ty.getElementType(), desiredLayout);
    return rewriter
        .create<ttir::ToLayoutOp>(loc, output.getType(), input, output)
        ->getResult(0);
  }

  Value bounce(PatternRewriter &rewriter, ToLayoutOp op,
               LayoutAttr bounceLayout) const {
    auto bounced =
        createToLayoutOp(rewriter, op.getLoc(), op.getInput(), bounceLayout);
    return rewriter.replaceOpWithNewOp<ttir::ToLayoutOp>(
        op, op.getOutput().getType(), bounced, op.getOutput());
  }

  LogicalResult matchAndRewrite(ToLayoutOp op,
                                PatternRewriter &rewriter) const final {
    auto components = op.compoundComponents();
    bool isCompound = (static_cast<int>(components.isLayoutChange) +
                       static_cast<int>(components.isGridChange) +
                       static_cast<int>(components.isFormatChange) +
                       static_cast<int>(components.isMemorySpaceChange) +
                       static_cast<int>(components.isMemoryLayoutChange)) > 1;

    if (!isCompound) {
      return failure();
    }

    auto inputType = mlir::cast<RankedTensorType>(op.getInput().getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getOutput().getType());
    auto inputLayout = mlir::cast<LayoutAttr>(inputType.getEncoding());
    auto outputLayout = mlir::cast<LayoutAttr>(outputType.getEncoding());

    bool inputL1 = inputLayout.getMemorySpace() == MemorySpace::DeviceL1;
    bool outputL1 = outputLayout.getMemorySpace() == MemorySpace::DeviceL1;

    // First prioritize moving the data into L1 so we can work with it in L1
    if (!inputL1) {
      // read first into L1, then format convert
      bounce(rewriter, op,
             inputLayout.withMemorySpace(rewriter.getContext(),
                                         MemorySpace::DeviceL1));
    } else if (!outputL1) {
      // format convert first in L1 first, then write
      assert(inputL1 && "input should guaranteed be in L1 because of the "
                        "previous case");
      bounce(rewriter, op,
             outputLayout.withMemorySpace(rewriter.getContext(),
                                          MemorySpace::DeviceL1));
    } else if (inputLayout.isTiled() != outputLayout.isTiled()) {
      // Prioritize moving tiled data
      if (inputLayout.isTiled()) {
        bounce(rewriter, op,
               outputLayout.withElementType(rewriter.getContext(),
                                            inputLayout.getElementType()));
      } else {
        assert(outputLayout.isTiled());
        bounce(rewriter, op,
               inputLayout.withElementType(rewriter.getContext(),
                                           outputLayout.getElementType()));
      }
    } else if (components.isLayoutChange && inputLayout.isTiled()) {
      // For now to flexibly support layout changes, we need to bounce to scalar
      // first
      bounce(rewriter, op,
             inputLayout.withElementType(rewriter.getContext(),
                                         inputLayout.getScalarElementType()));
    } else if (components.isGridChange) {
      assert(!components.isLayoutChange &&
             "Changing layout and grid at the same time is currently "
             "not supported");
      bounce(rewriter, op,
             outputLayout.withGrid(rewriter.getContext(), outputType,
                                   inputLayout.getGrid()));
    } else if (components.isMemoryLayoutChange) {
      bounce(rewriter, op,
             inputLayout.withMemoryLayout(rewriter.getContext(),
                                          outputLayout.getMemLayout()));
    } else {
      // Note we should eventually support DRAM <-> DRAM, or System <-> System
      // w/ format conversion via streaming supported
      assert(false && "Unsupported compound layout change");
      return failure();
    }

    return success();
  }
};

class TTIRSplitCompoundLayout
    : public impl::TTIRSplitCompoundLayoutBase<TTIRSplitCompoundLayout> {
public:
  using impl::TTIRSplitCompoundLayoutBase<
      TTIRSplitCompoundLayout>::TTIRSplitCompoundLayoutBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRSplitCompoundLayoutRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};

inline uint64_t getElementSizeBytes(Type ty) {
  uint64_t val = 0;
  if (isa<TileType>(ty)) {
    auto tileType = mlir::cast<TileType>(ty);
    val = tileType.getSizeBytes();
  } else {
    val = ty.getIntOrFloatBitWidth() / 8;
  }
  return val;
}

inline uint64_t getMemrefSizeBytes(MemRefType ty) {
  auto shape = ty.getShape();
  uint64_t size = getElementSizeBytes(ty.getElementType());
  return std::accumulate(shape.begin(), shape.end(), size,
                         std::multiplies<uint64_t>());
  return size;
}

inline uint64_t getLayoutMemrefSizeBytes(LayoutAttr layout) {
  return getMemrefSizeBytes(layout.getMemref());
}

inline uint64_t getTensorMemrefSizeBytes(RankedTensorType ty) {
  assert(ty.getEncoding());
  auto layout = mlir::cast<LayoutAttr>(ty.getEncoding());
  return getLayoutMemrefSizeBytes(layout);
}

class TTIRAllocate : public impl::TTIRAllocateBase<TTIRAllocate> {
  struct SimpleAllocator {
    static constexpr uint64_t kBaseAddress = 1llu << 18llu;
    uint64_t addressAlignment;

    SimpleAllocator(uint64_t addressAlignment)
        : addressAlignment(addressAlignment) {}

    SmallVector<uint64_t> currPtr = SmallVector<uint64_t>(
        getMaxEnumValForMemorySpace() + 1llu, kBaseAddress);

    uint64_t allocate(uint64_t size, MemorySpace memorySpace) {
      if (isSystemMemorySpace(memorySpace)) {
        return 0;
      }

      uint32_t index = static_cast<uint32_t>(memorySpace);
      assert(index < currPtr.size());
      uint64_t &ptr = currPtr[index];
      ptr = ttmlir::utils::alignUp(ptr, addressAlignment);
      auto result = ptr;
      ptr += size;
      return result;
    }
  };

public:
  using impl::TTIRAllocateBase<TTIRAllocate>::TTIRAllocateBase;

  std::pair<Operation *, Operation *>
  getStartEndOperationThroughDPSOps(const LivenessBlockInfo *livenessInfo,
                                    Value value) {
    auto *startOp = livenessInfo->getStartOperation(value);
    auto *endOp = livenessInfo->getEndOperation(value, startOp);
    auto *opOperandIter =
        llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
          return opOperand.is(value);
        });
    assert(opOperandIter != endOp->getOpOperands().end());
    while (
        isa<DestinationStyleOpInterface>(endOp) and
        cast<DestinationStyleOpInterface>(endOp).isDpsInit(&(*opOperandIter))) {
      assert(endOp->getResults().size() == 1);
      auto result = endOp->getResult(0);
      endOp = livenessInfo->getEndOperation(result, endOp);
      opOperandIter =
          llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
            return opOperand.is(result);
          });
      assert(opOperandIter != endOp->getOpOperands().end());
    }
    return std::make_pair(startOp, endOp);
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    module->walk([&](func::FuncOp func) {
      assert(func.getBody().hasOneBlock());
      auto systemDesc = getCurrentScopeSystemDesc(func);
      assert(systemDesc);
      auto addressAlignment = systemDesc.getAddressAlignBytes();
      SimpleAllocator allocator(addressAlignment);
      Liveness liveness(func.getOperation());
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(&func.getBody().front());

      mlir::SmallVector<Attribute> argumentAllocations;
      for (auto operand : func.getArguments()) {
        auto operandTy = mlir::cast<RankedTensorType>(operand.getType());
        assert(operandTy.getEncoding());
        auto memorySpace = getMemorySpace(operandTy);
        auto sizeBytes = getTensorMemrefSizeBytes(operandTy);
        auto address = allocator.allocate(sizeBytes, memorySpace);
        argumentAllocations.push_back(rewriter.getAttr<ArgumentAllocationAttr>(
            address, sizeBytes, memorySpace));
      }
      func->setDiscardableAttr(ArgumentAllocationAttr::name,
                               rewriter.getArrayAttr(argumentAllocations));

      func->walk([&](tensor::EmptyOp empty) {
        auto resultTy =
            mlir::cast<RankedTensorType>(empty.getResult().getType());
        assert(resultTy.getEncoding());

        auto [startOp, endOp] =
            getStartEndOperationThroughDPSOps(livenessInfo, empty.getResult());

        // Replace empty with allocate
        auto memorySpace = getMemorySpace(resultTy);
        auto sizeBytes = getTensorMemrefSizeBytes(resultTy);
        auto address = allocator.allocate(sizeBytes, memorySpace);
        rewriter.setInsertionPoint(startOp);
        auto alloc = rewriter.create<AllocOp>(startOp->getLoc(), resultTy,
                                              address, sizeBytes, memorySpace);
        rewriter.replaceOp(empty, alloc);

        // Insert deallocate unless this value is being returned
        if (isa<func::ReturnOp>(endOp)) {
          return;
        }
        rewriter.setInsertionPointAfter(endOp);
        rewriter.create<DeallocOp>(endOp->getLoc(), alloc.getResult());
      });
    });
  }
};

class TTIRGridSet : public impl::TTIRGridSetBase<TTIRGridSet> {
public:
  using impl::TTIRGridSetBase<TTIRGridSet>::TTIRGridSetBase;
  void runOnOperation() final {
    // Currently a placeholder pass for grid size optimization.
    // Goes through all the operations and sets the grid size to max supported
    // by target chip. Lacks:
    // - Constraint checking, whether the grid size is supported by the current
    // OP based on inputs and op type.
    //
    ModuleOp moduleOp = getOperation();

    // Get the max grid size from the system description.
    //
    assert(moduleOp->hasAttr(tt::DeviceAttr::name));
    GridAttr max_grid =
        mlir::cast<tt::DeviceAttr>(moduleOp->getAttr(tt::DeviceAttr::name))
            .getGrid();

    SystemDescAttr systemDesc = mlir::cast<tt::SystemDescAttr>(
        moduleOp->getAttr(tt::SystemDescAttr::name));
    ChipDescAttr chipDesc = systemDesc.getChipDescs()[0];
    llvm::DenseMap<Operation *, std::vector<LayoutAttr>> legalGrids;

    moduleOp->walk([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }

      RankedTensorType tensorType =
          mlir::cast<RankedTensorType>(op->getResult(0).getType());
      LegalGridAnalysis legalGridAnalysis =
          getChildAnalysis<LegalGridAnalysis>(op);
      legalGridAnalysis.init(LegalGridAnalysisInput(
          chipDesc, max_grid, tensorType, &overrideGridSizes));
      legalGrids[op] = legalGridAnalysis.getResult();
    });

    OptimalTargetGridAnalysis optimalTargetGridAnalysis =
        getAnalysis<OptimalTargetGridAnalysis>();
    optimalTargetGridAnalysis.init(
        OptimalTargetGridAnalysisInput(std::move(legalGrids)));

    // Pure application of determined grid sizes to the operations.
    // No further analysis.
    //
    moduleOp->walk([&](func::FuncOp func) {
      SmallVector<Type> funcResultTypes;
      func->walk([&](Operation *op) {
        if (op->getNumResults() == 0) {
          func::ReturnOp funcReturn = dyn_cast<func::ReturnOp>(op);
          if (funcReturn) {
            funcResultTypes.append(funcReturn.getOperandTypes().begin(),
                                   funcReturn.getOperandTypes().end());
          }
          return;
        }

        RankedTensorType tensorType =
            mlir::cast<RankedTensorType>(op->getResult(0).getType());
        llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

        // Update the output layout attribute with the new grid size.
        //
        op->getResult(0).setType(RankedTensorType::get(
            tensorShape, tensorType.getElementType(),
            optimalTargetGridAnalysis.getResult().at(op)));
      });

      // Update the function type to reflect the updated return operation's
      // result types.
      //
      FunctionType funcType = func.getFunctionType();
      FunctionType newFuncType = FunctionType::get(
          func.getContext(), funcType.getInputs(), funcResultTypes);
      func.setType(newFuncType);
    });
  }
};

class TTIRLoadSystemDesc
    : public impl::TTIRLoadSystemDescBase<TTIRLoadSystemDesc> {
public:
  using impl::TTIRLoadSystemDescBase<
      TTIRLoadSystemDesc>::TTIRLoadSystemDescBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    if (not path.empty()) {
      module->setAttr(tt::SystemDescAttr::name,
                      tt::SystemDescAttr::getFromPath(&getContext(), path));
    } else if (not module->hasAttr(tt::SystemDescAttr::name)) {
      module->setAttr(tt::SystemDescAttr::name,
                      tt::SystemDescAttr::getDefault(&getContext()));
    }
  }
};

} // namespace mlir::tt::ttir
