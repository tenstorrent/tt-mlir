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

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERIC
#define GEN_PASS_DEF_TTIRGENERICREGIONOPERANDSTOMEMREF
#define GEN_PASS_DEF_TTIRLAYOUT
#define GEN_PASS_DEF_TTIRALLOCATE
#define GEN_PASS_DEF_TTIRGRIDSET
#define GEN_PASS_DEF_TTIRIMPLICITDEVICE
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
                              systemDesc.cast<tt::SystemDescAttr>()));
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
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported Tosa operation for TTIR");
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
      return false;
    }
    auto rank = operands[0].getType().cast<RankedTensorType>().getRank();
    for (auto operand : operands) {
      if (operand.getType().cast<RankedTensorType>().getRank() != rank) {
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
    auto rank = operands[0].getType().cast<RankedTensorType>().getRank();
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
    auto rank = operands[0].getType().cast<RankedTensorType>().getRank();
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
    // Test if this generic op has already been lowered, todo find a better way
    if (op.getOperation()->getParentOp()->getName() ==
        OperationName("ttir.generic", rewriter.getContext())) {
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

class TTIRGeneric : public impl::TTIRGenericBase<TTIRGeneric> {
public:
  using impl::TTIRGenericBase<TTIRGeneric>::TTIRGenericBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRLinalgGenericRewriter, TTIRKernelGenericRewriter,
                 TTIRNamedToKernelRewriter<AddOp>,
                 TTIRNamedToKernelRewriter<MultiplyOp>,
                 TTIRNamedToKernelRewriter<SubtractOp>,
                 TTIRNamedToKernelRewriter<GreaterEqualOp>,
                 TTIRNamedToKernelRewriter<ReluOp>>(&getContext());
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

class TTIRGenericOperandsToMemrefRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    auto dpsInterface = cast<DestinationStyleOpInterface>(op.getOperation());
    for (auto &operand : op->getOpOperands()) {
      auto encoding = operand.get()
                          .getType()
                          .template cast<RankedTensorType>()
                          .getEncoding();
      if (not encoding) {
        return failure(); // Hasn't been type converted yet
      }

      auto blockArg = op.getRegion().getArgument(operand.getOperandNumber());
      if (blockArg.getType().isa<MemRefType>()) {
        return failure(); // Already lowered
      }

      // Rewire the operand to the layout op
      rewriter.modifyOpInPlace(op, [&]() {
        auto layout = operand.get()
                          .getType()
                          .template cast<RankedTensorType>()
                          .getEncoding()
                          .template cast<LayoutAttr>();

        blockArg.setType(layout.getMemref());

        bool isResult = dpsInterface.isDpsInit(&operand);
        if (isResult) {
          for (auto *user : blockArg.getUsers()) {
            dyn_cast<KernelOp>(user).getResult(0).setType(layout.getMemref());
          }
        }
      });
    }

    return success();
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
    patterns.add<TTIRGenericOperandsToMemrefRewriter>(&getContext());
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

inline MemorySpace getMemorySpace(MemRefType memref) {
  return memref.getMemorySpace().template cast<MemorySpaceAttr>().getValue();
}

inline MemorySpace getMemorySpace(LayoutAttr layout) {
  return getMemorySpace(layout.getMemref());
}

inline MemorySpace getMemorySpace(RankedTensorType ty) {
  assert(ty.getEncoding());
  auto layout = ty.getEncoding().template cast<LayoutAttr>();
  return getMemorySpace(layout);
}

inline MemorySpace uppermostMemorySpace(OperandConstraint operandConstraint) {
  if (bitEnumContainsAny(operandConstraint, OperandConstraint::L1)) {
    return MemorySpace::DeviceL1;
  }
  if (bitEnumContainsAny(operandConstraint, OperandConstraint::DRAM)) {
    return MemorySpace::DeviceDRAM;
  }
  return MemorySpace::System;
}

class TTIRLayoutTensorTypeConverter : public TypeConverter {
public:
  TTIRLayoutTensorTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](RankedTensorType type) -> Type {
      auto layout = type.getEncoding();
      if (layout) {
        return type;
      }
      auto newLayout = LayoutAttr::get(ctx, type);
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
createLayoutOp(PatternRewriter &rewriter, Location loc, Value input,
               OperandConstraint operandConstraint) {
  auto ty = input.getType().cast<RankedTensorType>();
  auto currLayout = ty.getEncoding().cast<LayoutAttr>();
  auto currMemorySpace = currLayout.getMemorySpace();
  auto desiredMemorySpace = uppermostMemorySpace(operandConstraint);
  if (currMemorySpace == desiredMemorySpace) {
    return std::nullopt;
  }

  auto desiredLayout = rewriter.getAttr<LayoutAttr>(ty, desiredMemorySpace);
  auto output = rewriter.create<tensor::EmptyOp>(
      loc, ty.getShape(), ty.getElementType(), desiredLayout);

  tensor::EmptyOp exising_empty = input.getDefiningOp<tensor::EmptyOp>();
  if (exising_empty) {
    rewriter.replaceOp(exising_empty, output);
    return output.getResult();
  }
  return rewriter.create<ttir::LayoutOp>(loc, output.getType(), input, output)
      ->getResult(0);
}

template <typename TTIROpTy>
class TTIRLayoutOperandsRewriter : public OpRewritePattern<TTIROpTy> {
public:
  using OpRewritePattern<TTIROpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTIROpTy op,
                                PatternRewriter &rewriter) const final {
    assert(op->template hasTrait<TTIROp::Trait>());
    auto dpsInterface = cast<DestinationStyleOpInterface>(op.getOperation());
    bool modified = false;
    for (auto &operand : op->getOpOperands()) {
      bool isResult = dpsInterface.isDpsInit(&operand);
      auto encoding = operand.get()
                          .getType()
                          .template cast<RankedTensorType>()
                          .getEncoding();
      if (not encoding) {
        return failure(); // Hasn't been type converted yet
      }

      auto operandConstraint =
          op.getOperandConstraints()[operand.getOperandNumber()]
              .template cast<OperandConstraintAttr>()
              .getValue();
      auto desiredLayout = createLayoutOp(rewriter, op.getLoc(), operand.get(),
                                          operandConstraint);

      if (desiredLayout) {
        rewriter.modifyOpInPlace(op, [&]() {
          modified = true;
          op->setOperand(operand.getOperandNumber(), *desiredLayout);
          std::optional<BlockArgument> blockArg;
          if (op->getNumRegions() == 1) {
            blockArg = op->getRegion(0).getArgument(operand.getOperandNumber());
            blockArg->setType(desiredLayout->getType());
          }
          if (isResult) {
            // If this is the output operand, update the result type
            op->getResult(0).setType(desiredLayout->getType());
            if (blockArg) {
              for (auto *user : blockArg->getUsers()) {
                dyn_cast<KernelOp>(user).getResult(0).setType(
                    desiredLayout->getType());
              }
            }
          }
        });
      }
    }

    return modified ? success() : failure();
  }
};

class TTIRLayoutFuncReturnRewriter
    : public OpRewritePattern<mlir::func::ReturnOp> {
public:
  using OpRewritePattern<mlir::func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    bool modified = false;
    for (auto &operand : op->getOpOperands()) {
      if (auto layout = createLayoutOp(rewriter, op.getLoc(), operand.get(),
                                       OperandConstraint::System);
          layout) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.setOperand(operand.getOperandNumber(), *layout); });
        modified = true;
      }
    }
    return modified ? success() : failure();
  }
};

class TTIRLayout : public impl::TTIRLayoutBase<TTIRLayout> {
public:
  using impl::TTIRLayoutBase<TTIRLayout>::TTIRLayoutBase;

  void runOnOperation() final {
    {
      TTIRLayoutTensorTypeConverter typeConverter(&getContext());
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
      patterns.add<
          TTIRLayoutOperandsRewriter<GenericOp>,
          TTIRLayoutOperandsRewriter<AddOp>,
          TTIRLayoutOperandsRewriter<MultiplyOp>,
          TTIRLayoutOperandsRewriter<SubtractOp>,
          TTIRLayoutOperandsRewriter<GreaterEqualOp>,
          TTIRLayoutOperandsRewriter<ReluOp>, TTIRLayoutOperandsRewriter<SumOp>,
          TTIRLayoutOperandsRewriter<SoftmaxOp>,
          TTIRLayoutOperandsRewriter<MatmulOp>, TTIRLayoutFuncReturnRewriter>(
          &getContext());
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

inline uint64_t getElementSizeBytes(Type ty) {
  assert(ty.isF32() && "Only support f32 for now");
  return 4;
}

inline uint64_t getMemrefSizeBytes(MemRefType ty) {
  auto shape = ty.getShape();
  uint64_t size = getElementSizeBytes(ty.getElementType());
  return std::accumulate(shape.begin(), shape.end(), size,
                         std::multiplies<uint64_t>());
  return size;
}

inline uint64_t getLayoutSizeBytes(LayoutAttr layout) {
  auto gridShape = layout.getGrid().getShape();
  auto gridVolume = std::accumulate(gridShape.begin(), gridShape.end(), 1,
                                    std::multiplies<uint64_t>());
  assert(gridVolume == 1 && "Only support grid shape of 1 for now");
  return getMemrefSizeBytes(layout.getMemref());
}

inline uint64_t getTensorSizeBytes(RankedTensorType ty) {
  assert(ty.getEncoding());
  auto layout = ty.getEncoding().template cast<LayoutAttr>();
  return getLayoutSizeBytes(layout);
}

class TTIRAllocate : public impl::TTIRAllocateBase<TTIRAllocate> {
  struct SimpleAllocator {
    static constexpr uint64_t kBaseAddress = 1llu << 18llu;

    SmallVector<uint64_t> currPtr = SmallVector<uint64_t>(
        getMaxEnumValForMemorySpace() + 1llu, kBaseAddress);

    uint64_t alignUp(uint64_t ptr, uint64_t alignment) {
      return (ptr + alignment - 1) & ~(alignment - 1);
    }

    uint64_t allocate(uint64_t size, MemorySpace memorySpace) {
      if (isSystemMemorySpace(memorySpace)) {
        return 0;
      }

      uint32_t index = static_cast<uint32_t>(memorySpace);
      assert(index < currPtr.size());
      uint64_t &ptr = currPtr[index];
      ptr = alignUp(ptr, 16);
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
      SimpleAllocator allocator;
      Liveness liveness(func.getOperation());
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(&func.getBody().front());
      func->walk([&](tensor::EmptyOp empty) {
        auto resultTy =
            empty.getResult().getType().template cast<RankedTensorType>();
        assert(resultTy.getEncoding());

        auto [startOp, endOp] =
            getStartEndOperationThroughDPSOps(livenessInfo, empty.getResult());

        // Replace empty with allocate
        auto memorySpace = getMemorySpace(resultTy);
        auto sizeBytes = getTensorSizeBytes(resultTy);
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
    GridAttr max_grid = moduleOp->getAttr(tt::DeviceAttr::name)
                            .cast<tt::DeviceAttr>()
                            .getGrid();

    SystemDescAttr systemDesc =
        moduleOp->getAttr(tt::SystemDescAttr::name).cast<tt::SystemDescAttr>();
    ChipDescAttr chipDesc = systemDesc.getChipDescs()[0];
    llvm::DenseMap<Operation *, std::vector<GridAttr>> legalGrids;

    moduleOp->walk([&](Operation *op) {
      if (op->getNumResults() == 0) {
        return;
      }

      RankedTensorType tensorType =
          op->getResult(0).getType().template cast<RankedTensorType>();
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
            op->getResult(0).getType().template cast<RankedTensorType>();
        LayoutAttr layout =
            tensorType.getEncoding().template cast<LayoutAttr>();
        llvm::ArrayRef<int64_t> tensorShape = tensorType.getShape();

        // Update the output layout attribute with the new grid size.
        //
        op->getResult(0).setType(RankedTensorType::get(
            tensorShape, tensorType.getElementType(),
            layout.withGrid(&getContext(), tensorShape,
                            optimalTargetGridAnalysis.getResult().at(op))));
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

} // namespace mlir::tt::ttir
