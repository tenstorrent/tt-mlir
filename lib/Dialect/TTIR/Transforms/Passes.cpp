// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
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

#include "ttmlir/Dialect/TTIR/Passes.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_CONVERTTOSATOTTIR
#define GEN_PASS_DEF_TTIRDISPATCH
#define GEN_PASS_DEF_TTIRLAYOUT
#define GEN_PASS_DEF_TTIRSHARD
#define GEN_PASS_DEF_TTIRALLOCATE
#include "ttmlir/Dialect/TTIR/Passes.h.inc"

template <typename TosaOp>
class TosaToTTIRKernelRewriter : public OpRewritePattern<TosaOp> {
public:
  using OpRewritePattern<TosaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TosaOp op,
                                PatternRewriter &rewriter) const final {
    StringRef kernelName;
    StringRef kernelKind;
    if constexpr (std::is_same<TosaOp, tosa::MulOp>::value) {
      assert(op.getShift() == 0);
      kernelName = "mulitply";
      kernelKind = "eltwise";
    } else if constexpr (std::is_same<TosaOp, tosa::MatMulOp>::value) {
      kernelName = "matmul";
      kernelKind = "matmul";
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "Unsupported Tosa operation for TTIR");
    }
    assert(kernelName.size() > 0);

    // Create empty output tensor for destination passing style (DPS)
    auto outputType =
        op.getResult().getType().template cast<RankedTensorType>();
    auto output = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), outputType.getShape(), outputType.getElementType());

    auto kernel = rewriter.create<ttir::KernelOp>(
        op.getLoc(), TypeRange(output.getType()), kernelName, kernelKind,
        op.getOperands(), ValueRange(output));

    rewriter.replaceOp(op, kernel);

    return success();
  }
};

class ConvertTosaToTTIR
    : public impl::ConvertTosaToTTIRBase<ConvertTosaToTTIR> {
public:
  using impl::ConvertTosaToTTIRBase<ConvertTosaToTTIR>::ConvertTosaToTTIRBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TosaToTTIRKernelRewriter<tosa::MulOp>,
                 TosaToTTIRKernelRewriter<tosa::MatMulOp>>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
      signalPassFailure();
    }
  }
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

class TTIRLinalgGenericToDispatchRewriter
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    return failure();
  }
};

class TTIRKernelDispatchRewriter : public OpRewritePattern<KernelOp> {
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
        OperationName("ttir.dispatch", rewriter.getContext())) {
      return failure();
    }

    // Create a dispatch op
    auto [indexingMaps, iteratorTypes] =
        createIndexingMaps(rewriter, op.getKind(), op.getOperands());
    auto constraints =
        createOperandConstraints(rewriter, op.getKind(), op.getOperands());
    auto dispatch = rewriter.create<ttir::DispatchOp>(
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

class TTIRDispatch : public impl::TTIRDispatchBase<TTIRDispatch> {
public:
  using impl::TTIRDispatchBase<TTIRDispatch>::TTIRDispatchBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<TTIRLinalgGenericToDispatchRewriter, TTIRKernelDispatchRewriter>(
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

inline SmallVector<int64_t> canonicalStride(::llvm::ArrayRef<int64_t> shape) {
  SmallVector<int64_t> stride;
  stride.push_back(1);
  for (auto iter = shape.rbegin(); iter != shape.rend(); ++iter) {
    stride.insert(stride.begin(),
                  stride.front() * *iter); // TODO(nsmith): alignup 16B
  }
  return stride;
}

inline LayoutAttr
canonicalLayoutAttr(MLIRContext *ctx, RankedTensorType ty,
                    MemorySpace memorySpace = MemorySpace::System) {
  auto map = AffineMap::getMultiDimIdentityMap(ty.getRank(), ctx);
  auto memref = MemRefType::get(ty.getShape(), ty.getElementType(), map,
                                MemorySpaceAttr::get(ctx, memorySpace));
  return LayoutAttr::get(ctx, canonicalStride(ty.getShape()), OOBVal::Undef,
                         GridAttr::get(ctx), memref);
}

inline LayoutAttr
canonicalLayoutAttr(PatternRewriter &rewriter, RankedTensorType ty,
                    MemorySpace memorySpace = MemorySpace::System) {
  auto map = rewriter.getMultiDimIdentityMap(ty.getRank());
  auto memref = MemRefType::get(ty.getShape(), ty.getElementType(), map,
                                rewriter.getAttr<MemorySpaceAttr>(memorySpace));
  return rewriter.getAttr<LayoutAttr>(canonicalStride(ty.getShape()),
                                      OOBVal::Undef,
                                      rewriter.getAttr<GridAttr>(), memref);
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
      auto newLayout = canonicalLayoutAttr(ctx, type);
      return RankedTensorType::get(type.getShape(), type.getElementType(),
                                   newLayout);
    });
  }
};

class TTIRLayoutTensorTypeRewriter : public RewritePattern {
public:
  TTIRLayoutTensorTypeRewriter(const TypeConverter &converter, MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        converter(converter) {}

  template <typename ValueRange>
  bool convertTypes(ValueRange valueRange, SmallVector<Type> &newTypes) const {
    bool updated = false;
    auto result = converter.convertTypes(valueRange.getTypes(), newTypes);
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
      ty = converter.convertType(ty);
    }
    for (Type &ty : outputTypes) {
      ty = converter.convertType(ty);
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

  const TypeConverter &converter;
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

  auto desiredLayout = canonicalLayoutAttr(rewriter, ty, desiredMemorySpace);
  auto output = rewriter.create<tensor::EmptyOp>(
      loc, ty.getShape(), ty.getElementType(), desiredLayout);

  tensor::EmptyOp exising_empty = input.getDefiningOp<tensor::EmptyOp>();
  if (exising_empty) {
    rewriter.replaceOp(exising_empty, output);
    return std::nullopt;
  }
  return rewriter.create<ttir::LayoutOp>(loc, output.getType(), input, output)
      ->getResult(0);
}

class TTIRLayoutDispatchOperandsRewriter : public OpRewritePattern<DispatchOp> {
public:
  using OpRewritePattern<DispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DispatchOp op,
                                PatternRewriter &rewriter) const final {
    auto dpsInterface = cast<DestinationStyleOpInterface>(op.getOperation());
    for (auto &operand : op->getOpOperands()) {
      auto encoding =
          operand.get().getType().cast<RankedTensorType>().getEncoding();
      if (not encoding) {
        return failure(); // Hasn't been type converted yet
      }

      auto blockArg = op.getRegion().getArgument(operand.getOperandNumber());
      if (blockArg.getType().isa<MemRefType>()) {
        return failure(); // Already lowered
      }

      auto operandConstraint =
          op.getOperandConstraints()[operand.getOperandNumber()]
              .template cast<OperandConstraintAttr>()
              .getValue();
      auto desiredLayout = createLayoutOp(rewriter, op.getLoc(), operand.get(),
                                          operandConstraint);

      if (desiredLayout) {
        rewriter.modifyOpInPlace(op, [&]() {
          op.setOperand(operand.getOperandNumber(), *desiredLayout);
        });
      }

      // Rewire the operand to the layout op
      rewriter.modifyOpInPlace(op, [&]() {
        auto layout = operand.get()
                          .getType()
                          .cast<RankedTensorType>()
                          .getEncoding()
                          .cast<LayoutAttr>();

        // Update the block arguments to use the underlying memref type
        blockArg.setType(layout.getMemref());

        bool isResult = dpsInterface.isDpsInit(&operand);
        if (isResult) {
          // If this is the output operand, update the result type for both
          // the kernel return type and the dispatch return type
          op.getResult(0).setType(operand.get().getType());
          for (auto *user : blockArg.getUsers()) {
            dyn_cast<KernelOp>(user).getResult(0).setType(layout.getMemref());
          }
        }
      });
    }

    return success();
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
      patterns.add<TTIRLayoutDispatchOperandsRewriter,
                   TTIRLayoutFuncReturnRewriter>(&getContext());
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

class TTIRShard : public impl::TTIRShardBase<TTIRShard> {
public:
  using impl::TTIRShardBase<TTIRShard>::TTIRShardBase;

  void runOnOperation() final { assert(false); }
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

} // namespace mlir::tt::ttir
