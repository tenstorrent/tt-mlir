// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <iostream>

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
#include "ttmlir/Dialect/TT/TTDialect.h"
#include "ttmlir/Dialect/TT/TTOpsTypes.h"

#include "ttmlir/Dialect/TTIR/TTIRPasses.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_CONVERTTOSATOTTIR
#define GEN_PASS_DEF_TTIRDISPATCH
#define GEN_PASS_DEF_TTIRLAYOUT
#define GEN_PASS_DEF_TTIRSHARD
#define GEN_PASS_DEF_TTIRALLOCATE
#include "ttmlir/Dialect/TTIR/TTIRPasses.h.inc"

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

class ConvertTosaToTTIR : public impl::ConvertTosaToTTIRBase<ConvertTosaToTTIR> {
public:
  using impl::ConvertTosaToTTIRBase<ConvertTosaToTTIR>::ConvertTosaToTTIRBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TosaToTTIRKernelRewriter<tosa::MulOp>,
                 TosaToTTIRKernelRewriter<tosa::MatMulOp>>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
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
    if (operands.empty())
      return false;
    auto rank = operands[0].getType().cast<RankedTensorType>().getRank();
    for (auto operand : operands) {
      if (operand.getType().cast<RankedTensorType>().getRank() != rank)
        return false;
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
    } else if (kind == "matmul") {
      return createMatmulIndexingMaps(rewriter, operands);
    } else {
      llvm_unreachable("Unsupported kernel kind");
    }
  }

  static ArrayAttr createOperandConstraints(PatternRewriter &rewriter,
                                            StringRef kind,
                                            mlir::OperandRange operands) {
    auto numOperands = operands.size();
    if (kind == "eltwise") {
      return rewriter.getArrayAttr(SmallVector<Attribute>(
          numOperands, rewriter.getAttr<OperandConstraintAttr>(
                           OperandConstraint::AnyDevice)));
    } else if (kind == "matmul") {
      return rewriter.getArrayAttr(SmallVector<Attribute>(
          numOperands, rewriter.getAttr<OperandConstraintAttr>(
                           OperandConstraint::AnyDeviceTile)));
    } else {
      llvm_unreachable("Unsupported kernel kind");
    }
  }

  LogicalResult matchAndRewrite(KernelOp op,
                                PatternRewriter &rewriter) const final {
    // Test if this generic op has already been lowered, todo find a better way
    if (op.getOperation()->getParentOp()->getName() ==
        OperationName("ttir.dispatch", rewriter.getContext()))
      return failure();

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
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
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
  if (bitEnumContainsAny(operandConstraint, OperandConstraint::L1))
    return MemorySpace::L1;
  else if (bitEnumContainsAny(operandConstraint, OperandConstraint::DRAM))
    return MemorySpace::DRAM;
  return MemorySpace::System;
}

inline SmallVector<int64_t> canonicalStride(::llvm::ArrayRef<int64_t> shape) {
  SmallVector<int64_t> stride;
  stride.push_back(1);
  for (auto iter = shape.rbegin(); iter != shape.rend(); ++iter) {
    stride.insert(stride.begin(),
                  stride.front() * *iter); // TODO: alignup 16B
  }
  return stride;
}

inline LayoutAttr
canonicalLayoutAttr(PatternRewriter &rewriter, RankedTensorType ty,
                    MemorySpace memorySpace = MemorySpace::System) {
  auto map = rewriter.getMultiDimIdentityMap(ty.getRank());
  auto memref =
      MemRefType::get(ty.getShape(), ty.getElementType(), map,
                      rewriter.getAttr<MemorySpaceAttr>(memorySpace));
  return rewriter.getAttr<LayoutAttr>(canonicalStride(ty.getShape()),
                                      OOBVal::Undef,
                                      rewriter.getAttr<GridAttr>(), memref);
}

class TTIRLayoutRewriter : public OpRewritePattern<DispatchOp> {
public:
  using OpRewritePattern<DispatchOp>::OpRewritePattern;

  std::optional<ttir::LayoutOp>
  createLayoutOp(PatternRewriter &rewriter, Location loc, Value input,
                 OperandConstraint operandConstraint) const {
    auto ty = input.getType().cast<RankedTensorType>();
    if (ty.getEncoding()) {
      assert(ty.getEncoding().isa<LayoutAttr>());
      return std::nullopt;
    }

    auto layoutEncoding = canonicalLayoutAttr(
        rewriter, ty, uppermostMemorySpace(operandConstraint));
    auto output = rewriter.create<tensor::EmptyOp>(
        loc, ty.getShape(), ty.getElementType(), layoutEncoding);

    tensor::EmptyOp exising_empty = input.getDefiningOp<tensor::EmptyOp>();
    if (exising_empty) {
      rewriter.replaceOp(exising_empty, output);
      return std::nullopt;
    } else {
      // Bounce it through casted system layout
      auto systemLayoutEncoding = canonicalLayoutAttr(rewriter, ty);
      auto systemOutput = rewriter.create<tensor::EmptyOp>(
          loc, ty.getShape(), ty.getElementType(), systemLayoutEncoding);
      auto bounce = rewriter.create<ttir::LayoutOp>(
          loc, systemOutput.getType(), input, systemOutput, true /*isCast*/);
      return rewriter.create<ttir::LayoutOp>(loc, output.getType(), bounce,
                                             output, false /*isCast*/);
    }
  }

  LogicalResult matchAndRewrite(DispatchOp op,
                                PatternRewriter &rewriter) const final {
    int operandIndex = 0;
    bool modified = false;
    for (auto operand : op.getOperands()) {
      auto operandConstraint = op.getOperandConstraints()[operandIndex]
                                   .template cast<OperandConstraintAttr>()
                                   .getValue();
      if (auto layout =
              createLayoutOp(rewriter, op.getLoc(), operand, operandConstraint);
          layout) {
        rewriter.modifyOpInPlace(
            op, [&]() { op.setOperand(operandIndex, *layout); });
        modified = true;
      }
      ++operandIndex;
    }

    // Update the region arguments to use the memref type
    if (modified) {
      operandIndex = 0;
      for (auto operand : op.getOperands()) {
        rewriter.modifyOpInPlace(op, [&]() {
          auto memref = operand.getType()
                            .cast<RankedTensorType>()
                            .getEncoding()
                            .cast<LayoutAttr>()
                            .getMemref();
          auto blockArg = op.getRegion().getArgument(operandIndex);
          blockArg.setType(memref);
          for (auto user : blockArg.getUsers()) {
            // This is kind of hacky, the last operand is the output so it'll be
            // last in setting the result type
            dyn_cast<KernelOp>(user).getResult(0).setType(memref);
          }

          // This is kind of hacky, the last operand is the output so it'll be
          // last in setting the result type
          op.getResult(0).setType(operand.getType());
        });
        ++operandIndex;
      }
    }

    return modified ? success() : failure();
  }
};

class TTIRUnlayoutFuncReturnRewriter
    : public OpRewritePattern<mlir::func::ReturnOp> {
public:
  using OpRewritePattern<mlir::func::ReturnOp>::OpRewritePattern;

  std::optional<Value> createUnlayout(PatternRewriter &rewriter, Location loc,
                                      Value input) const {
    auto ty = input.getType().cast<RankedTensorType>();
    if (not ty.getEncoding()) {
      return std::nullopt;
    }
    assert(ty.getEncoding().isa<LayoutAttr>());
    auto layout = ty.getEncoding().template cast<LayoutAttr>();
    auto output = rewriter.create<tensor::EmptyOp>(loc, ty.getShape(),
                                                   ty.getElementType());

    // Bounce it though system layout if coming from device
    if (layout.getMemref()
            .getMemorySpace()
            .template cast<MemorySpaceAttr>()
            .getValue() != MemorySpace::System) {
      auto systemLayout = canonicalLayoutAttr(rewriter, ty);
      auto systemOutput = rewriter.create<tensor::EmptyOp>(
          loc, ty.getShape(), ty.getElementType(), systemLayout);
      input = rewriter.create<ttir::LayoutOp>(loc, systemOutput.getType(),
                                              input, systemOutput, false);
    }
    return rewriter.create<ttir::LayoutOp>(loc, output.getType(), input, output,
                                           true);
  }

  LogicalResult matchAndRewrite(mlir::func::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    int operandIndex = 0;
    bool modified = false;
    for (auto operand : op.getOperands()) {
      if (auto layout = createUnlayout(rewriter, op.getLoc(), operand);
          layout) {
        rewriter.modifyOpInPlace(op, [&]() {
          op.setOperand(operandIndex, *layout);
        });
        modified = true;
      }
      ++operandIndex;
    }
    return modified ? success() : failure();
  }
};

class TTIRLayout : public impl::TTIRLayoutBase<TTIRLayout> {
public:
  using impl::TTIRLayoutBase<TTIRLayout>::TTIRLayoutBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRLayoutRewriter, TTIRUnlayoutFuncReturnRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
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
      if (memorySpace == MemorySpace::System)
        return 0;

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
    auto opOperandIter =
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
        // Skip for tensors without layout
        if (not resultTy.getEncoding())
          return;

        auto [startOp, endOp] =
            getStartEndOperationThroughDPSOps(livenessInfo, empty.getResult());

        // Replace empty with allocate
        auto sizeBytes = getTensorSizeBytes(resultTy);
        auto address = allocator.allocate(sizeBytes, getMemorySpace(resultTy));
        rewriter.setInsertionPoint(startOp);
        auto alloc =
            rewriter.create<AllocOp>(startOp->getLoc(), resultTy, address,
                                     sizeBytes, getMemorySpace(resultTy));
        rewriter.replaceOp(empty, alloc);

        // Insert deallocate
        rewriter.setInsertionPointAfter(endOp);
        rewriter.create<DeallocOp>(endOp->getLoc(), alloc.getResult());
      });
    });
  }
};

} // namespace mlir::tt::ttir
