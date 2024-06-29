// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "llvm/ADT/ScopeExit.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include "ttmlir/Dialect/TTNN/Passes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_CONVERTTTNNTOEMITC
#include "ttmlir/Dialect/TTNN/Passes.h.inc"

class TTNNToEmitCTypeConverter : public TypeConverter {
public:
  TTNNToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](DeviceType type) -> Type {
      return emitc::OpaqueType::get(ctx, "ttnn::Device");
    });
    addConversion([ctx](TensorType type) -> Type {
      return emitc::OpaqueType::get(ctx, "ttnn::Tensor");
    });
  }
};

class TTNNToEmitCTypeRewriter : public RewritePattern {
public:
  TTNNToEmitCTypeRewriter(const TypeConverter &converter, MLIRContext *ctx)
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

template <typename OpTy>
class TTNNToEmitCOpaqueRewriter : public OpRewritePattern<OpTy> {
public:
  using OpRewritePattern<OpTy>::OpRewritePattern;

  std::string getOpName(OpTy op) const {
    auto name = op.getOperation()->getName().getStringRef();
    if (name.starts_with("ttnn.")) {
      name = name.drop_front(5);
    }
    return "ttnn::" + name.str();
  }

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, op->getResultTypes(), getOpName(op), nullptr, nullptr,
        op->getOperands());
    return success();
  }
};

class ConvertTTNNToEmitC
    : public impl::ConvertTTNNToEmitCBase<ConvertTTNNToEmitC> {
public:
  using impl::ConvertTTNNToEmitCBase<
      ConvertTTNNToEmitC>::ConvertTTNNToEmitCBase;

  void runOnOperation() final {
    {
      TTNNToEmitCTypeConverter typeConverter(&getContext());
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNToEmitCTypeRewriter>(typeConverter, &getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
        return;
      }
    }

    {
      RewritePatternSet patterns(&getContext());
      patterns.add<TTNNToEmitCOpaqueRewriter<OpenDeviceOp>,
                   TTNNToEmitCOpaqueRewriter<FullOp>,
                   TTNNToEmitCOpaqueRewriter<ToMemoryConfigOp>,
                   TTNNToEmitCOpaqueRewriter<MultiplyOp>,
                   TTNNToEmitCOpaqueRewriter<MatmulOp>,
                   TTNNToEmitCOpaqueRewriter<CloseDeviceOp>>(&getContext());
      FrozenRewritePatternSet patternSet(std::move(patterns));
      if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
        signalPassFailure();
      }
    }

    {
      auto module = getOperation();
      OpBuilder builder(module);
      module.getBody()->push_front(builder.create<emitc::IncludeOp>(
          module.getLoc(), "ttnn/device.h", /*isStandard=*/false));
      module.getBody()->push_front(builder.create<emitc::IncludeOp>(
          module.getLoc(), "ttnn/operations/eltwise/binary/binary.hpp",
          /*isStandard=*/false));
      module.getBody()->push_front(builder.create<emitc::IncludeOp>(
          module.getLoc(), "ttnn/operations/core.hpp", /*isStandard=*/false));
      module.getBody()->push_front(builder.create<emitc::IncludeOp>(
          module.getLoc(), "ttnn/operations/creation.hpp",
          /*isStandard=*/false));
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttnn::TTNNDialect>();
    registry.insert<emitc::EmitCDialect>();
  }
};

LogicalResult emitTTNNAsCpp(ModuleOp origOp, llvm::raw_ostream &os) {
  ModuleOp op = cast<ModuleOp>(origOp->clone());
  auto cleanupDispatchClone = llvm::make_scope_exit([&op] { op->erase(); });

  auto pm = PassManager::on<ModuleOp>(op.getContext());
  pm.addPass(createConvertTTNNToEmitC());

  if (pm.run(op).failed()) {
    return failure();
  }

  if (emitc::translateToCpp(op, os).failed()) {
    return failure();
  }

  return success();
}
} // namespace mlir::tt::ttnn
