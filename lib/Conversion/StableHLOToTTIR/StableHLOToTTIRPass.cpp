// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/StableHLOToTTIR.h"

#include "ttmlir/Conversion/StableHLOToTTIR/EmptyOpTypeConversion.h"
#include "ttmlir/Conversion/StableHLOToTTIR/ShardyToTTIR.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOTTIR
#include "ttmlir/Conversion/Passes.h.inc"

namespace {

  static bool isZeroSizedTensor(Type ty) {
  if (auto rt = llvm::dyn_cast<RankedTensorType>(ty)) {
    for (int64_t d : rt.getShape())
      if (d == 0) return true;
  }
  return false;
}

static bool hasNonReadMemoryEffects(Operation *op) {
  if (op->hasTrait<OpTrait::HasRecursiveMemoryEffects>()) {
    if (auto mei = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance, 4> effs;
      mei.getEffects(effs);
      for (auto &e : effs)
        if (!isa<MemoryEffects::Read>(e.getEffect()))
          return true;
      return false;
    }
    return true;
  }
  return false;
}

static bool resultIsZeroElementTensor(Value v) {
  auto st = llvm::dyn_cast<ShapedType>(v.getType());
  if (!st || !st.hasRank()) return false;
  for (int64_t d : st.getShape())
    if (d == 0) return true;
  return false;
}

static Value makeZeroElementConst(OpBuilder &b, Location loc, Type ty) {
  auto st = llvm::cast<ShapedType>(ty);
  DenseElementsAttr emptyAttr = DenseElementsAttr::get(st, ArrayRef<Attribute>{});
  return b.create<stablehlo::ConstantOp>(loc, ty, emptyAttr).getResult();
}

struct ConvertStableHLOToTTIRPass
    : public ttir::impl::ConvertStableHLOToTTIRBase<
          ConvertStableHLOToTTIRPass> {

  ConvertStableHLOToTTIRPass() = default;
  ConvertStableHLOToTTIRPass(const ConvertStableHLOToTTIROptions &options)
      : Base(options) {}

  void runOnOperation() final {

      mlir::ModuleOp mod = getOperation();

  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation*> toErase;

    mod.walk([&](Operation *op) {
      if (isa<ModuleOp, func::FuncOp, func::ReturnOp, func::CallOp>(op)) return;
      bool hasZeroInput = llvm::any_of(op->getOperands(), [&](Value v) {
        return isZeroSizedTensor(v.getType());
      });
      if (!hasZeroInput) return;
      if (hasNonReadMemoryEffects(op)) return;

      if (op->getNumResults() == 0 ||
          llvm::all_of(op->getResults(), [](Value r){ return r.use_empty(); })) {
        toErase.push_back(op);
        changed = true;
        return;
      }

      SmallVector<Value> repls;
      repls.reserve(op->getNumResults());
      for (Value r : op->getResults()) {
        if (!resultIsZeroElementTensor(r)) {
          repls.clear();
          return;
        }
      }
      OpBuilder b(op);
      for (Value r : op->getResults())
        repls.push_back(makeZeroElementConst(b, op->getLoc(), r.getType()));

      op->replaceAllUsesWith(repls);
      toErase.push_back(op);
      changed = true;
    });

    for (Operation *op : toErase)
      op->erase();
  }

    mlir::ConversionTarget target(getContext());

    // Common legal/illegal ops/dialects for both partial and full conversion.
    target.addLegalDialect<mlir::quant::QuantDialect>();
    target.addLegalDialect<ttir::TTIRDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addLegalOp<mlir::tt::ttir::EmptyOp>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addIllegalOp<mlir::tensor::EmptyOp>();
    target.addIllegalDialect<mlir::sdy::SdyDialect>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(&getContext());

    addEmptyOpTypeConversionPattern(&getContext(), patterns, typeConverter);
    ::mlir::tt::populateStableHLOToTTIRPatterns(&getContext(), patterns,
                                                typeConverter);
    populateShardyToTTIRPatterns(&getContext(), patterns, typeConverter);

    // Function type conversions.
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    target.addDynamicallyLegalOp<func::CallOp>(
        [&](func::CallOp op) { return typeConverter.isLegal(op); });

    if (enablePartialConversion) {
      // For partial conversion, we can leave stablehlo dialect as neither
      // explicitly legal so the patterns run, nor explicitly illegal so
      // leftover ops won't throw an error.
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns)))) {
        signalPassFailure();
      }
    } else {
      // Full conversion implies stablehlo dialect is fully illegal afterwards.
      target.addIllegalDialect<mlir::stablehlo::StablehloDialect>();

      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        signalPassFailure();
      }
    }
  }
};
} // namespace
} // namespace mlir::tt::ttir

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass() {
  return std::make_unique<ttir::ConvertStableHLOToTTIRPass>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertStableHLOToTTIRPass(
    const ttir::ConvertStableHLOToTTIROptions &options) {
  return std::make_unique<ttir::ConvertStableHLOToTTIRPass>(options);
}

} // namespace mlir::tt
