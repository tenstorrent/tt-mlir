// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTKernel/TTIRToTTKernel.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/Analysis/CBProducerConsumer.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOTTKERNEL
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttir

namespace {
struct ConvertTTIRToTTKernel
    : public ttir::impl::ConvertTTIRToTTKernelBase<ConvertTTIRToTTKernel> {
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<ttmetal::TTMetalDialect>();
    target.addLegalDialect<ttcore::TTCoreDialect>();
    target.addLegalDialect<ttkernel::TTKernelDialect>();
    target.addLegalDialect<ttnn::TTNNDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<math::MathDialect>();
    target.addIllegalDialect<ttir::TTIRDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();

    target.addLegalOp<ttir::ToLayoutOp>();
    target.addLegalOp<ttir::StreamLayoutOp>();
    target.addLegalOp<ttir::ViewLayoutOp>();
    target.addLegalOp<ttir::GenericOp>();

    // Inputs to matmul_block. Will be folded in this pass.
    target.addLegalOp<memref::CastOp>();
    target.addLegalOp<memref::SubViewOp>();

    // Allow loads and stores to integer element types.
    //   i.e. riscv accesses to L1.
    target.addDynamicallyLegalOp<memref::LoadOp>([&](memref::LoadOp op) {
      return op.getMemRefType().getElementType().isIntOrIndex();
    });
    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      return op.getMemRefType().getElementType().isIntOrIndex();
    });
    target.addLegalOp<memref::AllocOp>();
    target.addLegalOp<memref::DeallocOp>();
    target.addLegalOp<memref::CopyOp>();
    target.addLegalOp<memref::GlobalOp>();
    target.addLegalOp<memref::GetGlobalOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return !op->hasAttr(ttir::ThreadAttr::name) ||
             (op.getFunctionType().getNumInputs() == 0);
    });

    target.addLegalOp<ttir::TTNNMetalLayoutCastOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](ttcore::TileType tile) {
      return IndexType::get(tile.getContext());
    });
    typeConverter.addConversion([](ttir::MemTxType memtx) {
      return IndexType::get(memtx.getContext());
    });
    typeConverter.addConversion([](MemRefType memref) -> Type {
      if (ttcore::getMemorySpace(memref) == ttcore::MemorySpace::RegisterDst) {
        return IndexType::get(memref.getContext());
      }
      return ttkernel::CBType::get(memref.getContext(), memref);
    });
    typeConverter.addConversion([](ttir::SemaphoreType semaphore) {
      return ttkernel::SemaphoreType::get(semaphore.getContext());
    });

    auto materializeAsUnrealizedCast = [](OpBuilder &builder, Type resultType,
                                          ValueRange inputs,
                                          Location loc) -> Value {
      if (inputs.size() != 1) {
        return Value();
      }
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeAsUnrealizedCast);
    typeConverter.addTargetMaterialization(materializeAsUnrealizedCast);

    ttir::AssociatedDMAWaits associatedDMAWaits =
        getAnalysis<ttir::AssociatedDMAWaits>();

    ttir::CBProducerConsumer cbProducerConsumer =
        getAnalysis<ttir::CBProducerConsumer>();

    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateTTIRToTTKernelPatterns(&getContext(), patterns, typeConverter,
                                   associatedDMAWaits, cbProducerConsumer);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  };
};
} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTKernelPass() {
  return std::make_unique<ConvertTTIRToTTKernel>();
}

} // namespace mlir::tt
