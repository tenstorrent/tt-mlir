// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTKernel/D2MToTTKernel.h"

#include "ttmlir/Dialect/D2M/Analysis/CBProducerConsumer.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
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

namespace mlir::tt::d2m {

#define GEN_PASS_DEF_CONVERTD2MTOTTKERNEL
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::d2m

namespace {
struct ConvertD2MToTTKernel
    : public d2m::impl::ConvertD2MToTTKernelBase<ConvertD2MToTTKernel> {
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
    target.addIllegalDialect<d2m::D2MDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();

    target.addLegalOp<d2m::ToLayoutOp>();
    target.addLegalOp<d2m::StreamLayoutOp>();
    target.addLegalOp<d2m::ViewLayoutOp>();
    target.addLegalOp<d2m::GenericOp>();
    target.addLegalOp<d2m::EmptyOp>();

    target.addLegalOp<ttir::TTNNMetalLayoutCastOp>();
    target.addLegalOp<ttir::MeshShardOp>();

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
      return !op->hasAttr(d2m::ThreadAttr::name) ||
             (op.getFunctionType().getNumInputs() == 0);
    });

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });
    typeConverter.addConversion([](ttcore::TileType tile) {
      return IndexType::get(tile.getContext());
    });
    typeConverter.addConversion([](d2m::MemTxType memtx) {
      return IndexType::get(memtx.getContext());
    });
    typeConverter.addConversion([](MemRefType memref) -> Type {
      if (ttcore::getMemorySpace(memref) == ttcore::MemorySpace::RegisterDst) {
        return IndexType::get(memref.getContext());
      }
      return ttkernel::CBType::get(memref.getContext(), memref);
    });
    typeConverter.addConversion([](d2m::SemaphoreType semaphore) {
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

    d2m::AssociatedDMAWaits associatedDMAWaits =
        getAnalysis<d2m::AssociatedDMAWaits>();

    d2m::CBProducerConsumer cbProducerConsumer =
        getAnalysis<d2m::CBProducerConsumer>();

    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateD2MToTTKernelPatterns(&getContext(), patterns, typeConverter,
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

std::unique_ptr<OperationPass<ModuleOp>> createConvertD2MToTTKernelPass() {
  return std::make_unique<ConvertD2MToTTKernel>();
}

} // namespace mlir::tt
