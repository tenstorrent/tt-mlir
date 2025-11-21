// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTKernel/D2MToTTKernel.h"

#include "ttmlir/Dialect/D2M/Analysis/CBProducerConsumer.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
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

  using Base = d2m::impl::ConvertD2MToTTKernelBase<ConvertD2MToTTKernel>;

  ConvertD2MToTTKernel() = default;
  ConvertD2MToTTKernel(
      const mlir::tt::d2m::ConvertD2MToTTKernelOptions &options)
      : Base(options) {}

  ConvertD2MToTTKernel(const ConvertD2MToTTKernel &rhs) : Base(rhs) {
    // Workaround: Passes are required to be copy-constructible but autogen'ed
    // base class copy constructors ignore Pass option fields.
    this->ttnnMode = rhs.ttnnMode;
  }

  // TODO(wenbinlyuTT): remove this WA once unary_bcast DST issue is fixed.
  // If a compute op consumes the output of a unary bcast, that unary bcast must
  // happen before all copy_tile calls whose outputs are also consumed by that
  // compute op.
  void reorderUnaryBcastOps(func::FuncOp func) {
    func.walk([&](Block *block) {
      for (Operation &op : *block) {
        // Collect all unary_bcast(_init).
        SmallVector<
            std::pair<ttkernel::UnaryBcastInitOp, ttkernel::UnaryBcastTileOp>>
            bcastOps;
        if (auto bcastOp = mlir::dyn_cast<ttkernel::UnaryBcastTileOp>(op)) {
          auto bcastInitOp =
              mlir::cast<ttkernel::UnaryBcastInitOp>(bcastOp->getPrevNode());
          bcastOps.emplace_back(bcastInitOp, bcastOp);
        }

        for (auto [bcastInitOp, bcastOp] : bcastOps) {
          Value bcastDstIdx = bcastOp.getDstTileIndex();

          Operation *consumer = nullptr;
          SmallVector<Value> consumerOperands;

          // Find the one and only consumer.
          for (auto *searchOp = bcastOp->getNextNode(); searchOp;
               searchOp = searchOp->getNextNode()) {
            if (searchOp->hasTrait<ttkernel::TTKernelFPUOpTrait>() ||
                searchOp->hasTrait<ttkernel::TTKernelSFPUOpTrait>()) {
              for (auto operand : searchOp->getOperands()) {
                consumerOperands.push_back(operand);
                if (operand == bcastDstIdx) {
                  assert(consumer == nullptr);
                  consumer = searchOp;
                }
              }
            }

            if (!consumer) {
              consumerOperands.clear();
            } else {
              break;
            }
          }
          assert(consumer);

          // Find all copy_tile ops that provides inputs for the consumer.
          SmallVector<std::pair<ttkernel::CopyTileInitOp, ttkernel::CopyTileOp>>
              copyTileOps;
          for (auto *searchOp = consumer->getPrevNode(); searchOp;
               searchOp = searchOp->getPrevNode()) {
            if (auto copyOp = mlir::dyn_cast<ttkernel::CopyTileOp>(searchOp)) {
              Value copyDstIdx = copyOp.getTileIndexDst();
              if (llvm::is_contained(consumerOperands, copyDstIdx)) {
                consumerOperands.erase(
                    llvm::remove_if(consumerOperands,
                                    [&](Value v) { return v == copyDstIdx; }),
                    consumerOperands.end());
                auto copyInitOp =
                    mlir::cast<ttkernel::CopyTileInitOp>(copyOp->getPrevNode());
                copyTileOps.emplace_back(copyInitOp, copyOp);
              }
            }
          }
          if (copyTileOps.empty()) {
            continue;
          }

          // Move unary_bcast(_init) to the front.
          Operation *firstCopyInitOp = nullptr;
          for (auto [copyInitOp, _] : copyTileOps) {
            if (!firstCopyInitOp ||
                copyInitOp->isBeforeInBlock(firstCopyInitOp)) {
              firstCopyInitOp = copyInitOp;
            }
          }
          if (firstCopyInitOp->isBeforeInBlock(bcastInitOp)) {
            bcastInitOp->moveBefore(firstCopyInitOp);
            bcastOp->moveAfter(bcastInitOp);
          }
        }
      }
    });
  }

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
    target.addLegalOp<d2m::MeshShardOp>();

    target.addLegalOp<ttir::TTNNMetalLayoutCastOp>();

    if (ttnnMode) {
      target.addLegalDialect<ttnn::TTNNDialect>();
    }

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
      auto memorySpace = ttcore::getMemorySpace(memref);
      if (mlir::isa<ttcore::DeviceLayoutInterface>(memref.getLayout())) {
        // This memref has a device layout meaning it's an address.
        return IntegerType::get(memref.getContext(), 32);
      }

      if (memorySpace == ttcore::MemorySpace::RegisterDst) {
        // This memref abstracts tile indices in dst register, convert to index
        // type.
        return IndexType::get(memref.getContext());
      }

      if (mlir::isa<StridedLayoutAttr>(memref.getLayout())) {
        // This memref abstracts index offsets into a subview.
        return IndexType::get(memref.getContext());
      }

      // Since none of the above is true, this memref abstracts cb backing.
      return ttkernel::CBType::get(memref);
    });
    typeConverter.addConversion([](d2m::CBType cb) -> Type {
      return ttkernel::CBType::get(cb.getUnderlyingAs<MemRefType>());
    });
    typeConverter.addConversion([](d2m::SemaphoreType semaphore) {
      return ttkernel::SemaphoreType::get(semaphore.getContext());
    });

    d2m::AssociatedDMAWaits associatedDMAWaits =
        getAnalysis<d2m::AssociatedDMAWaits>();

    d2m::CBProducerConsumer cbProducerConsumer =
        getAnalysis<d2m::CBProducerConsumer>();

    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateD2MToTTKernelPatterns(&getContext(), patterns, typeConverter,
                                  associatedDMAWaits, cbProducerConsumer,
                                  ttnnMode);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    getOperation().walk([this](func::FuncOp func) {
      auto threadType = func->getAttrOfType<ttkernel::ThreadTypeAttr>(
          ttkernel::ThreadTypeAttr::name);
      if (threadType &&
          threadType.getValue() == ttkernel::ThreadType::Compute) {
        reorderUnaryBcastOps(func);
      }
    });
  };
};
} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertD2MToTTKernelPass() {
  return std::make_unique<ConvertD2MToTTKernel>();
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertD2MToTTKernelPass(
    const d2m::ConvertD2MToTTKernelOptions &options) {
  return std::make_unique<ConvertD2MToTTKernel>(options);
}

} // namespace mlir::tt
