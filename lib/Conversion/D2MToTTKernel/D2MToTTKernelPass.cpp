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
    // Temp tile allocations stay as alloc ops, but their result type gets converted by type converter
    target.addLegalOp<memref::AllocOp>();
    target.addLegalOp<memref::DeallocOp>();
    // Convert memref.copy when it's between tile-typed memrefs (CB-to-CB copy)
    target.addDynamicallyLegalOp<memref::CopyOp>([&](memref::CopyOp op) {
      auto srcType = mlir::dyn_cast<MemRefType>(op.getSource().getType());
      auto dstType = mlir::dyn_cast<MemRefType>(op.getTarget().getType());
      if (!srcType || !dstType) return true;  // Legal if not memrefs

      // Illegal if both are tile-typed L1 memrefs (need CB-to-CB copy)
      bool srcIsTileCB = mlir::isa<ttcore::TileType>(srcType.getElementType()) &&
                         srcType.getMemorySpace() &&
                         ttcore::getMemorySpace(op.getSource()) == ttcore::MemorySpace::DeviceL1;
      bool dstIsTileCB = mlir::isa<ttcore::TileType>(dstType.getElementType()) &&
                         dstType.getMemorySpace() &&
                         ttcore::getMemorySpace(op.getTarget()) == ttcore::MemorySpace::DeviceL1;

      return !(srcIsTileCB && dstIsTileCB);  // Illegal if both are tile CBs
    });
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
      // Check if memref has memory space attribute before accessing it
      // Temp allocations from d2m.empty() inside linalg.generic may not have memspace
      auto memspaceAttr = memref.getMemorySpace();

      if (mlir::isa<ttcore::DeviceLayoutInterface>(memref.getLayout())) {
        // This memref has a device layout meaning it's an address.
        return IntegerType::get(memref.getContext(), 32);
      }

      // Handle memrefs WITH memory space attribute
      if (memspaceAttr) {
        auto memorySpace = ttcore::getMemorySpace(memref);

        // DST memrefs stay as index (for copy_tile, pack_tile operations)
        if (memorySpace == ttcore::MemorySpace::RegisterDst) {
          // This memref abstracts tile indices in dst register, convert to index type.
          return IndexType::get(memref.getContext());
        }

        // L1 tile-typed memrefs (including temp allocs) convert to CB
        // This handles temp allocations from d2m.empty() that got L1 memspace
        if (memorySpace == ttcore::MemorySpace::DeviceL1 &&
            mlir::isa<ttcore::TileType>(memref.getElementType())) {
          return ttkernel::CBType::get(memref);
        }
      }

      // Tile-typed memrefs WITHOUT memspace (temp allocs before our fix) → CB
      if (mlir::isa<ttcore::TileType>(memref.getElementType()) && !memspaceAttr) {
        return ttkernel::CBType::get(memref);
      }

      if (mlir::isa<StridedLayoutAttr>(memref.getLayout())) {
        // This memref abstracts index offsets into a subview.
        return IndexType::get(memref.getContext());
      }

      // Since none of the above is true, this memref abstracts cb backing.
      // This includes:
      // - Memrefs with DeviceL1 memspace (real CBs)
      // - Memrefs without memspace (temp allocs - treat as L1 CBs)
      return ttkernel::CBType::get(memref);
    });
    typeConverter.addConversion([](d2m::CBType cb) -> Type {
      return ttkernel::CBType::get(cb.getUnderlyingAs<MemRefType>());
    });
    typeConverter.addConversion([](d2m::SemaphoreType semaphore) {
      return ttkernel::SemaphoreType::get(semaphore.getContext());
    });

    // Add source materialization for memref→CB conversion
    // This handles cases where memref.alloc results need to be used as CBs
    typeConverter.addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                              ValueRange inputs,
                                              Location loc) -> Value {
      if (inputs.size() == 1 && mlir::isa<ttkernel::CBType>(resultType)) {
        // Forward the input - the type converter will handle it
        return inputs[0];
      }
      return nullptr;
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
