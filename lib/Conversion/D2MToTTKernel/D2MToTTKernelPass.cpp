// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/D2MToTTKernel/D2MToTTKernel.h"

#include "ttmlir/Dialect/D2M/Analysis/CBProducerConsumer.h"
#include "ttmlir/Dialect/D2M/IR/D2M.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/DMAUtils.h"
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
    this->useTensorAccessorDMA = rhs.useTensorAccessorDMA;
  }

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
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

    target.addLegalOp<d2m::ToDeviceOp>();
    target.addLegalOp<d2m::ToHostOp>();
    target.addLegalOp<d2m::ViewLayoutOp>();
    target.addLegalOp<d2m::GenericOp>();
    target.addLegalOp<d2m::EmptyOp>();
    target.addLegalOp<d2m::MeshShardOp>();
    target.addLegalOp<d2m::CreateGlobalSemaphoreOp>();
    target.addLegalOp<d2m::ResetGlobalSemaphoreOp>();
    target.addLegalOp<d2m::CreateLocalSemaphoreOp>();
    target.addLegalOp<d2m::SpatialOp>();
    target.addLegalOp<d2m::OperandAliasOp>();

    if (ttnnMode) {
      target.addLegalOp<ttir::TTNNMetalLayoutCastOp>();
      target.addLegalDialect<ttnn::TTNNDialect>();
    }

    auto isHostScalarLoadStore = [](Operation *op, MemRefType memrefType) {
      auto func = op->getParentOfType<func::FuncOp>();
      if (!func || func->hasAttr(d2m::ThreadAttr::name)) {
        return false;
      }

      return ttcore::getMemorySpace(memrefType) ==
                 ttcore::MemorySpace::System &&
             memrefType.hasStaticShape() && memrefType.getNumElements() == 1 &&
             !mlir::isa<ttcore::TileType>(memrefType.getElementType());
    };

    // Allow loads and stores to integer element types, i.e. riscv accesses to
    // L1. Host command-function scalar load/stores are also legal; those are
    // CPU-side buffer operations and must not be lowered as circular-buffer
    // tile accesses.
    target.addDynamicallyLegalOp<memref::LoadOp>([&](memref::LoadOp op) {
      return op.getMemRefType().getElementType().isIntOrIndex() ||
             isHostScalarLoadStore(op, op.getMemRefType());
    });
    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      return op.getMemRefType().getElementType().isIntOrIndex() ||
             isHostScalarLoadStore(op, op.getMemRefType());
    });
    target.addLegalOp<memref::AllocOp>();
    target.addLegalOp<memref::DeallocOp>();
    target.addLegalOp<memref::CopyOp>();
    target.addLegalOp<memref::GlobalOp>();
    target.addLegalOp<memref::GetGlobalOp>();

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return !op->hasAttr(d2m::ThreadAttr::name) ||
             op->hasAttr(ttkernel::ThreadTypeAttr::name);
    });

    if (failed(d2m::utils::checkBackendDatamovementProcessorSupport(
            moduleOp, "D2MToTTKernel"))) {
      signalPassFailure();
      return;
    }

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
    typeConverter.addConversion([](d2m::LocalSemaphoreType semaphore) {
      return ttkernel::LocalSemaphoreType::get(semaphore.getContext());
    });
    typeConverter.addConversion([](d2m::GlobalSemaphoreType globalSemaphore) {
      return ttkernel::L1AddrType::get(globalSemaphore.getContext());
    });

    d2m::CBProducerConsumer cbProducerConsumer =
        getAnalysis<d2m::CBProducerConsumer>();

    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateD2MToTTKernelPatterns(&getContext(), patterns, typeConverter,
                                  cbProducerConsumer, ttnnMode,
                                  useTensorAccessorDMA);
    scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter,
                                                         patterns, target);

    auto isFabricOp = [](Operation *op) {
      return llvm::isa<d2m::DeviceSynchronizeOp>(op) ||
             (llvm::isa<d2m::DMAWriteOp>(op) &&
              llvm::cast<d2m::DMAWriteOp>(op).getStartDevice().size() > 0) ||
             (llvm::isa<d2m::SemaphoreIncOp>(op) &&
              llvm::cast<d2m::SemaphoreIncOp>(op).getStartDevice().size() >
                  0) ||
             (llvm::isa<d2m::SemaphoreSetOp>(op) &&
              llvm::cast<d2m::SemaphoreSetOp>(op).getStartDevice().size() > 0);
    };
    // Ops that lower to use the fabric connection manager (so the fcm must
    // dominate them): the fabric ops plus mesh_position (lowers to
    // get_my_logical_mesh_position(fcm)). The fcm anchor is the common ancestor
    // of these, so a router-gated kernel must keep all of them -- including any
    // mesh_position -- inside the is_router_core() branch.
    auto isFcmUser = [&](Operation *op) {
      return isFabricOp(op) || llvm::isa<d2m::MeshPositionOp>(op);
    };

    // True if the generic whose thread is this func declares a router_cores
    // subset (fabric on fewer cores than the grid). When it does, the fabric
    // connection manager must be gated to the router cores; otherwise it is
    // created for the whole grid (legacy behavior). See
    // tools/d2m-jit/fabric_router_cores_design.md (step 2).
    auto funcHasRouterSubset = [](func::FuncOp func) {
      StringRef name = func.getSymName();
      bool found = false;
      if (auto module = func->getParentOfType<ModuleOp>()) {
        module.walk([&](d2m::GenericOp generic) {
          auto cfg = generic.getFabricConnectionConfigAttr();
          if (!cfg || cfg.getRouterCores().empty()) {
            return WalkResult::advance();
          }
          for (Attribute t : generic.getThreads()) {
            auto sym = llvm::cast<d2m::ThreadAttr>(t).getKernelSymbol();
            if (sym && sym.getLeafReference() == name) {
              found = true;
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
      }
      return found;
    };

    // Deepest block that is an ancestor of every op. For a router-gated kernel
    // (all fabric ops inside one `if is_router_core()`), this is that scf.if
    // body, so the fcm lifecycle lands inside the gate; for an ungated kernel
    // (fabric ops at the func top) it is the func body.
    auto commonAncestorBlock = [](ArrayRef<Operation *> ops) -> Block * {
      auto chain = [](Operation *op) {
        SmallVector<Block *> blocks;
        for (Block *b = op->getBlock(); b;) {
          blocks.push_back(b);
          Operation *parent = b->getParentOp();
          b = parent ? parent->getBlock() : nullptr;
        }
        return blocks; // innermost-first
      };
      SmallVector<Block *> common = chain(ops[0]);
      for (Operation *op : ops.drop_front()) {
        SmallVector<Block *> c = chain(op);
        llvm::SmallPtrSet<Block *, 8> set(c.begin(), c.end());
        SmallVector<Block *> next;
        for (Block *b : common) {
          if (set.contains(b)) {
            next.push_back(b);
          }
        }
        common = std::move(next);
      }
      return common.empty() ? nullptr : common.front();
    };

    // If there are any cross-device fabric ops, create/setup the fabric
    // connection manager before them and close it after, anchored so that a
    // router_cores-gated kernel opens the connection only on its router cores.
    moduleOp->walk([&](func::FuncOp func) {
      SmallVector<Operation *> fabricOps, fcmUsers;
      func.walk([&](Operation *op) {
        if (isFabricOp(op)) {
          fabricOps.push_back(op);
        }
        if (isFcmUser(op)) {
          fcmUsers.push_back(op);
        }
      });
      // Create the fcm whenever the func has any fcm user, not just a fabric
      // op: a thread can use mesh_position (an fcm user, lowers to
      // get_my_logical_mesh_position(fcm)) without itself doing a cross-device
      // fabric op -- e.g. a local output store whose grid index is derived from
      // mesh_position lands on a different NoC thread than the fabric send. Such
      // a thread still needs the fcm to dominate the mesh_position. The setup
      // only opens `num_send_dir` connections, which is 0 for a non-sending
      // thread, so this is just the (cheap) topology build that mesh_position
      // needs -- no stray fabric connection.
      if (fcmUsers.empty()) {
        return;
      }

      // Default (no router subset): func top, the whole grid opens a
      // connection. With a router subset: anchor at the fcm users' common
      // ancestor, so the connection manager is created only where the gated
      // fabric work runs. (The fcm must still dominate every user, hence the
      // common ancestor of *all* fcm users, mesh_position included, not just
      // the fabric ops.)
      Block *anchor = &func.getBody().front();
      if (funcHasRouterSubset(func)) {
        if (Block *common = commonAncestorBlock(fcmUsers)) {
          anchor = common;
        }
      }

      OpBuilder builder(func.getContext());
      builder.setInsertionPointToStart(anchor);
      auto fabricConnectionManager =
          builder
              .create<ttkernel::CreateFabricConnectionManagerOp>(func.getLoc())
              .getResult();
      builder.create<ttkernel::SetupFabricConnectionsOp>(
          func.getLoc(), fabricConnectionManager);
      builder.setInsertionPoint(anchor->getTerminator());
      builder.create<ttkernel::CloseFabricConnectionsOp>(
          func.getLoc(), fabricConnectionManager);
    });

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // The d2m.thread attr is kept until the end of this pass, when body
    // rewrites have consumed nocIndex.
    getOperation()->walk(
        [](func::FuncOp funcOp) { funcOp->removeAttr(d2m::ThreadAttr::name); });
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
