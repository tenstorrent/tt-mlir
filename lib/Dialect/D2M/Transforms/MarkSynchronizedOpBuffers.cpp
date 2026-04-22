// SPDX-FileCopyrightText: (c) 2026wr Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMARKSYNCHRONIZEDOPBUFFERS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MMarkSynchronizedOpBuffers
    : public impl::D2MMarkSynchronizedOpBuffersBase<
          D2MMarkSynchronizedOpBuffers> {
public:
  using impl::D2MMarkSynchronizedOpBuffersBase<
      D2MMarkSynchronizedOpBuffers>::D2MMarkSynchronizedOpBuffersBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    PatternRewriter rewriter(&getContext());

    moduleOp->walk([&](d2m::GenericOp genericOp) {
      if (failed(wrapComputeInSynchronizedRegion(genericOp, rewriter))) {
        emitError(
            genericOp->getLoc(),
            "Failed to wrap compute in synchronized region for genericop: ")
            << genericOp;
      }
      return WalkResult::advance();
    });
  }

private:
  Value traceComputeMemrefToCB(Value value, GenericOp genericOp) {
    llvm::errs() << "tracing value: " << value << "\n";
    while (value) {
      // check if its a cb (hoisted generic arg with cb layout attr),
      if (auto memrefType = mlir::dyn_cast<MemRefType>(value.getType())) {
        if (mlir::isa<ttcore::CBLayoutAttr>(memrefType.getLayout())) {
          return value;
        }
      }

      // if we are no longer inside the generic or have reached the root, stop
      // tracing and return nullptr
      Operation *definingOp = value.getDefiningOp();
      if (!definingOp || !genericOp->isProperAncestor(definingOp)) {
        llvm::errs() << "definingOp is not a proper ancestor of genericOp: "
                     << *definingOp << "\n";
        return nullptr;
      }

      // Otherwise keep tracing up the chain, if we reach an op we don't
      // support, stop tracing and return nullptr
      if (auto subviewOp = mlir::dyn_cast<memref::SubViewOp>(definingOp)) {
        value = subviewOp.getSource();
        continue;
      } else {
        llvm::errs() << "definingOp is not a subview op: " << *definingOp
                     << "\n";
        return nullptr;
      }
    }
    llvm::errs() << "value is not a cb: " << value << "\n";
    return nullptr;
  }

  LogicalResult wrapComputeInSynchronizedRegion(GenericOp genericOp,
                                                PatternRewriter &rewriter) {
    // Look for a D2M_GenericRegionComputeOp, and collect the outermost ops that
    // contain them in the generic op
    DenseSet<Operation *> outermostOps;
    genericOp.getRegion(0).walk([&](Operation *op) {
      if (!op->hasTrait<D2MGenericRegionComputeOpTrait>()) {
        return WalkResult::advance();
      }

      // go up loops until we reach generic op
      Operation *outermostLoopOp = op;
      while (outermostLoopOp->getParentOp() != genericOp.getOperation()) {
        outermostLoopOp = outermostLoopOp->getParentOp();
        if (!mlir::isa<scf::ForOp>(outermostLoopOp) &&
            !mlir::isa<affine::AffineForOp>(outermostLoopOp)) {
          llvm::errs() << "op " << *outermostLoopOp << "\n";
          assert(false && "outermost loop op is not a scf.for op");
        }
      }

      outermostOps.insert(outermostLoopOp);
      return WalkResult::advance();
    });

    for (Operation *outermostOp : outermostOps) {
      // for memref load and stores, trace to cb operand to get producers and
      // consumers
      DenseSet<Value> loadedCBOperands;
      DenseSet<Value> storedCBOperands;

      // and store in list of loaded operands
      outermostOp->walk([&](affine::AffineLoadOp loadOp) {
        Value cb = traceComputeMemrefToCB(loadOp.getMemref(), genericOp);
        if (cb) {
          loadedCBOperands.insert(cb);
        }
        return WalkResult::advance();
      });

      llvm::errs() << "outermost op: " << *outermostOp << "\n";
      llvm::errs() << "loadedCBOperands: ";
      for (Value loadedCBOperand : loadedCBOperands) {
        llvm::errs() << loadedCBOperand << " ";
      }
      llvm::errs() << "\n";

      // for store trace up to list of allocs store into and then
      outermostOp->walk([&](affine::AffineStoreOp storeOp) {
        Value cb = traceComputeMemrefToCB(storeOp.getMemref(), genericOp);
        if (cb) {
          storedCBOperands.insert(cb);
        }
        return WalkResult::advance();
      });

      llvm::errs() << "\n";
      llvm::errs() << "storedCBOperands: ";
      for (Value storedCBOperand : storedCBOperands) {
        llvm::errs() << storedCBOperand << " ";
      }
      llvm::errs() << "\n";

      // remove allocs in load that are in store since this is output cb reuse
      // and not an actual input
      for (Value storedCBOperand : storedCBOperands) {
        if (loadedCBOperands.contains(storedCBOperand)) {
          loadedCBOperands.erase(storedCBOperand);
        }
      }

      llvm::errs() << "outermost op: " << *outermostOp << "\n";
      llvm::errs() << "loadedCBOperands after cleanup: ";
      for (Value loadedCBOperand : loadedCBOperands) {
        llvm::errs() << loadedCBOperand << " ";
      }
      llvm::errs() << "\n";
      llvm::errs() << "storedCBOperands after cleanup: ";
      for (Value storedCBOperand : storedCBOperands) {
        llvm::errs() << storedCBOperand << " ";
      }
      llvm::errs() << "\n";
      // wrapInSynchronizedRegion(
      //   rewriter, outermostOp, SmallVector<Value>(loadedCBOperands.begin(),
      //   loadedCBOperands.end()), SmallVector<Value>(storedCBOperands.begin(),
      //   storedCBOperands.end()));
    }

    return success();
  }
};

} // namespace
} // namespace mlir::tt::d2m
