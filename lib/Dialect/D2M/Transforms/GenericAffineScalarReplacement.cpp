// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Utils/GenericAffineUtils.h"

#define DEBUG_TYPE "D2MGenericAffineScalarReplacement"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICAFFINESCALARREPLACEMENT
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

/// Check if a GenericOp input operand is an intermediate that can be
/// internalized. An input is intermediate if the only top-level operation
/// that uses it is the GenericOp itself; remote_load/remote_store uses
/// inside generic ops don't count as top-level uses.
static bool isIntermediateInput(GenericOp genericOp, Value operandVal) {
  for (Operation *user : operandVal.getUsers()) {
    if (user == genericOp.getOperation()) {
      continue;
    }
    if (isa<RemoteLoadOp, RemoteStoreOp>(user) &&
        user->getParentOfType<GenericOp>()) {
      continue;
    }
    return false;
  }
  return true;
}

/// Check that every remote_load from the intermediate inside the generic is
/// dominated by a remote_store to it. This ensures scalrep can forward all
/// loads. If any load appears before a store (e.g., accumulation patterns),
/// internalization is unsafe.
static bool allLoadsStoreDominated(GenericOp genericOp, Value operandVal) {
  bool storeReached = false;
  bool safe = true;
  genericOp.getRegion(0).walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!safe) {
      return;
    }
    if (auto store = dyn_cast<RemoteStoreOp>(op)) {
      if (store.getMemref() == operandVal) {
        storeReached = true;
      }
    }
    if (auto load = dyn_cast<RemoteLoadOp>(op)) {
      if (load.getMemref() == operandVal && !storeReached) {
        safe = false;
      }
    }
  });
  return safe;
}

/// Preprocess a GenericOp by internalizing intermediate operands.
/// Intermediate inputs are replaced with local memref allocs inside the
/// generic body, and their operands and block arg CBs are removed.
static void internalizeIntermediates(GenericOp genericOp, OpBuilder &builder) {
  if (genericOp.getRegions().empty() || genericOp.getRegion(0).empty()) {
    return;
  }

  Block &body = genericOp.getRegion(0).front();
  unsigned numInputs = genericOp.getInputs().size();

  // Collect intermediate input indices and their values.
  SmallVector<std::pair<unsigned, Value>> intermediates;

  for (unsigned i = 0; i < numInputs; ++i) {
    Value operandVal = genericOp.getInputs()[i];

    // Only locally-allocated memrefs can be intermediates.
    if (!isa_and_nonnull<memref::AllocOp>(operandVal.getDefiningOp())) {
      continue;
    }

    if (!isIntermediateInput(genericOp, operandVal)) {
      continue;
    }

    // Every remote_load from the intermediate must be preceded by a
    // remote_store; otherwise scalrep cannot forward all loads (e.g.,
    // accumulation patterns that load partial results before storing).
    if (!allLoadsStoreDominated(genericOp, operandVal)) {
      continue;
    }

    // The associated CB block arg must have no uses to safely remove it.
    BlockArgument cbArg = body.getArgument(i);
    if (!cbArg.use_empty()) {
      continue;
    }

    intermediates.emplace_back(i, operandVal);
  }

  if (intermediates.empty()) {
    return;
  }

  // Create local allocs for each intermediate and replace uses inside the
  // generic region before any structural mutations.
  for (auto &[inputIdx, operandVal] : intermediates) {
    builder.setInsertionPointToStart(&body);
    auto allocOp = builder.create<memref::AllocOp>(
        genericOp.getLoc(), cast<MemRefType>(operandVal.getType()));

    operandVal.replaceUsesWithIf(allocOp.getResult(), [&](OpOperand &use) {
      return genericOp->isAncestor(use.getOwner());
    });
  }

  // Remove operands, block args, and indexing maps in reverse index order.
  SmallVector<Attribute> maps(genericOp.getIndexingMaps().getValue());

  for (auto it = intermediates.rbegin(); it != intermediates.rend(); ++it) {
    unsigned idx = it->first;
    body.eraseArgument(idx);
    if (!maps.empty()) {
      maps.erase(maps.begin() + idx);
    }
    genericOp.getInputsMutable().erase(idx);
  }

  if (!maps.empty()) {
    genericOp.setIndexingMapsAttr(builder.getArrayAttr(maps));
  }
}

/// Convert RemoteLoadOps from destination-passing style to direct style:
/// replace post-load uses of localBuffer with the load result, so
/// affineScalarReplace can properly forward stored values through the
/// result SSA value.
static void convertRemoteLoadToDirectStyle(func::FuncOp funcOp) {
  funcOp.walk([&](RemoteLoadOp loadOp) {
    Value localBuf = loadOp.getLocalBuffer();
    Value result = loadOp.getResult();
    if (!localBuf || !result) {
      return;
    }

    localBuf.replaceUsesWithIf(result, [&](OpOperand &use) {
      Operation *user = use.getOwner();
      if (user == loadOp.getOperation()) {
        return false;
      }
      // Only replace uses that come after the load in the same block.
      return loadOp->getBlock() == user->getBlock() &&
             loadOp->isBeforeInBlock(user);
    });
  });
}

/// Convert RemoteLoadOps from direct style back to destination-passing
/// style: replace post-load uses of the result with the localBuffer
/// operand.
static void convertRemoteLoadToDPSStyle(func::FuncOp funcOp) {
  funcOp.walk([&](RemoteLoadOp loadOp) {
    Value localBuf = loadOp.getLocalBuffer();
    Value result = loadOp.getResult();
    if (!localBuf || !result) {
      return;
    }

    result.replaceUsesWithIf(localBuf, [&](OpOperand &use) {
      Operation *user = use.getOwner();
      if (user == loadOp.getOperation()) {
        return false;
      }
      return loadOp->getBlock() == user->getBlock() &&
             loadOp->isBeforeInBlock(user);
    });
  });
}

class D2MGenericAffineScalarReplacement
    : public impl::D2MGenericAffineScalarReplacementBase<
          D2MGenericAffineScalarReplacement> {
public:
  using D2MGenericAffineScalarReplacementBase::
      D2MGenericAffineScalarReplacementBase;

  void runOnOperation() final {
    getOperation()->walk([&](func::FuncOp funcOp) {
      OpBuilder builder(funcOp.getContext());

      // Collect only fused generic ops (those with d2m.affine_fused attr).
      // This must be rerun after each micro-pass as generics are being
      // rewritten multiple times.
      SmallVector<GenericOp> genericOps;
      auto collectFusedOps = [&]() {
        genericOps.clear();
        funcOp.walk([&](GenericOp op) {
          if (op->hasAttr("d2m.affine_fused")) {
            genericOps.push_back(op);
          }
        });
      };

      // Preprocess: internalize intermediate operands in each generic op.
      collectFusedOps();
      for (GenericOp op : genericOps) {
        internalizeIntermediates(op, builder);
      }

      // Convert all fused d2m.generic ops to affine-compatible form.
      collectFusedOps();
      for (GenericOp op : genericOps) {
        utils::convertToAffineCompatibilityForm(op, builder);
      }

      // Convert RemoteLoadOps to direct style for store-to-load forwarding.
      convertRemoteLoadToDirectStyle(funcOp);

      // Run affine scalar replacement on the function.
      auto &domInfo = getAnalysis<DominanceInfo>();
      auto &postDomInfo = getAnalysis<PostDominanceInfo>();
      auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
      affine::affineScalarReplace(funcOp, domInfo, postDomInfo, aliasAnalysis);

      // Convert surviving RemoteLoadOps back to DPS style.
      convertRemoteLoadToDPSStyle(funcOp);

      // Convert all fused d2m.generic ops back from affine-compatible form.
      collectFusedOps();
      for (GenericOp op : genericOps) {
        utils::convertFromAffineCompatibilityForm(op, builder);
      }
    });
  }
};
} // namespace

} // namespace mlir::tt::d2m
