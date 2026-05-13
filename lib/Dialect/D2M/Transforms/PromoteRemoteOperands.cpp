// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MPROMOTEREMOTEOPERANDS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Promote memref operands of remote_load/remote_store ops in a d2m.generic's
// body from `additionalArgs` to `ins`/`outs`.
//
// Frontends (notably the Triton-to-D2M path) emit d2m.generic ops in explicit-
// datamovement form with a placeholder `outs` and every DRAM tensor handle in
// `additionalArgs`; the real I/O is encoded in remote_load/remote_store ops
// inside the body. Downstream passes (D2MAllocate, D2MToTTMetal) assume the
// canonical layout where memref operands that participate in data movement
// live in `ins`/`outs`. This pass restores that invariant by walking the body
// and rewriting the operand layout.
//
// Rules:
//  - Memrefs read by `remote_load` go to `ins`.
//  - Memrefs written by `remote_store` go to `outs`.
//  - A memref used both ways follows DPS convention and goes to `outs`.
//  - The op's verifier requires exactly one `outs` operand. If there are 0 or
//    >1 distinct store targets we can't unambiguously canonicalize and leave
//    the op alone.
//  - Only operates on explicit-datamovement-form generics (empty block
//    factors / indexing maps / iterator types); other forms have ins/outs
//    semantics tied to the indexing maps and must not be reshuffled.
class D2MPromoteRemoteOperandsRewriter : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const final {
    if (!genericOp.isExplicitDatamovementForm()) {
      return failure();
    }

    llvm::SmallSetVector<Value, 4> loadMemrefs;
    llvm::SmallSetVector<Value, 4> storeMemrefs;
    for (Region &region : genericOp.getRegions()) {
      region.walk([&](RemoteLoadOp loadOp) {
        loadMemrefs.insert(loadOp.getMemref());
      });
      region.walk([&](RemoteStoreOp storeOp) {
        storeMemrefs.insert(storeOp.getMemref());
      });
    }

    // The verifier requires exactly one outs operand.
    if (storeMemrefs.size() != 1) {
      return failure();
    }

    Value canonicalOut = storeMemrefs.front();
    llvm::SmallSetVector<Value, 4> canonicalIns;
    for (Value v : loadMemrefs) {
      if (v != canonicalOut) {
        canonicalIns.insert(v);
      }
    }

    // Skip if already canonical: current ins exactly equals canonicalIns
    // (as a set) and the single current out equals canonicalOut.
    ValueRange currentIns = genericOp.getInputs();
    OperandRange currentOuts = genericOp.getOutputs();
    if (currentOuts.size() == 1 && currentOuts[0] == canonicalOut &&
        currentIns.size() == canonicalIns.size() &&
        llvm::all_of(currentIns,
                     [&](Value v) { return canonicalIns.contains(v); })) {
      return failure();
    }

    // Drop any value from additionalArgs that's now in ins/outs to avoid
    // referencing the same SSA value twice in the op.
    SmallVector<Value> newAdditionalArgs;
    for (Value v : genericOp.getAdditionalArgs()) {
      if (!canonicalIns.contains(v) && v != canonicalOut) {
        newAdditionalArgs.push_back(v);
      }
    }

    SmallVector<Value> newIns(canonicalIns.begin(), canonicalIns.end());
    SmallVector<Value> newOuts{canonicalOut};

    auto newGeneric = rewriter.create<GenericOp>(
        genericOp.getLoc(), genericOp.getResultTypes(), newIns, newOuts,
        newAdditionalArgs, genericOp.getGrid(), genericOp.getBlockFactors(),
        genericOp.getIndexingMaps(), genericOp.getIteratorTypes(),
        genericOp.getThreads(), genericOp.getFabricConnectionConfigAttr(),
        genericOp.getNumRegions());

    for (auto [oldRegion, newRegion] :
         llvm::zip(genericOp.getRegions(), newGeneric.getRegions())) {
      newRegion.takeBody(oldRegion);
    }

    rewriter.replaceOp(genericOp, newGeneric.getResults());
    return success();
  }
};

class D2MPromoteRemoteOperands
    : public impl::D2MPromoteRemoteOperandsBase<D2MPromoteRemoteOperands> {
public:
  using impl::D2MPromoteRemoteOperandsBase<
      D2MPromoteRemoteOperands>::D2MPromoteRemoteOperandsBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<D2MPromoteRemoteOperandsRewriter>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::d2m
