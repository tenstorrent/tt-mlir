// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MDISTRIBUTECOMPUTETHREADS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Matmul-shape gate: a linalg.generic with exactly 3 loops, iterator types
// [parallel, parallel, reduction] in any order, 2 inputs, 1 output. Strong
// enough for the bring-up matmul; can be tightened to isaMatmulOpInterface
// later.
static bool isMatmulShaped(linalg::GenericOp op) {
  if (op.getNumLoops() != 3) {
    return false;
  }
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) {
    return false;
  }
  auto iterTypes = op.getIteratorTypesArray();
  int numParallel = 0;
  int numReduction = 0;
  for (mlir::utils::IteratorType iter : iterTypes) {
    if (iter == mlir::utils::IteratorType::parallel) {
      ++numParallel;
    } else if (iter == mlir::utils::IteratorType::reduction) {
      ++numReduction;
    }
  }
  return numParallel == 2 && numReduction == 1;
}

// Return the iteration-space dim index that corresponds to either matmul "m"
// or "n", based on the output indexing map. For a matmul with output map
// (...) -> (m, n), getResult(0) is m and getResult(1) is n.
static FailureOr<unsigned> getMatmulDimIndex(linalg::GenericOp op,
                                             StringRef which) {
  AffineMap outMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  if (outMap.getNumResults() != 2) {
    return op->emitOpError("matmul-shaped linalg.generic must have rank-2 "
                            "output indexing map");
  }
  unsigned resultIdx = (which == "n") ? 1 : 0;
  auto dim = mlir::dyn_cast<AffineDimExpr>(outMap.getResult(resultIdx));
  if (!dim) {
    return op->emitOpError("output indexing map result is not a plain dim");
  }
  return dim.getPosition();
}

// Find the unique matmul-shaped linalg.generic inside a d2m.generic's compute
// region, if any. Returns nullptr if there are 0 or >1 linalg.generic ops, or
// if the single one isn't matmul-shaped.
static linalg::GenericOp findSoleMatmul(d2m::GenericOp genericOp) {
  linalg::GenericOp result = nullptr;
  for (Region &region : genericOp->getRegions()) {
    region.walk([&](linalg::GenericOp linalgOp) {
      if (result) {
        // Found more than one — give up.
        result = nullptr;
        return WalkResult::interrupt();
      }
      result = linalgOp;
      return WalkResult::advance();
    });
    if (!result) {
      continue;
    }
  }
  if (!result || !isMatmulShaped(result)) {
    return nullptr;
  }
  return result;
}

class D2MDistributeComputeThreads
    : public impl::D2MDistributeComputeThreadsBase<
          D2MDistributeComputeThreads> {
public:
  using impl::D2MDistributeComputeThreadsBase<
      D2MDistributeComputeThreads>::D2MDistributeComputeThreadsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Validate options.
    if (numComputeThreads <= 0) {
      moduleOp->emitError() << "num-compute-threads must be positive, got "
                            << numComputeThreads;
      return signalPassFailure();
    }
    if (splitDim != "m" && splitDim != "n") {
      moduleOp->emitError() << "split-dim must be 'm' or 'n', got '"
                            << splitDim << "'";
      return signalPassFailure();
    }

    // Collect targets first; tiling mutates the IR.
    SmallVector<std::pair<d2m::GenericOp, linalg::GenericOp>> targets;
    moduleOp->walk([&](d2m::GenericOp genericOp) {
      if (linalg::GenericOp matmul = findSoleMatmul(genericOp)) {
        targets.emplace_back(genericOp, matmul);
      }
    });

    IRRewriter rewriter(&getContext());
    for (auto [genericOp, matmul] : targets) {
      if (failed(distribute(rewriter, matmul))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult distribute(IRRewriter &rewriter, linalg::GenericOp matmul) {
    FailureOr<unsigned> dimIndex = getMatmulDimIndex(matmul, splitDim);
    if (failed(dimIndex)) {
      return failure();
    }

    // Build per-loop num_threads: 0 for everything except the chosen dim.
    SmallVector<OpFoldResult> numThreads(matmul.getNumLoops(),
                                         rewriter.getIndexAttr(0));
    numThreads[*dimIndex] = rewriter.getIndexAttr(numComputeThreads);

    // Build the mapping attribute array.
    SmallVector<Attribute> mapping = {
        rewriter.getAttr<ComputeThreadMappingAttr>(numComputeThreads)};

    scf::SCFTilingOptions opts;
    opts.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    opts.setNumThreads(numThreads);
    opts.setMapping(mapping);

    auto tilingInterface = cast<TilingInterface>(matmul.getOperation());
    rewriter.setInsertionPoint(matmul);
    FailureOr<scf::SCFTilingResult> result =
        scf::tileUsingSCF(rewriter, tilingInterface, opts);
    if (failed(result)) {
      return matmul->emitOpError("failed to tile matmul to scf.forall");
    }

    rewriter.replaceOp(matmul, result->replacements);
    return success();
  }
};

} // namespace
} // namespace mlir::tt::d2m
