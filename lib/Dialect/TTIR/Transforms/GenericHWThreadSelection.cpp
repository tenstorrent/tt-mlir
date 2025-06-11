// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Utils.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICHWTHREADSELECTION
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
class TTIRGenericMoveTrivialOutputThreadToComputeRewritePattern
    : public OpRewritePattern<GenericOp> {
public:
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    // construct new threads by copying original threads and deleting excess datamovement threads
    // this preserves block naming and other important attributes
    constexpr size_t num_hw_dm_threads = 2;
    SmallVector<Attribute> new_threads(op.getThreads().getValue());
    size_t num_dm_regions = llvm::count_if(new_threads, [](const Attribute &t) {
      return mlir::cast<ThreadAttr>(t).getThreadType() ==
             ThreadType::Datamovement;
    });
    size_t num_deleted_dm_regions =
        std::max(0, static_cast<int>(num_dm_regions - num_hw_dm_threads));
    size_t num_new_regions = op.getNumRegions() - num_deleted_dm_regions;
    if (num_deleted_dm_regions > 0) {
      new_threads.erase(new_threads.begin() + num_hw_dm_threads,
                        new_threads.begin() + num_dm_regions);
    }

    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), op.getGrid(), op.getIndexingMaps(),
        op.getIteratorTypes(), rewriter.getArrayAttr(new_threads),
        num_new_regions);

    // move the first HW_THREADS dm regions and the compute region to the new generic
    // remaining dm blocks will be merged into these copied regions 
    unsigned new_region_index = 0;
    size_t start_del_dm_regions = num_dm_regions - num_deleted_dm_regions;
    size_t end_del_dm_regions = num_dm_regions;
    for (mlir::Region &region : op.getRegions()) {
      auto region_index = region.getRegionNumber();
      // skip deleted dm regions
      if (region_index >= start_del_dm_regions &&
          region_index < end_del_dm_regions) {
        assert(mlir::cast<ThreadAttr>(op.getThreads()[region_index])
                   .getThreadType() == ThreadType::Datamovement);
        continue;
      }
      rewriter.modifyOpInPlace(op, [&] {
        newGeneric.getRegion(new_region_index++).takeBody(region);
      });
    }

    // merge remaining dm regions into already copied dm regions
    size_t dest_region_index = 0;
    for (size_t src_region_index = start_del_dm_regions; src_region_index < end_del_dm_regions; src_region_index++) {
      Block *dest_block = &newGeneric.getRegions()[dest_region_index].front();
      Block *src_block = &op.getRegions()[src_region_index].front();
      rewriter.mergeBlocks(src_block, dest_block, dest_block->getArguments());
      // alternate merging between all dm regions 
      dest_region_index = (dest_region_index == num_hw_dm_threads - 1)
                              ? 0
                              : dest_region_index + 1;
    }

    rewriter.replaceOp(op, newGeneric.getResults());
    return success();
  }
};
} // namespace

namespace {
class TTIRGenericHWThreadSelection
    : public impl::TTIRGenericHWThreadSelectionBase<
          TTIRGenericHWThreadSelection> {
public:
  using impl::TTIRGenericHWThreadSelectionBase<
      TTIRGenericHWThreadSelection>::TTIRGenericHWThreadSelectionBase;

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericMoveTrivialOutputThreadToComputeRewritePattern>(
        &getContext());
    walkAndApplyPatterns(getOperation(), std::move(patterns));

    ModuleOp moduleOp = getOperation();
    auto systemDesc =
        moduleOp->getAttrOfType<SystemDescAttr>(SystemDescAttr::name);
    auto chipDesc = systemDesc.getChipDescs().front();
    moduleOp.walk([&](GenericOp op) {
      // assert that the op has a valid HW thread selection
      if (op.getNumRegions() > (chipDesc.getNumComputeThreads() +
                                chipDesc.getNumDatamovementThreads())) {
        op.emitError("invalid number of regions (")
            << op.getNumRegions() << "), expected at most ("
            << (chipDesc.getNumComputeThreads() +
                chipDesc.getNumDatamovementThreads())
            << ")";
        signalPassFailure();
      }
    });
  }
};
} // namespace

} // namespace mlir::tt::ttir
