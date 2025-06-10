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

  static Block *getIfTrivialBlock(Region &region,
                                  unsigned outputOperandsIndex) {
    assert(region.getBlocks().size() == 1);
    auto &block = region.front();
    if (block.getOperations().size() != 1) {
      return nullptr;
    }
    ttir::AwaitOp awaitOp = dyn_cast<ttir::AwaitOp>(block.front());
    if (!awaitOp) {
      return nullptr;
    }

    if (!llvm::all_of(awaitOp.getOperands(), [&](Value operand) {
          return mlir::dyn_cast<BlockArgument>(operand) &&
                 mlir::cast<BlockArgument>(operand).getOwner() == &block &&
                 mlir::cast<BlockArgument>(operand).getArgNumber() ==
                     outputOperandsIndex;
        })) {
      return nullptr;
    }

    return &block;
  }

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
 
    // collect all datamovement blocks
    SmallVector<Attribute> threads(op.getThreads().getValue());
    SmallVector<Block*> dm_blocks;
    for (unsigned index = 0; index < threads.size(); index++) {
      if (mlir::cast<ThreadAttr>(threads[index]).getThreadType() == ThreadType::Datamovement) {
        Region& r = op.getRegion(index);
        assert(r.getBlocks().size() == 1);
        dm_blocks.push_back(&r.getBlocks().back());
      }
    }

    constexpr size_t HW_THREADS = 2;
    size_t num_dm_regions = std::min(HW_THREADS,dm_blocks.size());
    size_t max_dm_blocks_per_thread =
        std::max(1ul, static_cast<size_t>(std::ceil(
                          static_cast<double>(dm_blocks.size()) / HW_THREADS)));
    constexpr size_t num_compute_regions = 1;
    size_t num_regions = num_dm_regions + num_compute_regions;

    // construct new threads array
    SmallVector<Attribute> new_threads;
    for (size_t i = 0; i < num_dm_regions; i++) {
      new_threads.push_back(
          ThreadAttr::get(getContext(), ThreadType::Datamovement));
    }
    for (size_t i = 0; i < num_compute_regions; i++) {
      new_threads.push_back(ThreadAttr::get(getContext(), ThreadType::Compute));
    }

    auto newGeneric = rewriter.create<GenericOp>(
        op.getLoc(), op.getResults().getTypes(), op.getInputs(),
        op.getOutputs(), op.getGrid(), op.getIndexingMaps(),
        op.getIteratorTypes(), rewriter.getArrayAttr(new_threads),
        num_regions);

    // TODO: figure out why all blocks and arguments are being renamed?

    size_t curr_region_index = 0;
    size_t curr_blocks_merged = 0;
    for (mlir::Region &src_region : op.getRegions()) {
      if (mlir::cast<ThreadAttr>(threads[src_region.getRegionNumber()])
              .getThreadType() == ThreadType::Datamovement) {
        assert(src_region.getBlocks().size() == 1);

        Region &dest_region = newGeneric.getRegion(curr_region_index);

        bool has_blocks = dest_region.getBlocks().size() > 0;
        if (!has_blocks) {
          rewriter.modifyOpInPlace(op,
                                   [&] { dest_region.takeBody(src_region); });
          curr_blocks_merged++;
        } else {
          Block *dest_block = &dest_region.front();
          rewriter.mergeBlocks(&src_region.front(), dest_block,
                               dest_block->getArguments());
          curr_blocks_merged++;
        }

        if (curr_blocks_merged == max_dm_blocks_per_thread) {
          curr_region_index++;
          curr_blocks_merged = 0;
        }
      } else if (mlir::cast<ThreadAttr>(threads[src_region.getRegionNumber()])
                     .getThreadType() == ThreadType::Compute) {
        Region &dest_region = newGeneric.getRegions().back();
        rewriter.modifyOpInPlace(op, [&] { dest_region.takeBody(src_region); });
      }
    }

    llvm::dbgs() << "\n------- output -------\n";
    llvm::dbgs() << newGeneric;
    llvm::dbgs() << "\n------- output -------\n";

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

    //ModuleOp moduleOp = getOperation();
    //auto systemDesc =
    //    moduleOp->getAttrOfType<SystemDescAttr>(SystemDescAttr::name);
    //auto chipDesc = systemDesc.getChipDescs().front();
    //moduleOp.walk([&](GenericOp op) {
    //  // assert that the op has a valid HW thread selection
    //  if (op.getNumRegions() > (chipDesc.getNumComputeThreads() +
    //                            chipDesc.getNumDatamovementThreads())) {
    //    op.emitError("invalid number of regions (")
    //        << op.getNumRegions() << "), expected at most ("
    //        << (chipDesc.getNumComputeThreads() +
    //            chipDesc.getNumDatamovementThreads())
    //        << ")";
    //    signalPassFailure();
    //  }
    //});
  }
};
} // namespace

} // namespace mlir::tt::ttir
