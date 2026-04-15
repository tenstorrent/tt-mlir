// SPDX-FileCopyrightText: (c) 2026wr Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MMARKSYNCHRONIZEDOPBUFFERS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

class D2MMarkSynchronizedOpBuffers : public impl::D2MMarkSynchronizedOpBuffersBase<D2MMarkSynchronizedOpBuffers> {
public:
  using impl::D2MMarkSynchronizedOpBuffersBase<D2MMarkSynchronizedOpBuffers>::D2MMarkSynchronizedOpBuffersBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp->walk(
        [&](d2m::GenericOp genericOp) { 
            if (failed(markSynchronizedOpBuffers(rewriter, genericOp))) {
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
        });
  }

  private:

  LogicalResult insertCBLayoutAttr(RewriterBase &rewriter, memref::AllocOp allocOp, llvm::SmallVector<Operation*> &producers, bool adjustResult) {
    auto buffer = allocOp->getResult(0);
    auto bufferType = mlir::cast<MemRefType>(buffer.getType());
    auto shardShape = bufferType.getShape();
    llvm::errs() << "\n";
    for (int64_t shape : shardShape) {
      llvm::errs() << shape << " ";
    }
    llvm::errs() << "\n";
    Type elementType = bufferType.getElementType();
    llvm::errs() << "element type: " << elementType << "\n";

    auto cbLayout = ttcore::CBLayoutAttr::get(
        bufferType.getContext(), shardShape,
        ttcore::getElementSizeBytes(elementType), 2); // TODO: remove hardcoded value, actually change to just add d2m.double_buffered attribute to alloc op
    
    auto newMemRefType = MemRefType::get(shardShape, elementType, cbLayout,
        bufferType.getMemorySpace());
    rewriter.setInsertionPoint(allocOp);
    auto newAllocOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(allocOp, 
        newMemRefType);
    Value newBuffer = newAllocOp->getResult(0);

    // check ops that have this buffer as a dps init, we need to change their output type to also have cb layout
    if (adjustResult) {
      for (auto& producer: producers) {
        if (auto destinationStyleOp = dyn_cast<DestinationStyleOpInterface>(producer)) {
          for (auto& operand: destinationStyleOp->getOpOperands()) {
            if (destinationStyleOp.isDpsInit(&operand) && operand.get() == newBuffer) {
                // update op result that matches dps init
                OpResult tiedResult = destinationStyleOp.getTiedOpResult(&operand);
                tiedResult.setType(newMemRefType);
            }
          }
        }
      }
    }

    // Transfer address and alignment from the old alloc (assigned
    // by the planner in assignAllocAddresses).
    //if (auto addrAttr = allocOp->getAttr("address")) {
    //  newAllocOp->setAttr("address", addrAttr);
    //}
    //if (auto alignAttr = allocOp.getAlignmentAttr()) {
    //  newAllocOp.setAlignmentAttr(alignAttr);
    //}
    // TODO: can remove this above addr and alignement set once reordering 
    // cb layout attr insertion to top of allocator methods
    return success();
  }

  LogicalResult markSynchronizedOpBuffers(IRRewriter &rewriter, d2m::GenericOp genericOp) {   
    // print IR at this point
    llvm::errs() << "IR at this point: " << "\n";
    genericOp->print(llvm::errs());
    llvm::errs() << "\n";


    // hack adjust linalg and tile ops to be synchronized region ops
    genericOp->getRegions().front().walk([&](Operation *op) {
      if (auto linalgGenericOp = dyn_cast<mlir::linalg::GenericOp>(op)) {
        // wrap in synchronized region op
        rewriter.setInsertionPoint(op);
        rewriter.create<d2m::SynchronizedRegionOp>(linalgGenericOp->getLoc(), 
        linalgGenericOp.getOutputs().getTypes(), linalgGenericOp.getInputs(), linalgGenericOp.getOutputs(), 
          [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc, mlir::ValueRange bbArgs) {
            auto newLinalgGenericOp = bbBuilder.create<mlir::linalg::GenericOp>(
              bbLoc, TypeRange(linalgGenericOp->getResultTypes()), 
              bbArgs.take_front(linalgGenericOp.getInputs().size()), 
              bbArgs.take_back(linalgGenericOp.getOutputs().size()), 
              linalgGenericOp.getIndexingMapsArray(), linalgGenericOp.getIteratorTypesArray());
            IRMapping mapping;
            linalgGenericOp.getRegion().cloneInto(&newLinalgGenericOp.getRegion(), mapping);
            bbBuilder.create<d2m::YieldOp>(bbLoc, newLinalgGenericOp.getOutputs());
          });
        
          
        rewriter.eraseOp(linalgGenericOp);
      }
      else if (auto tileUntilizeBlockOp = dyn_cast<d2m::TileUntilizeBlockOp>(op)) {
        // wrap in synchronized region op
        rewriter.setInsertionPoint(op);
        auto newSynchronizedRegionOp = rewriter.create<d2m::SynchronizedRegionOp>(tileUntilizeBlockOp->getLoc(), tileUntilizeBlockOp->getResultTypes(), tileUntilizeBlockOp.getInput(), tileUntilizeBlockOp.getOutput(), 
          [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc, mlir::ValueRange bbArgs) {
            IRMapping mapping;
            mapping.map(tileUntilizeBlockOp.getInput(), bbArgs[0]);
            mapping.map(tileUntilizeBlockOp.getOutput(), bbArgs[1]);
            auto newTileUntilizeBlockOp = bbBuilder.clone(*tileUntilizeBlockOp, mapping);
            bbBuilder.create<d2m::YieldOp>(bbLoc, newTileUntilizeBlockOp->getResults());
          });
        rewriter.replaceOp(tileUntilizeBlockOp, newSynchronizedRegionOp.getOperation());
      }
      else if (auto tileTilizeBlockOp = dyn_cast<d2m::TileTilizeBlockOp>(op)) {
        // wrap in synchronized region op
        rewriter.setInsertionPoint(op);
        auto newSynchronizedRegionOp = rewriter.create<d2m::SynchronizedRegionOp>(tileTilizeBlockOp->getLoc(), tileTilizeBlockOp->getResultTypes(), tileTilizeBlockOp.getInput(), tileTilizeBlockOp.getOutput(), 
          [&](mlir::OpBuilder &bbBuilder, mlir::Location bbLoc, mlir::ValueRange bbArgs) {
            IRMapping mapping;
            mapping.map(tileTilizeBlockOp.getInput(), bbArgs[0]);
            mapping.map(tileTilizeBlockOp.getOutput(), bbArgs[1]);
            auto newTileTilizeBlockOp = bbBuilder.clone(*tileTilizeBlockOp, mapping);
            bbBuilder.create<d2m::YieldOp>(bbLoc, newTileTilizeBlockOp->getResults());
          });
        rewriter.replaceOp(tileTilizeBlockOp, newSynchronizedRegionOp.getOperation());
      }

      return WalkResult::advance();
    });

    auto cbUsageInfo = getCBUsageInfo(genericOp.getRegion(0));
    

    // loop over memrefs in generic, separate out to separate pass or rewriter?
    // go through ops with synchronizable interface being used as input and attach CB layout aatribute on their 
    // operands using interface to tell which are Synchronized inputs and which are Synchronized outputs
    // separate out handling for allocs inside generic and outside
    for (auto& [cb, usageInfo]: cbUsageInfo) {
      // TODO: add check on num consumers and producers
      llvm::errs() << "allocating CB for op x: true" << "\n";
      //TODO: this is is an inconsistency, most remote load/store will have to correctly be bufferized to not return a result 
      bool adjustResult = !mlir::isa<mlir::linalg::GenericOp>(usageInfo.producers[0]);
      if (failed(insertCBLayoutAttr(rewriter, mlir::cast<memref::AllocOp>(cb.getDefiningOp()), usageInfo.producers, adjustResult))) {
        return failure();
      }
    }

    

    return success();
  }

};

} // namespace
} // namespace mlir::tt::d2m
