// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsInterfaces.h"
#include "ttmlir/FunctionTypes.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MNORMALIZETHREADARGS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

static void rewriteOperand(IRRewriter &rewriter, Operation *op,
                           OpOperand &operand, unsigned operandIndex) {
  MemRefType memref = mlir::cast<MemRefType>(operand.get().getType());
  if (operand.get().getDefiningOp()) {
    std::tie(memref, std::ignore) = applyViews(operand.get().getDefiningOp());
  }
  rewriter.setInsertionPoint(op);
  Operation *buf = rewriter.create<GetArgOp>(
      op->getLoc(), memref, operandIndex,
      ResolutionStageAttr::get(rewriter.getContext(),
                               ResolutionStage::Compile));
  operand.set(buf->getResult(0));
}

// For each DMA op inside a generic, if its src or dst directly references one
// of the generic's ins/outs operands, replace it with a get_arg op.
static std::optional<unsigned> getCapturedOperandIndex(GenericOp generic,
                                                       Value operand) {
  for (OpOperand &opOperand : generic->getOpOperands()) {
    if (opOperand.get() == operand) {
      return opOperand.getOperandNumber();
    }
  }
  return std::nullopt;
}

static void rewriteCapturedIndexedRowCopyOperands(IRRewriter &rewriter,
                                                  GenericOp generic,
                                                  IndexedRowCopyOp copyOp) {
  for (OpOperand &operand : copyOp->getOpOperands()) {
    if (operand.get() == copyOp.getIndexScratch() ||
        operand.get() == copyOp.getRowScratch()) {
      continue;
    }
    auto operandIndex = getCapturedOperandIndex(generic, operand.get());
    if (operandIndex) {
      rewriteOperand(rewriter, copyOp.getOperation(), operand, *operandIndex);
    }
  }
}

// This pass normalizes thread arguments by inserting d2m.get_arg ops inside
// each thread block, and replacing all in-region uses of those args with the
// op results. d2m.get_cb ops are left untouched.
// operand_index for inserted d2m.get_arg ops follows the normalized
// enqueue_program/thread-arg numbering used by this pass, not raw GenericOp
// operand numbering. In particular, memref CB args are accessed via
// d2m.get_cb, and non-CB additionalArgs are indexed after excluding any
// preceding memref CB additionalArgs.
class D2MNormalizeThreadArgs
    : public impl::D2MNormalizeThreadArgsBase<D2MNormalizeThreadArgs> {
public:
  using impl::D2MNormalizeThreadArgsBase<
      D2MNormalizeThreadArgs>::D2MNormalizeThreadArgsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Fill in resolution_stage = compile on any existing get_* ops that don't
    // already have the attribute set.
    auto compileAttr =
        ResolutionStageAttr::get(&getContext(), ResolutionStage::Compile);
    moduleOp->walk([&](Operation *op) {
      TypeSwitch<Operation *>(op)
          .Case<GetCBOp>([&](GetCBOp cbOp) {
            if (!cbOp.getResolutionStageAttr()) {
              rewriter.modifyOpInPlace(
                  cbOp, [&]() { cbOp.setResolutionStageAttr(compileAttr); });
            }
          })
          .Case<GetArgOp>([&](GetArgOp argOp) {
            if (!argOp.getResolutionStageAttr()) {
              rewriter.modifyOpInPlace(
                  argOp, [&]() { argOp.setResolutionStageAttr(compileAttr); });
            }
          });
    });

    moduleOp->walk([&](GenericOp generic) {
      generic.walk([&](IndexedRowCopyOp copyOp) {
        rewriteCapturedIndexedRowCopyOperands(rewriter, generic, copyOp);
      });

      for (auto [i, arg] : llvm::enumerate(generic.getOperands())) {
        Type argType = arg.getType();

        if (!mlir::isa<MemRefType, d2m::LocalSemaphoreType,
                       d2m::GlobalSemaphoreType, IndexType, IntegerType,
                       FloatType>(argType)) {
          generic.emitOpError(
              "unsupported argument type in d2m.generic operands: ")
              << argType
              << "; only memref, semaphore, and scalar types are supported";
          signalPassFailure();
          return;
        }

        for (auto &region : generic.getRegions()) {
          // Skip this region if the arg is not used anywhere inside it.
          bool usedInRegion = llvm::any_of(arg.getUses(), [&](OpOperand &use) {
            Block *ownerBlock = use.getOwner()->getBlock();
            return ownerBlock &&
                   region.findAncestorBlockInRegion(*ownerBlock) != nullptr;
          });
          if (!usedInRegion) {
            continue;
          }

          for (auto &block : region) {
            rewriter.setInsertionPointToStart(&block);

            auto compileTimeAttr = ResolutionStageAttr::get(
                &getContext(), ResolutionStage::Compile);
            Value replacement = rewriter.create<GetArgOp>(arg.getLoc(), argType,
                                                          i, compileTimeAttr);
            rewriter.setInsertionPointAfter(replacement.getDefiningOp());

            rewriter.replaceUsesWithIf(arg, replacement, [&](OpOperand &use) {
              Block *ownerBlock = use.getOwner()->getBlock();
              return ownerBlock &&
                     region.findAncestorBlockInRegion(*ownerBlock) != nullptr;
            });
          }
        }
      }
    });
  }
};
} // namespace

} // namespace mlir::tt::d2m
