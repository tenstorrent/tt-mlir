// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// ElementTypeNormalization converts i1 → bf16 because TT hardware doesn't
// support boolean storage. On the CPU path this is unnecessary and wastes
// memory (e.g. a 32768×32768 causal mask: 4 GiB in f32 vs 1 GiB in i1).
//
// This pass narrows comparison/logical ops back to i1 in the CPU module,
// propagating i1 through shape-transparent ops (slice, reshape, permute,
// broadcast) and inserting typecasts only at arithmetic boundaries.

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRCPUBOOLEANNARROWING
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Ops through which i1 can propagate: logical ops and shape-manipulation ops
// that preserve element values.
static bool isBooleanTransparent(Operation *op) {
  return isa<
      // Logical ops (consume and produce booleans).
      LogicalAndOp, LogicalOrOp, LogicalXorOp, LogicalNotOp,
      // Boolean reductions.
      ReduceAndOp, ReduceOrOp,
      // Shape-manipulation (data-movement, values unchanged).
      SliceStaticOp, ReshapeOp, PermuteOp, BroadcastOp, ConcatOp, RepeatOp,
      RepeatInterleaveOp, SqueezeOp, UnsqueezeOp, ReverseOp, TransposeOp>(op);
}

class TTIRCPUBooleanNarrowing
    : public impl::TTIRCPUBooleanNarrowingBase<TTIRCPUBooleanNarrowing> {
public:
  using impl::TTIRCPUBooleanNarrowingBase<
      TTIRCPUBooleanNarrowing>::TTIRCPUBooleanNarrowingBase;

  void runOnOperation() final {
    OpBuilder builder(getOperation().getContext());

    getOperation().walk([&](Operation *op) {
      if (!isBooleanProducer(op)) {
        return;
      }
      for (OpResult result : op->getResults()) {
        if (!cast<RankedTensorType>(result.getType())
                 .getElementType()
                 .isInteger(1)) {
          narrowResult(result, builder);
        }
      }
    });
  }

private:
  // Ops whose results are boolean by semantics (entry points for narrowing).
  static bool isBooleanProducer(Operation *op) {
    return isa<EqualOp, NotEqualOp, GreaterEqualOp, GreaterThanOp, LessEqualOp,
               LessThanOp, LogicalAndOp, LogicalOrOp, LogicalXorOp,
               LogicalNotOp, ReduceAndOp, ReduceOrOp>(op);
  }

  // Narrow a result to i1 and propagate through transparent ops.
  // Insert typecasts only at non-transparent boundaries.
  void narrowResult(OpResult result, OpBuilder &builder) {
    auto origType = cast<RankedTensorType>(result.getType());
    result.setType(
        RankedTensorType::get(origType.getShape(), builder.getI1Type()));

    SmallVector<OpOperand *> nonTransparentUses;
    for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
      if (isBooleanTransparent(use.getOwner())) {
        for (OpResult r : use.getOwner()->getResults()) {
          if (!cast<RankedTensorType>(r.getType())
                   .getElementType()
                   .isInteger(1)) {
            narrowResult(r, builder);
          }
        }
      } else {
        nonTransparentUses.push_back(&use);
      }
    }

    if (!nonTransparentUses.empty()) {
      builder.setInsertionPointAfterValue(result);
      Value cast = builder.create<TypecastOp>(result.getLoc(), origType, result,
                                              /*conservativeFolding=*/false);
      for (OpOperand *use : nonTransparentUses) {
        use->set(cast);
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttir
