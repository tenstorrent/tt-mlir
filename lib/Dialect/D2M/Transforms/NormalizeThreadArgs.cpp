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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MNORMALIZETHREADARGS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Returns true if `type` is one of the supported scalar underlying types:
// bool (i1), ui8, si8, ui16, si16, ui32, si32, f16, bf16, f32, or index.
static bool isSupportedScalarType(Type type) {
  if (auto intTy = mlir::dyn_cast<IntegerType>(type)) {
    unsigned w = intTy.getWidth();
    return w == 1 || w == 8 || w == 16 || w == 32;
  }
  return mlir::isa<Float32Type, Float16Type, BFloat16Type, IndexType>(type);
}

// Inserts a get_arg op for a single DMA operand that directly references a
// generic's ins/outs operand, replacing the direct reference with the result.
static void rewriteOperand(IRRewriter &rewriter, DMAOpInterface dma,
                           OpOperand &dmaOperand, unsigned operandIndex) {
  MemRefType memref = mlir::cast<MemRefType>(dmaOperand.get().getType());
  if (dmaOperand.get().getDefiningOp()) {
    std::tie(memref, std::ignore) =
        applyViews(dmaOperand.get().getDefiningOp());
  }
  rewriter.setInsertionPoint(dma);
  Operation *buf = rewriter.create<GetArgOp>(
      dma.getLoc(), memref, operandIndex,
      ResolutionStageAttr::get(rewriter.getContext(),
                               ResolutionStage::Compile));
  dmaOperand.set(buf->getResult(0));
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

static void rewriteCapturedDMAOperands(IRRewriter &rewriter, GenericOp generic,
                                       DMAOpInterface dma) {
  auto srcIndex = getCapturedOperandIndex(generic, dma.getSrc());
  auto dstIndex = getCapturedOperandIndex(generic, dma.getDst());

  if (srcIndex) {
    rewriteOperand(rewriter, dma, dma.getSrcMutable(), *srcIndex);
  }
  if (dstIndex) {
    rewriteOperand(rewriter, dma, dma.getDstMutable(), *dstIndex);
  }
}

// This pass normalizes thread arguments by inserting d2m.get_arg ops inside
// each thread block, and replacing all in-region uses of those args with the
// op results. d2m.get_cb ops are left untouched.
// operand_index for the inserted ops matches the arg's position in the
// generic's combined operand list (ins + outs + additionalArgs).
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
      // Rewrite DMA ops that directly capture ins/outs operands.
      generic.walk([&](DMAOpInterface dma) {
        rewriteCapturedDMAOperands(rewriter, generic, dma);
      });

      int64_t baseIndex = static_cast<int64_t>(generic.getInputs().size() +
                                               generic.getOutputs().size());

      // Memref additional args (hoisted CBs) are placed in the enqueue_program
      // cbs list, not the args list. Track how many precede the current arg so
      // semaphore/scalar operandIndices reflect their actual position in args.
      int64_t numPrecedingCBArgs = 0;

      for (auto [i, arg] : llvm::enumerate(generic.getAdditionalArgs())) {
        Type argType = arg.getType();
        // Memref additional args use the full D2M position (baseIndex + i).
        // Non-memref args (semaphores, scalars) use the effective args-list
        // position, excluding preceding CB memrefs that won't appear in args.
        int64_t operandIndex =
            mlir::isa<MemRefType>(argType)
                ? baseIndex + static_cast<int64_t>(i)
                : baseIndex + static_cast<int64_t>(i) - numPrecedingCBArgs;

        if (mlir::isa<MemRefType>(argType)) {
          // Buffer additional args: insert get_arg before each in-region use.
          for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
            if (use.getOwner() == generic.getOperation()) {
              continue;
            }
            if (!generic->isAncestor(use.getOwner())) {
              continue;
            }
            rewriter.setInsertionPoint(use.getOwner());
            auto buf = rewriter.create<GetArgOp>(
                use.getOwner()->getLoc(), argType, operandIndex,
                ResolutionStageAttr::get(&getContext(),
                                         ResolutionStage::Compile));
            use.set(buf.getResult());
          }
          numPrecedingCBArgs++;
          continue;
        }

        assert((isSupportedScalarType(argType) ||
                mlir::isa<d2m::LocalSemaphoreType>(argType) ||
                mlir::isa<d2m::GlobalSemaphoreType>(argType)) &&
               "Additional argument type must be a supported scalar, "
               "semaphore, or buffer type");

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
            Value replacement = rewriter.create<GetArgOp>(
                arg.getLoc(), argType, operandIndex, compileTimeAttr);
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
