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

// Inserts a get_buffer op for a single DMA operand that directly references a
// generic's ins/outs operand, replacing the direct reference with the result.
static void rewriteOperand(IRRewriter &rewriter, DMAOpInterface dma,
                           OpOperand &dmaOperand, unsigned operandIndex) {
  MemRefType memref = mlir::cast<MemRefType>(dmaOperand.get().getType());
  if (dmaOperand.get().getDefiningOp()) {
    std::tie(memref, std::ignore) =
        applyViews(dmaOperand.get().getDefiningOp());
  }
  rewriter.setInsertionPoint(dma);
  Operation *buf = rewriter.create<GetBufferOp>(
      dma.getLoc(), memref, operandIndex,
      ResolutionStageAttr::get(rewriter.getContext(),
                               ResolutionStage::Compile));
  dmaOperand.set(buf->getResult(0));
}

// For each DMA op inside a generic, if its src or dst directly references one
// of the generic's ins/outs operands, replace it with a get_buffer op.
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

// This pass normalizes thread arguments by inserting d2m.get_buffer,
// d2m.get_scalar, d2m.get_local_semaphore, or d2m.get_global_semaphore ops
// inside each thread block, and replacing all in-region uses of those args
// with the op results. d2m.get_cb ops are left untouched.
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
          .Case<GetScalarOp>([&](GetScalarOp scalarOp) {
            if (!scalarOp.getResolutionStageAttr()) {
              rewriter.modifyOpInPlace(scalarOp, [&]() {
                scalarOp.setResolutionStageAttr(compileAttr);
              });
            }
          })
          .Case<GetGlobalSemaphoreOp>([&](GetGlobalSemaphoreOp semOp) {
            if (!semOp.getResolutionStageAttr()) {
              rewriter.modifyOpInPlace(
                  semOp, [&]() { semOp.setResolutionStageAttr(compileAttr); });
            }
          })
          .Case<GetLocalSemaphoreOp>([&](GetLocalSemaphoreOp semOp) {
            if (!semOp.getResolutionStageAttr()) {
              rewriter.modifyOpInPlace(
                  semOp, [&]() { semOp.setResolutionStageAttr(compileAttr); });
            }
          })
          .Case<GetBufferOp>([&](GetBufferOp bufOp) {
            if (!bufOp.getResolutionStageAttr()) {
              rewriter.modifyOpInPlace(
                  bufOp, [&]() { bufOp.setResolutionStageAttr(compileAttr); });
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

      for (auto [i, arg] : llvm::enumerate(generic.getAdditionalArgs())) {
        Type argType = arg.getType();
        int64_t operandIndex = baseIndex + static_cast<int64_t>(i);

        if (mlir::isa<MemRefType>(argType)) {
          // Buffer additional args: insert get_buffer before each in-region
          // use.
          for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
            if (use.getOwner() == generic.getOperation()) {
              continue;
            }
            if (!generic->isAncestor(use.getOwner())) {
              continue;
            }
            rewriter.setInsertionPoint(use.getOwner());
            auto buf = rewriter.create<GetBufferOp>(
                use.getOwner()->getLoc(), argType, operandIndex,
                ResolutionStageAttr::get(&getContext(),
                                         ResolutionStage::Compile));
            use.set(buf.getResult());
          }
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

            Value replacement;
            auto compileTimeAttr = ResolutionStageAttr::get(
                &getContext(), ResolutionStage::Compile);
            if (mlir::isa<d2m::GlobalSemaphoreType>(argType)) {
              replacement = rewriter.create<GetGlobalSemaphoreOp>(
                  arg.getLoc(), argType, operandIndex, compileTimeAttr);
            } else if (mlir::isa<d2m::LocalSemaphoreType>(argType)) {
              replacement = rewriter.create<GetLocalSemaphoreOp>(
                  arg.getLoc(), argType, operandIndex, compileTimeAttr);
            } else {
              replacement = rewriter.create<GetScalarOp>(
                  arg.getLoc(), argType, operandIndex, compileTimeAttr);
            }
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
