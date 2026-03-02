// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MCAPTUREADDITIONALARGSINTHREADS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {

// Returns true if `type` is one of the supported scalar underlying types:
// bool (i1), ui8, si8, ui16, si16, ui32, si32, f16, bf16, f32.
static bool isSupportedScalarType(Type type) {
  if (auto intTy = mlir::dyn_cast<IntegerType>(type)) {
    unsigned w = intTy.getWidth();
    return w == 1 || w == 8 || w == 16 || w == 32;
  }
  return mlir::isa<Float32Type, Float16Type, BFloat16Type>(type);
}

// Returns true if `type` is !d2m.scalar<ui32>.
static bool isUi32ScalarType(Type type) {
  auto intTy = mlir::dyn_cast<IntegerType>(type);
  return intTy && intTy.getWidth() == 32 && intTy.isUnsigned();
}

// Additional arguments to thread can be:
// 1. Global semaphores
// 2. Local semaphores
// 3. Scalar values (ui32, si32, ui16, si16, ui8, si8, f32, f16, bf16)
// This pass takes the additional argument list, appends it to the thread
// argument list as !d2m.scalar<ui32>, inserts a d2m.reinterpret_cast for
// non-ui32 types, and replaces all uses of the additional arguments in the
// thread body with the (cast) block argument.
class D2MCaptureAdditionalArgsInThreads
    : public impl::D2MCaptureAdditionalArgsInThreadsBase<
          D2MCaptureAdditionalArgsInThreads> {
public:
  using impl::D2MCaptureAdditionalArgsInThreadsBase<
      D2MCaptureAdditionalArgsInThreads>::
      D2MCaptureAdditionalArgsInThreadsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    Type ui32Type =
        IntegerType::get(&getContext(), 32, IntegerType::Unsigned);
    Type scalarUi32Type = ScalarType::get(&getContext(), ui32Type);

    moduleOp->walk([&](GenericOp generic) {
      for (auto arg : generic.getAdditionalArgs()) {
        Type argType = arg.getType();
        assert((isSupportedScalarType(argType) ||
                mlir::isa<d2m::LocalSemaphoreType>(argType) ||
                mlir::isa<d2m::GlobalSemaphoreType>(argType)) &&
               "Additional argument type must be a supported scalar or "
               "semaphore type");

        for (auto &region : generic.getRegions()) {
          for (auto &block : region) {
            Type blockArgType;
            if (mlir::isa<d2m::LocalSemaphoreType>(argType) ||
                mlir::isa<d2m::GlobalSemaphoreType>(argType)) {
              blockArgType = argType;
            } else {
              // All scalar types use !d2m.scalar<ui32> as the block arg type.
              blockArgType = scalarUi32Type;
            }

            auto newArg = block.addArgument(blockArgType, arg.getLoc());

            // For non-ui32 scalar types, insert a reinterpret_cast at the
            // start of the block to convert !d2m.scalar<ui32> to the original
            // scalar type, and replace uses with the cast result.
            Value replacement = newArg;
            if (isSupportedScalarType(argType) && !isUi32ScalarType(argType)) {
              Type scalarOrigType = ScalarType::get(&getContext(), argType);
              rewriter.setInsertionPointToStart(&block);
              auto cast = rewriter.create<ReinterpretCastOp>(
                  arg.getLoc(), scalarOrigType, newArg);
              replacement = cast.getResult();
            }

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
