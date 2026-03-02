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

// Additional arguments to thread can be:
// 1. Global semaphores
// 2. Local semaphores
// 3. Scalar values
// This pass takes the additional argument list, and appends it to the thread argument list, and replaces all uses of the additional arguments in the thread body with the corresponding thread argument.
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

    moduleOp->walk([&](GenericOp generic) {
      for (auto arg : generic.getAdditionalArgs()) {
        assert((mlir::isa<IntegerType>(arg.getType()) ||
                mlir::isa<d2m::SemaphoreType>(arg.getType()) ||
                mlir::isa<d2m::GlobalSemaphoreType>(arg.getType())) &&
               "Additional argument type must be either integer or semaphore");

        // Iterate through all the threads and add the additional argument as a thread argument
        for (auto &region : generic.getRegions()) {
          for (auto &block : region) {
            // If it's a IntegerType, we want to create ScalarType
            Type argType = arg.getType();
            if (mlir::isa<IntegerType>(argType)) {
              argType = ScalarType::get(&getContext(), argType);
            }

            // Add the additional argument as a thread argument
            auto newArg = block.addArgument(argType, arg.getLoc());
            // Replace all uses of the additional argument within this region
            // with the newly added block argument.
            rewriter.replaceUsesWithIf(arg, newArg, [&](OpOperand &use) {
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
