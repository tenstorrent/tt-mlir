// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/D2M/IR/D2MGenericRegionOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICREGIONSTOFUNCS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
static SmallVector<Value>
mapVirtualToPhysicalCoreIndex(OpBuilder &builder, Location loc,
                              ttcore::GridAttr grid,
                              ValueRange virtualCoreIndex) {
  AffineMap map = grid.getVirtToPhysicalMap();
  if (!map || map.isEmpty()) {
    return SmallVector<Value>(virtualCoreIndex.begin(), virtualCoreIndex.end());
  }

  TT_assertv(map.getNumDims() == virtualCoreIndex.size(),
             "Expected virtual-to-physical grid map input rank to match core "
             "index rank.");
  unsigned firstCoreResult =
      map.getNumResults() == virtualCoreIndex.size() ? 0 : 1;
  TT_assertv(map.getNumResults() >= firstCoreResult + virtualCoreIndex.size(),
             "Expected virtual-to-physical grid map to have enough core "
             "coordinate results.");

  SmallVector<Value> physicalCoreIndex;
  physicalCoreIndex.reserve(virtualCoreIndex.size());
  for (unsigned i = 0; i < virtualCoreIndex.size(); ++i) {
    AffineMap selectedMap = AffineMap::get(
        map.getNumDims(), map.getNumSymbols(),
        {map.getResult(firstCoreResult + i)}, builder.getContext());
    physicalCoreIndex.push_back(builder.create<affine::AffineApplyOp>(
        loc, selectedMap, virtualCoreIndex));
  }
  return physicalCoreIndex;
}

static void annotateCoreIndexOpsWithPhysicalToVirtualMaps(GenericOp generic) {
  AffineMap physicalToVirtualMap = generic.getGrid().getPhysicalToVirtMap();
  if (!physicalToVirtualMap || physicalToVirtualMap.isEmpty()) {
    return;
  }

  generic.walk([&](CoreIndexOp coreIndexOp) {
    if (coreIndexOp->getParentOfType<GenericOp>() != generic) {
      return;
    }
    coreIndexOp.setPhysToVirtMapAttr(AffineMapAttr::get(physicalToVirtualMap));
  });
}

static void
materializeCoreCoordinateOperandsInPhysicalSpace(GenericOp generic,
                                                 OpBuilder &builder) {
  ttcore::GridAttr grid = generic.getGrid();
  if (!grid.getVirtToPhysicalMap() || grid.getVirtToPhysicalMap().isEmpty()) {
    return;
  }

  generic.walk([&](DMAWriteOp dmaWriteOp) {
    if (dmaWriteOp->getParentOfType<GenericOp>() != generic ||
        dmaWriteOp.getMcastStartIndex().empty() ||
        !dmaWriteOp.getStartDevice().empty()) {
      return;
    }
    builder.setInsertionPoint(dmaWriteOp);
    SmallVector<Value> physicalMcastStartIndex = mapVirtualToPhysicalCoreIndex(
        builder, dmaWriteOp.getLoc(), grid, dmaWriteOp.getMcastStartIndex());
    dmaWriteOp.getMcastStartIndexMutable().assign(physicalMcastStartIndex);
  });

  auto rewriteSemaphoreCoreIndex = [&](auto semaphoreOp) {
    if (semaphoreOp->template getParentOfType<GenericOp>() != generic ||
        semaphoreOp.getDstCoreIndex().empty() ||
        !semaphoreOp.getStartDevice().empty()) {
      return;
    }
    builder.setInsertionPoint(semaphoreOp);
    SmallVector<Value> physicalDstCoreIndex = mapVirtualToPhysicalCoreIndex(
        builder, semaphoreOp.getLoc(), grid, semaphoreOp.getDstCoreIndex());
    semaphoreOp.getDstCoreIndexMutable().assign(physicalDstCoreIndex);
  };
  generic.walk([&](SemaphoreSetOp semaphoreSetOp) {
    rewriteSemaphoreCoreIndex(semaphoreSetOp);
  });
  generic.walk([&](SemaphoreIncOp semaphoreIncOp) {
    rewriteSemaphoreCoreIndex(semaphoreIncOp);
  });
}

static int32_t resolveDmCoreIndex(ThreadAttr thread,
                                  ttcore::ChipDescAttr chipDesc,
                                  int &unassignedDmCoreCounter) {
  int32_t dmCoreIndex = thread.getDmCoreIndex();
  // Handle unassigned DM core.
  if (thread.getThreadType() == ThreadType::Datamovement && dmCoreIndex < 0) {
    const auto arch = chipDesc.getArch().getValue();
    const auto nDmCores = chipDesc.getNumDatamovementThreads();
    if (arch == ttcore::Arch::Quasar) {
      // For Quasar, the downstream passes will force assign NoC0.
      dmCoreIndex = unassignedDmCoreCounter++ % nDmCores;
    } else {
      // For WH & BH, alternate between Core1-NoC0 and Core0-NoC1.
      const int32_t nocIdx = unassignedDmCoreCounter++ % nDmCores;
      dmCoreIndex = 1 - nocIdx;
      // Make sure this is the inverse of ttcore::getDmCoreDefaultNoc.
      TT_assertv(static_cast<int32_t>(
                     ttcore::getDmCoreDefaultNoc(arch, dmCoreIndex)) == nocIdx,
                 "Fallback DM-core assignment disagrees with "
                 "ttcore::getDmCoreDefaultNoc.");
    }
  }
  return dmCoreIndex;
}

static void materializeCapturedConstants(func::FuncOp func) {
  OpBuilder builder(func.getContext());
  Block &body = func.getBody().front();
  llvm::DenseMap<Operation *, Value> clonedConstants;

  func.walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      Operation *definingOp = operand.get().getDefiningOp();
      if (!definingOp || definingOp->getParentOfType<func::FuncOp>() == func) {
        continue;
      }

      if (!mlir::isa<arith::ConstantOp>(definingOp)) {
        continue;
      }

      auto [it, inserted] = clonedConstants.try_emplace(definingOp, Value{});
      if (inserted) {
        builder.setInsertionPointToStart(&body);
        Operation *clonedOp = builder.clone(*definingOp);
        it->second = clonedOp->getResult(0);
      }
      operand.set(it->second);
    }
  });
}

class D2MGenericRegionsToFuncs
    : public impl::D2MGenericRegionsToFuncsBase<D2MGenericRegionsToFuncs> {
public:
  using impl::D2MGenericRegionsToFuncsBase<
      D2MGenericRegionsToFuncs>::D2MGenericRegionsToFuncsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());
    int unique = 0;
    moduleOp->walk([&](GenericOp generic) {
      annotateCoreIndexOpsWithPhysicalToVirtualMaps(generic);
      materializeCoreCoordinateOperandsInPhysicalSpace(generic, builder);

      SmallVector<Attribute> threads;
      auto origThreads = generic.getThreadsAttr().getValue();
      const auto chipDesc = ttcore::getOpChipDescAttr(generic);
      int unassignedDmCoreCounter = 0;
      for (Region &region : generic.getRegions()) {
        builder.setInsertionPoint(moduleOp.getBody(),
                                  moduleOp.getBody()->end());
        auto origThreadAttr =
            mlir::cast<ThreadAttr>(origThreads[region.getRegionNumber()]);
        ThreadType threadType = origThreadAttr.getThreadType();
        const int32_t dmCoreIndex = resolveDmCoreIndex(origThreadAttr, chipDesc,
                                                       unassignedDmCoreCounter);
        std::string symbolName =
            stringifyEnum(threadType).str() + "_kernel" + Twine(unique++).str();
        auto threadAttrWithSym = builder.getAttr<ThreadAttr>(
            threadType, builder.getAttr<SymbolRefAttr>(symbolName),
            dmCoreIndex);
        auto threadAttrWithoutSym =
            builder.getAttr<ThreadAttr>(threadType, nullptr, dmCoreIndex);
        Location loc = region.getNumArguments() > 0
                           ? region.getArgument(0).getLoc()
                           : generic.getLoc();
        auto func = builder.create<func::FuncOp>(
            loc, symbolName,
            FunctionType::get(builder.getContext(), region.getArgumentTypes(),
                              {}));
        func.setPrivate();
        func->setAttr(d2m::ThreadAttr::name, threadAttrWithoutSym);
        func.getBody().takeBody(region);
        materializeCapturedConstants(func);
        ttmlir::utils::setFunctionType(func,
                                       ttmlir::utils::FunctionType::Kernel);
        builder.setInsertionPointToEnd(&func.getBody().front());
        builder.create<func::ReturnOp>(generic.getLoc());
        threads.push_back(threadAttrWithSym);
      }

      builder.setInsertionPoint(generic);
      auto symbolicGeneric = builder.create<GenericOp>(
          generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
          generic.getOutputs(), generic.getAdditionalArgs(), generic.getGrid(),
          generic.getBlockFactors(), generic.getIndexingMaps(),
          generic.getIteratorTypes(), builder.getArrayAttr(threads),
          generic.getFabricConnectionConfigAttr(),
          /*numRegions*/ 0);

      generic.replaceAllUsesWith(symbolicGeneric);
      generic.erase();
    });
  }
};
} // namespace

} // namespace mlir::tt::d2m
