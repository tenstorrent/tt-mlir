// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRGENERICREGIONSTOFUNCS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

static std::optional<unsigned> getCapturedOperandIndex(GenericOp op,
                                                       Value operand) {
  for (OpOperand &opOperand : op->getOpOperands()) {
    if (opOperand.get() == operand) {
      return opOperand.getOperandNumber();
    }
  }
  return std::nullopt;
}

static void rewriteOperand(OpBuilder &builder, DMAOpInterface dma,
                           OpOperand &dmaOperand, unsigned operandIndex) {
  MemRefType memref = mlir::cast<MemRefType>(dmaOperand.get().getType());
  AffineMap affineMapView = builder.getMultiDimIdentityMap(memref.getRank());
  if (dmaOperand.get().getDefiningOp()) {
    std::tie(memref, affineMapView) =
        applyViews(dmaOperand.get().getDefiningOp());
  }
  Operation *globalOperand =
      builder.create<GetGlobalOperandOp>(dma.getLoc(), memref, operandIndex);
  dmaOperand.set(globalOperand->getResult(0));
}

static void rewriteCapturedDMAOperands(OpBuilder &builder, GenericOp generic,
                                       DMAOpInterface dma) {

  auto srcIndex = getCapturedOperandIndex(generic, dma.getSrc());
  auto dstIndex = getCapturedOperandIndex(generic, dma.getDst());

  builder.setInsertionPoint(dma);
  if (srcIndex) {
    rewriteOperand(builder, dma, dma.getSrcMutable(), *srcIndex);
  }

  if (dstIndex) {
    rewriteOperand(builder, dma, dma.getDstMutable(), *dstIndex);
  }
}

namespace {
class TTIRGenericRegionsToFuncs
    : public impl::TTIRGenericRegionsToFuncsBase<TTIRGenericRegionsToFuncs> {
public:
  using impl::TTIRGenericRegionsToFuncsBase<
      TTIRGenericRegionsToFuncs>::TTIRGenericRegionsToFuncsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(&getContext());
    int unique = 0;
    moduleOp->walk([&](GenericOp generic) {
      generic.walk([&](DMAOpInterface dma) {
        rewriteCapturedDMAOperands(builder, generic, dma);
      });

      SmallVector<Attribute> threads;
      for (Region &region : generic.getRegions()) {
        builder.setInsertionPoint(moduleOp.getBody(),
                                  moduleOp.getBody()->end());
        ThreadType threadType =
            generic.getRegionThreadType(region.getRegionNumber());
        std::string symbolName =
            stringifyEnum(generic.getRegionThreadType(region.getRegionNumber()))
                .str() +
            "_kernel" + Twine(unique++).str();
        auto threadAttrWithSym = builder.getAttr<ThreadAttr>(
            threadType, builder.getAttr<SymbolRefAttr>(symbolName));
        auto threadAttrWithoutSym =
            builder.getAttr<ThreadAttr>(threadType, nullptr);
        Location loc = region.getNumArguments() > 0
                           ? region.getArgument(0).getLoc()
                           : generic.getLoc();
        auto func = builder.create<func::FuncOp>(
            loc, symbolName,
            FunctionType::get(builder.getContext(), region.getArgumentTypes(),
                              {}));
        func.setPrivate();
        func->setAttr(ttir::ThreadAttr::name, threadAttrWithoutSym);
        func.getBody().takeBody(region);
        builder.setInsertionPointToEnd(&func.getBody().front());
        builder.create<func::ReturnOp>(generic.getLoc());
        threads.push_back(threadAttrWithSym);
      }

      builder.setInsertionPoint(generic);
      auto symbolicGeneric = builder.create<GenericOp>(
          generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
          generic.getOutputs(), generic.getGrid(), generic.getBlockFactors(),
          generic.getIndexingMaps(), generic.getIteratorTypes(),
          builder.getArrayAttr(threads),
          /*numRegions*/ 0);

      generic.replaceAllUsesWith(symbolicGeneric);
      generic.erase();
    });
  }
};
} // namespace

} // namespace mlir::tt::ttir
