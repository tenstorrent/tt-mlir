// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/GenericInterchangeAnalysis.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

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

static void rewriteOperand(OpBuilder &builder, DMAOp dma, OpOperand &dmaOperand,
                           unsigned operandIndex) {
  auto globalOperand = builder.create<GetGlobalOperandOp>(
      dma.getLoc(), dmaOperand.get().getType(), operandIndex);
  dmaOperand.set(globalOperand.getResult());
}

static void rewriteCapturedDMAOperands(OpBuilder &builder, GenericOp generic,
                                       DMAOp dma) {
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
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());
    static int unique = 0;
    module->walk([&](GenericOp generic) {
      generic.walk([&](DMAOp dma) {
        rewriteCapturedDMAOperands(builder, generic, dma);
      });

      SmallVector<Attribute> symbols;
      for (Region &region : generic.getRegions()) {
        builder.setInsertionPoint(module.getBody(), module.getBody()->end());
        std::string symbolName = "kernel" + std::to_string(unique++);
        auto func = builder.create<func::FuncOp>(
            generic.getLoc(), symbolName,
            FunctionType::get(builder.getContext(), region.getArgumentTypes(),
                              {}));
        func.setPrivate();
        func.getBody().takeBody(region);
        builder.setInsertionPointToEnd(&func.getBody().front());
        builder.create<func::ReturnOp>(generic.getLoc());

        symbols.push_back(builder.getAttr<SymbolRefAttr>(symbolName));
      }

      builder.setInsertionPoint(generic);
      auto symbolicGeneric = builder.create<GenericOp>(
          generic->getLoc(), generic.getResultTypes(), generic.getInputs(),
          generic.getOutputs(), generic.getGrid(), generic.getIndexingMaps(),
          generic.getIteratorTypes(), builder.getArrayAttr(symbols),
          /*numRegions*/ 0);

      generic.replaceAllUsesWith(symbolicGeneric);
      generic.erase();
    });
  }
};
} // namespace

} // namespace mlir::tt::ttir
