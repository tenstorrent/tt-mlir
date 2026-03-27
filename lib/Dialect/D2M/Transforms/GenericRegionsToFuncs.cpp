// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MGENERICREGIONSTOFUNCS
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"

namespace {
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
      SmallVector<Attribute> threads;
      auto origThreads = generic.getThreadsAttr().getValue();
      for (Region &region : generic.getRegions()) {
        builder.setInsertionPoint(moduleOp.getBody(),
                                  moduleOp.getBody()->end());
        auto origThreadAttr =
            mlir::cast<ThreadAttr>(origThreads[region.getRegionNumber()]);
        ThreadType threadType = origThreadAttr.getThreadType();
        int32_t nocIndex = origThreadAttr.getNocIndex();
        std::string symbolName =
            stringifyEnum(threadType).str() + "_kernel" + Twine(unique++).str();
        auto threadAttrWithSym = builder.getAttr<ThreadAttr>(
            threadType, builder.getAttr<SymbolRefAttr>(symbolName), nocIndex);
        auto threadAttrWithoutSym =
            builder.getAttr<ThreadAttr>(threadType, nullptr, nocIndex);
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
          generic.getScratchInputsAttr(),
          /*numRegions*/ 0);

      generic.replaceAllUsesWith(symbolicGeneric);
      generic.erase();
    });
  }
};
} // namespace

} // namespace mlir::tt::d2m
