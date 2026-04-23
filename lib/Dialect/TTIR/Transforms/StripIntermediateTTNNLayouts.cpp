// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRSTRIPINTERMEDIATETTNNLAYOUTS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// collect all values that should be preserved.  These include the function
// arguments, return operands, and any ToLayout op results.
static llvm::DenseSet<Value> collectPreservedValues(func::FuncOp funcOp) {
  llvm::DenseSet<Value> preserved;

  for (BlockArgument arg : funcOp.getArguments()) {
    preserved.insert(arg);
  }

  for (Block &block : funcOp.getBody()) {
    if (auto returnOp = mlir::dyn_cast<func::ReturnOp>(block.getTerminator())) {
      for (Value returnedValue : returnOp.getOperands()) {
        preserved.insert(returnedValue);
      }
    }
    // We need to respect the ttnn layout of any to_layout op, so preserve the
    // ttnn layout on the to_layout op and
    // its init (empty op)
    for (ttir::ToLayoutOp toLayoutOp : block.getOps<ttir::ToLayoutOp>()) {
      for (Value result : toLayoutOp.getResults()) {
        preserved.insert(result);
      }
      for (Value init : toLayoutOp.getDpsInits()) {
        preserved.insert(init);
      }
    }
  }

  return preserved;
}

// return a new RankedTensorType with the ttnn layout removed
static RankedTensorType stripTTNNLayout(RankedTensorType tensorType) {
  if (!mlir::isa_and_nonnull<ttnn::TTNNLayoutAttr>(tensorType.getEncoding())) {
    return tensorType;
  }

  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType());
}

class TTIRStripIntermediateTTNNLayouts
    : public impl::TTIRStripIntermediateTTNNLayoutsBase<
          TTIRStripIntermediateTTNNLayouts> {
public:
  using impl::TTIRStripIntermediateTTNNLayoutsBase<
      TTIRStripIntermediateTTNNLayouts>::TTIRStripIntermediateTTNNLayoutsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    moduleOp.walk([&](func::FuncOp func) {
      llvm::DenseSet<Value> preservedValues = collectPreservedValues(func);

      func.walk([&](Operation *op) {
        if (!isa<TTIRDialect>(op->getDialect())) {
          return;
        }
        for (Value result : op->getResults()) {
          if (preservedValues.contains(result)) {
            continue;
          }

          auto rankedTensor =
              mlir::dyn_cast<RankedTensorType>(result.getType());
          if (!rankedTensor) {
            continue;
          }

          RankedTensorType strippedTensorType = stripTTNNLayout(rankedTensor);
          if (strippedTensorType == rankedTensor) {
            continue;
          }

          result.setType(strippedTensorType);
        }
      });
    });
  }
};

} // namespace
} // namespace mlir::tt::ttir
