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

// collect all values that are preserved from the function arguments, return
// operands, and DPS init operands
static llvm::DenseSet<Value> collectPreservedValues(func::FuncOp funcOp) {
  llvm::DenseSet<Value> preserved;
  llvm::SmallVector<Value> frontier;

  // Add `v` to the preserved set if not already present, and enqueue it on
  // the propagation frontier so its DPS-init operands (if any) get visited.
  auto markPreserved = [&](Value v) {
    if (preserved.insert(v).second) {
      frontier.push_back(v);
    }
  };

  for (BlockArgument arg : funcOp.getArguments()) {
    markPreserved(arg);
  }

  for (Block &block : funcOp.getBody()) {
    auto returnOp = mlir::dyn_cast<func::ReturnOp>(block.getTerminator());
    if (!returnOp) {
      continue;
    }
    for (Value returnedValue : returnOp.getOperands()) {
      markPreserved(returnedValue);
    }
  }

  // For any preserved value defined by a DPS op, the
  // DestinationStyleOpInterface verifier requires the op's DPS init operands to
  // share their result's type (including the ttnn_layout encoding). Stripping
  // such an init operand while keeping its result encoded would produce an
  // invalid op, so we propagate preservation backward through DPS init
  // operands.
  while (!frontier.empty()) {
    Value v = frontier.pop_back_val();
    Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      continue;
    }
    auto dps = mlir::dyn_cast<DestinationStyleOpInterface>(defOp);
    if (!dps) {
      continue;
    }
    for (Value init : dps.getDpsInits()) {
      markPreserved(init);
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

          auto rt = mlir::dyn_cast<RankedTensorType>(result.getType());
          if (!rt) {
            continue;
          }

          RankedTensorType stripped = stripTTNNLayout(rt);
          if (stripped == rt) {
            continue;
          }

          result.setType(stripped);
        }
      });
    });
  }
};

} // namespace
} // namespace mlir::tt::ttir
