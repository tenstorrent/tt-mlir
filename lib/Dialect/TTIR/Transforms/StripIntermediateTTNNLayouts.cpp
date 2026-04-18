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
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRSTRIPINTERMEDIATETTNNLAYOUTS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {

// Flip to true to enable verbose tracing on llvm::errs(). When false, all
// debug prints below are dead-code-eliminated by the optimizer because the
// guard is constexpr.
static constexpr bool kDebug = false;

// collect all values that are preserved from the function arguments, return
// operands, and DPS init operands
static llvm::DenseSet<Value> collectPreservedValues(func::FuncOp funcOp) {
  if (kDebug) {
    llvm::errs() << "[strip-layouts] collectPreservedValues: entering func @"
                 << funcOp.getName() << "\n";
  }

  llvm::DenseSet<Value> preserved;
  llvm::SmallVector<Value> frontier;

  // Add `v` to the preserved set if not already present, and enqueue it on
  // the propagation frontier so its DPS-init operands (if any) get visited.
  auto markPreserved = [&](Value v, llvm::StringRef reason) {
    bool inserted = preserved.insert(v).second;
    if (inserted) {
      if (kDebug) {
        llvm::errs() << "[strip-layouts]   preserving (" << reason
                     << ") : " << v.getType() << "\n";
      }
      frontier.push_back(v);
    }
  };

  for (BlockArgument arg : funcOp.getArguments()) {
    markPreserved(arg, "func arg");
  }

  for (Block &block : funcOp.getBody()) {
    auto returnOp = mlir::dyn_cast<func::ReturnOp>(block.getTerminator());
    if (!returnOp) {
      if (kDebug) {
        llvm::errs() << "[strip-layouts]   block terminator is not "
                        "func.return; skipping\n";
      }
      continue;
    }
    for (Value returnedValue : returnOp.getOperands()) {
      markPreserved(returnedValue, "return operand");
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
      markPreserved(init, "DPS init of preserved op");
    }
  }

  if (kDebug) {
    llvm::errs() << "[strip-layouts] collectPreservedValues: returning "
                 << preserved.size() << " preserved values from func @"
                 << funcOp.getName() << "\n";
  }
  return preserved;
}

// return a new RankedTensorType with the ttnn layout removed
static RankedTensorType stripTTNNLayout(RankedTensorType tensorType) {
  if (kDebug) {
    llvm::errs() << "[strip-layouts] stripTTNNLayout: input type = "
                 << tensorType << "\n";
  }

  if (!mlir::isa_and_nonnull<ttnn::TTNNLayoutAttr>(tensorType.getEncoding())) {
    if (kDebug) {
      llvm::errs() << "[strip-layouts]   no ttnn_layout encoding; "
                      "returning input unchanged\n";
    }
    return tensorType;
  }

  RankedTensorType stripped =
      RankedTensorType::get(tensorType.getShape(), tensorType.getElementType());
  if (kDebug) {
    llvm::errs() << "[strip-layouts]   stripped to: " << stripped << "\n";
  }
  return stripped;
}

class TTIRStripIntermediateTTNNLayouts
    : public impl::TTIRStripIntermediateTTNNLayoutsBase<
          TTIRStripIntermediateTTNNLayouts> {
public:
  using impl::TTIRStripIntermediateTTNNLayoutsBase<
      TTIRStripIntermediateTTNNLayouts>::TTIRStripIntermediateTTNNLayoutsBase;

  void runOnOperation() final {
    if (kDebug) {
      llvm::errs()
          << "[strip-layouts] runOnOperation: starting pass on module\n";
    }

    ModuleOp moduleOp = getOperation();
    moduleOp.walk([&](func::FuncOp func) {
      if (kDebug) {
        llvm::errs() << "[strip-layouts] visiting func @" << func.getName()
                     << "\n";
      }

      llvm::DenseSet<Value> preservedValues = collectPreservedValues(func);

      func.walk([&](Operation *op) {
        for (Value result : op->getResults()) {
          if (preservedValues.contains(result)) {
            if (kDebug) {
              llvm::errs() << "[strip-layouts]   skip (preserved): "
                           << op->getName()
                           << " result type = " << result.getType() << "\n";
            }
            continue;
          }

          auto rt = mlir::dyn_cast<RankedTensorType>(result.getType());
          if (!rt) {
            if (kDebug) {
              llvm::errs() << "[strip-layouts]   skip (non-tensor): "
                           << op->getName()
                           << " result type = " << result.getType() << "\n";
            }
            continue;
          }

          RankedTensorType stripped = stripTTNNLayout(rt);
          if (stripped == rt) {
            if (kDebug) {
              llvm::errs() << "[strip-layouts]   skip (no change): "
                           << op->getName() << " result type = " << rt << "\n";
            }
            continue;
          }

          if (kDebug) {
            llvm::errs() << "[strip-layouts]   rewriting " << op->getName()
                         << " result : " << rt << " -> " << stripped << "\n";
          }
          result.setType(stripped);
        }
      });
    });

    if (kDebug) {
      llvm::errs() << "[strip-layouts] runOnOperation: done\n";
    }
  }
};

} // namespace
} // namespace mlir::tt::ttir
