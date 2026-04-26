// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRPruneToOutput/TTIRPruneToOutput.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/BitVector.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt {

#define GEN_PASS_DEF_TTIRPRUNETOOUTPUT
#define GEN_PASS_DEF_FUNCDROPUNUSEDARGS
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt

namespace {

struct TTIRPruneToOutputPass
    : public mlir::tt::impl::TTIRPruneToOutputBase<TTIRPruneToOutputPass> {
  using Base = mlir::tt::impl::TTIRPruneToOutputBase<TTIRPruneToOutputPass>;

  TTIRPruneToOutputPass() : Base() {}

  TTIRPruneToOutputPass(const mlir::tt::TTIRPruneToOutputOptions &opts)
      : Base() {
    this->keepOutput = opts.keepOutput;
  }

  TTIRPruneToOutputPass(const TTIRPruneToOutputPass &rhs) : Base(rhs) {
    this->keepOutput = rhs.keepOutput;
  }

  void runOnOperation() override {
    if (keepOutput.empty()) {
      getOperation().emitError()
          << "ttir-prune-to-output: keep-output must be specified";
      return signalPassFailure();
    }

    // First pass: find the index in any function whose results have
    // hw.port_name attrs matching keepOutput. We assume all functions in the
    // module derive from the same hw.module port order, so the index is
    // shared (even if the SMT-lowered reference has lost its port_name attrs).
    ssize_t portIdx = -1;
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      ArrayAttr resAttrs = fn.getAllResultAttrs();
      if (!resAttrs)
        continue;
      for (size_t i = 0; i < resAttrs.size(); ++i) {
        auto dict = dyn_cast<DictionaryAttr>(resAttrs[i]);
        if (!dict)
          continue;
        if (auto nameAttr = dict.getAs<StringAttr>("hw.port_name")) {
          if (nameAttr.getValue() == keepOutput) {
            portIdx = static_cast<ssize_t>(i);
            break;
          }
        }
      }
      if (portIdx >= 0)
        break;
    }
    if (portIdx < 0) {
      getOperation().emitError() << "ttir-prune-to-output: no func.func has a "
                                    "result with hw.port_name = '"
                                 << keepOutput << "'";
      return signalPassFailure();
    }

    // Second pass: identify all functions whose result count is large enough
    // for `portIdx` to be valid.
    DenseMap<func::FuncOp, size_t> prunedFuncs;
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      if (fn.getNumResults() > static_cast<unsigned>(portIdx))
        prunedFuncs[fn] = static_cast<size_t>(portIdx);
    }

    // Drop any function whose body calls another function we're pruning,
    // because the call site would have stale result count after pruning.
    // These are typically thin wrappers (e.g., ui8/ui32 typecasts) that
    // aren't needed for LEC. We mutate the prunedFuncs map.
    SmallVector<func::FuncOp> toDrop;
    DenseSet<StringRef> candidateNames;
    for (auto &kv : prunedFuncs)
      candidateNames.insert(kv.first.getSymName());

    for (auto &kv : prunedFuncs) {
      func::FuncOp fn = kv.first;
      bool callsCandidate = false;
      fn.walk([&](func::CallOp call) {
        if (candidateNames.contains(call.getCallee()))
          callsCandidate = true;
      });
      if (callsCandidate)
        toDrop.push_back(fn);
    }
    for (auto fn : toDrop) {
      prunedFuncs.erase(fn);
      fn.erase();
    }

    // Third pass: prune the target functions themselves.
    for (auto &kv : prunedFuncs) {
      func::FuncOp fn = kv.first;
      size_t keepIdx = kv.second;

      unsigned numResults = fn.getNumResults();
      llvm::BitVector toErase(numResults, true);
      toErase.reset(static_cast<unsigned>(keepIdx));

      fn.walk([&](func::ReturnOp ret) {
        SmallVector<Value> kept;
        kept.push_back(ret.getOperand(static_cast<unsigned>(keepIdx)));
        OpBuilder b(ret);
        func::ReturnOp::create(b, ret.getLoc(), kept);
        ret.erase();
      });

      if (failed(fn.eraseResults(toErase))) {
        fn.emitError() << "failed to erase results";
        return signalPassFailure();
      }
    }

  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>>
createTTIRPruneToOutputPass(const TTIRPruneToOutputOptions &options) {
  return std::make_unique<TTIRPruneToOutputPass>(options);
}

namespace {
struct FuncDropUnusedArgsPass
    : public mlir::tt::impl::FuncDropUnusedArgsBase<FuncDropUnusedArgsPass> {
  void runOnOperation() override {
    for (auto fn : getOperation().getOps<func::FuncOp>()) {
      unsigned numArgs = fn.getNumArguments();
      llvm::BitVector argsToErase(numArgs, false);
      for (unsigned i = 0; i < numArgs; ++i) {
        if (fn.getArgument(i).use_empty())
          argsToErase.set(i);
      }
      if (argsToErase.none())
        continue;
      // Don't drop args of functions that are called from elsewhere — the
      // call sites would break.
      bool hasCallers = false;
      getOperation().walk([&](func::CallOp call) {
        if (call.getCallee() == fn.getSymName())
          hasCallers = true;
      });
      if (hasCallers)
        continue;
      if (failed(fn.eraseArguments(argsToErase))) {
        fn.emitError() << "failed to erase unused arguments";
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createFuncDropUnusedArgsPass() {
  return std::make_unique<FuncDropUnusedArgsPass>();
}

} // namespace mlir::tt
