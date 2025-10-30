// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// PassTracker - Compiled with -fno-rtti to match MLIR

#include "mlir-c/Pass.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"

namespace {

// Custom PrettyStackTrace entry for MLIR pass execution
// This entry is persistent and updated rather than created/destroyed
// to avoid stack ordering issues
class PassPrettyStackTraceEntry : public llvm::PrettyStackTraceEntry {
  mlir::Pass *pass = nullptr;
  mlir::Operation *op = nullptr;
  bool active = false;

public:
  void setPass(mlir::Pass *p, mlir::Operation *o) {
    pass = p;
    op = o;
    active = true;
  }

  void clear() {
    pass = nullptr;
    op = nullptr;
    active = false;
  }

  void print(llvm::raw_ostream &OS) const override {
    if (!active || !pass) {
      return;
    }

    OS << "Running pass '" << pass->getName() << "'";
    if (op) {
      OS << " on operation '" << op->getName() << "'";
      if (auto loc = op->getLoc(); !mlir::isa<mlir::UnknownLoc>(loc)) {
        OS << " at " << loc;
      }
    }
    OS << "\n";
  }
};

// PassInstrumentation that tracks current pass for crash reporting
class PassTrackerInstrumentation : public mlir::PassInstrumentation {
  // Persistent entry that stays registered for the lifetime of the
  // instrumentation
  PassPrettyStackTraceEntry traceEntry;

public:
  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override {
    traceEntry.setPass(pass, op);
  }

  void runAfterPass(mlir::Pass *, mlir::Operation *) override {
    traceEntry.clear();
  }

  void runAfterPassFailed(mlir::Pass *, mlir::Operation *) override {
    traceEntry.clear();
  }
};

} // namespace

// C-style function to add instrumentation (callable from Python with RTTI
// enabled)
extern "C" void ttmlirAddPassTracking(MlirPassManager pm) {
  auto *passManager = static_cast<mlir::PassManager *>(pm.ptr);
  passManager->addInstrumentation(
      std::make_unique<PassTrackerInstrumentation>());
}
