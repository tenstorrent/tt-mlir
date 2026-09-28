// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDUMPMODULE
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// Writes the module to a file, unchanged.
//
// Deliberately non-fatal: this is a debugging aid, so a path that cannot be
// opened emits a warning and the compile carries on. Failing the pass would let
// a stale or read-only dump directory break an otherwise good build.
class TTNNDumpModule : public impl::TTNNDumpModuleBase<TTNNDumpModule> {
public:
  using impl::TTNNDumpModuleBase<TTNNDumpModule>::TTNNDumpModuleBase;

  void runOnOperation() override {
    if (path.empty()) {
      return;
    }

    llvm::StringRef parent = llvm::sys::path::parent_path(path);
    if (!parent.empty()) {
      if (std::error_code ec = llvm::sys::fs::create_directories(parent)) {
        getOperation().emitWarning()
            << "ttnn-dump-module: cannot create " << parent << ": "
            << ec.message();
        return;
      }
    }

    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      getOperation().emitWarning()
          << "ttnn-dump-module: cannot open " << path << ": " << ec.message();
      return;
    }

    getOperation().print(os);
  }
};

} // namespace mlir::tt::ttnn
