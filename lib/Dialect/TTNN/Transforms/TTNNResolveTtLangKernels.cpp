// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Pass: --ttnn-resolve-tt-lang-kernels
//
// Resolve deferred `ttnn.tt_lang_op` kernels by invoking the tt-lang
// Python compiler through an embedded interpreter, and attach the
// returned artifact bytes back as the op's `kernel_artifact` attribute.
//
// This is the tt-mlir-owned successor to tt-xla's
// `tt_lang_bridge::resolveKernels`: the IR walk and the per-operand
// marshalling live here so any tt-mlir consumer (tt-xla today;
// tt-forge, ttrt, etc. tomorrow) that emits `ttnn.tt_lang_op` gets the
// same resolve semantics for free.
//
// File split. pybind11 needs `-frtti` because it uses `typeid` for
// type erasure, but the rest of MLIR (including `mlir::Pass` itself)
// is compiled `-fno-rtti`. Mixing them in a single TU triggers
// `undefined typeinfo for mlir::Pass` at link time, because the
// compiler emits typeinfo *references* whenever it sees a class
// hierarchy in a `-frtti` TU, but the typeinfo *definition* is
// only emitted by the TU that owns the base. So we split:
//
//   * TTNNResolveTtLangKernels.cpp   (this file, default `-fno-rtti`
//                                     -fno-exceptions): owns the
//                                     `mlir::Pass`-derived class and
//                                     the IR walk. Knows nothing
//                                     about pybind11.
//   * TTNNResolveTtLangKernelsPython.cpp (`-frtti -fexceptions`,
//                                         gated on the
//                                         `TTMLIR_ENABLE_TT_LANG_PYBIND_RESOLVER`
//                                         CMake flag): owns all the
//                                         pybind11 calls and exposes
//                                         a plain C++ entry point.
//
// The two TUs talk through `resolveTtLangKernelsViaPython` declared
// below. When the CMake flag is OFF, a stub of that function lives in
// `TTNNResolveTtLangKernelsPythonStub.cpp` (so the pass registration
// is uniform across builds; a module with no `ttnn.tt_lang_op` is
// unaffected and a module with one gets a clear "rebuild with the
// flag" diagnostic).

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <vector>

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNRESOLVETTLANGKERNELS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// Defined in TTNNResolveTtLangKernelsPython.cpp (real implementation,
// built with `-frtti -fexceptions` when
// `TTMLIR_ENABLE_TT_LANG_PYBIND_RESOLVER=ON`) or in
// TTNNResolveTtLangKernelsPythonStub.cpp (build-config-error stub,
// always built). Both expect the caller to have already populated
// `unresolvedOps` with ops whose `kernel_artifact` is unset.
mlir::LogicalResult resolveTtLangKernelsViaPython(
    llvm::ArrayRef<TtLangOp> unresolvedOps,
    llvm::ArrayRef<std::uint32_t> meshShape, llvm::StringRef pythonModule,
    llvm::StringRef pythonFunction);

namespace {

// Parse the comma-separated mesh-shape pipeline option into a
// concrete vector. Empty input -> empty vector (the Python side
// defaults to `[1]`). Whitespace/empty components are rejected so a
// typo doesn't silently degrade to `[]`.
mlir::FailureOr<std::vector<std::uint32_t>>
parseMeshShape(llvm::StringRef s, mlir::Operation *errorAnchor) {
  std::vector<std::uint32_t> out;
  if (s.empty()) {
    return out;
  }
  llvm::SmallVector<llvm::StringRef, 4> parts;
  s.split(parts, ',');
  for (llvm::StringRef part : parts) {
    if (part.empty()) {
      return errorAnchor->emitError()
             << "mesh-shape contains an empty component: \"" << s
             << "\". Use a comma-separated list of non-negative integers, "
                "e.g. \"1,1\" or \"2,4\".";
    }
    std::uint32_t v = 0;
    if (part.getAsInteger(/*Radix=*/10, v)) {
      return errorAnchor->emitError()
             << "mesh-shape component \"" << part << "\" is not a "
             << "non-negative integer (full value: \"" << s << "\").";
    }
    out.push_back(v);
  }
  return out;
}

class TTNNResolveTtLangKernels
    : public impl::TTNNResolveTtLangKernelsBase<TTNNResolveTtLangKernels> {
public:
  using impl::TTNNResolveTtLangKernelsBase<
      TTNNResolveTtLangKernels>::TTNNResolveTtLangKernelsBase;

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();

    // Collect all unresolved `ttnn.tt_lang_op`s. Ops whose
    // `kernel_artifact` is already populated are skipped -- e.g. by a
    // pre-bake step or by a previous invocation of this pass -- so the
    // pass is composable with future ahead-of-time artifact paths.
    llvm::SmallVector<TtLangOp> unresolvedOps;
    moduleOp.walk([&](TtLangOp op) {
      if (auto attr = op.getKernelArtifactAttr();
          !attr || attr.asArrayRef().empty()) {
        unresolvedOps.push_back(op);
      }
    });

    if (unresolvedOps.empty()) {
      return;
    }

    // Parse mesh shape once; surface any error on the first unresolved
    // op so the diagnostic has a click-through source location.
    auto meshOr = parseMeshShape(meshShape, unresolvedOps.front());
    if (mlir::failed(meshOr)) {
      return signalPassFailure();
    }

    if (mlir::failed(resolveTtLangKernelsViaPython(
            unresolvedOps, *meshOr, pythonModule, pythonFunction))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttnn
