// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Stub implementation of `mlir::tt::ttnn::resolveTtLangKernelsViaPython`
// used when tt-mlir is built WITHOUT
// `TTMLIR_ENABLE_TT_LANG_PYBIND_RESOLVER`. It keeps the
// `--ttnn-resolve-tt-lang-kernels` pass registration uniform across
// builds; modules without any `ttnn.tt_lang_op` are unaffected, and
// modules that do contain one get a clear "rebuild with the flag"
// diagnostic anchored on the first offending op.
//
// The real (pybind11-backed) implementation lives in
// `TTNNResolveTtLangKernelsPython.cpp` and is enabled by the CMake
// flag.

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir::tt::ttnn {

mlir::LogicalResult resolveTtLangKernelsViaPython(
    llvm::ArrayRef<TtLangOp> unresolvedOps,
    llvm::ArrayRef<std::uint32_t> /*meshShape*/,
    llvm::StringRef /*pythonModule*/, llvm::StringRef /*pythonFunction*/) {
  if (unresolvedOps.empty()) {
    return mlir::success();
  }
  // Non-const local handle so we can call the non-const `emitError`
  // (the `ArrayRef::front()` overload returns a const reference).
  TtLangOp firstOp = unresolvedOps.front();
  firstOp.emitError(
      "ttnn.tt_lang_op present but the tt-mlir build does not include "
      "the tt-lang Python resolver. Rebuild tt-mlir with "
      "`-DTTMLIR_ENABLE_TT_LANG_PYBIND_RESOLVER=ON`, or pre-bake the "
      "`kernel_artifact` attribute before running "
      "`--ttnn-resolve-tt-lang-kernels`.");
  return mlir::failure();
}

} // namespace mlir::tt::ttnn
