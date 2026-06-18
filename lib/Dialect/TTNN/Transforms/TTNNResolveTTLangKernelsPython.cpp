// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// pybind11-backed implementation of
// `mlir::tt::ttnn::resolveTTLangKernelsViaPython`. Compiled with
// `-frtti -fexceptions` (see lib/Dialect/TTNN/Transforms/CMakeLists.txt)
// so it stays isolated from the rest of `MLIRTTNNTransforms`, which is
// `-fno-rtti -fno-exceptions` along with the bulk of MLIR.
//
// Only built when `TTMLIR_ENABLE_TT_LANG_PYBIND_RESOLVER=ON`.
//
// Lifecycle. We do NOT call `Py_Initialize` here. The host process is
// expected to already be running CPython (the typical case when
// tt-xla's `pjrt_plugin_tt.so` is loaded from a `jax` / `torch_xla`
// program). The function acquires the GIL on the existing interpreter
// via `py::gil_scoped_acquire`. If no interpreter is alive, the
// pybind11 import below throws and we surface that as a clear "no
// Python interpreter" diagnostic.

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <exception>
#include <string>
#include <utility>

// pybind11 headers trip a few warnings that LLVM/MLIR's default flags
// promote to errors (`-Werror=covered-switch-default`,
// `-Werror=unused-parameter`, ...). They are not surfaced via
// `-isystem` so we silence them locally rather than relaxing the
// project-wide flags.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#pragma clang diagnostic ignored "-Wunused-parameter"
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#pragma clang diagnostic pop

namespace mlir::tt::ttnn {

namespace {

std::string mlirTypeToString(mlir::Type type) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  type.print(os);
  return os.str();
}

// Resolve a single `ttnn.tt_lang_op` by calling the Python resolver
// entry point. Caller holds the GIL.
//
// On success, attaches `kernel_artifact` to `op` and returns
// `success`. On any failure (missing required attribute, raising
// resolver, empty artifact) emits a diagnostic on `op` and returns
// `failure` -- the caller signalPassFailure()s.
mlir::LogicalResult resolveOneKernel(TTLangOp op,
                                     pybind11::object &resolveKernel,
                                     llvm::ArrayRef<std::uint32_t> meshShape) {
  namespace py = pybind11;
  mlir::MLIRContext *ctx = op.getContext();

  // Required string attributes. These come from the StableHLO
  // frontend (preserved verbatim through TTIR -> TTNN by the
  // conversion patterns), so a missing attribute here means someone
  // hand-built an invalid op or there is an upstream bug.
  llvm::StringRef kernelId = op.getKernelId();
  if (kernelId.empty()) {
    return op.emitError(
        "ttnn.tt_lang_op missing required `kernel_id` attribute.");
  }
  llvm::StringRef versionTag = op.getVersionTag();
  if (versionTag.empty()) {
    return op.emitError("ttnn.tt_lang_op '")
           << kernelId << "' missing required `version_tag` attribute.";
  }
  llvm::StringRef argRoles = op.getArgRoles();
  llvm::StringRef shardSpec = op.getShardSpec();

  try {
    // Build operand metadata. Shape comes from the now-final
    // shard-local tensor types; dtype is the element-type printed
    // form (e.g. "f32", "bf16"); layout is the printed
    // `ttnn.ttnn_layout` encoding, which carries memory space
    // (DRAM/L1), buffer type (interleaved/sharded), tensor layout
    // (row-major/tile), and grid info. Operands without an encoding
    // (the ttir-only path hasn't yet attached one) get an empty
    // string so the Python side can decide whether to default.
    py::list shapes;
    py::list dtypes;
    py::list layouts;
    for (mlir::Value operand : op->getOperands()) {
      py::list shape;
      std::string dtypeStr;
      std::string layoutStr;
      mlir::Type ty = operand.getType();
      if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(ty)) {
        for (int64_t d : ranked.getShape()) {
          shape.append(static_cast<int64_t>(d));
        }
        dtypeStr = mlirTypeToString(ranked.getElementType());
        if (mlir::Attribute encoding = ranked.getEncoding()) {
          std::string buf;
          llvm::raw_string_ostream os(buf);
          encoding.print(os);
          layoutStr = std::move(os.str());
        }
      } else {
        dtypeStr = mlirTypeToString(ty);
      }
      shapes.append(std::move(shape));
      dtypes.append(std::move(dtypeStr));
      layouts.append(py::str(layoutStr));
    }

    py::list mesh;
    if (meshShape.empty()) {
      mesh.append(static_cast<std::uint32_t>(1));
    } else {
      for (std::uint32_t d : meshShape) {
        mesh.append(d);
      }
    }

    py::dict kwargs;
    // The op's id lives in the `kernel_id` MLIR attribute (wire-level name)
    // but the Python resolver's parameter is `operation_id`.
    kwargs["operation_id"] = py::str(kernelId.str());
    kwargs["version_tag"] = py::str(versionTag.str());
    kwargs["shapes"] = shapes;
    kwargs["dtypes"] = dtypes;
    kwargs["layouts"] = layouts;
    kwargs["mesh_shape"] = mesh;
    // `arg_roles` is required by the TTNN op verifier (StrAttr, not
    // OptionalAttr). Pass it through unconditionally so the Python
    // resolver doesn't have to defensively default.
    kwargs["arg_roles"] = py::str(argRoles.str());
    if (!shardSpec.empty()) {
      kwargs["shard_spec"] = py::str(shardSpec.str());
    }

    py::bytes artifact = resolveKernel(**kwargs);
    std::string buf = artifact;
    if (buf.empty()) {
      return op.emitError("tt-lang resolve returned an empty artifact "
                          "for kernel '")
             << kernelId << "'.";
    }

    // The artifact is the resolver's JSON payload (text); store it as a
    // `StringAttr`. `StringRef(buf.data(), buf.size())` preserves the
    // exact bytes (including any embedded NULs) rather than truncating
    // at the first NUL the way a C-string would.
    op.setKernelArtifactAttr(
        mlir::StringAttr::get(ctx, llvm::StringRef(buf.data(), buf.size())));
    return mlir::success();
  } catch (const std::exception &e) {
    return op.emitError("tt-lang resolve failed for kernel '")
           << kernelId << "': " << e.what();
  }
}

} // namespace

mlir::LogicalResult
resolveTTLangKernelsViaPython(llvm::ArrayRef<TTLangOp> unresolvedOps,
                              llvm::ArrayRef<std::uint32_t> meshShape,
                              llvm::StringRef pythonModule,
                              llvm::StringRef pythonFunction) {
  namespace py = pybind11;

  if (unresolvedOps.empty()) {
    return mlir::success();
  }

  // CRITICAL: every `py::object` (including `resolveKernel` below) MUST
  // be destroyed while the GIL is held -- pybind11 asserts on dec_ref
  // without the GIL and the process aborts (`pybind11::handle::dec_ref()
  // PyGILState_Check() failure`). Holding a single `gil_scoped_acquire`
  // across both the import and the per-op resolution keeps the
  // destruction of `resolveKernel` (at end of scope) under the GIL.
  py::gil_scoped_acquire gil;

  py::object resolveKernel;
  try {
    resolveKernel = py::module_::import(pythonModule.str().c_str())
                        .attr(pythonFunction.str().c_str());
  } catch (const std::exception &e) {
    // `TTLangOp` is a thin handle (it holds an `Operation*`); we need
    // a non-const local because `emitError()` is non-const, and
    // `ArrayRef::front()` returns `const TTLangOp&`.
    TTLangOp firstOp = unresolvedOps.front();
    firstOp.emitError()
        << "module contains " << unresolvedOps.size()
        << " unresolved ttnn.tt_lang_op(s) but the tt-lang resolver "
           "entry point is not importable: `"
        << pythonModule << "." << pythonFunction << "`: " << e.what()
        << ". Make sure the tt_torch wheel is installed in the same "
           "Python interpreter that loaded this compiler, and that an "
           "interpreter is already initialised in this process (the "
           "pass does not call Py_Initialize).";
    return mlir::failure();
  }

  for (TTLangOp op : unresolvedOps) {
    if (mlir::failed(resolveOneKernel(op, resolveKernel, meshShape))) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

} // namespace mlir::tt::ttnn
