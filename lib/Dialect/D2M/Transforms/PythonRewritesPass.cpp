// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Transforms/Passes.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MPYTHONREWRITES
#include "ttmlir/Dialect/D2M/Transforms/Passes.h.inc"
} // namespace mlir::tt::d2m

#ifdef TTMLIR_ENABLE_D2M_JIT

// We use the raw CPython C-API rather than nanobind/pybind11 here because
// MLIRD2MTransforms inherits LLVM's -fno-rtti / -fno-exceptions, which
// nanobind doesn't support.
//
// The pass does NOT cross C++/Python via a shared MlirModule pointer —
// ttmlir-opt is statically linked against MLIR while the ttmlir Python
// bindings ship their own copy of MLIR (libTTMLIRPythonCAPI.so). The
// two MLIR runtimes can't share opaque storage (StringAttr uniquer
// etc.), so we fall back to a text round-trip:
//
//   1. C++: print the host ModuleOp to a string.
//   2. Python: parse the string in *its* ir.Context, run apply_patterns,
//      print the rewritten module back to a string.
//   3. C++: parse the result string into the host context as a new
//      module, then swap its body into the original ModuleOp.

#include <Python.h>

#include <atomic>
#include <string>

namespace mlir::tt::d2m {
namespace {

// RAII wrapper for Py_DECREF on a PyObject*.
class PyRef {
public:
  PyRef() = default;
  explicit PyRef(PyObject *obj) : obj(obj) {}
  PyRef(const PyRef &) = delete;
  PyRef &operator=(const PyRef &) = delete;
  PyRef(PyRef &&other) noexcept : obj(other.obj) { other.obj = nullptr; }
  PyRef &operator=(PyRef &&other) noexcept {
    if (this != &other) {
      reset();
      obj = other.obj;
      other.obj = nullptr;
    }
    return *this;
  }
  ~PyRef() { reset(); }

  void reset(PyObject *newObj = nullptr) {
    Py_XDECREF(obj);
    obj = newObj;
  }
  PyObject *get() const { return obj; }
  explicit operator bool() const { return obj != nullptr; }

private:
  PyObject *obj = nullptr;
};

// RAII GIL acquire/release. Cheaper than thread-state save/restore.
class PyGIL {
public:
  PyGIL() : state(PyGILState_Ensure()) {}
  PyGIL(const PyGIL &) = delete;
  PyGIL &operator=(const PyGIL &) = delete;
  ~PyGIL() { PyGILState_Release(state); }

private:
  PyGILState_STATE state;
};

// Ensure Python is initialized exactly once per process.
static void ensurePythonInitialized() {
  if (Py_IsInitialized()) {
    return;
  }
  Py_Initialize();
  PyEval_SaveThread();
}

// Capture the most recent Python exception as a std::string and clear it.
static std::string capturePythonError() {
  if (!PyErr_Occurred()) {
    return "<no python error>";
  }
  PyObject *type = nullptr;
  PyObject *value = nullptr;
  PyObject *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);
  std::string out;
  if (value) {
    PyRef str(PyObject_Str(value));
    if (str) {
      const char *utf8 = PyUnicode_AsUTF8(str.get());
      if (utf8) {
        out = utf8;
      }
    }
  }
  if (out.empty()) {
    out = "<unrepresentable python exception>";
  }
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  return out;
}

class D2MPythonRewritesPass
    : public impl::D2MPythonRewritesBase<D2MPythonRewritesPass> {
public:
  using impl::D2MPythonRewritesBase<
      D2MPythonRewritesPass>::D2MPythonRewritesBase;

  void runOnOperation() override {
    if (modulePaths.empty()) {
      // No-op when invoked without any patterns.
      return;
    }

    ModuleOp module = getOperation();

    // Step 1: print the host module to text.
    std::string inputText;
    {
      llvm::raw_string_ostream os(inputText);
      OpPrintingFlags flags;
      // Generic form is portable across C++/Python MLIR runtimes.
      module.print(os, flags);
    }

    // Step 2: call into Python.
    ensurePythonInitialized();
    PyGIL gil;

    PyRef rewriteMod(PyImport_ImportModule("d2m_jit._src.rewrite"));
    if (!rewriteMod) {
      emitPyError("could not import d2m_jit._src.rewrite");
      return;
    }
    PyRef applyText(
        PyObject_GetAttrString(rewriteMod.get(), "apply_patterns_text"));
    if (!applyText) {
      emitPyError("d2m_jit._src.rewrite has no apply_patterns_text");
      return;
    }

    PyRef inputPy(
        PyUnicode_FromStringAndSize(inputText.data(), inputText.size()));
    if (!inputPy) {
      emitPyError("could not convert input text to Python str");
      return;
    }

    PyRef pathsList(PyList_New(modulePaths.size()));
    if (!pathsList) {
      emitPyError("could not allocate pattern-path list");
      return;
    }
    for (size_t i = 0; i < modulePaths.size(); ++i) {
      PyObject *path = PyUnicode_FromString(modulePaths[i].c_str());
      if (!path) {
        emitPyError("could not convert pattern path to Python str");
        return;
      }
      PyList_SET_ITEM(pathsList.get(), static_cast<Py_ssize_t>(i), path);
    }

    PyRef resultPy(PyObject_CallFunctionObjArgs(applyText.get(), inputPy.get(),
                                                pathsList.get(), nullptr));
    if (!resultPy) {
      emitPyError("d2m_jit apply_patterns_text raised");
      return;
    }
    if (!PyUnicode_Check(resultPy.get())) {
      module.emitError("d2m-python-rewrites: apply_patterns_text did not "
                       "return a string");
      signalPassFailure();
      return;
    }
    Py_ssize_t resultLen = 0;
    const char *resultUtf8 =
        PyUnicode_AsUTF8AndSize(resultPy.get(), &resultLen);
    if (!resultUtf8) {
      emitPyError("could not extract result string from Python");
      return;
    }
    std::string outputText(resultUtf8, resultUtf8 + resultLen);

    // Step 3: parse the result back into the host context, replace
    // module body. We keep the GIL held through the parse — it's
    // synchronous and our pass doesn't expect parallelism here.
    OwningOpRef<ModuleOp> newModule =
        parseSourceString<ModuleOp>(outputText, &getContext());

    if (!newModule) {
      module.emitError(
          "d2m-python-rewrites: failed to re-parse rewritten module text. "
          "This usually means the Python rewrites produced IR that the host "
          "MLIR context can't understand (missing dialect registration, etc.)");
      signalPassFailure();
      return;
    }

    // Swap bodies: erase the host module's body ops, splice in the
    // re-parsed module's body. Both modules have a single block in
    // their region.
    Block &dst = module.getBodyRegion().front();
    Block &src = newModule->getBodyRegion().front();

    // Erase existing ops; iterating backwards because erase mutates the
    // list. Skip the implicit terminator if any (ModuleOp doesn't have
    // one).
    for (auto it = dst.rbegin(); it != dst.rend();) {
      Operation &op = *it;
      ++it;
      op.erase();
    }
    // Splice the source's ops to the destination.
    auto &srcOps = src.getOperations();
    while (!srcOps.empty()) {
      Operation &op = srcOps.front();
      op.moveBefore(&dst, dst.end());
    }
    // newModule is now empty; drop it.
  }

private:
  void emitPyError(const std::string &prefix) {
    std::string detail = capturePythonError();
    getOperation().emitError("d2m-python-rewrites: ")
        << prefix << ": " << detail;
    signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tt::d2m

#else // TTMLIR_ENABLE_D2M_JIT

namespace mlir::tt::d2m {
namespace {

// Stub when d2m-jit isn't enabled. The pass entry still exists so
// ttmlir-opt can advertise the flag, but invoking it fails fast.
class D2MPythonRewritesPass
    : public impl::D2MPythonRewritesBase<D2MPythonRewritesPass> {
public:
  using impl::D2MPythonRewritesBase<
      D2MPythonRewritesPass>::D2MPythonRewritesBase;

  void runOnOperation() override {
    getOperation().emitError(
        "d2m-python-rewrites requires the build to be configured with "
        "-DTTMLIR_ENABLE_D2M_JIT=ON; rebuild and retry.");
    signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tt::d2m

#endif // TTMLIR_ENABLE_D2M_JIT
