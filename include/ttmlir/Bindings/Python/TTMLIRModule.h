// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H
#define TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "ttmlir-c/Dialects.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

namespace py = pybind11;

namespace mlir::ttmlir::python {
void populateTTModule(py::module &m);
void populateTTKernelModule(py::module &m);
} // namespace mlir::ttmlir::python

#endif // TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H
