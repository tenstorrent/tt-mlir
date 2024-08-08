// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H
#define TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
// #include "mlir/Parser/Parser.h"
#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/CAPI/IR.h"
#include "ttmlir-c/Dialects.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNToCpp.h"
#include "ttmlir/RegisterAll.h"
// #include <cstdint>
// #include <string>
// #include <unordered_map>

namespace py = pybind11;

namespace mlir::ttmlir::python {
void populateTTModule(py::module &m);
void populateTTKernelModule(py::module &m);
void populateOverridesModule(py::module &m);
void populatePassesModule(py::module &m);
} // namespace mlir::ttmlir::python

#endif // TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H
