// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_BINDINGS_PYTHON_OVERRIDES_H
#define TTMLIR_BINDINGS_PYTHON_OVERRIDES_H

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "ttmlir-c/Dialects.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include <cstdint>
#include <string>
#include <unordered_map>

namespace py = pybind11;

namespace mlir::ttmlir::python {
void populateOverridesModule(py::module &m);
} // namespace mlir::ttmlir::python

#endif // TTMLIR_BINDINGS_PYTHON_OVERRIDES_H
