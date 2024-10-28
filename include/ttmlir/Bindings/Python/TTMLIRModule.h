// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H
#define TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "ttmlir-c/Dialects.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNToCpp.h"
#include "ttmlir/RegisterAll.h"
#include "llvm/Support/CommandLine.h"

namespace py = pybind11;

namespace mlir::ttmlir::python {
void populateTTModule(py::module &m);
void populateTTIRModule(py::module &m);
void populateTTKernelModule(py::module &m);
void populateTTNNModule(py::module &m);
void populateOverridesModule(py::module &m);
void populatePassesModule(py::module &m);
} // namespace mlir::ttmlir::python

#endif // TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H
