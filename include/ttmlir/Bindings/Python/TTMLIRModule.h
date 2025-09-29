// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H
#define TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "ttmlir-c/Dialects.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/TTNNToCpp.h"
#include "ttmlir/RegisterAll.h"
#include "llvm/Support/CommandLine.h"

#include <nanobind/stl/variant.h>
#include <variant>

namespace nb = nanobind;

namespace mlir::ttmlir::python {

template <typename T>
nb::class_<T> tt_attribute_class(nb::module_ &m, const char *class_name) {
  nb::class_<T> cls(m, class_name);
  cls.def_static("maybe_downcast",
                 [](MlirAttribute attr) -> std::variant<T, nb::object> {
                   auto res = mlir::dyn_cast<T>(unwrap(attr));
                   if (res) {
                     return res;
                   }
                   return nb::none();
                 });
  return cls;
}

template <typename T>
nb::class_<T> tt_type_class(nb::module_ &m, const char *class_name) {
  nb::class_<T> cls(m, class_name);
  cls.def_static("maybe_downcast",
                 [](MlirType type) -> std::variant<T, nb::object> {
                   auto res = mlir::dyn_cast<T>(unwrap(type));
                   if (res) {
                     return res;
                   }
                   return nb::none();
                 });
  return cls;
}

void populateTTModule(nb::module_ &m);
void populateD2MModule(nb::module_ &m);
void populateTTKernelModule(nb::module_ &m);
void populateTTNNModule(nb::module_ &m);
void populateOverridesModule(nb::module_ &m);
void populateOptimizerOverridesModule(nb::module_ &m);
void populatePassesModule(nb::module_ &m);
void populateUtilModule(nb::module_ &m);
} // namespace mlir::ttmlir::python

#endif // TTMLIR_BINDINGS_PYTHON_TTMLIRMODULE_H
