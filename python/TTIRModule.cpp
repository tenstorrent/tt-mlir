// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/IR.h"
#include "ttmlir-c/TTIRAttrs.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include <nanobind/stl/optional.h>

namespace mlir::ttmlir::python {
void populateTTIRModule(nb::module_ &m) {
  tt_attribute_class<tt::ttir::ThreadAttr>(m, "ThreadAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t threadType) {
            return ttmlirTTIRThreadAttrGet(ctx, threadType,
                                           MlirAttribute{nullptr});
          },
          nb::arg("ctx"), nb::arg("threadType"));
}
} // namespace mlir::ttmlir::python
