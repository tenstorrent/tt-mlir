// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/IR.h"
#include "ttmlir-c/TTIRTypes.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROpsTypes.h"

namespace mlir::ttmlir::python {
void populateTTIRModule(nb::module_ &m) {
  tt_attribute_class<tt::ttir::ThreadAttr>(m, "ThreadAttr")
      .def_prop_ro_static("name",
                          [](nb::handle /*unused*/) {
                            return std::string(tt::ttir::ThreadAttr::name);
                          })
      .def_static("get", [](MlirContext ctx, std::string threadTypeStr) {
        tt::ttir::ThreadType threadType;
        if (threadTypeStr == "compute") {
          threadType = tt::ttir::ThreadType::Compute;
        } else if (threadTypeStr == "datamovement") {
          threadType = tt::ttir::ThreadType::Datamovement;
        } else {
          throw std::runtime_error("Unknown thread type " + threadTypeStr);
        }
        return ttmlirTTIRThreadTypeAttrGet(
            ctx, static_cast<std::underlying_type_t<tt::ttir::ThreadType>>(
                     threadType));
      });

  tt_type_class<tt::ttir::SemaphoreType>(m, "SemaphoreType")
      .def_static("get", [](MlirContext ctx) {
        return wrap(tt::ttir::SemaphoreType::get(unwrap(ctx)));
      });
}
} // namespace mlir::ttmlir::python
