// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/IR.h"
#include "ttmlir-c/D2MTypes.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsTypes.h"

namespace mlir::ttmlir::python {
void populateD2MModule(nb::module_ &m) {
  tt_attribute_class<tt::d2m::ThreadAttr>(m, "ThreadAttr")
      .def_prop_ro_static("name",
                          [](nb::handle /*unused*/) {
                            return std::string(tt::d2m::ThreadAttr::name);
                          })
      .def_static("get", [](MlirContext ctx, std::string threadTypeStr) {
        tt::d2m::ThreadType threadType;
        if (threadTypeStr == "compute") {
          threadType = tt::d2m::ThreadType::Compute;
        } else if (threadTypeStr == "datamovement") {
          threadType = tt::d2m::ThreadType::Datamovement;
        } else {
          throw std::runtime_error("Unknown thread type " + threadTypeStr);
        }
        return ttmlirD2MThreadTypeAttrGet(
            ctx, static_cast<std::underlying_type_t<tt::d2m::ThreadType>>(
                     threadType));
      });

  tt_type_class<tt::d2m::SemaphoreType>(m, "SemaphoreType")
      .def_static("get", [](MlirContext ctx) {
        return wrap(tt::d2m::SemaphoreType::get(unwrap(ctx)));
      });
}
} // namespace mlir::ttmlir::python
