// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/IR.h"
#include "ttmlir-c/TTKernelTypes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

namespace mlir::ttmlir::python {
void populateTTKernelModule(nb::module_ &m) {
  tt_type_class<tt::ttkernel::CBType>(m, "CBType")
      .def_static("get",
                  [](MlirContext ctx, uint64_t address, uint64_t port,
                     MlirType memrefType) {
                    return ttmlirTTKernelCBTypeGet(ctx, address, port,
                                                   memrefType);
                  })
      .def_static("cast",
                  [](MlirType &ty) {
                    return mlir::cast<tt::ttkernel::CBType>(unwrap(ty));
                  })
      .def_prop_ro("shape",
                   [](tt::ttkernel::CBType &cb) {
                     return std::vector<int64_t>(cb.getShape());
                   })
      .def_prop_ro("memref", &tt::ttkernel::CBType::getMemref);

  tt_attribute_class<tt::ttkernel::ThreadTypeAttr>(m, "ThreadTypeAttr")
      .def_prop_ro_static("name",
                          [](nb::handle /*unused*/) {
                            return std::string(
                                tt::ttkernel::ThreadTypeAttr::name);
                          })
      .def_static("get", [](MlirContext ctx, std::string threadTypeStr) {
        tt::ttkernel::ThreadType threadType;
        if (threadTypeStr == "compute") {
          threadType = tt::ttkernel::ThreadType::Compute;
        } else if (threadTypeStr == "noc") {
          threadType = tt::ttkernel::ThreadType::Noc;
        } else {
          throw std::runtime_error("Unknown thread type " + threadTypeStr);
        }

        return ttmlirTTKernelThreadTypeAttrGet(
            ctx, static_cast<std::underlying_type_t<tt::ttkernel::ThreadType>>(
                     threadType));
      });
}
} // namespace mlir::ttmlir::python
