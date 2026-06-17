// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "ttmlir-c/D2MTypes.h"

#include "ttmlir/AffineMapUtils.h"
#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsTypes.h"
#include "ttmlir/Dialect/D2M/Utils/VirtualGrid.h"

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
        } else if (threadTypeStr == "unified") {
          threadType = tt::d2m::ThreadType::Unified;
        } else {
          throw std::runtime_error("Unknown thread type " + threadTypeStr);
        }
        return ttmlirD2MThreadTypeAttrGet(
            ctx, static_cast<std::underlying_type_t<tt::d2m::ThreadType>>(
                     threadType));
      });

  tt_type_class<tt::d2m::LocalSemaphoreType>(m, "LocalSemaphoreType")
      .def_static("get", [](MlirContext ctx) {
        return wrap(tt::d2m::LocalSemaphoreType::get(unwrap(ctx)));
      });

  tt_type_class<tt::d2m::GlobalSemaphoreType>(m, "GlobalSemaphoreType")
      .def_static("get", [](MlirContext ctx) {
        return wrap(tt::d2m::GlobalSemaphoreType::get(unwrap(ctx)));
      });

  tt_type_class<tt::d2m::CBType>(m, "CBType")
      .def_static("get",
                  [](MlirContext ctx, MlirType shapedType) {
                    return ttmlirD2MCBTypeGet(ctx, shapedType);
                  })
      .def_static(
          "cast",
          [](MlirType &ty) { return mlir::cast<tt::d2m::CBType>(unwrap(ty)); })
      .def("get_underlying",
           [](tt::d2m::CBType self) { return wrap(self.getUnderlying()); });

  m.def("calculate_reblock_map", [](std::vector<int64_t> inputShape,
                                    std::vector<int64_t> outputShape,
                                    MlirContext ctx) {
    return wrap(::ttmlir::utils::calculateReblockMap(inputShape, outputShape,
                                                     unwrap(ctx)));
  });

  // Forward (virtual->physical) and inverse (physical->virtual) affine maps that
  // implement a virtual grid as a physical-view pair. Mirrors GridSelection's
  // use of grids::createCoreVirtMaps so d2m-jit can target a logical grid (e.g.
  // 64x1) that folds onto the physical worker grid (e.g. 8x8).
  m.def("create_core_virt_maps", [](std::vector<int64_t> virtualGrid,
                                    std::vector<int64_t> targetGrid,
                                    MlirContext ctx) {
    auto [forwardMap, inverseMap] =
        ::ttmlir::d2m::utils::grids::createCoreVirtMaps(unwrap(ctx), virtualGrid,
                                                        targetGrid);
    return std::make_pair(wrap(forwardMap), wrap(inverseMap));
  });
}
} // namespace mlir::ttmlir::python
