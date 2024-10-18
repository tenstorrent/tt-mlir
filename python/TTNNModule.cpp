// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

namespace mlir::ttmlir::python {
void populateTTNNModule(py::module &m) {

  py::class_<tt::ttnn::CoreRangeAttr>(m, "CoreRangeAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> offset,
                     std::vector<int64_t> size) {
                    return wrap(tt::ttnn::CoreRangeAttr::get(unwrap(ctx),
                                                             offset, size));
                  })
      .def_static(
          "get_with_grid",
          [](MlirContext ctx, MlirAttribute grid, std::vector<int64_t> offset) {
            llvm::SmallVector<int64_t> offsetVec{0, 0};
            if (offset.size() == 2 && not(offset[0] == 0 && offset[1] == 0)) {
              offsetVec[0] = offset[0];
              offsetVec[1] = offset[1];
            }
            return wrap(tt::ttnn::CoreRangeAttr::get(
                unwrap(ctx), mlir::cast<tt::GridAttr>(unwrap(grid)),
                offsetVec));
          },
          py::arg("ctx"), py::arg("grid"),
          py::arg("offset") = std::vector<int64_t>{0, 0});
  py::class_<tt::ttnn::LayoutAttr>(m, "LayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t layout) {
                    return wrap(tt::ttnn::LayoutAttr::get(
                        unwrap(ctx), static_cast<tt::ttnn::Layout>(layout)));
                  })
      .def_property_readonly("value", [](tt::ttnn::LayoutAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });
  py::class_<tt::ttnn::TensorMemoryLayoutAttr>(m, "TensorMemoryLayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t tensorMemoryLayout) {
                    return wrap(tt::ttnn::TensorMemoryLayoutAttr::get(
                        unwrap(ctx), static_cast<tt::ttnn::TensorMemoryLayout>(
                                         tensorMemoryLayout)));
                  })
      .def_property_readonly("value",
                             [](tt::ttnn::TensorMemoryLayoutAttr self) {
                               return static_cast<uint32_t>(self.getValue());
                             });
  py::class_<tt::ttnn::BufferTypeAttr>(m, "BufferTypeAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t bufferType) {
            return wrap(tt::ttnn::BufferTypeAttr::get(
                unwrap(ctx), static_cast<tt::ttnn::BufferType>(bufferType)));
          })
      .def_property_readonly("value", [](tt::ttnn::BufferTypeAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });
  py::class_<tt::ttnn::MemoryConfigAttr>(m, "MemoryConfigAttr")
      .def_static("get",
                  [](MlirContext ctx,
                     tt::ttnn::TensorMemoryLayoutAttr tensorMemoryLayoutAttr,
                     tt::ttnn::BufferTypeAttr bufferTypeAttr) {
                    return wrap(tt::ttnn::MemoryConfigAttr::get(
                        unwrap(ctx), tensorMemoryLayoutAttr, bufferTypeAttr));
                  })
      .def_static(
          "get_by_value",
          [](MlirContext ctx, uint32_t tensorMemoryLayout,
             uint32_t bufferType) {
            return wrap(tt::ttnn::MemoryConfigAttr::get(
                unwrap(ctx),
                tt::ttnn::TensorMemoryLayoutAttr::get(
                    unwrap(ctx), static_cast<tt::ttnn::TensorMemoryLayout>(
                                     tensorMemoryLayout)),
                tt::ttnn::BufferTypeAttr::get(
                    unwrap(ctx),
                    static_cast<tt::ttnn::BufferType>(bufferType))));
          })
      .def_property_readonly("tensor_memory_layout",
                             &tt::ttnn::MemoryConfigAttr::getTensorMemoryLayout)
      .def_property_readonly("buffer_type",
                             &tt::ttnn::MemoryConfigAttr::getBufferType);
  py::class_<tt::ttnn::ShapeAttr>(m, "ShapeAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> shape) {
                    return wrap(tt::ttnn::ShapeAttr::get(unwrap(ctx), shape));
                  })
      .def_property_readonly("shape", [](tt::ttnn::ShapeAttr self) {
        return std::vector<int64_t>(self.getShape().begin(),
                                    self.getShape().end());
      });
  py::class_<tt::ttnn::MeshShapeAttr>(m, "MeshShapeAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return wrap(
                        tt::ttnn::MeshShapeAttr::get(unwrap(ctx), y, x));
                  })
      .def_property_readonly("y", &tt::ttnn::MeshShapeAttr::getY)
      .def_property_readonly("x", &tt::ttnn::MeshShapeAttr::getX);
}
} // namespace mlir::ttmlir::python
