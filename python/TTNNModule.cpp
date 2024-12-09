// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/CAPI/AffineMap.h"
#include "ttmlir/Bindings/Python/TTMLIRModule.h"

namespace mlir::ttmlir::python {
void populateTTNNModule(py::module &m) {

  tt_attribute_class<tt::ttnn::CoreRangeAttr>(m, "CoreRangeAttr")
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
          py::arg("offset") = std::vector<int64_t>{0, 0})
      .def_property_readonly(
          "offset",
          [](tt::ttnn::CoreRangeAttr self) { return self.getOffset().vec(); })
      .def_property_readonly("size", [](tt::ttnn::CoreRangeAttr self) {
        return self.getSize().vec();
      });

  tt_attribute_class<tt::ttnn::LayoutAttr>(m, "LayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t layout) {
                    return wrap(tt::ttnn::LayoutAttr::get(
                        unwrap(ctx), static_cast<tt::ttnn::Layout>(layout)));
                  })
      .def_property_readonly("value", [](tt::ttnn::LayoutAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::ttnn::TensorMemoryLayoutAttr>(m,
                                                       "TensorMemoryLayoutAttr")
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
  tt_attribute_class<tt::ttnn::BufferTypeAttr>(m, "BufferTypeAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t bufferType) {
            return wrap(tt::ttnn::BufferTypeAttr::get(
                unwrap(ctx), static_cast<tt::ttnn::BufferType>(bufferType)));
          })
      .def_property_readonly("value", [](tt::ttnn::BufferTypeAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::ttnn::ShardSpecAttr>(m, "ShardSpecAttr")
      .def_static("get",
                  [](MlirContext ctx, tt::ttnn::ShapeAttr shardShape) {
                    return wrap(
                        tt::ttnn::ShardSpecAttr::get(unwrap(ctx), shardShape));
                  })
      .def_property_readonly("shard_shape",
                             &tt::ttnn::ShardSpecAttr::getShardShape);

  tt_attribute_class<tt::ttnn::MemoryConfigAttr>(m, "MemoryConfigAttr")
      .def_static("get",
                  [](MlirContext ctx,
                     tt::ttnn::TensorMemoryLayoutAttr tensorMemoryLayoutAttr,
                     tt::ttnn::BufferTypeAttr bufferTypeAttr,
                     tt::ttnn::ShardSpecAttr shardSpecAttr) {
                    return wrap(tt::ttnn::MemoryConfigAttr::get(
                        unwrap(ctx), bufferTypeAttr, shardSpecAttr,
                        tensorMemoryLayoutAttr));
                  })
      .def_static(
          "get_by_value",
          [](MlirContext ctx, uint32_t tensorMemoryLayout, uint32_t bufferType,
             std::vector<int64_t> shardShape) {
            tt::ttnn::TensorMemoryLayoutAttr layoutAttr =
                tt::ttnn::TensorMemoryLayoutAttr::get(
                    unwrap(ctx), static_cast<tt::ttnn::TensorMemoryLayout>(
                                     tensorMemoryLayout));

            return wrap(tt::ttnn::MemoryConfigAttr::get(
                unwrap(ctx),
                tt::ttnn::BufferTypeAttr::get(
                    unwrap(ctx), static_cast<tt::ttnn::BufferType>(bufferType)),
                tt::ttnn::ShardSpecAttr::get(
                    unwrap(ctx),
                    tt::ttnn::ShapeAttr::get(unwrap(ctx), shardShape)),
                layoutAttr));
          })
      .def_property_readonly("tensor_memory_layout",
                             &tt::ttnn::MemoryConfigAttr::getTensorMemoryLayout)
      .def_property_readonly("buffer_type",
                             &tt::ttnn::MemoryConfigAttr::getBufferType)
      .def_property_readonly("shard_spec",
                             &tt::ttnn::MemoryConfigAttr::getShardSpec);

  tt_attribute_class<tt::ttnn::ShapeAttr>(m, "ShapeAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> shape) {
                    return wrap(tt::ttnn::ShapeAttr::get(unwrap(ctx), shape));
                  })
      .def_property_readonly("shape", [](tt::ttnn::ShapeAttr self) {
        return std::vector<int64_t>(self.getShape().begin(),
                                    self.getShape().end());
      });

  tt_attribute_class<tt::ttnn::MeshShapeAttr>(m, "MeshShapeAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return wrap(
                        tt::ttnn::MeshShapeAttr::get(unwrap(ctx), y, x));
                  })
      .def_property_readonly("y", &tt::ttnn::MeshShapeAttr::getY)
      .def_property_readonly("x", &tt::ttnn::MeshShapeAttr::getX);

  tt_attribute_class<tt::ttnn::TTNNLayoutAttr>(m, "TTNNLayoutAttr")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAffineMap linear, MlirAttribute grid,
             MlirType memref,
             std::optional<unsigned> memLayout = std::nullopt) {
            tt::ttnn::TensorMemoryLayoutAttr memLayoutAttr;
            if (memLayout.has_value()) {
              memLayoutAttr = tt::ttnn::TensorMemoryLayoutAttr::get(
                  unwrap(ctx),
                  static_cast<tt::ttnn::TensorMemoryLayout>(memLayout.value()));
            }
            return wrap(tt::ttnn::TTNNLayoutAttr::get(
                unwrap(ctx), mlir::cast<AffineMap>(unwrap(linear)),
                mlir::cast<tt::GridAttr>(unwrap(grid)),
                mlir::cast<MemRefType>(unwrap(memref)), memLayoutAttr));
          })
      .def_property_readonly(
          "linear",
          [](tt::ttnn::TTNNLayoutAttr self) { return wrap(self.getLinear()); })
      .def_property_readonly("grid_attr", &tt::ttnn::TTNNLayoutAttr::getGrid)
      .def_property_readonly(
          "memref",
          [](tt::ttnn::TTNNLayoutAttr self) { return wrap(self.getMemref()); })
      .def_property_readonly(
          "memory_layout_as_int", [](tt::ttnn::TTNNLayoutAttr self) {
            if (!self.getMemLayout()) {
              assert(false && "Memory layout is not set");
            }
            return static_cast<uint32_t>(self.getMemLayout().getValue());
          });
}
} // namespace mlir::ttmlir::python
