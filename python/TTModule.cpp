// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/IR.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::ttmlir::python {
void populateTTModule(py::module &m) {
  py::enum_<tt::DataType>(m, "DataType")
      .value("Float32", tt::DataType::Float32)
      .value("Float16", tt::DataType::Float16)
      .value("BFloat16", tt::DataType::BFloat16)
      .value("BFP_Float8", tt::DataType::BFP_Float8)
      .value("BFP_BFloat8", tt::DataType::BFP_BFloat8)
      .value("BFP_Float4", tt::DataType::BFP_Float4)
      .value("BFP_BFloat4", tt::DataType::BFP_BFloat4)
      .value("BFP_Float2", tt::DataType::BFP_Float2)
      .value("BFP_BFloat2", tt::DataType::BFP_BFloat2)
      .value("UInt32", tt::DataType::UInt32)
      .value("UInt16", tt::DataType::UInt16)
      .value("UInt8", tt::DataType::UInt8);

  py::enum_<tt::MemorySpace>(m, "MemorySpace")
      .value("System", tt::MemorySpace::System)
      .value("SystemMMIO", tt::MemorySpace::SystemMMIO)
      .value("DeviceDRAM", tt::MemorySpace::DeviceDRAM)
      .value("DeviceL1", tt::MemorySpace::DeviceL1)
      .value("MIN", tt::MemorySpace::System)
      .value(
          "MAX",
          tt::MemorySpace::DeviceL1); // Referenced from Generated MemorySpace

  py::enum_<tt::OOBVal>(m, "OOBVal")
      .value("Zero", tt::OOBVal::Zero)
      .value("One", tt::OOBVal::One)
      .value("Undef", tt::OOBVal::Undef)
      .value("Inf", tt::OOBVal::Inf)
      .value("NegInf", tt::OOBVal::NegInf)
      .value("MIN", tt::OOBVal::Undef)
      .value("MAX",
             tt::OOBVal::NegInf); // Referenced from types_generated.h as well

  py::class_<llvm::ArrayRef<int64_t>>(m, "ArrayRefInt64")
      .def(py::init<const int64_t *, size_t>())
      .def("__getitem__",
           [](const llvm::ArrayRef<int64_t> &arr, size_t index) {
             if (index >= arr.size()) {
               throw py::index_error("Index out of range");
             }
             return arr[index];
           })
      .def("__len__", &llvm::ArrayRef<int64_t>::size);

  py::class_<tt::GridAttr>(m, "GridAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> shape) {
                    return wrap(tt::GridAttr::get(unwrap(ctx), shape));
                  })
      .def_property_readonly("name",
                             [](tt::GridAttr const &ga) { return ga.name; })
      .def_property_readonly("shape", &tt::GridAttr::getShape)
      .def_property_readonly("physical_grid_mapping",
                             &tt::GridAttr::getPhysicalGridMapping);

  py::class_<MemRefType>(m, "MemRefType")
      .def_property_readonly("shape", &MemRefType::getShape)
      .def_property_readonly(
          "element_type",
          &MemRefType::getElementType) // mlir::Type not supported
      .def_property_readonly("name",
                             [](MemRefType const &mt) {
                               return mt.name;
                             }) // llvm::StringLiteral not supported
      .def_property_readonly(
          "memory_space",
          &MemRefType::getMemorySpace); // mlir::Attribute not supported

  py::class_<tt::LayoutAttr>(m, "LayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, MlirType rankedTensorType,
                     tt::MemorySpace memorySpace, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals,
                     tt::OOBVal oobVal) {
                    return wrap(tt::LayoutAttr::get(
                        unwrap(ctx),
                        unwrap(rankedTensorType).cast<RankedTensorType>(),
                        memorySpace, unwrap(grid).cast<tt::GridAttr>(),
                        collapseIntervals, oobVal));
                  })
      .def_static("with_grid",
                  [](MlirContext ctx, MlirAttribute self,
                     std::vector<std::int64_t> tensorShape, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals) {
                    return wrap(unwrap(self).cast<tt::LayoutAttr>().withGrid(
                        unwrap(ctx), tensorShape,
                        unwrap(grid).cast<tt::GridAttr>(), collapseIntervals));
                  })
      .def_static("with_grid_",
                  [](MlirContext ctx, MlirAttribute self,
                     std::vector<std::int64_t> tensorShape, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals) {
                    return unwrap(self).cast<tt::LayoutAttr>().withGrid(
                        unwrap(ctx), tensorShape,
                        unwrap(grid).cast<tt::GridAttr>(), collapseIntervals);
                  })
      .def_static(
          "with_element_type",
          [](MlirContext ctx, MlirAttribute self, MlirType elementType) {
            return wrap(unwrap(self).cast<tt::LayoutAttr>().withElementType(
                unwrap(ctx), unwrap(elementType)));
          })
      .def_static(
          "with_element_type_",
          [](MlirContext ctx, MlirAttribute self, MlirType elementType) {
            return unwrap(self).cast<tt::LayoutAttr>().withElementType(
                unwrap(ctx), unwrap(elementType));
          })
      .def("getLayout",
           [](MlirType &type) {
             assert(isa<RankedTensorType>(
                 unwrap(type))); // Make sure that this is operating on a
                                 // RankedTensorType object
             RankedTensorType tensor = unwrap(type).cast<RankedTensorType>();
             assert(tensor.getEncoding()); // Make sure that this Tensor has an
                                           // encoding value
             tt::LayoutAttr layout =
                 tensor.getEncoding().template cast<tt::LayoutAttr>();
             return layout;
           })
      .def("wrapped", [](tt::LayoutAttr const &self) { return wrap(self); })
      .def_property_readonly("stride",
                             [](tt::LayoutAttr const &self) {
                               auto stride = self.getStride();
                               return std::vector<std::int64_t>(stride.begin(),
                                                                stride.end());
                             })
      .def_property_readonly("oobval", &tt::LayoutAttr::getOobVal)
      .def_property_readonly("grid_attr", &tt::LayoutAttr::getGrid)
      .def_property_readonly("memref", &tt::LayoutAttr::getMemref)
      .def_property_readonly("memory_space", &tt::LayoutAttr::getMemorySpace)
      .def_property_readonly("shard_shape", &tt::LayoutAttr::getShardShape);

  py::class_<tt::TileType>(m, "TileType")
      .def_static("get",
                  [](MlirContext ctx, uint64_t height, uint64_t width,
                     tt::DataType dataType) {
                    return wrap(tt::TileType::get(unwrap(ctx), height, width,
                                                  dataType));
                  })
      .def_property_readonly(
          "data_type",
          [](tt::TileType const &tile) { return tile.getDataType(); })
      .def_property_readonly("shape", [](tt::TileType const &tile) {
        return std::vector<int64_t>({tile.getHeight(), tile.getWidth()});
      });
}
} // namespace mlir::ttmlir::python
