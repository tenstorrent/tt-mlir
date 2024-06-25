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
      .value("BC_Float8", tt::DataType::BC_Float8)
      .value("BC_BFloat8", tt::DataType::BC_BFloat8)
      .value("BC_Float4", tt::DataType::BC_Float4)
      .value("BC_BFloat4", tt::DataType::BC_BFloat4)
      .value("BC_Float2", tt::DataType::BC_Float2)
      .value("BC_BFloat2", tt::DataType::BC_BFloat2)
      .value("UInt32", tt::DataType::UInt32)
      .value("UInt16", tt::DataType::UInt16)
      .value("UInt8", tt::DataType::UInt8);

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
