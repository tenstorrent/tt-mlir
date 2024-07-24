// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/IR.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::ttmlir::python {
void populateTTModule(py::module &m) {
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

  py::class_<tt::GridAttr>(m, "GridAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> shape) {
                    return wrap(tt::GridAttr::get(unwrap(ctx), shape));
                  })
      .def_property_readonly("shape", [](tt::GridAttr const &ga) {
        return std::vector<int64_t>(ga.getShape().begin(), ga.getShape().end());
      });

  py::class_<tt::ChipCapabilityAttr>(m, "ChipCapabilityAttr")
      .def_static("get", [](MlirContext ctx, uint32_t chipCapability) {
        return wrap(tt::ChipCapabilityAttr::get(
            unwrap(ctx), static_cast<tt::ChipCapability>(chipCapability)));
      });

  py::class_<tt::ArchAttr>(m, "ArchAttr")
      .def_static("get", [](MlirContext ctx, uint32_t arch) {
        return wrap(
            tt::ArchAttr::get(unwrap(ctx), static_cast<tt::Arch>(arch)));
      });

  py::class_<tt::ChipDescAttr>(m, "ChipDescAttr")
      .def_static("get", [](MlirContext ctx, MlirAttribute arch,
                            MlirAttribute grid, unsigned l1Size,
                            unsigned numDramChannels, unsigned dramChannelSize,
                            unsigned nocL1AddressAlignBytes,
                            unsigned pcieAddressAlignBytes,
                            unsigned nocDRAMAddressAlignBytes) {
        return wrap(tt::ChipDescAttr::get(
            unwrap(ctx), unwrap(arch).cast<tt::ArchAttr>(),
            unwrap(grid).cast<tt::GridAttr>(), l1Size, numDramChannels,
            dramChannelSize, nocL1AddressAlignBytes, pcieAddressAlignBytes,
            nocDRAMAddressAlignBytes));
      });

  py::class_<tt::ChipCoordAttr>(m, "ChipCoordAttr")
      .def_static("get", [](MlirContext ctx, unsigned rack, unsigned shelf,
                            unsigned y, unsigned x) {
        return wrap(tt::ChipCoordAttr::get(unwrap(ctx), rack, shelf, y, x));
      });

  py::class_<tt::ChipChannelAttr>(m, "ChipChannelAttr")
      .def_static("get",
                  [](MlirContext ctx, unsigned endpoint0, unsigned endpoint1) {
                    return wrap(tt::ChipChannelAttr::get(unwrap(ctx), endpoint0,
                                                         endpoint1));
                  });

  py::class_<tt::SystemDescAttr>(m, "SystemDescAttr")
      .def_static("get", [](MlirContext ctx,
                            std::vector<MlirAttribute> chipDescs,
                            std::vector<unsigned> chipDescIndices,
                            std::vector<MlirAttribute> chipCapabilities,
                            std::vector<MlirAttribute> chipCoords,
                            std::vector<MlirAttribute> chipChannels) {
        std::vector<tt::ChipDescAttr> chipDescsUnwrapped;
        for (auto chipDesc : chipDescs) {
          chipDescsUnwrapped.push_back(
              unwrap(chipDesc).cast<tt::ChipDescAttr>());
        }
        std::vector<tt::ChipCapabilityAttr> chipCapabilitiesUnwrapped;
        for (auto chipCapability : chipCapabilities) {
          chipCapabilitiesUnwrapped.push_back(
              unwrap(chipCapability).cast<tt::ChipCapabilityAttr>());
        }
        std::vector<tt::ChipCoordAttr> chipCoordsUnwrapped;
        for (auto chipCoord : chipCoords) {
          chipCoordsUnwrapped.push_back(
              unwrap(chipCoord).cast<tt::ChipCoordAttr>());
        }
        std::vector<tt::ChipChannelAttr> chipChannelsUnwrapped;
        for (auto chipChannel : chipChannels) {
          chipChannelsUnwrapped.push_back(
              unwrap(chipChannel).cast<tt::ChipChannelAttr>());
        }
        return wrap(tt::SystemDescAttr::get(
            unwrap(ctx), chipDescsUnwrapped, chipDescIndices,
            chipCapabilitiesUnwrapped, chipCoordsUnwrapped,
            chipChannelsUnwrapped));
      });

  py::class_<tt::MemorySpaceAttr>(m, "MemorySpaceAttr")
      .def_static("get", [](MlirContext ctx, uint32_t memorySpace) {
        return wrap(tt::MemorySpaceAttr::get(
            unwrap(ctx), static_cast<tt::MemorySpace>(memorySpace)));
      });

  py::class_<tt::OOBValAttr>(m, "OOBValAttr")
      .def_static("get", [](MlirContext ctx, uint32_t oobVal) {
        return wrap(
            tt::OOBValAttr::get(unwrap(ctx), static_cast<tt::OOBVal>(oobVal)));
      });

  py::class_<tt::IteratorTypeAttr>(m, "IteratorTypeAttr")
      .def_static("get", [](MlirContext ctx, uint32_t iteratorType) {
        return wrap(tt::IteratorTypeAttr::get(
            unwrap(ctx), static_cast<tt::IteratorType>(iteratorType)));
      });

  py::class_<tt::OperandConstraintAttr>(m, "OperandConstraintAttr")
      .def_static("get", [](uint32_t operandConstraint, MlirContext ctx) {
        return wrap(tt::OperandConstraintAttr::get(
            unwrap(ctx),
            static_cast<tt::OperandConstraint>(operandConstraint)));
      });

  py::class_<tt::DeviceType>(m, "DeviceType")
      .def_static("get", [](MlirContext ctx, MlirAttribute grid,
                            unsigned chipIds, size_t chipIdsSize) {
        llvm::ArrayRef<unsigned> chipIdsRef(&chipIds, chipIds + chipIdsSize);
        return wrap(tt::DeviceType::get(
            unwrap(ctx), unwrap(grid).cast<tt::GridAttr>(), chipIdsRef));
      });

  py::class_<tt::TileType>(m, "TileType")
      .def_static("get",
                  [](MlirContext ctx, unsigned height, unsigned width,
                     uint32_t dataType) {
                    return wrap(
                        tt::TileType::get(unwrap(ctx), height, width,
                                          static_cast<tt::DataType>(dataType)));
                  })
      .def_property_readonly("data_type", &tt::TileType::getDataType)
      .def_property_readonly("shape", [](tt::TileType const &tile) {
        return std::vector<int64_t>({tile.getHeight(), tile.getWidth()});
      });
}
} // namespace mlir::ttmlir::python
