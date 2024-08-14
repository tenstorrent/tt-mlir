// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <mlir/IR/BuiltinAttributes.h>
#include <vector>

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::ttmlir::python {
void populateTTModule(py::module &m) {
  py::class_<tt::LayoutAttr>(m, "LayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, MlirType rankedTensorType,
                     uint32_t memorySpaceValue, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals,
                     uint32_t oobValValue) {
                    return wrap(tt::LayoutAttr::get(
                        unwrap(ctx),
                        mlir::cast<RankedTensorType>(unwrap(rankedTensorType)),
                        static_cast<tt::MemorySpace>(memorySpaceValue),
                        mlir::cast<tt::GridAttr>(unwrap(grid)),
                        collapseIntervals,
                        static_cast<tt::OOBVal>(oobValValue)));
                  })
      .def_static("with_grid",
                  [](MlirContext ctx, MlirAttribute self,
                     std::vector<std::int64_t> tensorShape, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals) {
                    return wrap(
                        mlir::cast<tt::LayoutAttr>(unwrap(self))
                            .withGrid(unwrap(ctx), tensorShape,
                                      mlir::cast<tt::GridAttr>(unwrap(grid)),
                                      collapseIntervals));
                  })
      .def_static("with_grid_",
                  [](MlirContext ctx, MlirAttribute self,
                     std::vector<std::int64_t> tensorShape, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals) {
                    return mlir::cast<tt::LayoutAttr>(unwrap(self))
                        .withGrid(unwrap(ctx), tensorShape,
                                  mlir::cast<tt::GridAttr>(unwrap(grid)),
                                  collapseIntervals);
                  })
      .def_static(
          "with_element_type",
          [](MlirContext ctx, MlirAttribute self, MlirType elementType) {
            return wrap(mlir::cast<tt::LayoutAttr>(unwrap(self))
                            .withElementType(unwrap(ctx), unwrap(elementType)));
          })
      .def_static(
          "with_element_type_",
          [](MlirContext ctx, MlirAttribute self, MlirType elementType) {
            return mlir::cast<tt::LayoutAttr>(unwrap(self))
                .withElementType(unwrap(ctx), unwrap(elementType));
          })
      .def("getLayout",
           [](MlirType &type) {
             assert(isa<RankedTensorType>(
                 unwrap(type))); // Make sure that this is operating on a
                                 // RankedTensorType object
             RankedTensorType tensor =
                 mlir::cast<RankedTensorType>(unwrap(type));
             assert(tensor.getEncoding()); // Make sure that this Tensor has an
                                           // encoding value
             tt::LayoutAttr layout =
                 mlir::cast<tt::LayoutAttr>(tensor.getEncoding());
             return layout;
           })
      .def("wrapped", [](tt::LayoutAttr const &self) { return wrap(self); })
      .def_property_readonly(
          "stride",
          [](tt::LayoutAttr const &self, std::vector<int64_t> logicalShape) {
            auto stride = self.getStride(logicalShape);
            return std::vector<std::int64_t>(stride.begin(), stride.end());
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
                            std::vector<int64_t> grid, unsigned l1Size,
                            unsigned numDramChannels, unsigned dramChannelSize,
                            unsigned nocL1AddressAlignBytes,
                            unsigned pcieAddressAlignBytes,
                            unsigned nocDRAMAddressAlignBytes) {
        return wrap(tt::ChipDescAttr::get(
            unwrap(ctx), mlir::cast<tt::ArchAttr>(unwrap(arch)), grid, l1Size,
            numDramChannels, dramChannelSize, nocL1AddressAlignBytes,
            pcieAddressAlignBytes, nocDRAMAddressAlignBytes));
      });

  py::class_<tt::ChipCoordAttr>(m, "ChipCoordAttr")
      .def_static("get", [](MlirContext ctx, unsigned rack, unsigned shelf,
                            unsigned y, unsigned x) {
        return wrap(tt::ChipCoordAttr::get(unwrap(ctx), rack, shelf, y, x));
      });

  py::class_<tt::ChipChannelAttr>(m, "ChipChannelAttr")
      .def_static("get", [](MlirContext ctx, unsigned deviceId0,
                            std::vector<int64_t> ethernetCoreCoord0,
                            unsigned deviceId1,
                            std::vector<int64_t> ethernetCoreCoord1) {
        return wrap(tt::ChipChannelAttr::get(unwrap(ctx), deviceId0,
                                             ethernetCoreCoord0, deviceId1,
                                             ethernetCoreCoord1));
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
              mlir::cast<tt::ChipDescAttr>(unwrap(chipDesc)));
        }
        std::vector<tt::ChipCapabilityAttr> chipCapabilitiesUnwrapped;
        for (auto chipCapability : chipCapabilities) {
          chipCapabilitiesUnwrapped.push_back(
              mlir::cast<tt::ChipCapabilityAttr>(unwrap(chipCapability)));
        }
        std::vector<tt::ChipCoordAttr> chipCoordsUnwrapped;
        for (auto chipCoord : chipCoords) {
          chipCoordsUnwrapped.push_back(
              mlir::cast<tt::ChipCoordAttr>(unwrap(chipCoord)));
        }
        std::vector<tt::ChipChannelAttr> chipChannelsUnwrapped;
        for (auto chipChannel : chipChannels) {
          chipChannelsUnwrapped.push_back(
              mlir::cast<tt::ChipChannelAttr>(unwrap(chipChannel)));
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
      .def_static("get", [](MlirContext ctx, MlirAttribute deviceAttr) {
        return wrap(tt::DeviceType::get(
            unwrap(ctx), mlir::cast<tt::DeviceAttr>(unwrap(deviceAttr))));
      });

  py::class_<tt::DeviceAttr>(m, "DeviceAttr")
      .def_static(
          "get",
          [](MlirContext ctx, std::vector<int64_t> shape,
             MlirAffineMap physicalGridMapping, std::vector<unsigned> chipIds) {
            return wrap(tt::DeviceAttr::get(
                unwrap(ctx), shape, unwrap(physicalGridMapping), chipIds));
          })
      .def("unwrap", [](MlirAttribute const &self) {
        return mlir::cast<tt::DeviceAttr>(unwrap(self));
      });

  py::class_<tt::TileType>(m, "TileType")
      .def_static("get",
                  [](MlirContext ctx, std::int64_t height, std::int64_t width,
                     uint32_t dataType) {
                    return wrap(tt::TileType::get(
                        unwrap(ctx), SmallVector<std::int64_t>{height, width},
                        static_cast<tt::DataType>(dataType)));
                  })
      .def_property_readonly("size_bytes", &tt::TileType::getSizeBytes)
      .def_property_readonly("data_type", &tt::TileType::getDataType)
      .def_property_readonly("shape", [](tt::TileType const &tile) {
        return std::vector<int64_t>({tile.getHeight(), tile.getWidth()});
      });

  py::class_<tt::CoreRangeAttr>(m, "CoreRangeAttr")
      .def_static("get", [](MlirContext ctx, std::vector<int64_t> offset,
                            std::vector<int64_t> size) {
        return wrap(tt::CoreRangeAttr::get(unwrap(ctx), offset, size));
      });

  py::class_<tt::CircularBufferAttributesAttr>(m,
                                               "CircularBufferAttributesAttr")
      .def_static("get", [](MlirContext ctx, uint8_t cb_id,
                            MlirAttribute core_range, uint32_t total_size,
                            uint32_t page_size, uint32_t data_format) {
        return wrap(tt::CircularBufferAttributesAttr::get(
            unwrap(ctx), static_cast<tt::CB>(cb_id),
            mlir::cast<tt::CoreRangeAttr>(unwrap(core_range)), total_size,
            page_size, static_cast<tt::DataType>(data_format)));
      });

  py::class_<tt::DataMovementConfigAttr>(m, "DataMovementConfigAttr")
      .def_static("get", [](MlirContext ctx, uint8_t data_movement_type,
                            std::vector<uint32_t> compile_args) {
        return wrap(tt::DataMovementConfigAttr::get(
            unwrap(ctx), static_cast<tt::DataMovementType>(data_movement_type),
            compile_args));
      });

  py::class_<tt::DataMovementAttributesAttr>(m, "DataMovementAttributesAttr")
      .def_static("get", [](MlirContext ctx, MlirAttribute core_range,
                            std::string kernel_path,
                            MlirAttribute data_movement_config) {
        auto *context = unwrap(ctx);

        return wrap(tt::DataMovementAttributesAttr::get(
            context, mlir::cast<tt::CoreRangeAttr>(unwrap(core_range)),
            mlir::StringAttr::get(context, kernel_path),
            mlir::cast<tt::DataMovementConfigAttr>(
                unwrap(data_movement_config))));
      });

  py::class_<tt::ComputeConfigAttr>(m, "ComputeConfigAttr")
      .def_static("get", [](MlirContext ctx, uint8_t math_fidelity,
                            bool fp32_dest_acc_en, bool preserve_fp32_precision,
                            bool math_approx_mode,
                            std::vector<uint32_t> compile_args,
                            std::map<std::string, std::string> defines) {
        auto *context = unwrap(ctx);

        SmallVector<NamedAttribute> namedAttributes;
        for (const auto &[name, value] : defines) {
          namedAttributes.emplace_back(mlir::StringAttr::get(context, name),
                                       mlir::StringAttr::get(context, value));
        }

        return wrap(tt::ComputeConfigAttr::get(
            context, static_cast<tt::MathFidelity>(math_fidelity),
            mlir::BoolAttr::get(context, fp32_dest_acc_en),
            mlir::BoolAttr::get(context, preserve_fp32_precision),
            mlir::BoolAttr::get(context, math_approx_mode), compile_args,
            mlir::DictionaryAttr::get(context, namedAttributes)));
      });

  py::class_<tt::ComputeAttributesAttr>(m, "ComputeAttributesAttr")
      .def_static("get", [](MlirContext ctx, MlirAttribute core_range,
                            std::string kernel_path,
                            MlirAttribute compute_config) {
        auto *context = unwrap(ctx);

        return wrap(tt::ComputeAttributesAttr::get(
            context, mlir::cast<tt::CoreRangeAttr>(unwrap(core_range)),
            mlir::StringAttr::get(context, kernel_path),
            mlir::cast<tt::ComputeConfigAttr>(unwrap(compute_config))));
      });
}
} // namespace mlir::ttmlir::python
