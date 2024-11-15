// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Target/Common/Target.h"
#include "ttmlir/Utils.h"

namespace mlir::ttmlir::python {
void populateTTModule(py::module &m) {
  tt_attribute_class<tt::LayoutAttr>(m, "LayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, MlirType rankedTensorType,
                     uint32_t memorySpaceValue, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals,
                     uint32_t oobValValue, uint32_t memLayoutValue) {
                    return wrap(tt::LayoutAttr::get(
                        unwrap(ctx),
                        mlir::cast<RankedTensorType>(unwrap(rankedTensorType)),
                        static_cast<tt::MemorySpace>(memorySpaceValue),
                        mlir::cast<tt::GridAttr>(unwrap(grid)),
                        collapseIntervals, static_cast<tt::OOBVal>(oobValValue),
                        static_cast<tt::TensorMemoryLayout>(memLayoutValue)));
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
      .def_property_readonly("oobval_as_int",
                             [](tt::LayoutAttr la) {
                               return static_cast<uint32_t>(la.getOobVal());
                             })
      .def_property_readonly("grid_attr", &tt::LayoutAttr::getGrid)
      .def_property_readonly("memref", &tt::LayoutAttr::getMemref)
      .def_property_readonly("memory_space", &tt::LayoutAttr::getMemorySpace)
      .def_property_readonly("memory_space_as_int",
                             [](tt::LayoutAttr la) {
                               return static_cast<uint32_t>(
                                   la.getMemorySpace());
                             })
      .def_property_readonly("shard_shape", &tt::LayoutAttr::getShardShape)
      .def_property_readonly("memory_layout", &tt::LayoutAttr::getMemLayout)
      .def_property_readonly("memory_layout_as_int", [](tt::LayoutAttr la) {
        return static_cast<uint32_t>(la.getMemLayout());
      });

  tt_attribute_class<tt::GridAttr>(m, "GridAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> shape) {
                    return wrap(tt::GridAttr::get(unwrap(ctx), shape));
                  })
      .def_property_readonly(
          "shape", [](tt::GridAttr const &ga) { return ga.getShape().vec(); });

  tt_attribute_class<tt::ChipCapabilityAttr>(m, "ChipCapabilityAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t chipCapability) {
            return wrap(tt::ChipCapabilityAttr::get(
                unwrap(ctx), static_cast<tt::ChipCapability>(chipCapability)));
          })
      .def_property_readonly("capability_as_int",
                             [](tt::ChipCapabilityAttr self) {
                               return static_cast<uint32_t>(self.getValue());
                             });

  tt_attribute_class<tt::ArchAttr>(m, "ArchAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t arch) {
                    return wrap(tt::ArchAttr::get(unwrap(ctx),
                                                  static_cast<tt::Arch>(arch)));
                  })
      .def_property_readonly("arch_as_int", [](tt::ArchAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::DataTypeAttr>(m, "DataTypeAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint16_t *supportedDataTypes) {
            return wrap(tt::DataTypeAttr::get(
                unwrap(ctx), static_cast<tt::DataType>(*supportedDataTypes)));
          })
      .def_property_readonly("data_type_as_int", [](tt::DataTypeAttr self) {
        return static_cast<uint16_t>(self.getValue());
      });

  tt_attribute_class<tt::ChipDescAttr>(m, "ChipDescAttr")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAttribute arch, std::vector<int64_t> grid,
             unsigned l1Size, unsigned numDramChannels,
             unsigned dramChannelSize, unsigned nocL1AddressAlignBytes,
             unsigned pcieAddressAlignBytes, unsigned nocDRAMAddressAlignBytes,
             unsigned l1UnreservedBase, unsigned eriscL1UnreservedBase,
             unsigned dramUnreservedBase, unsigned dramUnreservedEnd,
             MlirAttribute chipPhysicalCores, MlirAttribute supportedDataTypes,
             MlirAttribute supportedTileSizes, unsigned numCBs) {
            return wrap(tt::ChipDescAttr::get(
                unwrap(ctx), mlir::cast<tt::ArchAttr>(unwrap(arch)), grid,
                l1Size, numDramChannels, dramChannelSize,
                nocL1AddressAlignBytes, pcieAddressAlignBytes,
                nocDRAMAddressAlignBytes, l1UnreservedBase,
                eriscL1UnreservedBase, dramUnreservedBase, dramUnreservedEnd,
                mlir::dyn_cast<tt::ChipPhysicalCoresAttr>(
                    unwrap(chipPhysicalCores)),
                mlir::cast<tt::DataTypeAttr>(unwrap(supportedDataTypes)),
                mlir::cast<tt::TileSizeAttr>(unwrap(supportedTileSizes)),
                numCBs));
          })
      .def_property_readonly("usable_l1_size",
                             &tt::ChipDescAttr::getUsableL1Size)
      .def_property_readonly("usable_dram_channel_size",
                             &tt::ChipDescAttr::getUsableDramChannelSize)
      .def_property_readonly("arch", &tt::ChipDescAttr::getArch)
      .def_property_readonly(
          "grid", [](tt::ChipDescAttr self) { return self.getGrid().vec(); })
      .def_property_readonly("l1_size", &tt::ChipDescAttr::getL1Size)
      .def_property_readonly("num_dram_channels",
                             &tt::ChipDescAttr::getNumDramChannels)
      .def_property_readonly("dram_channel_size",
                             &tt::ChipDescAttr::getDramChannelSize)
      .def_property_readonly("noc_l1_address_align_bytes",
                             &tt::ChipDescAttr::getNocL1AddressAlignBytes)
      .def_property_readonly("pcie_address_align_bytes",
                             &tt::ChipDescAttr::getPcieAddressAlignBytes)
      .def_property_readonly("noc_dram_address_align_bytes",
                             &tt::ChipDescAttr::getNocDRAMAddressAlignBytes)
      .def_property_readonly("l1_unreserved_base",
                             &tt::ChipDescAttr::getL1UnreservedBase)
      .def_property_readonly("erisc_l1_unreserved_base",
                             &tt::ChipDescAttr::getEriscL1UnreservedBase)
      .def_property_readonly("dram_unreserved_base",
                             &tt::ChipDescAttr::getDramUnreservedBase)
      .def_property_readonly("dram_unreserved_end",
                             &tt::ChipDescAttr::getDramUnreservedEnd)
      .def_property_readonly("chip_physical_cores",
                             &tt::ChipDescAttr::getChipPhysicalCores)
      .def_property_readonly("supported_data_types",
                             [](tt::ChipDescAttr self) {
                               return self.getSupportedDataTypes().vec();
                             })
      .def_property_readonly("supported_tile_sizes",
                             [](tt::ChipDescAttr self) {
                               return self.getSupportedTileSizes().vec();
                             })
      .def_property_readonly("num_cbs", &tt::ChipDescAttr::getNumCBs);

  tt_attribute_class<tt::TileSizeAttr>(m, "TileSizeAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return wrap(tt::TileSizeAttr::get(unwrap(ctx), y, x));
                  })
      .def_property_readonly("y", &tt::TileSizeAttr::getY)
      .def_property_readonly("x", &tt::TileSizeAttr::getX);

  tt_attribute_class<tt::ChipPhysicalCoresAttr>(m, "ChipPhysicalCoresAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<tt::CoreCoordAttr> worker,
                     std::vector<tt::CoreCoordAttr> dram,
                     std::vector<tt::CoreCoordAttr> eth,
                     std::vector<tt::CoreCoordAttr> eth_inactive) {
                    return wrap(tt::ChipPhysicalCoresAttr::get(
                        unwrap(ctx), worker, dram, eth, eth_inactive));
                  })
      .def_property_readonly(
          "worker",
          [](tt::ChipPhysicalCoresAttr self) { return self.getWorker().vec(); })
      .def_property_readonly(
          "dram",
          [](tt::ChipPhysicalCoresAttr self) { return self.getDram().vec(); })
      .def_property_readonly(
          "eth",
          [](tt::ChipPhysicalCoresAttr self) { return self.getEth().vec(); })
      .def_property_readonly("eth_inactive",
                             [](tt::ChipPhysicalCoresAttr self) {
                               return self.getEthInactive().vec();
                             });

  tt_attribute_class<tt::ChipCoordAttr>(m, "ChipCoordAttr")
      .def_static("get",
                  [](MlirContext ctx, unsigned rack, unsigned shelf, unsigned y,
                     unsigned x) {
                    return wrap(
                        tt::ChipCoordAttr::get(unwrap(ctx), rack, shelf, y, x));
                  })
      .def_property_readonly("rack", &tt::ChipCoordAttr::getRack)
      .def_property_readonly("shelf", &tt::ChipCoordAttr::getShelf)
      .def_property_readonly("y", &tt::ChipCoordAttr::getY)
      .def_property_readonly("x", &tt::ChipCoordAttr::getX);

  tt_attribute_class<tt::ChipChannelAttr>(m, "ChipChannelAttr")
      .def_static(
          "get",
          [](MlirContext ctx, unsigned deviceId0,
             std::vector<int64_t> ethernetCoreCoord0, unsigned deviceId1,
             std::vector<int64_t> ethernetCoreCoord1) {
            return wrap(tt::ChipChannelAttr::get(unwrap(ctx), deviceId0,
                                                 ethernetCoreCoord0, deviceId1,
                                                 ethernetCoreCoord1));
          })
      .def_property_readonly("device_id0", &tt::ChipChannelAttr::getDeviceId0)
      .def_property_readonly("ethernet_core_coord0",
                             [](tt::ChipChannelAttr self) {
                               return self.getEthernetCoreCoord0().vec();
                             })
      .def_property_readonly("device_id1", &tt::ChipChannelAttr::getDeviceId1)
      .def_property_readonly("ethernet_core_coord1",
                             [](tt::ChipChannelAttr self) {
                               return self.getEthernetCoreCoord1().vec();
                             });

  tt_attribute_class<tt::SystemDescAttr>(m, "SystemDescAttr")
      .def_static("get_default",
                  [](MlirContext ctx) {
                    return wrap(tt::SystemDescAttr::getDefault(unwrap(ctx)));
                  })
      .def_static("get", [](MlirContext ctx,
                            std::vector<MlirAttribute> chipDescs,
                            std::vector<unsigned> chipDescIndices,
                            std::vector<MlirAttribute> chipCapabilities,
                            std::vector<MlirAttribute> chipCoords,
                            std::vector<MlirAttribute> chipChannels,
                            std::vector<MlirAttribute> cpuDescs) {
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
        std::vector<tt::CPUDescAttr> cpuDescsUnwrapped;
        for (auto cpuDesc : cpuDescs)
        {
            cpuDescsUnwrapped.push_back(mlir::cast<tt::CPUDescAttr>(cpuDesc));
        }
        return wrap(tt::SystemDescAttr::get(
            unwrap(ctx), chipDescsUnwrapped, chipDescIndices,
            chipCapabilitiesUnwrapped, chipCoordsUnwrapped,
            chipChannelsUnwrapped, cpuDescsUnwrapped));
      });

  tt_attribute_class<tt::MemorySpaceAttr>(m, "MemorySpaceAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t memorySpace) {
            return wrap(tt::MemorySpaceAttr::get(
                unwrap(ctx), static_cast<tt::MemorySpace>(memorySpace)));
          })
      .def_property_readonly("memory_space_as_int",
                             [](tt::MemorySpaceAttr self) {
                               return static_cast<uint32_t>(self.getValue());
                             });

  tt_attribute_class<tt::OOBValAttr>(m, "OOBValAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t oobVal) {
                    return wrap(tt::OOBValAttr::get(
                        unwrap(ctx), static_cast<tt::OOBVal>(oobVal)));
                  })
      .def_property_readonly("oob_val_as_int", [](tt::OOBValAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::TensorMemoryLayoutAttr>(m, "TensorMemoryLayoutAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t memLayout) {
            return wrap(tt::TensorMemoryLayoutAttr::get(
                unwrap(ctx), static_cast<tt::TensorMemoryLayout>(memLayout)));
          })
      .def_property_readonly("mem_layout_as_int",
                             [](tt::TensorMemoryLayoutAttr self) {
                               return static_cast<uint32_t>(self.getValue());
                             });

  tt_attribute_class<tt::IteratorTypeAttr>(m, "IteratorTypeAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t iteratorType) {
            return wrap(tt::IteratorTypeAttr::get(
                unwrap(ctx), static_cast<tt::IteratorType>(iteratorType)));
          })
      .def_property_readonly("iterator_type_as_int",
                             [](tt::IteratorTypeAttr self) {
                               return static_cast<uint32_t>(self.getValue());
                             });

  tt_attribute_class<tt::OperandConstraintAttr>(m, "OperandConstraintAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t operandConstraint) {
                    return wrap(tt::OperandConstraintAttr::get(
                        unwrap(ctx),
                        static_cast<tt::OperandConstraint>(operandConstraint)));
                  })
      .def_static(
          "get",
          [](MlirContext ctx, std::vector<MlirAttribute> attributesArray) {
            return ::ttmlir::utils::wrapArrayOfMlirAttributesAsAttribute(
                ctx, attributesArray);
          })
      .def_property_readonly("operand_constraint_as_int",
                             [](tt::OperandConstraintAttr self) {
                               return static_cast<uint32_t>(self.getValue());
                             });

  tt_type_class<tt::DeviceType>(m, "DeviceType")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAttribute deviceAttr) {
            return wrap(tt::DeviceType::get(
                unwrap(ctx), mlir::cast<tt::DeviceAttr>(unwrap(deviceAttr))));
          })
      .def_property_readonly("device_attr", [](tt::DeviceType const &self) {
        return self.getDesc();
      });

  tt_attribute_class<tt::DeviceAttr>(m, "DeviceAttr")
      .def_static("from_system_desc",
                  [](MlirContext ctx, MlirAttribute systemDesc,
                     std::vector<int64_t> meshShape) {
                    return wrap(tt::DeviceAttr::get(
                        unwrap(ctx),
                        mlir::cast<tt::SystemDescAttr>(unwrap(systemDesc)),
                        meshShape));
                  })
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> gridShape,
                     MlirAffineMap workerGridMapping, MlirAffineMap l1Map,
                     MlirAffineMap dramMap, std::vector<int64_t> meshShape,
                     std::vector<unsigned> chipIds) {
                    return wrap(tt::DeviceAttr::get(
                        unwrap(ctx),
                        tt::GridAttr::get(unwrap(ctx), gridShape,
                                          unwrap(workerGridMapping)),
                        unwrap(l1Map), unwrap(dramMap), meshShape, chipIds));
                  })
      .def("unwrap",
           [](MlirAttribute const &self) {
             return mlir::cast<tt::DeviceAttr>(unwrap(self));
           })
      .def_property_readonly("grid_attr", &tt::DeviceAttr::getWorkerGrid)
      .def_property_readonly("l1_map", &tt::DeviceAttr::getL1Map)
      .def_property_readonly("dram_map", &tt::DeviceAttr::getDramMap)
      .def_property_readonly(
          "mesh_shape",
          [](tt::DeviceAttr const &self) { return self.getMeshShape().vec(); })
      .def_property_readonly("chip_ids", [](tt::DeviceAttr const &self) {
        return self.getChipIds().vec();
      });

  tt_type_class<tt::TileType>(m, "TileType")
      .def_static("get",
                  [](MlirContext ctx, std::int64_t height, std::int64_t width,
                     uint32_t dataType) {
                    return wrap(tt::TileType::get(
                        unwrap(ctx), SmallVector<std::int64_t>{height, width},
                        static_cast<tt::DataType>(dataType)));
                  })
      .def_property_readonly("data_type", &tt::TileType::getDataType)
      .def_property_readonly("shape", [](tt::TileType const &tile) {
        return std::vector<int64_t>({tile.getHeight(), tile.getWidth()});
      });
}
} // namespace mlir::ttmlir::python
