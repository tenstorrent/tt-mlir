// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Target/Common/Target.h"
#include "ttmlir/Utils.h"

namespace mlir::ttmlir::python {
void populateTTModule(nb::module_ &m) {
  tt_attribute_class<tt::MetalLayoutAttr>(m, "MetalLayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> logicalShape,
                     uint32_t oobValValue, uint32_t memorySpaceValue) {
                    return wrap(tt::MetalLayoutAttr::get(
                        unwrap(ctx), ArrayRef<int64_t>(logicalShape),
                        logicalShape.size(),
                        static_cast<tt::OOBVal>(oobValValue),
                        static_cast<tt::MemorySpace>(memorySpaceValue)));
                  })
      .def("getLayout",
           [](MlirType &type) -> std::variant<tt::MetalLayoutAttr, nb::object> {
             // Make sure that this is operating on a RankedTensorType object
             if (!isa<RankedTensorType>(unwrap(type))) {
               return nb::none();
             }
             RankedTensorType tensor =
                 mlir::cast<RankedTensorType>(unwrap(type));
             // Make sure that this Tensor has an encoding value
             if (!tensor.getEncoding()) {
               return nb::none();
             }
             tt::MetalLayoutAttr layout =
                 mlir::cast<tt::MetalLayoutAttr>(tensor.getEncoding());
             return layout;
           })
      .def("wrapped",
           [](const tt::MetalLayoutAttr &self) { return wrap(self); })
      // Properties
      .def_prop_ro("logical_shape",
                   [](const tt::MetalLayoutAttr &self) {
                     auto shape = self.getLogicalShape();
                     return std::vector<int64_t>(shape.begin(), shape.end());
                   })
      .def_prop_ro("dim_alignments",
                   [](const tt::MetalLayoutAttr &self)
                       -> std::optional<std::vector<int64_t>> {
                     if (auto align = self.getDimAlignments(); !align.empty()) {
                       return std::vector<int64_t>(align.begin(), align.end());
                     }
                     return std::nullopt;
                   })
      .def_prop_ro("oobval", &tt::MetalLayoutAttr::getOobVal)
      .def_prop_ro("oobval_as_int",
                   [](tt::MetalLayoutAttr la) {
                     return static_cast<uint32_t>(la.getOobVal());
                   })
      .def_prop_ro("memory_space", &tt::MetalLayoutAttr::getMemorySpace)
      .def_prop_ro("memory_space_as_int", [](tt::MetalLayoutAttr la) {
        return static_cast<uint32_t>(la.getMemorySpace());
      });

  tt_attribute_class<tt::GridAttr>(m, "GridAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> shape) {
                    return wrap(tt::GridAttr::get(unwrap(ctx), shape));
                  })
      .def_prop_ro("shape",
                   [](const tt::GridAttr &ga) { return ga.getShape().vec(); });

  tt_attribute_class<tt::ChipCapabilityAttr>(m, "ChipCapabilityAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t chipCapability) {
            return wrap(tt::ChipCapabilityAttr::get(
                unwrap(ctx), static_cast<tt::ChipCapability>(chipCapability)));
          })
      .def_prop_ro("capability_as_int", [](tt::ChipCapabilityAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::ArchAttr>(m, "ArchAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t arch) {
                    return wrap(tt::ArchAttr::get(unwrap(ctx),
                                                  static_cast<tt::Arch>(arch)));
                  })
      .def_prop_ro("arch_as_int", [](tt::ArchAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::DataTypeAttr>(m, "DataTypeAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint16_t *supportedDataTypes) {
            return wrap(tt::DataTypeAttr::get(
                unwrap(ctx), static_cast<tt::DataType>(*supportedDataTypes)));
          })
      .def_prop_ro("data_type_as_int", [](tt::DataTypeAttr self) {
        return static_cast<uint16_t>(self.getValue());
      });

  tt_attribute_class<tt::ChipDescAttr>(m, "ChipDescAttr")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAttribute arch, std::vector<int64_t> grid,
             std::vector<int64_t> coordTranslationOffsets, unsigned l1Size,
             unsigned numDramChannels, unsigned dramChannelSize,
             unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
             unsigned nocDRAMAddressAlignBytes, unsigned l1UnreservedBase,
             unsigned eriscL1UnreservedBase, unsigned dramUnreservedBase,
             unsigned dramUnreservedEnd, MlirAttribute chipPhysicalHelperCores,
             MlirAttribute supportedDataTypes, MlirAttribute supportedTileSizes,
             unsigned dstRegisterSizeTiles, unsigned numCBs,
             unsigned numComputeThreads, unsigned numDatamovementThreads) {
            return wrap(tt::ChipDescAttr::get(
                unwrap(ctx), mlir::cast<tt::ArchAttr>(unwrap(arch)), grid,
                coordTranslationOffsets, l1Size, numDramChannels,
                dramChannelSize, nocL1AddressAlignBytes, pcieAddressAlignBytes,
                nocDRAMAddressAlignBytes, l1UnreservedBase,
                eriscL1UnreservedBase, dramUnreservedBase, dramUnreservedEnd,
                mlir::dyn_cast<tt::ChipPhysicalHelperCoresAttr>(
                    unwrap(chipPhysicalHelperCores)),
                mlir::cast<tt::DataTypeAttr>(unwrap(supportedDataTypes)),
                mlir::cast<tt::TileSizeAttr>(unwrap(supportedTileSizes)),
                dstRegisterSizeTiles, numCBs, numComputeThreads,
                numDatamovementThreads));
          })
      .def_prop_ro("usable_l1_size", &tt::ChipDescAttr::getUsableL1Size)
      .def_prop_ro("usable_dram_channel_size",
                   &tt::ChipDescAttr::getUsableDramChannelSize)
      .def_prop_ro("arch", &tt::ChipDescAttr::getArch)
      .def_prop_ro("grid",
                   [](tt::ChipDescAttr self) { return self.getGrid().vec(); })
      .def_prop_ro("coord_translation_offsets",
                   [](tt::ChipDescAttr self) {
                     return self.getCoordTranslationOffsets().vec();
                   })
      .def_prop_ro("l1_size", &tt::ChipDescAttr::getL1Size)
      .def_prop_ro("num_dram_channels", &tt::ChipDescAttr::getNumDramChannels)
      .def_prop_ro("dram_channel_size", &tt::ChipDescAttr::getDramChannelSize)
      .def_prop_ro("noc_l1_address_align_bytes",
                   &tt::ChipDescAttr::getNocL1AddressAlignBytes)
      .def_prop_ro("pcie_address_align_bytes",
                   &tt::ChipDescAttr::getPcieAddressAlignBytes)
      .def_prop_ro("noc_dram_address_align_bytes",
                   &tt::ChipDescAttr::getNocDRAMAddressAlignBytes)
      .def_prop_ro("l1_unreserved_base", &tt::ChipDescAttr::getL1UnreservedBase)
      .def_prop_ro("erisc_l1_unreserved_base",
                   &tt::ChipDescAttr::getEriscL1UnreservedBase)
      .def_prop_ro("dram_unreserved_base",
                   &tt::ChipDescAttr::getDramUnreservedBase)
      .def_prop_ro("dram_unreserved_end",
                   &tt::ChipDescAttr::getDramUnreservedEnd)
      .def_prop_ro("chip_physical_helper_cores",
                   &tt::ChipDescAttr::getChipPhysicalHelperCores)
      .def_prop_ro("supported_data_types",
                   [](tt::ChipDescAttr self) {
                     return self.getSupportedDataTypes().vec();
                   })
      .def_prop_ro("supported_tile_sizes",
                   [](tt::ChipDescAttr self) {
                     return self.getSupportedTileSizes().vec();
                   })
      .def_prop_ro("num_cbs", &tt::ChipDescAttr::getNumCBs);

  tt_attribute_class<tt::TileSizeAttr>(m, "TileSizeAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return wrap(tt::TileSizeAttr::get(unwrap(ctx), y, x));
                  })
      .def_prop_ro("y", &tt::TileSizeAttr::getY)
      .def_prop_ro("x", &tt::TileSizeAttr::getX);

  tt_attribute_class<tt::ChipPhysicalHelperCoresAttr>(
      m, "ChipPhysicalHelperCoresAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<tt::CoreCoordAttr> dram,
                     std::vector<tt::CoreCoordAttr> eth,
                     std::vector<tt::CoreCoordAttr> eth_inactive) {
                    return wrap(tt::ChipPhysicalHelperCoresAttr::get(
                        unwrap(ctx), dram, eth, eth_inactive));
                  })
      .def_prop_ro("dram",
                   [](tt::ChipPhysicalHelperCoresAttr self) {
                     return self.getDram().vec();
                   })
      .def_prop_ro("eth",
                   [](tt::ChipPhysicalHelperCoresAttr self) {
                     return self.getEth().vec();
                   })
      .def_prop_ro("eth_inactive", [](tt::ChipPhysicalHelperCoresAttr self) {
        return self.getEthInactive().vec();
      });

  tt_attribute_class<tt::CoreCoordAttr>(m, "CoreCoordAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return wrap(tt::CoreCoordAttr::get(unwrap(ctx), y, x));
                  })
      .def_prop_ro("y", &tt::CoreCoordAttr::getY)
      .def_prop_ro("x", &tt::CoreCoordAttr::getX);

  tt_attribute_class<tt::ChipCoordAttr>(m, "ChipCoordAttr")
      .def_static("get",
                  [](MlirContext ctx, unsigned rack, unsigned shelf, unsigned y,
                     unsigned x) {
                    return wrap(
                        tt::ChipCoordAttr::get(unwrap(ctx), rack, shelf, y, x));
                  })
      .def_prop_ro("rack", &tt::ChipCoordAttr::getRack)
      .def_prop_ro("shelf", &tt::ChipCoordAttr::getShelf)
      .def_prop_ro("y", &tt::ChipCoordAttr::getY)
      .def_prop_ro("x", &tt::ChipCoordAttr::getX);

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
      .def_prop_ro("device_id0", &tt::ChipChannelAttr::getDeviceId0)
      .def_prop_ro("ethernet_core_coord0",
                   [](tt::ChipChannelAttr self) {
                     return self.getEthernetCoreCoord0().vec();
                   })
      .def_prop_ro("device_id1", &tt::ChipChannelAttr::getDeviceId1)
      .def_prop_ro("ethernet_core_coord1", [](tt::ChipChannelAttr self) {
        return self.getEthernetCoreCoord1().vec();
      });

  tt_attribute_class<tt::SystemDescAttr>(m, "SystemDescAttr")
      .def_static("get_default",
                  [](MlirContext ctx) {
                    return wrap(tt::SystemDescAttr::getDefault(unwrap(ctx)));
                  })
      .def_static(
          "get",
          [](MlirContext ctx, const std::vector<MlirAttribute> &cpuDescs,
             const std::vector<MlirAttribute> &chipDescs,
             const std::vector<unsigned> &chipDescIndices,
             const std::vector<MlirAttribute> &chipCapabilities,
             const std::vector<MlirAttribute> &chipCoords,
             const std::vector<MlirAttribute> &chipChannels) {
            std::vector<tt::ChipDescAttr> chipDescsUnwrapped;
            for (const auto &chipDesc : chipDescs) {
              chipDescsUnwrapped.push_back(
                  mlir::cast<tt::ChipDescAttr>(unwrap(chipDesc)));
            }
            std::vector<tt::ChipCapabilityAttr> chipCapabilitiesUnwrapped;
            for (const auto &chipCapability : chipCapabilities) {
              chipCapabilitiesUnwrapped.push_back(
                  mlir::cast<tt::ChipCapabilityAttr>(unwrap(chipCapability)));
            }
            std::vector<tt::ChipCoordAttr> chipCoordsUnwrapped;
            for (const auto &chipCoord : chipCoords) {
              chipCoordsUnwrapped.push_back(
                  mlir::cast<tt::ChipCoordAttr>(unwrap(chipCoord)));
            }
            std::vector<tt::ChipChannelAttr> chipChannelsUnwrapped;
            for (const auto &chipChannel : chipChannels) {
              chipChannelsUnwrapped.push_back(
                  mlir::cast<tt::ChipChannelAttr>(unwrap(chipChannel)));
            }
            std::vector<tt::CPUDescAttr> cpuDescsUnwrapped;
            for (const auto &cpuDesc : cpuDescs) {
              cpuDescsUnwrapped.push_back(
                  mlir::cast<tt::CPUDescAttr>(unwrap(cpuDesc)));
            }
            return wrap(tt::SystemDescAttr::get(
                unwrap(ctx), cpuDescsUnwrapped, chipDescsUnwrapped,
                chipDescIndices, chipCapabilitiesUnwrapped, chipCoordsUnwrapped,
                chipChannelsUnwrapped));
          })
      .def_prop_ro(
          "chip_descs",
          [](tt::SystemDescAttr self) { return self.getChipDescs().vec(); })
      .def_prop_ro("chip_desc_indices",
                   [](tt::SystemDescAttr self) {
                     return self.getChipDescIndices().vec();
                   })
      .def_prop_ro("chip_capabilities",
                   [](tt::SystemDescAttr self) {
                     return self.getChipCapabilities().vec();
                   })
      .def_prop_ro(
          "chip_coords",
          [](tt::SystemDescAttr self) { return self.getChipCoords().vec(); })
      .def_prop_ro("chip_channels", [](tt::SystemDescAttr self) {
        return self.getChipChannels().vec();
      });

  tt_attribute_class<tt::MemorySpaceAttr>(m, "MemorySpaceAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t memorySpace) {
            return wrap(tt::MemorySpaceAttr::get(
                unwrap(ctx), static_cast<tt::MemorySpace>(memorySpace)));
          })
      .def_prop_ro("memory_space_as_int", [](tt::MemorySpaceAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::OOBValAttr>(m, "OOBValAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t oobVal) {
                    return wrap(tt::OOBValAttr::get(
                        unwrap(ctx), static_cast<tt::OOBVal>(oobVal)));
                  })
      .def_prop_ro("oob_val_as_int", [](tt::OOBValAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::IteratorTypeAttr>(m, "IteratorTypeAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t iteratorType) {
            return wrap(tt::IteratorTypeAttr::get(
                unwrap(ctx), static_cast<tt::IteratorType>(iteratorType)));
          })
      .def_prop_ro("iterator_type_as_int", [](tt::IteratorTypeAttr self) {
        return static_cast<uint32_t>(self.getValue());
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
           [](const MlirAttribute &self) {
             return mlir::cast<tt::DeviceAttr>(unwrap(self));
           })
      .def_prop_ro("grid_attr", &tt::DeviceAttr::getWorkerGrid)
      .def_prop_ro("l1_map",
                   [](tt::DeviceAttr self) { return wrap(self.getL1Map()); })
      .def_prop_ro("dram_map",
                   [](tt::DeviceAttr self) { return wrap(self.getDramMap()); })
      .def_prop_ro(
          "mesh_shape",
          [](const tt::DeviceAttr &self) { return self.getMeshShape().vec(); })
      .def_prop_ro("chip_ids", [](const tt::DeviceAttr &self) {
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
      .def_prop_ro("data_type_as_int",
                   [](tt::TileType self) {
                     return static_cast<uint32_t>(self.getDataType());
                   })
      .def_prop_ro("shape", [](const tt::TileType &tile) {
        return std::vector<int64_t>({tile.getHeight(), tile.getWidth()});
      });
}
} // namespace mlir::ttmlir::python
