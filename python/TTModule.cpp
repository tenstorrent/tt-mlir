// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/AffineMap.h"

#include <cstdint>
#include <vector>

namespace mlir::ttmlir::python {
void populateTTModule(nb::module_ &m) {
  tt_attribute_class<tt::ttcore::MetalLayoutAttr>(m, "MetalLayoutAttr")
      // 5-arg overload (no index_map provided)
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> logicalShape,
                     uint32_t oobValValue, uint32_t memorySpaceValue) {
                    return wrap(tt::ttcore::MetalLayoutAttr::get(
                        unwrap(ctx), ArrayRef<int64_t>(logicalShape),
                        static_cast<tt::ttcore::OOBVal>(oobValValue),
                        static_cast<tt::ttcore::MemorySpace>(memorySpaceValue),
                        tt::ttcore::TensorMemoryLayout::Sharded));
                  })
      // 7-arg overload (override memory layout)
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> logicalShape,
                     uint32_t oobValValue, uint32_t memorySpaceValue,
                     uint32_t memoryLayoutValue) {
                    return wrap(tt::ttcore::MetalLayoutAttr::get(
                        unwrap(ctx), ArrayRef<int64_t>(logicalShape),
                        static_cast<tt::ttcore::OOBVal>(oobValValue),
                        static_cast<tt::ttcore::MemorySpace>(memorySpaceValue),
                        static_cast<tt::ttcore::TensorMemoryLayout>(
                            memoryLayoutValue)));
                  })
      .def("getLayout",
           [](MlirType &type)
               -> std::variant<tt::ttcore::MetalLayoutAttr, nb::object> {
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
             tt::ttcore::MetalLayoutAttr layout =
                 mlir::cast<tt::ttcore::MetalLayoutAttr>(tensor.getEncoding());
             return layout;
           })
      .def("wrapped",
           [](const tt::ttcore::MetalLayoutAttr &self) { return wrap(self); })
      .def("getDeviceShape",
           [](const tt::ttcore::MetalLayoutAttr &self,
              std::vector<int64_t> gridShape, std::vector<int64_t> tileShape) {
             const auto shape = self.getDeviceShape(gridShape, tileShape);
             return std::vector<int64_t>(shape.begin(), shape.end());
           })
      // Properties
      .def_prop_ro("logical_shape",
                   [](const tt::ttcore::MetalLayoutAttr &self) {
                     auto shape = self.getLogicalShape();
                     return std::vector<int64_t>(shape.begin(), shape.end());
                   })
      .def_prop_ro("dim_alignments",
                   [](const tt::ttcore::MetalLayoutAttr &self)
                       -> std::optional<std::vector<int64_t>> {
                     if (auto align = self.getDimAlignments(); !align.empty()) {
                       return std::vector<int64_t>(align.begin(), align.end());
                     }
                     return std::nullopt;
                   })
      .def_prop_ro("oobval", &tt::ttcore::MetalLayoutAttr::getOobVal)
      .def_prop_ro("oobval_as_int",
                   [](tt::ttcore::MetalLayoutAttr la) {
                     return static_cast<uint32_t>(la.getOobVal());
                   })
      .def_prop_ro("memory_space", &tt::ttcore::MetalLayoutAttr::getMemorySpace)
      .def_prop_ro("memory_space_as_int",
                   [](tt::ttcore::MetalLayoutAttr la) {
                     return static_cast<uint32_t>(la.getMemorySpace());
                   })
      .def_prop_ro("memory_layout",
                   &tt::ttcore::MetalLayoutAttr::getMemoryLayout)
      .def_prop_ro("memory_layout_as_int", [](tt::ttcore::MetalLayoutAttr la) {
        return static_cast<uint32_t>(la.getMemoryLayout());
      });

  tt_attribute_class<tt::ttcore::GridAttr>(m, "GridAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> shape) {
                    return wrap(tt::ttcore::GridAttr::get(unwrap(ctx), shape));
                  })
      .def_prop_ro("shape", [](const tt::ttcore::GridAttr &ga) {
        return ga.getShape().vec();
      });

  tt_attribute_class<tt::ttcore::ChipCapabilityAttr>(m, "ChipCapabilityAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t chipCapability) {
                    return wrap(tt::ttcore::ChipCapabilityAttr::get(
                        unwrap(ctx), static_cast<tt::ttcore::ChipCapability>(
                                         chipCapability)));
                  })
      .def_prop_ro("capability_as_int",
                   [](tt::ttcore::ChipCapabilityAttr self) {
                     return static_cast<uint32_t>(self.getValue());
                   });

  tt_attribute_class<tt::ttcore::ArchAttr>(m, "ArchAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t arch) {
                    return wrap(tt::ttcore::ArchAttr::get(
                        unwrap(ctx), static_cast<tt::ttcore::Arch>(arch)));
                  })
      .def_prop_ro("arch_as_int", [](tt::ttcore::ArchAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::ttcore::DataTypeAttr>(m, "DataTypeAttr")
      .def_static("get",
                  [](MlirContext ctx, uint16_t *supportedDataTypes) {
                    return wrap(tt::ttcore::DataTypeAttr::get(
                        unwrap(ctx), static_cast<tt::ttcore::DataType>(
                                         *supportedDataTypes)));
                  })
      .def_prop_ro("data_type_as_int", [](tt::ttcore::DataTypeAttr self) {
        return static_cast<uint16_t>(self.getValue());
      });

  tt_attribute_class<tt::ttcore::ChipDescAttr>(m, "ChipDescAttr")
      .def_static(
          "get",
          [](MlirContext ctx, MlirAttribute arch, std::vector<int64_t> grid,
             std::vector<int64_t> coordTranslationOffsets, unsigned l1Size,
             unsigned numDramChannels, unsigned dramChannelSize,
             unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
             unsigned nocDRAMAddressAlignBytes, unsigned l1UnreservedBase,
             unsigned eriscL1UnreservedBase, unsigned dramUnreservedBase,
             unsigned dramUnreservedEnd, MlirAttribute supportedDataTypes,
             MlirAttribute supportedTileSizes, unsigned dstPhysicalSizeTiles,
             unsigned numCBs, unsigned numComputeThreads,
             unsigned numDatamovementThreads) {
            return wrap(tt::ttcore::ChipDescAttr::get(
                unwrap(ctx), mlir::cast<tt::ttcore::ArchAttr>(unwrap(arch)),
                grid, coordTranslationOffsets, l1Size, numDramChannels,
                dramChannelSize, nocL1AddressAlignBytes, pcieAddressAlignBytes,
                nocDRAMAddressAlignBytes, l1UnreservedBase,
                eriscL1UnreservedBase, dramUnreservedBase, dramUnreservedEnd,
                mlir::cast<tt::ttcore::DataTypeAttr>(
                    unwrap(supportedDataTypes)),
                mlir::cast<tt::ttcore::TileSizeAttr>(
                    unwrap(supportedTileSizes)),
                dstPhysicalSizeTiles, numCBs, numComputeThreads,
                numDatamovementThreads));
          })
      .def_prop_ro("usable_l1_size", &tt::ttcore::ChipDescAttr::getUsableL1Size)
      .def_prop_ro("usable_dram_channel_size",
                   &tt::ttcore::ChipDescAttr::getUsableDramChannelSize)
      .def_prop_ro(
          "get_dst_logical_size_tiles",
          [](const tt::ttcore::ChipDescAttr &self, MlirType type,
             bool fullSyncEn, unsigned overridePhysicalSize) {
            return self.getDstLogicalSizeTiles(unwrap(type), fullSyncEn,
                                               overridePhysicalSize);
          },
          nb::arg("type"), nb::arg("full_sync_en") = false,
          nb::arg("override_physical_size") = 0)
      .def_prop_ro("arch", &tt::ttcore::ChipDescAttr::getArch)
      .def_prop_ro(
          "grid",
          [](tt::ttcore::ChipDescAttr self) { return self.getGrid().vec(); })
      .def_prop_ro("coord_translation_offsets",
                   [](tt::ttcore::ChipDescAttr self) {
                     return self.getCoordTranslationOffsets().vec();
                   })
      .def_prop_ro("l1_size", &tt::ttcore::ChipDescAttr::getL1Size)
      .def_prop_ro("num_dram_channels",
                   &tt::ttcore::ChipDescAttr::getNumDramChannels)
      .def_prop_ro("dram_channel_size",
                   &tt::ttcore::ChipDescAttr::getDramChannelSize)
      .def_prop_ro("noc_l1_address_align_bytes",
                   &tt::ttcore::ChipDescAttr::getNocL1AddressAlignBytes)
      .def_prop_ro("pcie_address_align_bytes",
                   &tt::ttcore::ChipDescAttr::getPcieAddressAlignBytes)
      .def_prop_ro("noc_dram_address_align_bytes",
                   &tt::ttcore::ChipDescAttr::getNocDRAMAddressAlignBytes)
      .def_prop_ro("l1_unreserved_base",
                   &tt::ttcore::ChipDescAttr::getL1UnreservedBase)
      .def_prop_ro("erisc_l1_unreserved_base",
                   &tt::ttcore::ChipDescAttr::getEriscL1UnreservedBase)
      .def_prop_ro("dram_unreserved_base",
                   &tt::ttcore::ChipDescAttr::getDramUnreservedBase)
      .def_prop_ro("dram_unreserved_end",
                   &tt::ttcore::ChipDescAttr::getDramUnreservedEnd)
      .def_prop_ro("supported_data_types",
                   [](tt::ttcore::ChipDescAttr self) {
                     return self.getSupportedDataTypes().vec();
                   })
      .def_prop_ro("supported_tile_sizes",
                   [](tt::ttcore::ChipDescAttr self) {
                     return self.getSupportedTileSizes().vec();
                   })
      .def_prop_ro("num_cbs", &tt::ttcore::ChipDescAttr::getNumCBs);

  tt_attribute_class<tt::ttcore::TileSizeAttr>(m, "TileSizeAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return wrap(
                        tt::ttcore::TileSizeAttr::get(unwrap(ctx), y, x));
                  })
      .def_prop_ro("y", &tt::ttcore::TileSizeAttr::getY)
      .def_prop_ro("x", &tt::ttcore::TileSizeAttr::getX);

  tt_attribute_class<tt::ttcore::CoreCoordAttr>(m, "CoreCoordAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return wrap(
                        tt::ttcore::CoreCoordAttr::get(unwrap(ctx), y, x));
                  })
      .def_prop_ro("y", &tt::ttcore::CoreCoordAttr::getY)
      .def_prop_ro("x", &tt::ttcore::CoreCoordAttr::getX);

  tt_attribute_class<tt::ttcore::ChipCoordAttr>(m, "ChipCoordAttr")
      .def_static("get",
                  [](MlirContext ctx, unsigned rack, unsigned shelf, unsigned y,
                     unsigned x) {
                    return wrap(tt::ttcore::ChipCoordAttr::get(
                        unwrap(ctx), rack, shelf, y, x));
                  })
      .def_prop_ro("rack", &tt::ttcore::ChipCoordAttr::getRack)
      .def_prop_ro("shelf", &tt::ttcore::ChipCoordAttr::getShelf)
      .def_prop_ro("y", &tt::ttcore::ChipCoordAttr::getY)
      .def_prop_ro("x", &tt::ttcore::ChipCoordAttr::getX);

  tt_attribute_class<tt::ttcore::ChipChannelAttr>(m, "ChipChannelAttr")
      .def_static("get",
                  [](MlirContext ctx, unsigned deviceId0,
                     std::vector<int64_t> ethernetCoreCoord0,
                     unsigned deviceId1,
                     std::vector<int64_t> ethernetCoreCoord1) {
                    return wrap(tt::ttcore::ChipChannelAttr::get(
                        unwrap(ctx), deviceId0, ethernetCoreCoord0, deviceId1,
                        ethernetCoreCoord1));
                  })
      .def_prop_ro("device_id0", &tt::ttcore::ChipChannelAttr::getDeviceId0)
      .def_prop_ro("ethernet_core_coord0",
                   [](tt::ttcore::ChipChannelAttr self) {
                     return self.getEthernetCoreCoord0().vec();
                   })
      .def_prop_ro("device_id1", &tt::ttcore::ChipChannelAttr::getDeviceId1)
      .def_prop_ro("ethernet_core_coord1",
                   [](tt::ttcore::ChipChannelAttr self) {
                     return self.getEthernetCoreCoord1().vec();
                   });

  tt_attribute_class<tt::ttcore::SystemDescAttr>(m, "SystemDescAttr")
      .def_static("get_default",
                  [](MlirContext ctx) {
                    return wrap(
                        tt::ttcore::SystemDescAttr::getDefault(unwrap(ctx)));
                  })
      .def_static(
          "get",
          [](MlirContext ctx, const std::vector<MlirAttribute> &cpuDescs,
             const std::vector<MlirAttribute> &chipDescs,
             const std::vector<unsigned> &chipDescIndices,
             const std::vector<MlirAttribute> &chipCapabilities,
             const std::vector<MlirAttribute> &chipCoords,
             const std::vector<MlirAttribute> &chipChannels) {
            std::vector<tt::ttcore::ChipDescAttr> chipDescsUnwrapped;
            for (const auto &chipDesc : chipDescs) {
              chipDescsUnwrapped.push_back(
                  mlir::cast<tt::ttcore::ChipDescAttr>(unwrap(chipDesc)));
            }
            std::vector<tt::ttcore::ChipCapabilityAttr>
                chipCapabilitiesUnwrapped;
            for (const auto &chipCapability : chipCapabilities) {
              chipCapabilitiesUnwrapped.push_back(
                  mlir::cast<tt::ttcore::ChipCapabilityAttr>(
                      unwrap(chipCapability)));
            }
            std::vector<tt::ttcore::ChipCoordAttr> chipCoordsUnwrapped;
            for (const auto &chipCoord : chipCoords) {
              chipCoordsUnwrapped.push_back(
                  mlir::cast<tt::ttcore::ChipCoordAttr>(unwrap(chipCoord)));
            }
            std::vector<tt::ttcore::ChipChannelAttr> chipChannelsUnwrapped;
            for (const auto &chipChannel : chipChannels) {
              chipChannelsUnwrapped.push_back(
                  mlir::cast<tt::ttcore::ChipChannelAttr>(unwrap(chipChannel)));
            }
            std::vector<tt::ttcore::CPUDescAttr> cpuDescsUnwrapped;
            for (const auto &cpuDesc : cpuDescs) {
              cpuDescsUnwrapped.push_back(
                  mlir::cast<tt::ttcore::CPUDescAttr>(unwrap(cpuDesc)));
            }
            return wrap(tt::ttcore::SystemDescAttr::get(
                unwrap(ctx), cpuDescsUnwrapped, chipDescsUnwrapped,
                chipDescIndices, chipCapabilitiesUnwrapped, chipCoordsUnwrapped,
                chipChannelsUnwrapped));
          })
      .def_prop_ro("chip_descs",
                   [](tt::ttcore::SystemDescAttr self) {
                     return self.getChipDescs().vec();
                   })
      .def_prop_ro("chip_desc_indices",
                   [](tt::ttcore::SystemDescAttr self) {
                     return self.getChipDescIndices().vec();
                   })
      .def_prop_ro("chip_capabilities",
                   [](tt::ttcore::SystemDescAttr self) {
                     return self.getChipCapabilities().vec();
                   })
      .def_prop_ro("chip_coords",
                   [](tt::ttcore::SystemDescAttr self) {
                     return self.getChipCoords().vec();
                   })
      .def_prop_ro("chip_channels", [](tt::ttcore::SystemDescAttr self) {
        return self.getChipChannels().vec();
      });

  tt_attribute_class<tt::ttcore::MemorySpaceAttr>(m, "MemorySpaceAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t memorySpace) {
                    return wrap(tt::ttcore::MemorySpaceAttr::get(
                        unwrap(ctx),
                        static_cast<tt::ttcore::MemorySpace>(memorySpace)));
                  })
      .def_prop_ro("memory_space_as_int", [](tt::ttcore::MemorySpaceAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::ttcore::OOBValAttr>(m, "OOBValAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t oobVal) {
                    return wrap(tt::ttcore::OOBValAttr::get(
                        unwrap(ctx), static_cast<tt::ttcore::OOBVal>(oobVal)));
                  })
      .def_prop_ro("oob_val_as_int", [](tt::ttcore::OOBValAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::ttcore::IteratorTypeAttr>(m, "IteratorTypeAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t iteratorType) {
                    return wrap(tt::ttcore::IteratorTypeAttr::get(
                        unwrap(ctx),
                        static_cast<tt::ttcore::IteratorType>(iteratorType)));
                  })
      .def_prop_ro("iterator_type_as_int",
                   [](tt::ttcore::IteratorTypeAttr self) {
                     return static_cast<uint32_t>(self.getValue());
                   });

  tt_attribute_class<tt::ttcore::DeviceAttr>(m, "DeviceAttr")
      .def_static(
          "from_system_desc",
          [](MlirContext ctx, MlirAttribute systemDesc,
             std::vector<int64_t> meshShape) {
            return wrap(tt::ttcore::DeviceAttr::get(
                unwrap(ctx),
                mlir::cast<tt::ttcore::SystemDescAttr>(unwrap(systemDesc)),
                meshShape));
          })
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> gridShape,
                     MlirAffineMap workerGridMapping, MlirAffineMap l1Map,
                     MlirAffineMap dramMap, std::vector<int64_t> meshShape,
                     std::vector<unsigned> chipIds) {
                    return wrap(tt::ttcore::DeviceAttr::get(
                        unwrap(ctx),
                        tt::ttcore::GridAttr::get(unwrap(ctx), gridShape,
                                                  unwrap(workerGridMapping)),
                        unwrap(l1Map), unwrap(dramMap), meshShape, chipIds));
                  })
      .def("unwrap",
           [](const MlirAttribute &self) {
             return mlir::cast<tt::ttcore::DeviceAttr>(unwrap(self));
           })
      .def_prop_ro("grid_attr", &tt::ttcore::DeviceAttr::getWorkerGrid)
      .def_prop_ro(
          "l1_map",
          [](tt::ttcore::DeviceAttr self) { return wrap(self.getL1Map()); })
      .def_prop_ro(
          "dram_map",
          [](tt::ttcore::DeviceAttr self) { return wrap(self.getDramMap()); })
      .def_prop_ro("mesh_shape",
                   [](const tt::ttcore::DeviceAttr &self) {
                     return self.getMeshShape().vec();
                   })
      .def_prop_ro("chip_ids", [](const tt::ttcore::DeviceAttr &self) {
        return self.getChipIds().vec();
      });

  nb::enum_<mlir::tt::ttcore::TensorMemoryLayout>(m, "TensorMemoryLayout")
      .value("Interleaved", mlir::tt::ttcore::TensorMemoryLayout::Interleaved)
      .value("Sharded", mlir::tt::ttcore::TensorMemoryLayout::Sharded);

  tt_type_class<tt::ttcore::TileType>(m, "TileType")
      .def_static("get",
                  [](MlirContext ctx, std::int64_t height, std::int64_t width,
                     uint32_t dataType) {
                    return wrap(tt::ttcore::TileType::get(
                        unwrap(ctx), SmallVector<std::int64_t>{height, width},
                        static_cast<tt::ttcore::DataType>(dataType)));
                  })
      .def_prop_ro("data_type_as_int",
                   [](tt::ttcore::TileType self) {
                     return static_cast<uint32_t>(self.getDataType());
                   })
      .def_prop_ro("shape", [](const tt::ttcore::TileType &tile) {
        return std::vector<int64_t>({tile.getHeight(), tile.getWidth()});
      });
}
} // namespace mlir::ttmlir::python
