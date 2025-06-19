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
  tt_attribute_class<ttcore::MetalLayoutAttr>(m, "MetalLayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, MlirType rankedTensorType,
                     uint32_t memorySpaceValue, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals,
                     uint32_t oobValValue) {
                    return wrap(ttcore::MetalLayoutAttr::get(
                        unwrap(ctx),
                        mlir::cast<RankedTensorType>(unwrap(rankedTensorType)),
                        static_cast<ttcore::MemorySpace>(memorySpaceValue),
                        mlir::cast<ttcore::GridAttr>(unwrap(grid)),
                        collapseIntervals,
                        static_cast<ttcore::OOBVal>(oobValValue)));
                  })
      .def_static("get",
                  [](MlirContext ctx, MlirType rankedTensorType,
                     MlirAttribute grid, bool tiled, uint32_t memorySpaceValue,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals,
                     uint32_t oobValValue) {
                    return wrap(ttcore::MetalLayoutAttr::get(
                        unwrap(ctx),
                        mlir::cast<RankedTensorType>(unwrap(rankedTensorType)),
                        mlir::cast<ttcore::GridAttr>(unwrap(grid)), tiled,
                        static_cast<ttcore::MemorySpace>(memorySpaceValue),
                        collapseIntervals,
                        static_cast<ttcore::OOBVal>(oobValValue)));
                  })
      .def_static(
          "with_grid",
          [](MlirContext ctx, MlirAttribute self,
             std::vector<std::int64_t> tensorShape, MlirAttribute grid,
             std::vector<std::pair<std::int64_t, std::int64_t>>
                 collapseIntervals) {
            return wrap(
                mlir::cast<ttcore::MetalLayoutAttr>(unwrap(self))
                    .withGrid(unwrap(ctx), tensorShape,
                              mlir::cast<ttcore::GridAttr>(unwrap(grid)),
                              collapseIntervals));
          })
      .def_static("with_grid_",
                  [](MlirContext ctx, MlirAttribute self,
                     std::vector<std::int64_t> tensorShape, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals) {
                    return mlir::cast<ttcore::MetalLayoutAttr>(unwrap(self))
                        .withGrid(unwrap(ctx), tensorShape,
                                  mlir::cast<ttcore::GridAttr>(unwrap(grid)),
                                  collapseIntervals);
                  })
      .def_static(
          "with_element_type",
          [](MlirContext ctx, MlirAttribute self, MlirType elementType) {
            return wrap(mlir::cast<ttcore::MetalLayoutAttr>(unwrap(self))
                            .withElementType(unwrap(ctx), unwrap(elementType)));
          })
      .def_static(
          "with_element_type_",
          [](MlirContext ctx, MlirAttribute self, MlirType elementType) {
            return mlir::cast<ttcore::MetalLayoutAttr>(unwrap(self))
                .withElementType(unwrap(ctx), unwrap(elementType));
          })
      .def("getLayout",
           [](MlirType &type)
               -> std::variant<ttcore::MetalLayoutAttr, nb::object> {
             // Make sure that this is operating on a RankedTensorType object
             if (not isa<RankedTensorType>(unwrap(type))) {
               return nb::none();
             }
             RankedTensorType tensor =
                 mlir::cast<RankedTensorType>(unwrap(type));
             // Make sure that this Tensor has an encoding value
             if (not tensor.getEncoding()) {
               return nb::none();
             }
             ttcore::MetalLayoutAttr layout =
                 mlir::cast<ttcore::MetalLayoutAttr>(tensor.getEncoding());
             return layout;
           })
      .def("wrapped",
           [](const ttcore::MetalLayoutAttr &self) { return wrap(self); })
      .def_prop_ro("stride",
                   [](const ttcore::MetalLayoutAttr &self,
                      std::vector<int64_t> logicalShape) {
                     auto stride = self.getStride(logicalShape);
                     return std::vector<std::int64_t>(stride.begin(),
                                                      stride.end());
                   })
      .def_prop_ro("oobval", &ttcore::MetalLayoutAttr::getOobVal)
      .def_prop_ro("oobval_as_int",
                   [](ttcore::MetalLayoutAttr la) {
                     return static_cast<uint32_t>(la.getOobVal());
                   })
      .def_prop_ro("grid_attr", &ttcore::MetalLayoutAttr::getGrid)
      .def_prop_ro(
          "memref",
          [](ttcore::MetalLayoutAttr self) { return wrap(self.getMemref()); })
      .def_prop_ro("memory_space", &ttcore::MetalLayoutAttr::getMemorySpace)
      .def_prop_ro("memory_space_as_int",
                   [](ttcore::MetalLayoutAttr la) {
                     return static_cast<uint32_t>(la.getMemorySpace());
                   })
      .def_prop_ro("shard_shape", &ttcore::MetalLayoutAttr::getShardShape)
      .def_prop_ro("linear", [](ttcore::MetalLayoutAttr self) {
        return wrap(self.getLinear());
      });

  tt_attribute_class<ttcore::GridAttr>(m, "GridAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> shape) {
                    return wrap(ttcore::GridAttr::get(unwrap(ctx), shape));
                  })
      .def_prop_ro("shape", [](const ttcore::GridAttr &ga) {
        return ga.getShape().vec();
      });

  tt_attribute_class<ttcore::ChipCapabilityAttr>(m, "ChipCapabilityAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t chipCapability) {
                    return wrap(ttcore::ChipCapabilityAttr::get(
                        unwrap(ctx),
                        static_cast<ttcore::ChipCapability>(chipCapability)));
                  })
      .def_prop_ro("capability_as_int", [](ttcore::ChipCapabilityAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<ttcore::ArchAttr>(m, "ArchAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t arch) {
                    return wrap(ttcore::ArchAttr::get(
                        unwrap(ctx), static_cast<ttcore::Arch>(arch)));
                  })
      .def_prop_ro("arch_as_int", [](ttcore::ArchAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<ttcore::DataTypeAttr>(m, "DataTypeAttr")
      .def_static("get",
                  [](MlirContext ctx, uint16_t *supportedDataTypes) {
                    return wrap(ttcore::DataTypeAttr::get(
                        unwrap(ctx),
                        static_cast<ttcore::DataType>(*supportedDataTypes)));
                  })
      .def_prop_ro("data_type_as_int", [](ttcore::DataTypeAttr self) {
        return static_cast<uint16_t>(self.getValue());
      });

  tt_attribute_class<ttcore::ChipDescAttr>(m, "ChipDescAttr")
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
            return wrap(ttcore::ChipDescAttr::get(
                unwrap(ctx), mlir::cast<ttcore::ArchAttr>(unwrap(arch)), grid,
                coordTranslationOffsets, l1Size, numDramChannels,
                dramChannelSize, nocL1AddressAlignBytes, pcieAddressAlignBytes,
                nocDRAMAddressAlignBytes, l1UnreservedBase,
                eriscL1UnreservedBase, dramUnreservedBase, dramUnreservedEnd,
                mlir::dyn_cast<ttcore::ChipPhysicalHelperCoresAttr>(
                    unwrap(chipPhysicalHelperCores)),
                mlir::cast<ttcore::DataTypeAttr>(unwrap(supportedDataTypes)),
                mlir::cast<ttcore::TileSizeAttr>(unwrap(supportedTileSizes)),
                dstRegisterSizeTiles, numCBs, numComputeThreads,
                numDatamovementThreads));
          })
      .def_prop_ro("usable_l1_size", &ttcore::ChipDescAttr::getUsableL1Size)
      .def_prop_ro("usable_dram_channel_size",
                   &ttcore::ChipDescAttr::getUsableDramChannelSize)
      .def_prop_ro("arch", &ttcore::ChipDescAttr::getArch)
      .def_prop_ro(
          "grid",
          [](ttcore::ChipDescAttr self) { return self.getGrid().vec(); })
      .def_prop_ro("coord_translation_offsets",
                   [](ttcore::ChipDescAttr self) {
                     return self.getCoordTranslationOffsets().vec();
                   })
      .def_prop_ro("l1_size", &ttcore::ChipDescAttr::getL1Size)
      .def_prop_ro("num_dram_channels",
                   &ttcore::ChipDescAttr::getNumDramChannels)
      .def_prop_ro("dram_channel_size",
                   &ttcore::ChipDescAttr::getDramChannelSize)
      .def_prop_ro("noc_l1_address_align_bytes",
                   &ttcore::ChipDescAttr::getNocL1AddressAlignBytes)
      .def_prop_ro("pcie_address_align_bytes",
                   &ttcore::ChipDescAttr::getPcieAddressAlignBytes)
      .def_prop_ro("noc_dram_address_align_bytes",
                   &ttcore::ChipDescAttr::getNocDRAMAddressAlignBytes)
      .def_prop_ro("l1_unreserved_base",
                   &ttcore::ChipDescAttr::getL1UnreservedBase)
      .def_prop_ro("erisc_l1_unreserved_base",
                   &ttcore::ChipDescAttr::getEriscL1UnreservedBase)
      .def_prop_ro("dram_unreserved_base",
                   &ttcore::ChipDescAttr::getDramUnreservedBase)
      .def_prop_ro("dram_unreserved_end",
                   &ttcore::ChipDescAttr::getDramUnreservedEnd)
      .def_prop_ro("chip_physical_helper_cores",
                   &ttcore::ChipDescAttr::getChipPhysicalHelperCores)
      .def_prop_ro("supported_data_types",
                   [](ttcore::ChipDescAttr self) {
                     return self.getSupportedDataTypes().vec();
                   })
      .def_prop_ro("supported_tile_sizes",
                   [](ttcore::ChipDescAttr self) {
                     return self.getSupportedTileSizes().vec();
                   })
      .def_prop_ro("num_cbs", &ttcore::ChipDescAttr::getNumCBs);

  tt_attribute_class<ttcore::TileSizeAttr>(m, "TileSizeAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return wrap(ttcore::TileSizeAttr::get(unwrap(ctx), y, x));
                  })
      .def_prop_ro("y", &ttcore::TileSizeAttr::getY)
      .def_prop_ro("x", &ttcore::TileSizeAttr::getX);

  tt_attribute_class<ttcore::ChipPhysicalHelperCoresAttr>(
      m, "ChipPhysicalHelperCoresAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<ttcore::CoreCoordAttr> dram,
                     std::vector<ttcore::CoreCoordAttr> eth,
                     std::vector<ttcore::CoreCoordAttr> eth_inactive) {
                    return wrap(ttcore::ChipPhysicalHelperCoresAttr::get(
                        unwrap(ctx), dram, eth, eth_inactive));
                  })
      .def_prop_ro("dram",
                   [](ttcore::ChipPhysicalHelperCoresAttr self) {
                     return self.getDram().vec();
                   })
      .def_prop_ro("eth",
                   [](ttcore::ChipPhysicalHelperCoresAttr self) {
                     return self.getEth().vec();
                   })
      .def_prop_ro("eth_inactive",
                   [](ttcore::ChipPhysicalHelperCoresAttr self) {
                     return self.getEthInactive().vec();
                   });

  tt_attribute_class<ttcore::CoreCoordAttr>(m, "CoreCoordAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return wrap(ttcore::CoreCoordAttr::get(unwrap(ctx), y, x));
                  })
      .def_prop_ro("y", &ttcore::CoreCoordAttr::getY)
      .def_prop_ro("x", &ttcore::CoreCoordAttr::getX);

  tt_attribute_class<ttcore::ChipCoordAttr>(m, "ChipCoordAttr")
      .def_static("get",
                  [](MlirContext ctx, unsigned rack, unsigned shelf, unsigned y,
                     unsigned x) {
                    return wrap(ttcore::ChipCoordAttr::get(unwrap(ctx), rack,
                                                           shelf, y, x));
                  })
      .def_prop_ro("rack", &ttcore::ChipCoordAttr::getRack)
      .def_prop_ro("shelf", &ttcore::ChipCoordAttr::getShelf)
      .def_prop_ro("y", &ttcore::ChipCoordAttr::getY)
      .def_prop_ro("x", &ttcore::ChipCoordAttr::getX);

  tt_attribute_class<ttcore::ChipChannelAttr>(m, "ChipChannelAttr")
      .def_static("get",
                  [](MlirContext ctx, unsigned deviceId0,
                     std::vector<int64_t> ethernetCoreCoord0,
                     unsigned deviceId1,
                     std::vector<int64_t> ethernetCoreCoord1) {
                    return wrap(ttcore::ChipChannelAttr::get(
                        unwrap(ctx), deviceId0, ethernetCoreCoord0, deviceId1,
                        ethernetCoreCoord1));
                  })
      .def_prop_ro("device_id0", &ttcore::ChipChannelAttr::getDeviceId0)
      .def_prop_ro("ethernet_core_coord0",
                   [](ttcore::ChipChannelAttr self) {
                     return self.getEthernetCoreCoord0().vec();
                   })
      .def_prop_ro("device_id1", &ttcore::ChipChannelAttr::getDeviceId1)
      .def_prop_ro("ethernet_core_coord1", [](ttcore::ChipChannelAttr self) {
        return self.getEthernetCoreCoord1().vec();
      });

  tt_attribute_class<ttcore::SystemDescAttr>(m, "SystemDescAttr")
      .def_static("get_default",
                  [](MlirContext ctx) {
                    return wrap(
                        ttcore::SystemDescAttr::getDefault(unwrap(ctx)));
                  })
      .def_static(
          "get",
          [](MlirContext ctx, const std::vector<MlirAttribute> &cpuDescs,
             const std::vector<MlirAttribute> &chipDescs,
             const std::vector<unsigned> &chipDescIndices,
             const std::vector<MlirAttribute> &chipCapabilities,
             const std::vector<MlirAttribute> &chipCoords,
             const std::vector<MlirAttribute> &chipChannels) {
            std::vector<ttcore::ChipDescAttr> chipDescsUnwrapped;
            for (const auto &chipDesc : chipDescs) {
              chipDescsUnwrapped.push_back(
                  mlir::cast<ttcore::ChipDescAttr>(unwrap(chipDesc)));
            }
            std::vector<ttcore::ChipCapabilityAttr> chipCapabilitiesUnwrapped;
            for (const auto &chipCapability : chipCapabilities) {
              chipCapabilitiesUnwrapped.push_back(
                  mlir::cast<ttcore::ChipCapabilityAttr>(
                      unwrap(chipCapability)));
            }
            std::vector<ttcore::ChipCoordAttr> chipCoordsUnwrapped;
            for (const auto &chipCoord : chipCoords) {
              chipCoordsUnwrapped.push_back(
                  mlir::cast<ttcore::ChipCoordAttr>(unwrap(chipCoord)));
            }
            std::vector<ttcore::ChipChannelAttr> chipChannelsUnwrapped;
            for (const auto &chipChannel : chipChannels) {
              chipChannelsUnwrapped.push_back(
                  mlir::cast<ttcore::ChipChannelAttr>(unwrap(chipChannel)));
            }
            std::vector<ttcore::CPUDescAttr> cpuDescsUnwrapped;
            for (const auto &cpuDesc : cpuDescs) {
              cpuDescsUnwrapped.push_back(
                  mlir::cast<ttcore::CPUDescAttr>(unwrap(cpuDesc)));
            }
            return wrap(ttcore::SystemDescAttr::get(
                unwrap(ctx), cpuDescsUnwrapped, chipDescsUnwrapped,
                chipDescIndices, chipCapabilitiesUnwrapped, chipCoordsUnwrapped,
                chipChannelsUnwrapped));
          })
      .def_prop_ro(
          "chip_descs",
          [](ttcore::SystemDescAttr self) { return self.getChipDescs().vec(); })
      .def_prop_ro("chip_desc_indices",
                   [](ttcore::SystemDescAttr self) {
                     return self.getChipDescIndices().vec();
                   })
      .def_prop_ro("chip_capabilities",
                   [](ttcore::SystemDescAttr self) {
                     return self.getChipCapabilities().vec();
                   })
      .def_prop_ro("chip_coords",
                   [](ttcore::SystemDescAttr self) {
                     return self.getChipCoords().vec();
                   })
      .def_prop_ro("chip_channels", [](ttcore::SystemDescAttr self) {
        return self.getChipChannels().vec();
      });

  tt_attribute_class<ttcore::MemorySpaceAttr>(m, "MemorySpaceAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t memorySpace) {
            return wrap(ttcore::MemorySpaceAttr::get(
                unwrap(ctx), static_cast<ttcore::MemorySpace>(memorySpace)));
          })
      .def_prop_ro("memory_space_as_int", [](ttcore::MemorySpaceAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<ttcore::OOBValAttr>(m, "OOBValAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t oobVal) {
                    return wrap(ttcore::OOBValAttr::get(
                        unwrap(ctx), static_cast<ttcore::OOBVal>(oobVal)));
                  })
      .def_prop_ro("oob_val_as_int", [](ttcore::OOBValAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<ttcore::IteratorTypeAttr>(m, "IteratorTypeAttr")
      .def_static(
          "get",
          [](MlirContext ctx, uint32_t iteratorType) {
            return wrap(ttcore::IteratorTypeAttr::get(
                unwrap(ctx), static_cast<ttcore::IteratorType>(iteratorType)));
          })
      .def_prop_ro("iterator_type_as_int", [](ttcore::IteratorTypeAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<ttcore::DeviceAttr>(m, "DeviceAttr")
      .def_static("from_system_desc",
                  [](MlirContext ctx, MlirAttribute systemDesc,
                     std::vector<int64_t> meshShape) {
                    return wrap(ttcore::DeviceAttr::get(
                        unwrap(ctx),
                        mlir::cast<ttcore::SystemDescAttr>(unwrap(systemDesc)),
                        meshShape));
                  })
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> gridShape,
                     MlirAffineMap workerGridMapping, MlirAffineMap l1Map,
                     MlirAffineMap dramMap, std::vector<int64_t> meshShape,
                     std::vector<unsigned> chipIds) {
                    return wrap(ttcore::DeviceAttr::get(
                        unwrap(ctx),
                        ttcore::GridAttr::get(unwrap(ctx), gridShape,
                                              unwrap(workerGridMapping)),
                        unwrap(l1Map), unwrap(dramMap), meshShape, chipIds));
                  })
      .def("unwrap",
           [](const MlirAttribute &self) {
             return mlir::cast<ttcore::DeviceAttr>(unwrap(self));
           })
      .def_prop_ro("grid_attr", &ttcore::DeviceAttr::getWorkerGrid)
      .def_prop_ro(
          "l1_map",
          [](ttcore::DeviceAttr self) { return wrap(self.getL1Map()); })
      .def_prop_ro(
          "dram_map",
          [](ttcore::DeviceAttr self) { return wrap(self.getDramMap()); })
      .def_prop_ro("mesh_shape",
                   [](const ttcore::DeviceAttr &self) {
                     return self.getMeshShape().vec();
                   })
      .def_prop_ro("chip_ids", [](const ttcore::DeviceAttr &self) {
        return self.getChipIds().vec();
      });

  tt_type_class<ttcore::TileType>(m, "TileType")
      .def_static("get",
                  [](MlirContext ctx, std::int64_t height, std::int64_t width,
                     uint32_t dataType) {
                    return wrap(ttcore::TileType::get(
                        unwrap(ctx), SmallVector<std::int64_t>{height, width},
                        static_cast<ttcore::DataType>(dataType)));
                  })
      .def_prop_ro("data_type_as_int",
                   [](ttcore::TileType self) {
                     return static_cast<uint32_t>(self.getDataType());
                   })
      .def_prop_ro("shape", [](const ttcore::TileType &tile) {
        return std::vector<int64_t>({tile.getHeight(), tile.getWidth()});
      });
}
} // namespace mlir::ttmlir::python
