// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "ttmlir-c/TTAttrs.h"

#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/AffineMap.h"

#include <cstdint>
#include <vector>

namespace mlir::ttmlir::python {

// Returns a vector containing elements with type T extracted from an attribute
// using the two provided callbacks.
template <typename T>
std::vector<T>
propertyVector(MlirAttribute attr,
               llvm::function_ref<intptr_t(MlirAttribute)> sizeFn,
               llvm::function_ref<T(MlirAttribute, intptr_t)> getFn) {
  std::vector<T> result;
  intptr_t size = sizeFn(attr);
  result.reserve(size);
  for (intptr_t i = 0; i < size; ++i) {
    result.push_back(getFn(attr, i));
  }
  return result;
}

void populateTTModule(nb::module_ &m) {
  mlir::python::nanobind_adaptors::mlir_attribute_subclass(m, "GridAttr",
                                                           ttmlirIsGridAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctx, std::vector<int64_t> shape) {
            return cls(ttmlirGridAttrGet(ctx, shape.size(), shape.data()));
          },
          nb::arg("cls"), nb::arg("ctx"), nb::arg("shape"))
      .def_classmethod(
          "from_attribute",
          [](nb::object cls, MlirAttribute attr) {
            if (!ttmlirIsGridAttr(attr)) {
              throw std::runtime_error("Attribute is not a GridAttr");
            }
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("attr"))
      .def_property_readonly("shape", [](MlirAttribute self) {
        return propertyVector<int64_t>(self, ttmlirGridAttrGetShapeSize,
                                       ttmlirGridAttrGetShapeElem);
      });

  nb::enum_<MlirReduceTypeEnum>(m, "ReduceType")
      .value("Sum", MlirReduceTypeEnum::MlirReduceTypeSum)
      .value("Mean", MlirReduceTypeEnum::MlirReduceTypeMean)
      .value("Max", MlirReduceTypeEnum::MlirReduceTypeMax)
      .value("Min", MlirReduceTypeEnum::MlirReduceTypeMin)
      .value("Std", MlirReduceTypeEnum::MlirReduceTypeStd)
      .value("Var", MlirReduceTypeEnum::MlirReduceTypeVar)
      .value("Prod", MlirReduceTypeEnum::MlirReduceTypeProd)
      .value("Invalid", MlirReduceTypeEnum::MlirReduceTypeInvalid);

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(
      m, "ReduceTypeAttr", ttmlirIsReduceTypeAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctx, MlirReduceTypeEnum value) {
            return cls(ttmlirReduceTypeAttrGet(ctx, value));
          },
          nb::arg("cls"), nb::arg("ctx"), nb::arg("value"))
      .def_classmethod(
          "from_attribute",
          [](nb::object cls, MlirAttribute attr) {
            if (!ttmlirIsReduceTypeAttr(attr)) {
              throw std::runtime_error("Attribute is not a ReduceTypeAttr");
            }
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("attr"))
      .def_property_readonly("value", [](MlirAttribute self) {
        return ttmlirReduceTypeAttrGetValue(self);
      });

  nb::enum_<MlirDataTypeEnum>(m, "DataType")
      .value("Float32", MlirDataTypeEnum::MlirDataTypeFloat32)
      .value("Float16", MlirDataTypeEnum::MlirDataTypeFloat16)
      .value("BFloat16", MlirDataTypeEnum::MlirDataTypeBFloat16)
      .value("BFP_Float8", MlirDataTypeEnum::MlirDataTypeBFP_Float8)
      .value("BFP_BFloat8", MlirDataTypeEnum::MlirDataTypeBFP_BFloat8)
      .value("BFP_Float4", MlirDataTypeEnum::MlirDataTypeBFP_Float4)
      .value("BFP_BFloat4", MlirDataTypeEnum::MlirDataTypeBFP_BFloat4)
      .value("BFP_Float2", MlirDataTypeEnum::MlirDataTypeBFP_Float2)
      .value("BFP_BFloat2", MlirDataTypeEnum::MlirDataTypeBFP_BFloat2)
      .value("UInt32", MlirDataTypeEnum::MlirDataTypeUInt32)
      .value("UInt16", MlirDataTypeEnum::MlirDataTypeUInt16)
      .value("UInt8", MlirDataTypeEnum::MlirDataTypeUInt8)
      .value("Int32", MlirDataTypeEnum::MlirDataTypeInt32)
      .value("Bool", MlirDataTypeEnum::MlirDataTypeBool);

  mlir::python::nanobind_adaptors::mlir_attribute_subclass(m, "DataTypeAttr",
                                                           ttmlirIsDataTypeAttr)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctx, MlirDataTypeEnum value) {
            return cls(ttmlirDataTypeAttrGet(ctx, value));
          },
          nb::arg("cls"), nb::arg("ctx"), nb::arg("value"))
      .def_classmethod(
          "from_attribute",
          [](nb::object cls, MlirAttribute attr) {
            if (!ttmlirIsDataTypeAttr(attr)) {
              throw std::runtime_error("Attribute is not a DataTypeAttr");
            }
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("attr"))
      .def_property_readonly("value", [](MlirAttribute self) {
        return ttmlirDataTypeAttrGetValue(self);
      });

  mlir::python::nanobind_adaptors::mlir_type_subclass(m, "TileType",
                                                      ttmlirIsTileType)
      .def_classmethod(
          "get",
          [](nb::object cls, MlirContext ctx, int64_t height, int64_t width,
             MlirAttribute dataType) {
            return cls(ttmlirTileTypeGet(ctx, height, width, dataType));
          },
          nb::arg("cls"), nb::arg("ctx"), nb::arg("height"), nb::arg("width"),
          nb::arg("dataType"))
      .def_classmethod(
          "from_attribute",
          [](nb::object cls, MlirType attr) {
            if (!ttmlirIsTileType(attr)) {
              throw std::runtime_error("Type is not a TileType");
            }
            return cls(attr);
          },
          nb::arg("cls"), nb::arg("attr"))
      .def_property_readonly(
          "height", [](MlirType self) { return ttmlirTileTypeGetHeight(self); })
      .def_property_readonly(
          "width", [](MlirType self) { return ttmlirTileTypeGetWidth(self); })
      .def_property_readonly("datatype", [](MlirType self) {
        return ttmlirTileTypeGetDataType(self);
      });

  tt_attribute_class<tt::ttcore::MetalLayoutAttr>(m, "MetalLayoutAttr")
      // 4-arg overload (no memory_layout provided, defaults to Sharded)
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> logicalShape,
                     uint32_t oobValValue, uint32_t memorySpaceValue) {
                    return wrap(tt::ttcore::MetalLayoutAttr::get(
                        unwrap(ctx), ArrayRef<int64_t>(logicalShape),
                        static_cast<tt::ttcore::OOBVal>(oobValValue),
                        static_cast<tt::ttcore::MemorySpace>(memorySpaceValue),
                        tt::ttcore::TensorMemoryLayout::Sharded));
                  })
      // 5-arg overload (override memory layout)
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
      // 6-arg overload (with index_map, computes defaults for collapseIntervals
      // and dimAlignments)
      .def_static(
          "get",
          [](MlirContext ctx, std::vector<int64_t> logicalShape,
             uint32_t oobValValue, uint32_t memorySpaceValue,
             uint32_t memoryLayoutValue, MlirAffineMap indexMap) {
            // Use [0, -1] as default collapsed intervals.
            auto *context = unwrap(ctx);
            auto intervalType =
                RankedTensorType::get({1, 2}, IntegerType::get(context, 64));
            auto collapsedIntervals = DenseIntElementsAttr::get(
                intervalType, llvm::ArrayRef<int64_t>({0, -1}));

            // Normalize intervals and compute alignments.
            auto normalizedIntervals =
                tt::ttcore::MetalLayoutAttr::normalizeAndFlattenIntervals(
                    collapsedIntervals, logicalShape.size());
            auto dimAlignments =
                tt::ttcore::MetalLayoutAttr::computeTileAlignments(
                    logicalShape, normalizedIntervals);

            return wrap(tt::ttcore::MetalLayoutAttr::get(
                context, ArrayRef<int64_t>(logicalShape),
                static_cast<tt::ttcore::OOBVal>(oobValValue),
                static_cast<tt::ttcore::MemorySpace>(memorySpaceValue),
                static_cast<tt::ttcore::TensorMemoryLayout>(memoryLayoutValue),
                collapsedIntervals, dimAlignments, unwrap(indexMap)));
          })
      // 8-arg overload (full specification with index_map)
      .def_static(
          "get",
          [](MlirContext ctx, std::vector<int64_t> logicalShape,
             uint32_t oobValValue, uint32_t memorySpaceValue,
             uint32_t memoryLayoutValue, MlirAttribute collapseIntervals,
             std::vector<int64_t> dimAlignments, MlirAffineMap indexMap) {
            return wrap(tt::ttcore::MetalLayoutAttr::get(
                unwrap(ctx), ArrayRef<int64_t>(logicalShape),
                static_cast<tt::ttcore::OOBVal>(oobValValue),
                static_cast<tt::ttcore::MemorySpace>(memorySpaceValue),
                static_cast<tt::ttcore::TensorMemoryLayout>(memoryLayoutValue),
                mlir::cast<DenseIntElementsAttr>(unwrap(collapseIntervals)),
                ArrayRef<int64_t>(dimAlignments), unwrap(indexMap)));
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

  nb::enum_<tt::ttcore::MeshShardType>(m, "MeshShardType")
      .value("Identity", tt::ttcore::MeshShardType::Identity)
      .value("Replicate", tt::ttcore::MeshShardType::Replicate)
      .value("Maximal", tt::ttcore::MeshShardType::Maximal)
      .value("Devices", tt::ttcore::MeshShardType::Devices);

  tt_attribute_class<tt::ttcore::MeshShardTypeAttr>(m, "MeshShardTypeAttr")
      .def_static("get",
                  [](MlirContext ctx, tt::ttcore::MeshShardType shardType) {
                    return wrap(tt::ttcore::MeshShardTypeAttr::get(unwrap(ctx),
                                                                   shardType));
                  })
      .def_prop_ro("value", [](tt::ttcore::MeshShardTypeAttr self) {
        return self.getValue();
        ;
      });

  nb::enum_<tt::ttcore::MeshShardDirection>(m, "MeshShardDirection")
      .value("FullToShard", tt::ttcore::MeshShardDirection::FullToShard)
      .value("ShardToFull", tt::ttcore::MeshShardDirection::ShardToFull);

  tt_attribute_class<tt::ttcore::MeshShardDirectionAttr>(
      m, "MeshShardDirectionAttr")
      .def_static(
          "get",
          [](MlirContext ctx, tt::ttcore::MeshShardDirection shardDirection) {
            return wrap(tt::ttcore::MeshShardDirectionAttr::get(
                unwrap(ctx), shardDirection));
          })
      .def_prop_ro("value", [](tt::ttcore::MeshShardDirectionAttr self) {
        return self.getValue();
        ;
      });

  tt_attribute_class<tt::ttcore::MeshAttr>(m, "MeshAttr")
      .def_static(
          "get",
          [](MlirContext ctx, std::string name, std::vector<int64_t> shape) {
            return wrap(tt::ttcore::MeshAttr::get(
                unwrap(ctx), mlir::StringAttr::get(unwrap(ctx), name),
                ArrayRef<int64_t>(shape)));
          })
      .def_prop_ro(
          "name",
          [](const tt::ttcore::MeshAttr &mesh) { return mesh.getName().str(); })
      .def_prop_ro("shape", [](const tt::ttcore::MeshAttr &mesh) {
        return std::vector<int64_t>(mesh.getShape().begin(),
                                    mesh.getShape().end());
      });

  tt_attribute_class<tt::ttcore::MeshesAttr>(m, "MeshesAttr")
      .def_static(
          "get",
          [](MlirContext ctx, std::vector<MlirAttribute> meshes) {
            std::vector<tt::ttcore::MeshAttr> meshAttrs;
            for (const auto &mesh : meshes) {
              meshAttrs.push_back(
                  mlir::cast<tt::ttcore::MeshAttr>(unwrap(mesh)));
            }
            return wrap(tt::ttcore::MeshesAttr::get(
                unwrap(ctx), ArrayRef<tt::ttcore::MeshAttr>(meshAttrs)));
          })
      .def_prop_ro("meshes", [](const tt::ttcore::MeshesAttr &meshes) {
        return meshes.getMeshes().vec();
      });
}
} // namespace mlir::ttmlir::python
