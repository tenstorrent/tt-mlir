// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/AffineMap.h"
#include "mlir/CAPI/IR.h"

#include "ttmlir-c/TTAttrs.h"
#include "ttmlir-c/TTTypes.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Target/Common/Target.h"
#include "ttmlir/Utils.h"

namespace mlir::ttmlir::python {
void populateTTModule(nb::module_ &m) {
  tt_attribute_class<tt::MetalLayoutAttr>(m, "MetalLayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, MlirType rankedTensorType,
                     uint32_t memorySpaceValue, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals,
                     uint32_t oobValValue) {
                    // Create AffineMap for linear mapping
                    auto affineMap = mlir::AffineMap::getMultiDimIdentityMap(
                        mlir::cast<RankedTensorType>(unwrap(rankedTensorType))
                            .getRank(),
                        unwrap(ctx));
                    return ttmlirTTMetalLayoutAttrGet(ctx, wrap(affineMap),
                                                      oobValValue, grid,
                                                      rankedTensorType);
                  })
      .def_static("with_grid",
                  [](MlirContext ctx, MlirAttribute self,
                     std::vector<std::int64_t> tensorShape, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals) {
                    return wrap(
                        mlir::cast<tt::MetalLayoutAttr>(unwrap(self))
                            .withGrid(unwrap(ctx), tensorShape,
                                      mlir::cast<tt::GridAttr>(unwrap(grid)),
                                      collapseIntervals));
                  })
      .def_static("with_grid_",
                  [](MlirContext ctx, MlirAttribute self,
                     std::vector<std::int64_t> tensorShape, MlirAttribute grid,
                     std::vector<std::pair<std::int64_t, std::int64_t>>
                         collapseIntervals) {
                    return mlir::cast<tt::MetalLayoutAttr>(unwrap(self))
                        .withGrid(unwrap(ctx), tensorShape,
                                  mlir::cast<tt::GridAttr>(unwrap(grid)),
                                  collapseIntervals);
                  })
      .def_static(
          "with_element_type",
          [](MlirContext ctx, MlirAttribute self, MlirType elementType) {
            return wrap(mlir::cast<tt::MetalLayoutAttr>(unwrap(self))
                            .withElementType(unwrap(ctx), unwrap(elementType)));
          })
      .def_static(
          "with_element_type_",
          [](MlirContext ctx, MlirAttribute self, MlirType elementType) {
            return mlir::cast<tt::MetalLayoutAttr>(unwrap(self))
                .withElementType(unwrap(ctx), unwrap(elementType));
          })
      .def("getLayout",
           [](MlirType &type) -> std::variant<tt::MetalLayoutAttr, nb::object> {
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
             tt::MetalLayoutAttr layout =
                 mlir::cast<tt::MetalLayoutAttr>(tensor.getEncoding());
             return layout;
           })
      .def("wrapped",
           [](tt::MetalLayoutAttr const &self) { return wrap(self); })
      .def_prop_ro("stride",
                   [](tt::MetalLayoutAttr const &self,
                      std::vector<int64_t> logicalShape) {
                     auto stride = self.getStride(logicalShape);
                     return std::vector<std::int64_t>(stride.begin(),
                                                      stride.end());
                   })
      .def_prop_ro("oobval", &tt::MetalLayoutAttr::getOobVal)
      .def_prop_ro("oobval_as_int",
                   [](tt::MetalLayoutAttr la) {
                     return static_cast<uint32_t>(la.getOobVal());
                   })
      .def_prop_ro("grid_attr", &tt::MetalLayoutAttr::getGrid)
      .def_prop_ro(
          "memref",
          [](tt::MetalLayoutAttr self) { return wrap(self.getMemref()); })
      .def_prop_ro("memory_space", &tt::MetalLayoutAttr::getMemorySpace)
      .def_prop_ro("memory_space_as_int",
                   [](tt::MetalLayoutAttr la) {
                     return static_cast<uint32_t>(la.getMemorySpace());
                   })
      .def_prop_ro("shard_shape", &tt::MetalLayoutAttr::getShardShape)
      .def_prop_ro("linear", [](tt::MetalLayoutAttr self) {
        return wrap(self.getLinear());
      });

  tt_attribute_class<tt::GridAttr>(m, "GridAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<int64_t> shape) {
                    return ttmlirTTGridAttrGet(ctx, shape.data(), shape.size());
                  })
      .def_prop_ro("shape",
                   [](tt::GridAttr const &ga) { return ga.getShape().vec(); });

  tt_attribute_class<tt::ChipCapabilityAttr>(m, "ChipCapabilityAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t chipCapability) {
                    return ttmlirTTChipCapabilityAttrGet(ctx, chipCapability);
                  })
      .def_prop_ro("capability_as_int", [](tt::ChipCapabilityAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::ArchAttr>(m, "ArchAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t arch) {
                    return ttmlirTTArchAttrGet(ctx, arch);
                  })
      .def_prop_ro("arch_as_int", [](tt::ArchAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::DataTypeAttr>(m, "DataTypeAttr")
      .def_static("get",
                  [](MlirContext ctx, uint16_t *supportedDataTypes) {
                    return ttmlirTTDataTypeAttrGet(ctx, supportedDataTypes);
                  })
      .def_prop_ro("data_type_as_int", [](tt::DataTypeAttr self) {
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
             MlirAttribute supportedTileSizes, unsigned numCBs,
             unsigned numComputeThreads, unsigned numDatamovementThreads) {
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
                numCBs, numComputeThreads, numDatamovementThreads));
          })
      .def_prop_ro("usable_l1_size", &tt::ChipDescAttr::getUsableL1Size)
      .def_prop_ro("usable_dram_channel_size",
                   &tt::ChipDescAttr::getUsableDramChannelSize)
      .def_prop_ro("arch", &tt::ChipDescAttr::getArch)
      .def_prop_ro("grid",
                   [](tt::ChipDescAttr self) { return self.getGrid().vec(); })
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
      .def_prop_ro("chip_physical_cores",
                   &tt::ChipDescAttr::getChipPhysicalCores)
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
                    return ttmlirTTTileSizeAttrGet(ctx, y, x);
                  })
      .def_prop_ro("y", &tt::TileSizeAttr::getY)
      .def_prop_ro("x", &tt::TileSizeAttr::getX);

  tt_attribute_class<tt::ChipPhysicalCoresAttr>(m, "ChipPhysicalCoresAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<tt::CoreCoordAttr> worker,
                     std::vector<tt::CoreCoordAttr> dram,
                     std::vector<tt::CoreCoordAttr> eth,
                     std::vector<tt::CoreCoordAttr> eth_inactive) {
                    return wrap(tt::ChipPhysicalCoresAttr::get(
                        unwrap(ctx), worker, dram, eth, eth_inactive));
                  })
      .def_prop_ro(
          "worker",
          [](tt::ChipPhysicalCoresAttr self) { return self.getWorker().vec(); })
      .def_prop_ro(
          "dram",
          [](tt::ChipPhysicalCoresAttr self) { return self.getDram().vec(); })
      .def_prop_ro(
          "eth",
          [](tt::ChipPhysicalCoresAttr self) { return self.getEth().vec(); })
      .def_prop_ro("eth_inactive", [](tt::ChipPhysicalCoresAttr self) {
        return self.getEthInactive().vec();
      });

  tt_attribute_class<tt::CoreCoordAttr>(m, "CoreCoordAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t y, int64_t x) {
                    return ttmlirTTCoreCoordAttrGet(ctx, y, x);
                  })
      .def_prop_ro("y", &tt::CoreCoordAttr::getY)
      .def_prop_ro("x", &tt::CoreCoordAttr::getX);

  tt_attribute_class<tt::ChipCoordAttr>(m, "ChipCoordAttr")
      .def_static("get",
                  [](MlirContext ctx, unsigned rack, unsigned shelf, unsigned y,
                     unsigned x) {
                    return ttmlirTTChipCoordAttrGet(ctx, rack, shelf, y, x);
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
      .def_static("get",
                  [](MlirContext ctx, uint32_t memorySpace) {
                    return ttmlirTTMemorySpaceAttrGet(ctx, memorySpace);
                  })
      .def_prop_ro("memory_space_as_int", [](tt::MemorySpaceAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::OOBValAttr>(m, "OOBValAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t oobVal) {
                    return ttmlirTTOOBValAttrGet(ctx, oobVal);
                  })
      .def_prop_ro("oob_val_as_int", [](tt::OOBValAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::IteratorTypeAttr>(m, "IteratorTypeAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t iteratorType) {
                    return ttmlirTTIteratorTypeAttrGet(ctx, iteratorType);
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
           [](MlirAttribute const &self) {
             return mlir::cast<tt::DeviceAttr>(unwrap(self));
           })
      .def_prop_ro("grid_attr", &tt::DeviceAttr::getWorkerGrid)
      .def_prop_ro("l1_map",
                   [](tt::DeviceAttr self) { return wrap(self.getL1Map()); })
      .def_prop_ro("dram_map",
                   [](tt::DeviceAttr self) { return wrap(self.getDramMap()); })
      .def_prop_ro(
          "mesh_shape",
          [](tt::DeviceAttr const &self) { return self.getMeshShape().vec(); })
      .def_prop_ro("chip_ids", [](tt::DeviceAttr const &self) {
        return self.getChipIds().vec();
      });

  tt_type_class<tt::TileType>(m, "TileType")
      .def_static("get",
                  [](MlirContext ctx, std::int64_t height, std::int64_t width,
                     uint32_t dataType) {
                    return ttmlirTTTileTypeGet(ctx, height, width, dataType);
                  })
      .def_prop_ro("data_type_as_int",
                   [](tt::TileType self) {
                     return static_cast<uint32_t>(self.getDataType());
                   })
      .def_prop_ro("shape", [](tt::TileType const &tile) {
        return std::vector<int64_t>({tile.getHeight(), tile.getWidth()});
      });
  // Add missing attribute classes
  tt_attribute_class<tt::CPURoleAttr>(m, "CPURoleAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t cpuRole) {
                    return ttmlirTTCPURoleAttrGet(ctx, cpuRole);
                  })
      .def_prop_ro("cpu_role_as_int", [](tt::CPURoleAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::CPUDescAttr>(m, "CPUDescAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t cpuRole,
                     const std::string &targetTriple) {
                    return ttmlirTTCPUDescAttrGet(ctx, cpuRole,
                                                  targetTriple.c_str());
                  })
      .def_prop_ro("role", &tt::CPUDescAttr::getRole)
      .def_prop_ro("target_triple", [](tt::CPUDescAttr self) {
        return self.getTargetTriple().str();
      });

  tt_attribute_class<tt::StreamLayoutAttr>(m, "StreamLayoutAttr")
      .def_static("get",
                  [](MlirContext ctx, MlirAffineMap affineMap) {
                    return ttmlirTTStreamLayoutAttrGet(ctx, affineMap);
                  })
      .def_prop_ro("affine_map", [](tt::StreamLayoutAttr self) {
        return wrap(self.getAffineMap());
      });

  tt_attribute_class<tt::ShardLayoutAttr>(m, "ShardLayoutAttr")
      .def_static(
          "get",
          [](MlirContext ctx, std::vector<int64_t> stride, uint32_t buffers) {
            return ttmlirTTShardLayoutAttrGet(ctx, stride.data(), stride.size(),
                                              buffers);
          })
      .def_prop_ro(
          "stride",
          [](tt::ShardLayoutAttr self) { return self.getStride().vec(); })
      .def_prop_ro("buffers", &tt::ShardLayoutAttr::getBuffers)
      .def_prop_ro("affine_map", [](tt::ShardLayoutAttr self) {
        return wrap(self.getAffineMap());
      });

  tt_attribute_class<tt::TensorMeshShardingAxisAttr>(
      m, "TensorMeshShardingAxisAttr")
      .def_static("get",
                  [](MlirContext ctx, int64_t shardShape, int64_t shardDim) {
                    return ttmlirTTTensorMeshShardingAxisAttrGet(
                        ctx, shardShape, shardDim);
                  })
      .def_prop_ro("shard_shape",
                   &tt::TensorMeshShardingAxisAttr::getShardShape);

  tt_attribute_class<tt::TensorMeshShardingAttr>(m, "TensorMeshShardingAttr")
      .def_static("get",
                  [](MlirContext ctx, const std::string &name,
                     std::vector<MlirAttribute> tensorMeshShardingAxis) {
                    return ttmlirTTTensorMeshShardingAttrGet(
                        ctx, name.c_str(), tensorMeshShardingAxis.data(),
                        tensorMeshShardingAxis.size());
                  })
      .def_prop_ro(
          "name",
          [](tt::TensorMeshShardingAttr self) { return self.getName().str(); })
      .def_prop_ro("tensor_mesh_sharding_axis",
                   [](tt::TensorMeshShardingAttr self) {
                     return self.getTensorMeshShardingAxis().vec();
                   });

  tt_attribute_class<tt::MeshAttr>(m, "MeshAttr")
      .def_static("get",
                  [](MlirContext ctx, const std::string &name,
                     std::vector<int64_t> shape) {
                    return ttmlirTTMeshAttrGet(ctx, name.c_str(), shape.data(),
                                               shape.size());
                  })
      .def_prop_ro("name",
                   [](tt::MeshAttr self) { return self.getName().str(); })
      .def_prop_ro("shape",
                   [](tt::MeshAttr self) { return self.getShape().vec(); });

  tt_attribute_class<tt::MeshesAttr>(m, "MeshesAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<MlirAttribute> meshes) {
                    return ttmlirTTMeshesAttrGet(ctx, meshes.data(),
                                                 meshes.size());
                  })
      .def_prop_ro("meshes",
                   [](tt::MeshesAttr self) { return self.getMeshes().vec(); })
      .def("get_mesh", [](tt::MeshesAttr self, const std::string &name) {
        return wrap(self.getMesh(name));
      });

  tt_attribute_class<tt::ArgumentTypeAttr>(m, "ArgumentTypeAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t argumentType) {
                    return ttmlirTTArgumentTypeAttrGet(ctx, argumentType);
                  })
      .def_prop_ro("argument_type_as_int", [](tt::ArgumentTypeAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  tt_attribute_class<tt::ArgumentAllocationAttr>(m, "ArgumentAllocationAttr")
      .def_static("get",
                  [](MlirContext ctx, uint64_t address, uint64_t size,
                     uint32_t memorySpace) {
                    return ttmlirTTArgumentAllocationAttrGet(ctx, address, size,
                                                             memorySpace);
                  })
      .def_prop_ro("address", &tt::ArgumentAllocationAttr::getAddress)
      .def_prop_ro("size", &tt::ArgumentAllocationAttr::getSize)
      .def_prop_ro("memory_space", &tt::ArgumentAllocationAttr::getMemorySpace)
      .def_prop_ro("memory_space_as_int", [](tt::ArgumentAllocationAttr self) {
        return static_cast<uint32_t>(self.getMemorySpace());
      });

  tt_attribute_class<tt::ReduceTypeAttr>(m, "ReduceTypeAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t reduceType) {
                    return ttmlirTTReduceTypeAttrGet(ctx, reduceType);
                  })
      .def_prop_ro("reduce_type_as_int", [](tt::ReduceTypeAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  // For ReduceTypeArrayAttr, we'll use a similar pattern as
  // IteratorTypeArrayAttr
  tt_attribute_class<mlir::ArrayAttr>(m, "ReduceTypeArrayAttr")
      .def_static("get",
                  [](MlirContext ctx, std::vector<uint32_t> reduceTypes) {
                    return ttmlirTTReduceTypeArrayAttrGet(
                        ctx, reduceTypes.data(), reduceTypes.size());
                  });

  tt_attribute_class<tt::MeshShardDirectionAttr>(m, "MeshShardDirectionAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t meshShardDirection) {
                    return ttmlirTTMeshShardDirectionAttrGet(
                        ctx, meshShardDirection);
                  })
      .def_prop_ro("mesh_shard_direction_as_int",
                   [](tt::MeshShardDirectionAttr self) {
                     return static_cast<uint32_t>(self.getValue());
                   });

  tt_attribute_class<tt::MeshShardTypeAttr>(m, "MeshShardTypeAttr")
      .def_static("get",
                  [](MlirContext ctx, uint32_t meshShardType) {
                    return ttmlirTTMeshShardTypeAttrGet(ctx, meshShardType);
                  })
      .def_prop_ro("mesh_shard_type_as_int", [](tt::MeshShardTypeAttr self) {
        return static_cast<uint32_t>(self.getValue());
      });

  // Add TupleType
  tt_type_class<mlir::TupleType>(m, "TupleType")
      .def_static("get",
                  [](MlirContext ctx, std::vector<MlirType> elements) {
                    return ttmlirTTTupleTypeGet(ctx, elements.data(),
                                                elements.size());
                  })
      .def_prop_ro("element_types", [](mlir::TupleType self) {
        std::vector<MlirType> elements;
        for (auto type : self.getTypes()) {
          elements.push_back(wrap(type));
        }
        return elements;
      });
}
} // namespace mlir::ttmlir::python
