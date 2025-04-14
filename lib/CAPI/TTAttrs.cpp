// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTAttrs.h"
#include "mlir/CAPI/IR.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

using namespace mlir::tt;

MlirAttribute ttmlirTTGridAttrGet(MlirContext ctx, int64_t *shape,
                                  size_t shapeSize) {
  return wrap(GridAttr::get(unwrap(ctx), {shape, shape + shapeSize},
                            mlir::AffineMap::get(unwrap(ctx))));
}

MlirAttribute ttmlirTTChipCapabilityAttrGet(MlirContext ctx,
                                            uint32_t chipCapability) {
  return wrap(ChipCapabilityAttr::get(
      unwrap(ctx), static_cast<ChipCapability>(chipCapability)));
}

MlirAttribute ttmlirTTArchAttrGet(MlirContext ctx, uint32_t arch) {
  return wrap(ArchAttr::get(unwrap(ctx), static_cast<Arch>(arch)));
}

MlirAttribute ttmlirTTDataTypeAttrGet(MlirContext ctx,
                                      uint16_t *supportedDataTypes) {
  return wrap(DataTypeAttr::get(unwrap(ctx),
                                static_cast<DataType>(*supportedDataTypes)));
}

MlirAttribute ttmlirTTChipDescAttrGet(
    MlirContext ctx, MlirAttribute arch, int64_t *grid, size_t gridSize,
    unsigned l1Size, unsigned numDramChannels, unsigned dramChannelSize,
    unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
    unsigned nocDRAMAddressAlignBytes, unsigned l1UnreservedBase,
    unsigned eriscL1UnreservedBase, unsigned dramUnreservedBase,
    unsigned dramUnreservedEnd, MlirAttribute chipPhysicalCores,
    MlirAttribute *supportedDataTypes, MlirAttribute *supportedTileSizes,
    unsigned numCBs, unsigned numComputeThreads,
    unsigned numDatamovementThreads) {
  std::vector<int64_t> gridVec(grid, grid + gridSize);
  return wrap(ChipDescAttr::get(
      unwrap(ctx), mlir::dyn_cast<ArchAttr>(unwrap(arch)), gridVec, l1Size,
      numDramChannels, dramChannelSize, nocL1AddressAlignBytes,
      pcieAddressAlignBytes, nocDRAMAddressAlignBytes, l1UnreservedBase,
      eriscL1UnreservedBase, dramUnreservedBase, dramUnreservedEnd,
      mlir::dyn_cast<ChipPhysicalCoresAttr>(unwrap(chipPhysicalCores)),
      mlir::dyn_cast<DataTypeAttr>(unwrap(*supportedDataTypes)),
      mlir::dyn_cast<TileSizeAttr>(unwrap(*supportedTileSizes)), numCBs,
      numComputeThreads, numDatamovementThreads));
}

MlirAttribute ttmlirTTChipCoordAttrGet(MlirContext ctx, unsigned rack,
                                       unsigned shelf, unsigned y, unsigned x) {
  return wrap(ChipCoordAttr::get(unwrap(ctx), rack, shelf, y, x));
}

MlirAttribute ttmlirTTChipChannelAttrGet(MlirContext ctx, unsigned deviceId0,
                                         int64_t *ethernetCoreCoord0,
                                         size_t ethernetCoreCoord0Size,
                                         unsigned deviceId1,
                                         int64_t *ethernetCoreCoord1,
                                         size_t ethernetCoreCoord1Size) {
  std::vector<int64_t> ethCoord0Vec(
      ethernetCoreCoord0, ethernetCoreCoord0 + ethernetCoreCoord0Size);
  std::vector<int64_t> ethCoord1Vec(
      ethernetCoreCoord1, ethernetCoreCoord1 + ethernetCoreCoord1Size);
  return wrap(ChipChannelAttr::get(unwrap(ctx), deviceId0, ethCoord0Vec,
                                   deviceId1, ethCoord1Vec));
}

MlirAttribute ttmlirTTSystemDescAttrGet(
    MlirContext ctx, MlirAttribute *cpuDescs, size_t cpuDescsSize,
    MlirAttribute *chipDescs, size_t chipDescsSize, unsigned *chipDescIndices,
    size_t chipDescIndicesSize, MlirAttribute *chipCapabilities,
    size_t chipCapabilitiesSize, MlirAttribute *chipCoords,
    size_t chipCoordsSize, MlirAttribute *chipChannels,
    size_t chipChannelsSize) {
  llvm::ArrayRef<MlirAttribute> cpuDescsRef(cpuDescs, cpuDescsSize),
      chipDescsRef(chipDescs, chipDescsSize),
      chipCapabilitiesRef(chipCapabilities, chipCapabilitiesSize),
      chipCoordsRef(chipCoords, chipCoordsSize),
      chipChannelsRef(chipChannels, chipChannelsSize);
  llvm::ArrayRef<unsigned> chipDescIndicesRef(chipDescIndices,
                                              chipDescIndicesSize);

  // Unwrap all of the MlirAttributes
  std::vector<ChipDescAttr> chipDescsUnwrapped;
  for (auto chipDesc : chipDescsRef) {
    chipDescsUnwrapped.push_back(mlir::cast<ChipDescAttr>(unwrap(chipDesc)));
  }

  std::vector<ChipCapabilityAttr> chipCapabilitiesUnwrapped;
  for (auto chipCapability : chipCapabilitiesRef) {
    chipCapabilitiesUnwrapped.push_back(
        mlir::cast<ChipCapabilityAttr>(unwrap(chipCapability)));
  }

  std::vector<ChipCoordAttr> chipCoordsUnwrapped;
  for (auto chipCoord : chipCoordsRef) {
    chipCoordsUnwrapped.push_back(mlir::cast<ChipCoordAttr>(unwrap(chipCoord)));
  }

  std::vector<ChipChannelAttr> chipChannelsUnwrapped;
  for (auto chipChannel : chipChannelsRef) {
    chipChannelsUnwrapped.push_back(
        mlir::cast<ChipChannelAttr>(unwrap(chipChannel)));
  }

  std::vector<CPUDescAttr> cpuDescsUnwrapped;
  for (auto cpuDesc : cpuDescsRef) {
    cpuDescsUnwrapped.push_back(mlir::cast<CPUDescAttr>(unwrap(cpuDesc)));
  }

  return wrap(SystemDescAttr::get(
      unwrap(ctx), cpuDescsUnwrapped, chipDescsUnwrapped, chipDescIndicesRef,
      chipCapabilitiesUnwrapped, chipCoordsUnwrapped, chipChannelsUnwrapped));
}

MlirAttribute ttmlirTTMetalLayoutAttrGet(MlirContext ctx, MlirAffineMap linear,
                                         unsigned oobVal, MlirAttribute grid,
                                         MlirType memref) {
  mlir::AffineMap affineMap = mlir::AffineMap::getFromOpaquePointer(linear.ptr);
  return wrap(
      MetalLayoutAttr::get(unwrap(ctx), affineMap, static_cast<OOBVal>(oobVal),
                           mlir::cast<GridAttr>(unwrap(grid)),
                           mlir::cast<mlir::MemRefType>(unwrap(memref))));
}

MlirAttribute ttmlirTTMemorySpaceAttrGet(MlirContext ctx,
                                         uint32_t memorySpace) {
  return wrap(
      MemorySpaceAttr::get(unwrap(ctx), static_cast<MemorySpace>(memorySpace)));
}

MlirAttribute ttmlirTTOOBValAttrGet(MlirContext ctx, uint32_t oobVal) {
  return wrap(OOBValAttr::get(unwrap(ctx), static_cast<OOBVal>(oobVal)));
}

MlirAttribute ttmlirTTIteratorTypeAttrGet(MlirContext ctx,
                                          uint32_t iteratorType) {
  return wrap(IteratorTypeAttr::get(unwrap(ctx),
                                    static_cast<IteratorType>(iteratorType)));
}

MlirAttribute ttmlirTTIteratorTypeArrayAttrGet(MlirContext ctx,
                                               uint32_t *iteratorTypes,
                                               size_t iteratorTypesSize) {
  std::vector<uint32_t> iteratorTypesEnumArray(
      iteratorTypes, iteratorTypes + iteratorTypesSize);
  std::vector<mlir::Attribute> iteratorTypesArray;

  for (auto iteratorEnum : iteratorTypesEnumArray) {
    iteratorTypesArray.push_back(IteratorTypeAttr::get(
        unwrap(ctx), static_cast<IteratorType>(iteratorEnum)));
  }

  return wrap(mlir::ArrayAttr::get(unwrap(ctx), iteratorTypesArray));
}

MlirAttribute ttmlirTTTileSizeAttrGet(MlirContext ctx, int64_t y, int64_t x) {
  return wrap(TileSizeAttr::get(unwrap(ctx), y, x));
}

MlirAttribute ttmlirTTChipPhysicalCoresAttrGet(
    MlirContext ctx, MlirAttribute *worker, size_t workerSize,
    MlirAttribute *dram, size_t dramSize, MlirAttribute *eth, size_t ethSize,
    MlirAttribute *eth_inactive, size_t eth_inactiveSize) {
  std::vector<CoreCoordAttr> workerVec, dramVec, ethVec, ethInactiveVec;
  for (size_t i = 0; i < workerSize; i++) {
    workerVec.push_back(mlir::cast<CoreCoordAttr>(unwrap(worker[i])));
  }

  for (size_t i = 0; i < dramSize; i++) {
    dramVec.push_back(mlir::cast<CoreCoordAttr>(unwrap(dram[i])));
  }

  for (size_t i = 0; i < ethSize; i++) {
    ethVec.push_back(mlir::cast<CoreCoordAttr>(unwrap(eth[i])));
  }

  for (size_t i = 0; i < eth_inactiveSize; i++) {
    ethInactiveVec.push_back(
        mlir::cast<CoreCoordAttr>(unwrap(eth_inactive[i])));
  }

  return wrap(ChipPhysicalCoresAttr::get(unwrap(ctx), workerVec, dramVec,
                                         ethVec, ethInactiveVec));
}

MlirAttribute ttmlirTTCoreCoordAttrGet(MlirContext ctx, int64_t y, int64_t x) {
  return wrap(CoreCoordAttr::get(unwrap(ctx), y, x));
}

MlirAttribute ttmlirTTCPURoleAttrGet(MlirContext ctx, uint32_t cpuRole) {
  return wrap(CPURoleAttr::get(unwrap(ctx), static_cast<CPURole>(cpuRole)));
}

MlirAttribute ttmlirTTCPUDescAttrGet(MlirContext ctx, uint32_t cpuRole,
                                     const char *target_triple) {
  auto targetTripleAttr = mlir::StringAttr::get(unwrap(ctx), target_triple);
  return wrap(CPUDescAttr::get(unwrap(ctx), static_cast<CPURole>(cpuRole),
                               targetTripleAttr));
}

MlirAttribute ttmlirTTStreamLayoutAttrGet(MlirContext ctx,
                                          MlirAffineMap affineMap) {
  return wrap(StreamLayoutAttr::get(
      unwrap(ctx), mlir::AffineMap::getFromOpaquePointer(affineMap.ptr)));
}

MlirAttribute ttmlirTTShardLayoutAttrGet(MlirContext ctx, int64_t *stride,
                                         size_t strideSize, uint32_t buffers) {
  std::vector<int64_t> strideVec(stride, stride + strideSize);
  return wrap(ShardLayoutAttr::get(unwrap(ctx), strideVec, buffers));
}

MlirAttribute ttmlirTTTensorMeshShardingAxisAttrGet(MlirContext ctx,
                                                    int64_t shardShape,
                                                    int64_t shardDim) {
  return wrap(
      TensorMeshShardingAxisAttr::get(unwrap(ctx), shardShape, shardDim));
}

MlirAttribute
ttmlirTTTensorMeshShardingAttrGet(MlirContext ctx, const char *name,
                                  MlirAttribute *tensorMeshShardingAxis,
                                  size_t tensorMeshShardingAxisSize) {
  auto nameAttr = mlir::StringAttr::get(unwrap(ctx), name);
  std::vector<TensorMeshShardingAxisAttr> axisVec;

  for (size_t i = 0; i < tensorMeshShardingAxisSize; i++) {
    axisVec.push_back(mlir::cast<TensorMeshShardingAxisAttr>(
        unwrap(tensorMeshShardingAxis[i])));
  }

  return wrap(TensorMeshShardingAttr::get(unwrap(ctx), nameAttr, axisVec));
}

MlirAttribute ttmlirTTMeshAttrGet(MlirContext ctx, const char *name,
                                  int64_t *shape, size_t shapeSize) {
  auto nameAttr = mlir::StringAttr::get(unwrap(ctx), name);
  std::vector<int64_t> shapeVec(shape, shape + shapeSize);
  return wrap(MeshAttr::get(unwrap(ctx), nameAttr, shapeVec));
}

MlirAttribute ttmlirTTMeshesAttrGet(MlirContext ctx, MlirAttribute *meshes,
                                    size_t meshesSize) {
  std::vector<MeshAttr> meshesVec;
  for (size_t i = 0; i < meshesSize; i++) {
    meshesVec.push_back(mlir::cast<MeshAttr>(unwrap(meshes[i])));
  }
  return wrap(MeshesAttr::get(unwrap(ctx), meshesVec));
}

MlirAttribute ttmlirTTArgumentTypeAttrGet(MlirContext ctx,
                                          uint32_t argumentType) {
  return wrap(ArgumentTypeAttr::get(unwrap(ctx),
                                    static_cast<ArgumentType>(argumentType)));
}

MlirAttribute ttmlirTTDeviceAttrGet(MlirContext ctx, MlirAttribute workerGrid,
                                    MlirAffineMap l1Map, MlirAffineMap dramMap,
                                    int64_t *meshShape, size_t meshShapeSize,
                                    unsigned *chipIds, size_t chipIdsSize) {
  mlir::AffineMap l1AffineMap =
      mlir::AffineMap::getFromOpaquePointer(l1Map.ptr);
  mlir::AffineMap dramAffineMap =
      mlir::AffineMap::getFromOpaquePointer(dramMap.ptr);
  std::vector<int64_t> meshShapeVec(meshShape, meshShape + meshShapeSize);
  std::vector<unsigned> chipIdsVec(chipIds, chipIds + chipIdsSize);

  return wrap(
      DeviceAttr::get(unwrap(ctx), mlir::cast<GridAttr>(unwrap(workerGrid)),
                      l1AffineMap, dramAffineMap, meshShapeVec, chipIdsVec));
}

MlirAttribute ttmlirTTArgumentAllocationAttrGet(MlirContext ctx,
                                                uint64_t address, uint64_t size,
                                                uint32_t memorySpace) {
  return wrap(ArgumentAllocationAttr::get(
      unwrap(ctx), address, size, static_cast<MemorySpace>(memorySpace)));
}

MlirAttribute ttmlirTTReduceTypeAttrGet(MlirContext ctx, uint32_t reduceType) {
  return wrap(
      ReduceTypeAttr::get(unwrap(ctx), static_cast<ReduceType>(reduceType)));
}

MlirAttribute ttmlirTTReduceTypeArrayAttrGet(MlirContext ctx,
                                             uint32_t *reduceTypes,
                                             size_t reduceTypesSize) {
  std::vector<uint32_t> reduceTypesEnumArray(reduceTypes,
                                             reduceTypes + reduceTypesSize);
  std::vector<mlir::Attribute> reduceTypesArray;

  for (auto reduceEnum : reduceTypesEnumArray) {
    reduceTypesArray.push_back(
        ReduceTypeAttr::get(unwrap(ctx), static_cast<ReduceType>(reduceEnum)));
  }

  return wrap(mlir::ArrayAttr::get(unwrap(ctx), reduceTypesArray));
}

MlirAttribute ttmlirTTMeshShardDirectionAttrGet(MlirContext ctx,
                                                uint32_t meshShardDirection) {
  return wrap(MeshShardDirectionAttr::get(
      unwrap(ctx), static_cast<MeshShardDirection>(meshShardDirection)));
}

MlirAttribute ttmlirTTMeshShardTypeAttrGet(MlirContext ctx,
                                           uint32_t meshShardType) {
  return wrap(MeshShardTypeAttr::get(
      unwrap(ctx), static_cast<MeshShardType>(meshShardType)));
}
