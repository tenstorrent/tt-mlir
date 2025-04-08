// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_C_TTATTRS_H
#define TTMLIR_C_TTATTRS_H

#include "mlir-c/AffineMap.h"
#include "ttmlir-c/Dialects.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTGridAttrGet(MlirContext ctx,
                                                     int64_t *shape,
                                                     size_t shapeSize);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTChipCapabilityAttrGet(MlirContext ctx, uint32_t chipCapability);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTArchAttrGet(MlirContext ctx,
                                                     uint32_t arch);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTDataTypeAttrGet(MlirContext ctx, uint16_t *supportedDataTypes);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipDescAttrGet(
    MlirContext ctx, MlirAttribute arch, int64_t *grid, size_t gridSize,
    unsigned l1Size, unsigned numDramChannels, unsigned dramChannelSize,
    unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
    unsigned nocDRAMAddressAlignBytes, unsigned l1UnreservedBase,
    unsigned eriscL1UnreservedBase, unsigned dramUnreservedBase,
    MlirAttribute chipPhysicalCores, MlirAttribute *supportedDataTypes,
    MlirAttribute *supportedTileSizes, unsigned numCBs,
    unsigned numComputeThreads, unsigned numDatamovementThreads);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipCoordAttrGet(
    MlirContext ctx, unsigned rack, unsigned shelf, unsigned y, unsigned x);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipChannelAttrGet(
    MlirContext ctx, unsigned deviceId0, int64_t *ethernetCoreCoord0,
    size_t ethernetCoreCoord0Size, unsigned deviceId1,
    int64_t *ethernetCoreCoord1, size_t ethernetCoreCoord1Size);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTSystemDescAttrGet(
    MlirContext ctx, MlirAttribute *cpuDescs, size_t cpuDescsSize,
    MlirAttribute *chipDescs, size_t chipDescsSize, unsigned *chipDescIndices,
    size_t chipDescIndicesSize, MlirAttribute *chipCapabilities,
    size_t chipCapabilitiesSize, MlirAttribute *chipCoords,
    size_t chipCoordsSize, MlirAttribute *chipChannels,
    size_t chipChannelsSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTMetalLayoutAttrGet(
    MlirContext ctx, MlirAffineMap linear, unsigned oobVal, MlirAttribute grid,
    MlirType memref);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTMemorySpaceAttrGet(MlirContext ctx, uint32_t memorySpace);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTOOBValAttrGet(MlirContext ctx,
                                                       uint32_t oobVal);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTIteratorTypeAttrGet(MlirContext ctx, uint32_t iteratorType);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTIteratorTypeArrayAttrGet(
    MlirContext ctx, uint32_t *iteratorTypes, size_t iteratorTypesSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTTileSizeAttrGet(MlirContext ctx,
                                                         int64_t y, int64_t x);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTChipPhysicalCoresAttrGet(
    MlirContext ctx, MlirAttribute *worker, size_t workerSize,
    MlirAttribute *dram, size_t dramSize, MlirAttribute *eth, size_t ethSize,
    MlirAttribute *eth_inactive, size_t eth_inactiveSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTCoreCoordAttrGet(MlirContext ctx,
                                                          int64_t y, int64_t x);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTCPURoleAttrGet(MlirContext ctx,
                                                        uint32_t cpuRole);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTCPUDescAttrGet(
    MlirContext ctx, uint32_t cpuRole, const char *target_triple);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTStreamLayoutAttrGet(MlirContext ctx, MlirAffineMap affineMap);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTShardLayoutAttrGet(MlirContext ctx,
                                                            int64_t *stride,
                                                            size_t strideSize,
                                                            uint32_t buffers);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTTensorMeshShardingAxisAttrGet(
    MlirContext ctx, int64_t shardShape, int64_t shardDim);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTTensorMeshShardingAttrGet(
    MlirContext ctx, const char *name, MlirAttribute *tensorMeshShardingAxis,
    size_t tensorMeshShardingAxisSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTMeshAttrGet(MlirContext ctx,
                                                     const char *name,
                                                     int64_t *shape,
                                                     size_t shapeSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTMeshesAttrGet(MlirContext ctx,
                                                       MlirAttribute *meshes,
                                                       size_t meshesSize);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTArgumentTypeAttrGet(MlirContext ctx, uint32_t argumentType);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTDeviceAttrGet(
    MlirContext ctx, MlirAttribute workerGrid, MlirAffineMap l1Map,
    MlirAffineMap dramMap, int64_t *meshShape, size_t meshShapeSize,
    unsigned *chipIds, size_t chipIdsSize);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTArgumentAllocationAttrGet(
    MlirContext ctx, uint64_t address, uint64_t size, uint32_t memorySpace);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTReduceTypeAttrGet(MlirContext ctx,
                                                           uint32_t reduceType);

MLIR_CAPI_EXPORTED MlirAttribute ttmlirTTReduceTypeArrayAttrGet(
    MlirContext ctx, uint32_t *reduceTypes, size_t reduceTypesSize);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTMeshShardDirectionAttrGet(MlirContext ctx, uint32_t meshShardDirection);

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTMeshShardTypeAttrGet(MlirContext ctx, uint32_t meshShardType);

#ifdef __cplusplus
}
#endif

#endif // TTMLIR_C_TTATTRS_H
