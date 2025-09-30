// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "ttmlir-c/TTAttrs.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

using namespace mlir::tt::ttcore;

MlirAttribute ttmlirTTGridAttrGet(MlirContext ctx, int64_t *shape,
                                  int shapeSize) {
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
    int64_t *coordTranslationOffsets, size_t coordTranslationOffsetsSize,
    unsigned l1Size, unsigned numDramChannels, unsigned dramChannelSize,
    unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
    unsigned nocDRAMAddressAlignBytes, unsigned l1UnreservedBase,
    unsigned eriscL1UnreservedBase, unsigned dramUnreservedBase,
    unsigned dramUnreservedEnd, MlirAttribute *supportedDataTypes,
    MlirAttribute *supportedTileSizes, unsigned dstRegisterSizeTiles,
    unsigned numCBs, unsigned numComputeThreads,
    unsigned numDatamovementThreads) {
  std::vector<int64_t> gridVec(grid, grid + gridSize);
  std::vector<int64_t> coordTranslationOffsetsVec(
      coordTranslationOffsets,
      coordTranslationOffsets + coordTranslationOffsetsSize);
  return wrap(ChipDescAttr::get(
      unwrap(ctx), mlir::dyn_cast<ArchAttr>(unwrap(arch)), gridVec,
      coordTranslationOffsetsVec, l1Size, numDramChannels, dramChannelSize,
      nocL1AddressAlignBytes, pcieAddressAlignBytes, nocDRAMAddressAlignBytes,
      l1UnreservedBase, eriscL1UnreservedBase, dramUnreservedBase,
      dramUnreservedEnd,
      mlir::dyn_cast<DataTypeAttr>(unwrap(*supportedDataTypes)),
      mlir::dyn_cast<TileSizeAttr>(unwrap(*supportedTileSizes)),
      dstRegisterSizeTiles, numCBs, numComputeThreads, numDatamovementThreads));
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

MlirAttribute ttmlirTTMetalLayoutAttrGet(MlirContext ctx, intptr_t logicalRank,
                                         const int64_t *logicalShape,
                                         intptr_t gridRank,
                                         const int64_t *gridShape,
                                         unsigned oobVal,
                                         unsigned memorySpace) {

  llvm::ArrayRef<int64_t> logicalShapeRef(logicalShape, logicalRank);
  llvm::ArrayRef<int64_t> gridShapeRef(gridShape, gridRank);

  return wrap(MetalLayoutAttr::get(
      unwrap(ctx), logicalShapeRef, gridShapeRef, static_cast<OOBVal>(oobVal),
      static_cast<MemorySpace>(memorySpace), TensorMemoryLayout::Sharded));
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

MlirAttribute ttmlirTTCoreCoordAttrGet(MlirContext ctx, int64_t y, int64_t x) {
  return wrap(CoreCoordAttr::get(unwrap(ctx), y, x));
}
