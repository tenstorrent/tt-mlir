// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTAttrs.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt {

MlirAttribute ttmlirTTGridAttrGet(MlirContext ctx, int64_t *shape,
                                  int shapeSize) {
  return wrap(GridAttr::get(unwrap(ctx), {shape, shape + shapeSize},
                            AffineMap::get(unwrap(ctx))));
}

MlirAttribute ttmlirTTChipCapabilityAttrGet(MlirContext ctx,
                                            uint32_t chipCapability) {
  return wrap(ChipCapabilityAttr::get(
      unwrap(ctx), static_cast<ChipCapability>(chipCapability)));
}

MlirAttribute ttmlirTTArchAttrGet(MlirContext ctx, uint32_t arch) {
  return wrap(ArchAttr::get(unwrap(ctx), static_cast<Arch>(arch)));
}

MlirAttribute ttmlirTTChipDescAttrGet(
    MlirContext ctx, MlirAttribute arch, int64_t *grid, size_t gridSize,
    unsigned l1Size, unsigned numDramChannels, unsigned dramChannelSize,
    unsigned nocL1AddressAlignBytes, unsigned pcieAddressAlignBytes,
    unsigned nocDRAMAddressAlignBytes, unsigned l1UnreservedBase,
    unsigned eriscL1UnreservedBase, unsigned dramUnreservedBase,
    MlirAttribute chipPhysicalCores) {
  std::vector<int64_t> gridVec(grid, grid + gridSize);
  return wrap(ChipDescAttr::get(
      unwrap(ctx), mlir::dyn_cast<ArchAttr>(unwrap(arch)), gridVec, l1Size,
      numDramChannels, dramChannelSize, nocL1AddressAlignBytes,
      pcieAddressAlignBytes, nocDRAMAddressAlignBytes, l1UnreservedBase,
      eriscL1UnreservedBase, dramUnreservedBase,
      mlir::dyn_cast<ChipPhysicalCoresAttr>(unwrap(chipPhysicalCores))));
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
    MlirContext ctx, MlirAttribute *chipDescs, size_t chipDescsSize,
    unsigned *chipDescIndices, size_t chipDescIndicesSize,
    MlirAttribute *chipCapabilities, size_t chipCapabilitiesSize,
    MlirAttribute *chipCoords, size_t chipCoordsSize,
    MlirAttribute *chipChannels, size_t chipChannelsSize) {
  llvm::ArrayRef<MlirAttribute> chipDescsRef(chipDescs, chipDescsSize),
      chipCapabilitiesRef(chipCapabilities, chipCapabilitiesSize),
      chipCoordsRef(chipCoords, chipCoordsSize),
      chipChannelsRef(chipChannels, chipChannelsSize);
  llvm::ArrayRef<unsigned> chipDescIndicesRef(chipDescIndices,
                                              chipDescIndicesSize);

  // Unwrap all of the MlirAttributes
  std::vector<tt::ChipDescAttr> chipDescsUnwrapped;
  for (auto chipDesc : chipDescsRef) {
    chipDescsUnwrapped.push_back(mlir::cast<ChipDescAttr>(unwrap(chipDesc)));
  }

  std::vector<tt::ChipCapabilityAttr> chipCapabilitiesUnwrapped;
  for (auto chipCapability : chipCapabilitiesRef) {
    chipCapabilitiesUnwrapped.push_back(
        mlir::cast<ChipCapabilityAttr>(unwrap(chipCapability)));
  }

  std::vector<tt::ChipCoordAttr> chipCoordsUnwrapped;
  for (auto chipCoord : chipCoordsRef) {
    chipCoordsUnwrapped.push_back(mlir::cast<ChipCoordAttr>(unwrap(chipCoord)));
  }

  std::vector<tt::ChipChannelAttr> chipChannelsUnwrapped;
  for (auto chipChannel : chipChannelsRef) {
    chipChannelsUnwrapped.push_back(
        mlir::cast<ChipChannelAttr>(unwrap(chipChannel)));
  }

  return wrap(SystemDescAttr::get(unwrap(ctx), chipDescsUnwrapped,
                                  chipDescIndicesRef, chipCapabilitiesUnwrapped,
                                  chipCoordsUnwrapped, chipChannelsUnwrapped));
}

MlirAttribute ttmlirTTLayoutAttrGet(MlirContext ctx, MlirAffineMap linear,
                                    unsigned oobVal, MlirAttribute grid,
                                    MlirType memref) {
  mlir::AffineMap affineMap = mlir::AffineMap::getFromOpaquePointer(linear.ptr);
  return wrap(LayoutAttr::get(unwrap(ctx), affineMap,
                              static_cast<OOBVal>(oobVal),
                              mlir::cast<GridAttr>(unwrap(grid)),
                              mlir::cast<MemRefType>(unwrap(memref))));
}

MlirAttribute ttmlirTTMemorySpaceAttrGet(MlirContext ctx,
                                         uint32_t memorySpace) {
  return wrap(MemorySpaceAttr::get(unwrap(ctx),
                                   static_cast<tt::MemorySpace>(memorySpace)));
}

MlirAttribute ttmlirTTOOBValAttrGet(MlirContext ctx, uint32_t oobVal) {
  return wrap(OOBValAttr::get(unwrap(ctx), static_cast<tt::OOBVal>(oobVal)));
}

MlirAttribute ttmlirTTIteratorTypeAttrGet(MlirContext ctx,
                                          uint32_t iteratorType) {
  return wrap(IteratorTypeAttr::get(
      unwrap(ctx), static_cast<tt::IteratorType>(iteratorType)));
}

MlirAttribute ttmlirTTIteratorTypeArrayAttrGet(MlirContext ctx,
                                               uint32_t *iteratorTypes,
                                               size_t iteratorTypesSize) {
  std::vector<uint32_t> iteratorTypesEnumArray(
      iteratorTypes, iteratorTypes + iteratorTypesSize);
  std::vector<mlir::Attribute> iteratorTypesArray;

  for (auto iteratorEnum : iteratorTypesEnumArray) {
    iteratorTypesArray.push_back(IteratorTypeAttr::get(
        unwrap(ctx), static_cast<tt::IteratorType>(iteratorEnum)));
  }

  return wrap(ArrayAttr::get(unwrap(ctx), iteratorTypesArray));
}

MlirAttribute ttmlirTTOperandConstraintAttrGet(MlirContext ctx,
                                               uint32_t operandConstraint) {
  return wrap(OperandConstraintAttr::get(
      unwrap(ctx), static_cast<tt::OperandConstraint>(operandConstraint)));
}

MlirAttribute
ttmlirTTOperandConstraintArrayAttrGet(MlirContext ctx,
                                      uint32_t *operandConstraints,
                                      size_t operandConstraintsSize) {
  std::vector<uint32_t> operandConstraintsEnumArray(
      operandConstraints, operandConstraints + operandConstraintsSize);
  std::vector<mlir::Attribute> operandConstraintsArray;

  for (auto operandEnum : operandConstraintsEnumArray) {
    operandConstraintsArray.push_back(OperandConstraintAttr::get(
        unwrap(ctx), static_cast<tt::OperandConstraint>(operandEnum)));
  }

  return wrap(ArrayAttr::get(unwrap(ctx), operandConstraintsArray));
}

} // namespace mlir::tt
