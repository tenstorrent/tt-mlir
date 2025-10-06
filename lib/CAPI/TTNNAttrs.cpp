// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTNNAttrs.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include <optional>

using namespace mlir::tt::ttnn;

MlirAttribute ttmlirTTNNCoreCoordAttrGet(MlirContext ctx, uint64_t y,
                                         uint64_t x) {
  return wrap(CoreCoordAttr::get(unwrap(ctx), y, x));
}

MlirAttribute ttmlirTTNNCoreRangeSetAttrGet(MlirContext ctx,
                                            MlirAttribute *coreRangesAttrs,
                                            size_t coreRangesAttrsSize) {
  std::vector<CoreRangeAttr> rangeVec;
  rangeVec.reserve(coreRangesAttrsSize);
  for (size_t i = 0; i < coreRangesAttrsSize; ++i) {
    rangeVec.push_back(mlir::cast<CoreRangeAttr>(unwrap(coreRangesAttrs[i])));
  }
  return wrap(CoreRangeSetAttr::get(unwrap(ctx), rangeVec));
}

MlirAttribute ttmlirTTNNUnaryWithParamAttr(MlirContext ctx, uint32_t opTypeEnum,
                                           MlirAttribute *params,
                                           size_t paramsSize) {
  std::vector<mlir::FloatAttr> paramsVec;
  paramsVec.reserve(paramsSize);
  for (size_t i = 0; i < paramsSize; ++i) {
    paramsVec.push_back(mlir::cast<mlir::FloatAttr>(unwrap(params[i])));
  }
  return wrap(UnaryWithParamAttr::get(
      unwrap(ctx), static_cast<UnaryOpType>(opTypeEnum), paramsVec));
}

MlirAttribute ttmlirTTNNMatmulMultiCoreReuseProgramConfigAttr(
    MlirContext ctx, MlirAttribute computeWithStorageGridSize,
    uint64_t in0BlockW, uint64_t outSubblockH, uint64_t outSubblockW,
    uint64_t perCoreM, uint64_t perCoreN) {
  return wrap(MatmulMultiCoreReuseProgramConfigAttr::get(
      unwrap(ctx),
      mlir::cast<CoreCoordAttr>(unwrap(computeWithStorageGridSize)), in0BlockW,
      outSubblockH, outSubblockW, perCoreM, perCoreN));
}

MlirAttribute ttmlirTTNNMatmulMultiCoreReuseMultiCastProgramConfigAttr(
    MlirContext ctx, MlirAttribute computeWithStorageGridSize,
    uint64_t in0BlockW, uint64_t outSubblockH, uint64_t outSubblockW,
    uint64_t outBlockH, uint64_t outBlockW, uint64_t perCoreM,
    uint64_t perCoreN, bool transposeMcast, MlirAttribute fusedActivation,
    bool fuseBatch) {
  return wrap(MatmulMultiCoreReuseMultiCastProgramConfigAttr::get(
      unwrap(ctx),
      mlir::cast<CoreCoordAttr>(unwrap(computeWithStorageGridSize)), in0BlockW,
      outSubblockH, outSubblockW, outBlockH, outBlockW, perCoreM, perCoreN,
      transposeMcast, mlir::cast<UnaryWithParamAttr>(unwrap(fusedActivation)),
      fuseBatch));
}

MlirAttribute ttmlirTTNNMatmulMultiCoreReuseMultiCast1DProgramConfigAttrGet(
    MlirContext ctx, MlirAttribute computeWithStorageGridSize,
    uint64_t in0BlockW, uint64_t outSubblockH, uint64_t outSubblockW,
    uint64_t outBlockH, uint64_t outBlockW, uint64_t perCoreM,
    uint64_t perCoreN, bool fuseBatch, MlirAttribute fusedActivation,
    bool mcastIn0, bool gatherIn0, MlirAttribute hopCores,
    uint64_t numGlobalCbReceivers, bool untilizeOut) {
  return wrap(MatmulMultiCoreReuseMultiCast1DProgramConfigAttr::get(
      unwrap(ctx),
      mlir::cast<CoreCoordAttr>(unwrap(computeWithStorageGridSize)), in0BlockW,
      outSubblockH, outSubblockW, outBlockH, outBlockW, perCoreM, perCoreN,
      fuseBatch, mlir::cast<UnaryWithParamAttr>(unwrap(fusedActivation)),
      mcastIn0, gatherIn0, mlir::cast<CoreRangeSetAttr>(unwrap(hopCores)),
      numGlobalCbReceivers, untilizeOut));
}

MlirAttribute
ttmlirTTNNMatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttrGet(
    MlirContext ctx, uint64_t in0BlockW, uint64_t perCoreM, uint64_t perCoreN,
    MlirAttribute fusedActivation) {
  return wrap(MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfigAttr::get(
      unwrap(ctx), in0BlockW, perCoreM, perCoreN,
      mlir::cast<UnaryWithParamAttr>(unwrap(fusedActivation))));
}

MlirAttribute ttmlirTTNNCoreRangeAttrGet(MlirContext ctx,
                                         MlirAttribute startCoord,
                                         MlirAttribute endCoord) {
  return wrap(CoreRangeAttr::get(unwrap(ctx),
                                 mlir::cast<CoreCoordAttr>(unwrap(startCoord)),
                                 mlir::cast<CoreCoordAttr>(unwrap(endCoord))));
}

MlirAttribute ttmlirTTNNCoreRangeArrayAttrGet(MlirContext ctx,
                                              MlirAttribute *coreRangeAttrs,
                                              size_t coreRangeAttrsSize) {
  std::vector<mlir::Attribute> coreRanges;
  for (size_t i = 0; i < coreRangeAttrsSize; i++) {
    coreRanges.push_back(mlir::cast<CoreRangeAttr>(unwrap(coreRangeAttrs[i])));
  }
  return wrap(mlir::ArrayAttr::get(unwrap(ctx), coreRanges));
}

MlirAttribute ttmlirTTNNLayoutAttrGet(MlirContext ctx, uint32_t layout) {
  return wrap(LayoutAttr::get(unwrap(ctx), static_cast<Layout>(layout)));
}

MlirAttribute ttmlirTTNNTensorMemoryLayoutAttrGet(MlirContext ctx,
                                                  uint32_t tensorMemoryLayout) {
  return wrap(TensorMemoryLayoutAttr::get(
      unwrap(ctx), static_cast<TensorMemoryLayout>(tensorMemoryLayout)));
}

MlirAttribute ttmlirTTNNBufferTypeAttrGet(MlirContext ctx,
                                          uint32_t bufferType) {
  return wrap(
      BufferTypeAttr::get(unwrap(ctx), static_cast<BufferType>(bufferType)));
}

MlirAttribute ttmlirTTNNShardSpecAttrGet(MlirContext ctx,
                                         MlirAttribute coreRangeSetAttr,
                                         MlirAttribute shapeAttr,
                                         MlirAttribute shardOrientationAttr) {
  return wrap(ShardSpecAttr::get(
      unwrap(ctx), mlir::cast<CoreRangeSetAttr>(unwrap(coreRangeSetAttr)),
      mlir::cast<ShapeAttr>(unwrap(shapeAttr)),
      mlir::cast<ShardOrientationAttr>(unwrap(shardOrientationAttr))));
}

MlirAttribute ttmlirTTNNMemoryConfigAttrGet(
    MlirContext ctx, MlirAttribute tensorMemoryLayoutAttr,
    MlirAttribute bufferTypeAttr, MlirAttribute shardSpecAttr) {
  return wrap(MemoryConfigAttr::get(
      unwrap(ctx),
      mlir::cast<TensorMemoryLayoutAttr>(unwrap(tensorMemoryLayoutAttr)),
      mlir::cast<BufferTypeAttr>(unwrap(bufferTypeAttr)),
      mlirAttributeIsNull(shardSpecAttr)
          ? std::nullopt
          : std::optional<mlir::tt::ttnn::ShardSpecAttr>(
                mlir::cast<ShardSpecAttr>(unwrap(shardSpecAttr)))));
}

MlirAttribute ttmlirTTNNShapeAttrGet(MlirContext ctx, int64_t *shape,
                                     size_t shapeSize) {
  return wrap(ShapeAttr::get(unwrap(ctx), {shape, shape + shapeSize}));
}

MlirAttribute ttmlirTTNNMeshShapeAttrGet(MlirContext ctx, int64_t y,
                                         int64_t x) {
  return wrap(MeshShapeAttr::get(unwrap(ctx), y, x));
}

// Get layout TTNNLayout attribute
//
// param ctx: mlir context
// param linear Affine map for mapping tensor from logical to physical space
// param grid Grid of cores where tensor is mapped to
// param memref Memref which holds shard size, shard scalar type and memory
// param memLayout Memory layout of the tensor
MlirAttribute ttmlirTTNNTTNNLayoutAttrGet(MlirContext ctx, MlirAffineMap linear,
                                          MlirAttribute grid, MlirType memref,
                                          unsigned *memLayout = nullptr) {
  mlir::AffineMap affineMap = mlir::AffineMap::getFromOpaquePointer(linear.ptr);
  TensorMemoryLayoutAttr memLayoutAttr;
  if (memLayout) {
    memLayoutAttr = TensorMemoryLayoutAttr::get(
        unwrap(ctx), static_cast<TensorMemoryLayout>(*memLayout));
  }

  mlir::tt::ttcore::TensorMeshAttr tensorMeshAttr;
  return wrap(
      TTNNLayoutAttr::get(unwrap(ctx), affineMap,
                          mlir::cast<mlir::tt::ttcore::GridAttr>(unwrap(grid)),
                          mlir::cast<mlir::MemRefType>(unwrap(memref)),
                          memLayoutAttr, tensorMeshAttr));
}