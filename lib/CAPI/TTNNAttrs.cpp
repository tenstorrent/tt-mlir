// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTNNAttrs.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

namespace mlir::tt::ttnn {

MlirAttribute ttmlirTTNNCoreRangeAttrGet(MlirContext ctx, int64_t *offset,
                                         size_t offsetSize, int64_t *size,
                                         size_t sizeSize) {
  return wrap(CoreRangeAttr::get(unwrap(ctx), {offset, offset + offsetSize},
                                 {size, size + sizeSize}));
}

MlirAttribute ttmlirTTNNCoreRangeArrayAttrGet(MlirContext ctx,
                                              MlirAttribute *coreRangeAttrs,
                                              size_t coreRangeAttrsSize) {
  std::vector<mlir::Attribute> coreRanges;
  for (size_t i = 0; i < coreRangeAttrsSize; i++) {
    coreRanges.push_back(mlir::cast<CoreRangeAttr>(unwrap(coreRangeAttrs[i])));
  }
  return wrap(ArrayAttr::get(unwrap(ctx), coreRanges));
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
                                         MlirAttribute shardShapeAttr) {
  return wrap(ShardSpecAttr::get(
      unwrap(ctx), mlir::cast<ShapeAttr>(unwrap(shardShapeAttr))));
}

MlirAttribute ttmlirTTNNMemoryConfigAttrGet(
    MlirContext ctx, MlirAttribute tensorMemoryLayoutAttr,
    MlirAttribute bufferTypeAttr, MlirAttribute shardSpecAttr) {
  return wrap(MemoryConfigAttr::get(
      unwrap(ctx), mlir::cast<BufferTypeAttr>(unwrap(bufferTypeAttr)),
      mlir::cast<ShardSpecAttr>(unwrap(shardSpecAttr)),
      mlir::cast<TensorMemoryLayoutAttr>(unwrap(tensorMemoryLayoutAttr))));
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
  return wrap(TTNNLayoutAttr::get(
      unwrap(ctx), affineMap, mlir::cast<GridAttr>(unwrap(grid)),
      mlir::cast<MemRefType>(unwrap(memref)), memLayoutAttr));
}

} // namespace mlir::tt::ttnn
