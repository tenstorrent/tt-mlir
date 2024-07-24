// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "ttmlir-c/TTAttrs.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

namespace mlir::tt {

MlirType ttmlirTTTileTypeGet(MlirContext ctx, unsigned height, unsigned width,
                             uint32_t dataType) {
  return wrap(TileType::get(unwrap(ctx), height, width,
                            static_cast<tt::DataType>(dataType)));
}

MlirType ttmlirTTDeviceTypeGet(MlirContext ctx, MlirAttribute grid,
                               unsigned *chipIds, size_t chipIdsSize) {
  llvm::ArrayRef<unsigned> chipIdsRef(chipIds, chipIds + chipIdsSize);
  return wrap(DeviceType::get(unwrap(ctx), unwrap(grid).cast<tt::GridAttr>(),
                              chipIdsRef));
}

} // namespace mlir::tt
