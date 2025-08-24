// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "ttmlir-c/TTAttrs.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

using namespace mlir::tt::ttcore;

MlirType ttmlirTTTileTypeGet(MlirContext ctx, unsigned height, unsigned width,
                             uint32_t dataType) {
  return wrap(TileType::get(unwrap(ctx),
                            llvm::SmallVector<std::int64_t>{height, width},
                            static_cast<DataType>(dataType)));
}
