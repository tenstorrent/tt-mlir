// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTTypes.h"
#include "mlir/CAPI/IR.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"

using namespace mlir::tt;

MlirType ttmlirTTTileTypeGet(MlirContext ctx, unsigned height, unsigned width,
                             uint32_t dataType) {
  return wrap(TileType::get(unwrap(ctx),
                            llvm::SmallVector<std::int64_t>{height, width},
                            static_cast<DataType>(dataType)));
}

MlirType ttmlirTTTupleTypeGet(MlirContext ctx, MlirType *elements,
                              size_t numElements) {
  llvm::SmallVector<mlir::Type> elementsVec;
  for (size_t i = 0; i < numElements; i++) {
    elementsVec.push_back(unwrap(elements[i]));
  }
  return wrap(mlir::TupleType::get(unwrap(ctx), elementsVec));
}
