// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTKernelTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

using namespace mlir::tt::ttkernel;

MlirType ttmlirTTKernelCBTypeGet(MlirContext ctx, MlirType memrefType) {
  return wrap(CBType::get(unwrap(ctx),
                          mlir::cast<mlir::MemRefType>(unwrap(memrefType))));
}

MLIR_CAPI_EXPORTED MlirAttribute
ttmlirTTKernelThreadTypeAttrGet(MlirContext ctx, uint32_t enumValue) {
  return wrap(
      ThreadTypeAttr::get(unwrap(ctx), static_cast<ThreadType>(enumValue)));
}
