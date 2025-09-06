// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTIRTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROpsTypes.h"

using namespace mlir::tt::ttir;

MlirAttribute ttmlirTTIRThreadTypeAttrGet(MlirContext ctx, uint32_t enumValue) {
  return wrap(ThreadAttr::get(unwrap(ctx), static_cast<ThreadType>(enumValue)));
}
