// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/D2MTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/D2M/IR/D2MOps.h"
#include "ttmlir/Dialect/D2M/IR/D2MOpsTypes.h"

using namespace mlir::tt::d2m;

MlirAttribute ttmlirD2MThreadTypeAttrGet(MlirContext ctx, uint32_t enumValue) {
  return wrap(ThreadAttr::get(unwrap(ctx), static_cast<ThreadType>(enumValue)));
}
