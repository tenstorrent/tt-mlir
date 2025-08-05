// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTIRAttrs.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

using namespace mlir::tt::ttir;

MlirAttribute ttmlirTTIRThreadAttrGet(MlirContext ctx, uint32_t threadType,
                                      MlirAttribute kernelSymbol) {
  mlir::SymbolRefAttr kernelSymbolAttr = nullptr;
  if (kernelSymbol.ptr != nullptr) {
    kernelSymbolAttr =
        mlir::dyn_cast<mlir::SymbolRefAttr>(unwrap(kernelSymbol));
  }
  return wrap(ThreadAttr::get(unwrap(ctx), static_cast<ThreadType>(threadType),
                              kernelSymbolAttr));
}
