// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir-c/TTKernelTypes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

using namespace mlir::tt::ttkernel;

MlirType ttmlirTTKernelCBTypeGet(MlirContext ctx, uint64_t address,
                                 uint64_t port, MlirType memrefType) {
  return wrap(CBType::get(unwrap(ctx), symbolizeCBPort(port).value(), address,
                          mlir::cast<mlir::MemRefType>(unwrap(memrefType))));
}
