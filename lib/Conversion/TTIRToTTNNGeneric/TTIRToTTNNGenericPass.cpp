// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTNNGeneric/TTIRToTTNNGeneric.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_CONVERTTTIRTOTTNNGENERIC
#include "ttmlir/Conversion/Passes.h.inc" // impl::ConvertTTIRToTTNNGenericBase

// ............................................................................
using namespace llvm;

namespace {

struct ConvertTTIRToTTNNGenericPass final
    : impl::ConvertTTIRToTTNNGenericBase<ConvertTTIRToTTNNGenericPass> {
  void runOnOperation() final {
    // TODO: populate and apply conversion here
  }
};

} // namespace
} // namespace mlir::tt::ttir
// ............................................................................
namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTIRToTTNNGenericPass() {
  return std::make_unique<ttir::ConvertTTIRToTTNNGenericPass>();
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
