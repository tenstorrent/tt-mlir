// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTIRToTTIRGeneric/TTIRToTTIRGeneric.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"

// ----------------------------------------------------------------------------
namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRTOTTIRGENERIC
#include "ttmlir/Conversion/Passes.h.inc" // impl::TTIRToTTIRGenericBase

} // namespace mlir::tt::ttir
// ............................................................................
namespace mlir::tt {

using namespace llvm;

namespace {
struct TTIRToTTIRGenericPass final
    : ttir::impl::TTIRToTTIRGenericBase<TTIRToTTIRGenericPass> {

  TTIRToTTIRGenericPass(const TTIRToTTIRGenericPass &other)
      : TTIRToTTIRGenericBase<TTIRToTTIRGenericPass>(other) {
    useTileMatmul = other.useTileMatmul;
  }
  TTIRToTTIRGenericPass() : TTIRToTTIRGenericBase<TTIRToTTIRGenericPass>() {}

  void runOnOperation() final {

    auto &ctx = getContext();
    auto op = getOperation();

    mlir::ConversionTarget target{ctx};
    {
      // Illegal.

      target.addIllegalDialect<ttir::TTIRDialect>();

      // Legal.

      target.addLegalDialect<mlir::BuiltinDialect>();
      target.addLegalDialect<mlir::func::FuncDialect>();
      target.addLegalDialect<mlir::linalg::LinalgDialect>();

      target.addLegalDialect<tt::TTCoreDialect>();

      // An explicit list of legal ttir.* ops.
      target.addLegalOp<ttir::GenericOp>();
      target.addLegalOp<ttir::ToLayoutOp>();
      target.addLegalOp<ttir::StreamLayoutOp>();
      target.addLegalOp<ttir::ViewLayoutOp>();
      target.addLegalOp<ttir::EmptyOp>();
      target.addLegalOp<ttir::ConstantOp>();

      target.addLegalOp<
#define GET_OP_LIST
#include "ttmlir/Dialect/TTIR/IR/TTIRGenericRegionOps.cpp.inc"
#undef GET_OP_LIST
          >();
    }

    DeviceAttr deviceAttr = lookupDevice(getOperation());

    TypeConverter typeConverter;
    {
      // Dialect conversion requires 1:1 (null) type conversion rule at a
      // minimum.
      typeConverter.addConversion([](Type type) { return type; });
    }

    mlir::RewritePatternSet patterns{&ctx};
    populateTTIRToTTIRGenericPatterns(&ctx, patterns, typeConverter,
                                      deviceAttr.getWorkerGrid().getRank(),
                                      useTileMatmul);

    if (failed(mlir::applyFullConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

protected:
  Option<bool> useTileMatmul{*this, "use-tile-matmul",
                             llvm::cl::desc("Use tile_matmul"),
                             llvm::cl::init(true)};

}; // end of class
} // namespace
// ............................................................................

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRGenericPass() {
  return std::make_unique<TTIRToTTIRGenericPass>();
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
