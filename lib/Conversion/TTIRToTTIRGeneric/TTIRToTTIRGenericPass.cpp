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

// ............................................................................
using namespace llvm;

namespace {

struct TTIRToTTIRGenericPass final
    : impl::TTIRToTTIRGenericBase<TTIRToTTIRGenericPass> {

  using Base = impl::TTIRToTTIRGenericBase<TTIRToTTIRGenericPass>;

  TTIRToTTIRGenericPass(const TTIRToTTIRGenericOptions &options)
      : Base(options) {}
  TTIRToTTIRGenericPass() = default;

  TTIRToTTIRGenericPass(const TTIRToTTIRGenericPass &rhs) : Base(rhs) {
    // Workaround: Passes are required to be copy-constructible but autogen'ed
    // base class copy constructors ignore Pass option fields.
    this->useTileMatmul = rhs.useTileMatmul;
    this->defaultInputMemSpace = rhs.defaultInputMemSpace;
    this->defaultOutputMemSpace = rhs.defaultOutputMemSpace;
  };

  void runOnOperation() final {
    auto &ctx = getContext();
    auto moduleOp = getOperation();

    mlir::ConversionTarget target{ctx};
    {
      // Illegal.

      target.addIllegalDialect<ttir::TTIRDialect>();

      // Legal.

      target.addLegalDialect<mlir::BuiltinDialect>();
      target.addLegalDialect<mlir::func::FuncDialect>();
      target.addLegalDialect<mlir::linalg::LinalgDialect>();

      target.addLegalDialect<ttcore::TTCoreDialect>();

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

    ttcore::DeviceAttr deviceAttr = ttcore::lookupDevice(moduleOp);

    TypeConverter typeConverter;
    {
      // Dialect conversion requires 1:1 (null) type conversion rule at a
      // minimum.
      typeConverter.addConversion([](Type type) { return type; });
    }

    mlir::RewritePatternSet patterns{&ctx};
    populateTTIRToTTIRGenericPatterns(
        &ctx, patterns, typeConverter,
        {useTileMatmul, defaultInputMemSpace, defaultOutputMemSpace},
        deviceAttr.getWorkerGrid().getRank());

    if (failed(
            mlir::applyFullConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

}; // end of class
} // namespace
} // namespace mlir::tt::ttir
// ............................................................................
namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createTTIRToTTIRGenericPass() {
  return std::make_unique<ttir::TTIRToTTIRGenericPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createTTIRToTTIRGenericPass(const ttir::TTIRToTTIRGenericOptions &options) {
  return std::make_unique<ttir::TTIRToTTIRGenericPass>(options);
}

} // namespace mlir::tt
// ----------------------------------------------------------------------------
