// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cassert>

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_CONVERTTTNNTOEMITC
#include "ttmlir/Conversion/Passes.h.inc"

} // namespace mlir::tt::ttnn

namespace {

class TTNNToEmitCTypeConverter : public TypeConverter {
public:
  TTNNToEmitCTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([ctx](tt::DeviceType type) -> emitc::PointerType {
      return emitc::PointerType::get(
          emitc::OpaqueType::get(ctx, "ttnn::Device"));
    });
    addConversion([ctx](mlir::TensorType type) -> emitc::OpaqueType {
      return emitc::OpaqueType::get(ctx, "ttnn::Tensor");
    });
  }
};

struct ConvertTTNNToEmitCPass
    : public ttnn::impl::ConvertTTNNToEmitCBase<ConvertTTNNToEmitCPass> {
  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());

    // EmitC is legal, TTNN is illegal
    //
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<ttnn::TTNNDialect>();

    // mlir::ModuleOp is legal only if no attributes are present on it
    //
    target.addDynamicallyLegalOp<mlir::ModuleOp>(
        [&](mlir::ModuleOp op) { return op->getAttrs().empty(); });

    // Add header imports to front of module
    //
    {
      mlir::ModuleOp module = getOperation();
      OpBuilder builder(module);

      if (module.getBodyRegion().empty()) {
        // Parent module is empty, nothing to do here
        //
        signalPassFailure();
      }

      // Set insertion point to start of first module child
      //
      builder.setInsertionPointToStart(module.getBody(0));

      // Include headers
      //
      builder.create<emitc::IncludeOp>(module.getLoc(), "ttnn-precompiled.hpp",
                                       /*isStandard=*/false);
    }

    // TTNN -> EmitC
    //
    {
      TTNNToEmitCTypeConverter typeConverter(&getContext());
      RewritePatternSet patterns(&getContext());

      // Func dialect handling
      //
      populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
          patterns, typeConverter);
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
      });
      populateReturnOpTypeConversionPattern(patterns, typeConverter);
      target.addDynamicallyLegalOp<func::ReturnOp>(
          [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
      populateCallOpTypeConversionPattern(patterns, typeConverter);
      target.addDynamicallyLegalOp<func::CallOp>(
          [&](func::CallOp op) { return typeConverter.isLegal(op); });

      // TTNN -> EmitC patterns
      //
      populateTTNNToEmitCPatterns(&getContext(), patterns, typeConverter);

      // Apply conversion
      //
      if (failed(applyFullConversion(getOperation(), target,
                                     std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // // Add tensor creation functions
    // //
    // {
    //   mlir::ModuleOp module = getOperation();
    //   OpBuilder builder(module);

    //   assert(module->getRegions().size() ==
    //          1); // TODO: should this be an assert?

    //   assert(module->getRegion(0).getBlocks().size() ==
    //          1); // TODO: should this be an assert?

    //   // Get the first block of the region at index 0
    //   //
    //   Block *block = module.getBody(0);

    //   // Find all the func.func ops in the module
    //   //
    //   SmallVector<func::FuncOp, 1> funcOps;
    //   for (mlir::Operation &op : block->getOperations()) {
    //     if (mlir::func::FuncOp funcOp = dyn_cast<func::FuncOp>(op)) {

    //       // Skip functions that are used
    //       //
    //       // This will skip utility functions that are used by other
    //       functions,
    //       // only top-level "forward" functions should be considered
    //       //
    //       if (!funcOp->getUses().empty()) {
    //         continue;
    //       }

    //       funcOps.push_back(funcOp);
    //     }
    //   }

    //   // Iterate over all the func ops and add tensor creation functions
    //   //
    //   for (mlir::func::FuncOp funcOp : funcOps) {
    //     // Set insertion point to end of first module child
    //     //
    //     builder.setInsertionPointToEnd(block);
    //   }

    //   // Set insertion point to end of first module child
    //   //
    //   builder.setInsertionPointToEnd(block);

    //   // Create tensor creation functions
    //   //
    //   {
    //     // CreateTensor function
    //     //
    //     {
    //       // Create function
    //       //
    //       auto createTensorFunc = builder.create<emitc::FuncOp>(
    //           module.getLoc(), "createTensor",
    //           builder.getFunctionType(
    //               {emitc::PointerType::get(
    //                   emitc::OpaqueType::get(getContext(), "ttnn::Shape"))},
    //               {emitc::OpaqueType::get(getContext(), "ttnn::Tensor")}));

    //       // Set insertion point to start of function
    //       //
    //       builder.setInsertionPointToStart(createTensorFunc.addEntryBlock());

    //       // Create tensor
    //       //
    //       auto tensor = builder.create<emitc::CallOpaqueOp>(
    //           module.getLoc(),
    //           emitc::OpaqueType::get(getContext(), "ttnn::Tensor"),
    //           "ttnn::createTensor", nullptr, nullptr,
    //           createTensorFunc.getArguments());

    //       // Return tensor
    //       //
    //       builder.create<emitc::ReturnOp>(module.getLoc(),
    //       tensor.getResult(0));
    //     }

    //     // CreateTensorOnDevice function
    //     //
    //     {
    //       // Create function
    //       //
    //       auto createTensorOnDeviceFunc = builder.create<emitc::FuncOp>(
    //           module.getLoc(), "createTensorOnDevice",
    //           builder.getFunctionType(
    //               {emitc::PointerType::get(
    //                    emitc::OpaqueType::get(getContext(), "ttnn::Shape")),
    //                emitc::PointerType::get(
    //                    emitc::OpaqueType::get(getContext(),
    //                    "ttnn::Device"))},
    //               {emitc::OpaqueType::get(getContext(), "ttnn::Tensor")}));

    //       // Set insertion point to start of function
    //       //
    //       builder.setInsertionPointToStart(
    //           createTensorOnDeviceFunc.addEntryBlock());

    //       // Create tensor
    //       //
    //       auto tensor = builder.create<emitc::CallOpaqueOp>(
    //           module.getLoc(),
    //           emitc::OpaqueType::get(getContext(), "ttnn::Tensor"),
    //           "ttnn::createTensorOnDevice", nullptr, nullptr,
    //           createTensorOnDeviceFunc.getArguments());

    //       // Return tensor
    //       //
    //       builder.create<emitc::ReturnOp>(module.getLoc(),
    //       tensor.getResult(0));
    //   }
    // }
    // }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitCPass() {
  return std::make_unique<ConvertTTNNToEmitCPass>();
}

} // namespace mlir::tt
