// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTNNToEmitC/TTNNToEmitC.h"

#include "ttmlir/Conversion/TTNNToEmitC/EmitCConversion.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNN.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

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
    addConversion([ctx](mlir::tt::ttnn::DeviceType type) -> emitc::PointerType {
      return emitc::PointerType::get(
          emitc::OpaqueType::get(ctx, "ttnn::distributed::MeshDevice"));
    });
    addConversion([ctx](mlir::RankedTensorType type) -> emitc::OpaqueType {
      if (mlir::isa_and_present<mlir::tt::ttnn::TraceIdAttr>(
              type.getEncoding())) {
        return emitc::OpaqueType::get(ctx, "ttnn::MeshTraceId");
      }
      return emitc::OpaqueType::get(ctx,
                                    ttnn_to_emitc::TypeNameV<::ttnn::Tensor>);
    });
    addConversion(
        [ctx](mlir::tt::ttnn::GlobalSemaphoreType type) -> emitc::OpaqueType {
          return emitc::OpaqueType::get(ctx, "::ttnn::GlobalSemaphore");
        });
    addConversion([ctx](mlir::TupleType type) -> emitc::OpaqueType {
      return emitc::OpaqueType::get(
          ctx, ttnn_to_emitc::TypeNameV<std::vector<::ttnn::Tensor>>);
    });
  }
};

// Tensor vectors are materialized as named `std::vector<::ttnn::Tensor>`
// locals: `util_create_vec` builds one for a variadic operand list, and ops
// such as `ttnn::max_pool2d` return one that the generated code indexes into.
// Either way the local lives until the enclosing function returns, so it keeps
// a second reference to every tensor it holds. `ttnn::deallocate` frees a
// buffer only when its handle is the last one referencing it, so the
// deallocations the compiler emitted for those tensors become silent no-ops and
// their L1 buffers are never released. The flatbuffer runtime does not have
// this problem: it holds tensors in a pool and drops its reference right after
// the op runs.
//
// Restore that behaviour by releasing each vector as soon as its last consumer
// has run, which re-establishes the sole ownership that the deallocation passes
// assume. A vector that escapes the function is owned by the caller and is left
// untouched.
static void releaseVariadicOperandVectors(mlir::ModuleOp module) {
  module->walk([](func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return;
    }

    // Only values produced by a call are considered: that covers both
    // `util_create_vec` and ops returning a tensor vector, while leaving reads
    // of the const-eval result globals alone, since those are owned by the
    // cache rather than by the function.
    auto vectorTy = emitc::OpaqueType::get(
        funcOp.getContext(),
        mlir::tt::ttnn_to_emitc::TypeNameV<std::vector<::ttnn::Tensor>>);

    llvm::SmallVector<emitc::CallOpaqueOp> vectorOps;
    funcOp.walk([&](emitc::CallOpaqueOp callOp) {
      if (callOp.getNumResults() == 1 &&
          callOp.getResult(0).getType() == vectorTy) {
        vectorOps.push_back(callOp);
      }
    });

    OpBuilder builder(funcOp.getContext());
    for (emitc::CallOpaqueOp vectorOp : vectorOps) {
      Value vector = vectorOp.getResult(0);

      // Find the last consumer of the vector within the block that defines it.
      // A use in another block, or no use at all, means the lifetime cannot be
      // reasoned about locally, so the vector is left alone.
      //
      // Uses are followed through ops that yield a reference into the vector
      // rather than a value copied out of it: an element is read as a
      // `subscript` producing an lvalue followed by a `load` consuming it, and
      // both fold into a single `vec[i]` expression when the C++ is emitted.
      // Only the `load` copies the element out, so releasing after the
      // `subscript` would clear the vector before it is read.
      Operation *lastUser = nullptr;
      bool releasable = !vector.use_empty();
      llvm::SmallVector<Value> worklist{vector};
      llvm::SmallPtrSet<Operation *, 8> visited;
      while (releasable && !worklist.empty()) {
        Value value = worklist.pop_back_val();
        for (Operation *user : value.getUsers()) {
          if (user->getBlock() != vectorOp->getBlock() ||
              mlir::isa<func::ReturnOp>(user)) {
            releasable = false;
            break;
          }
          if (!lastUser || lastUser->isBeforeInBlock(user)) {
            lastUser = user;
          }
          if (!visited.insert(user).second) {
            continue;
          }
          for (Value result : user->getResults()) {
            if (mlir::isa<emitc::LValueType>(result.getType())) {
              worklist.push_back(result);
            }
          }
        }
      }
      if (!releasable || !lastUser) {
        continue;
      }

      builder.setInsertionPointAfter(lastUser);
      builder.create<emitc::CallOpaqueOp>(
          vectorOp.getLoc(), TypeRange{},
          mlir::tt::ttnn_to_emitc::kReleaseVectorFunctionName,
          /*args=*/nullptr, /*template_args=*/nullptr, ValueRange{vector});
    }
  });
}

struct ConvertTTNNToEmitCPass
    : public mlir::tt::ttnn::impl::ConvertTTNNToEmitCBase<
          ConvertTTNNToEmitCPass> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::ConversionTarget target(getContext());

    // EmitC is legal, TTNN is illegal
    //
    target.addLegalDialect<emitc::EmitCDialect>();
    target.addIllegalDialect<mlir::tt::ttnn::TTNNDialect>();

    // mlir::ModuleOp is legal only if no attributes are present on it
    //
    target.addDynamicallyLegalOp<mlir::ModuleOp>(
        [&](mlir::ModuleOp op) { return op->getAttrs().empty(); });

    // Add header imports to front of module
    //
    {
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
      // Disallow arg attrs on func op
      //
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody()) &&
               (!op.getArgAttrs().has_value() ||
                op.getArgAttrs().value().empty());
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
      if (failed(applyFullConversion(module, target, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    releaseVariadicOperandVectors(module);
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTTNNToEmitCPass() {
  return std::make_unique<ConvertTTNNToEmitCPass>();
}

} // namespace mlir::tt
