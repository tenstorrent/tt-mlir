// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>

namespace mlir::tt::ttkernel {

// Class used to add includes and other boilerplate code to the generated
// kernel.
namespace {
class ScopedModuleHelper {
public:
  ScopedModuleHelper(OpBuilder *builder, Location loc, ThreadType threadType)
      : builder(builder), loc(loc), threadType(threadType) {
    builder->create<emitc::IncludeOp>(loc, "cstdint",
                                      /*isStandard=*/true);
    if (threadType == ThreadType::Noc) {

      builder->create<emitc::IncludeOp>(loc, "dataflow_api.h",
                                        /*isStandard=*/false);
    }
    if (threadType == ThreadType::Tensix) {
      builder->create<emitc::IncludeOp>(loc, "llk_defs.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/common.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/tilize.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/untilize.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc,
                                        "compute_kernel_api/eltwise_binary.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api.h", // max ops
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc,
                                        "compute_kernel_api/tile_move_copy.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/eltwise_unary.h",
          /*isStandard=*/false);
      // TODO (kmitrovic) exp.h is an ExpOp-specific include. Every op has one,
      // should be handled in general, not like this.
      // Issue: https://github.com/tenstorrent/tt-mlir/issues/772
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/exp.h",
          /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/sfpu_split_includes.h",
          /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/recip.h",
          /*isStandard=*/false);
      // Must define macros REDUCE_OP and REDUCE_DIM before including reduce.h
      // because they are default template parameters values in reduce api.
      builder->create<emitc::VerbatimOp>(loc,
                                         "#define REDUCE_OP PoolType::SUM");
      builder->create<emitc::VerbatimOp>(
          loc, "#define REDUCE_DIM ReduceDim::REDUCE_COL");
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/reduce.h",
                                        /*isStandard=*/false);
      builder->create<emitc::VerbatimOp>(loc, "namespace NAMESPACE {");
    }
  }

  ~ScopedModuleHelper() {
    if (threadType == ThreadType::Tensix) {
      builder->create<emitc::VerbatimOp>(loc, "void MAIN { kernel_main(); }");
      builder->create<emitc::VerbatimOp>(loc,
                                         "}"); // close namespace NAMESPACE
    }
  }

private:
  OpBuilder *builder;
  Location loc;
  ThreadType threadType;
};
} // namespace

static FailureOr<mlir::ModuleOp>
cloneEntryIntoStandaloneModule(func::FuncOp origEntry, ThreadType threadType) {
  auto *ctx = origEntry.getContext();
  Region *region = &origEntry.getBody();
  auto loc = origEntry.getLoc();

  OpBuilder builder(ctx);

  // We will wrap everything in a standalone module op so that we can run the
  // translation.
  auto moduleWrapper = builder.create<mlir::ModuleOp>(loc, "module_wrapper");
  builder.setInsertionPointToStart(moduleWrapper.getBody());

  Region *kernelMainRegion;
  {
    ScopedModuleHelper threadConfigHelper(&builder, loc, threadType);

    // Clone 'region' into a new func op nested inside 'moduleWrapper':
    auto kernelMain = builder.create<func::FuncOp>(
        loc, "kernel_main",
        builder.getType<FunctionType>(region->getArgumentTypes(), TypeRange()));
    kernelMainRegion = &kernelMain.getBody();
  }

  IRMapping irMapper;
  region->cloneInto(kernelMainRegion, irMapper);
  return moduleWrapper;
}

LogicalResult translateKernelFuncToCpp(func::FuncOp entry,
                                       llvm::raw_ostream &os) {
  if (!entry->hasAttr(ThreadTypeAttr::name)) {
    return failure();
  }

  ThreadType threadType =
      entry->getAttrOfType<ThreadTypeAttr>(ThreadTypeAttr::name).getValue();
  auto kernelModule = cloneEntryIntoStandaloneModule(entry, threadType);
  if (failed(kernelModule)) {
    return failure();
  }
  return emitc::translateToCpp(*kernelModule, os);
}

LogicalResult translateTopLevelKernelToCpp(ModuleOp moduleOp,
                                           llvm::raw_ostream &os,
                                           StringRef symbolName) {
  SymbolTable symbolTable(moduleOp);
  auto entry = symbolTable.lookup<func::FuncOp>(symbolName);
  if (!entry) {
    return failure();
  }
  return translateKernelFuncToCpp(entry, os);
}

LogicalResult translateTopLevelKernelsToCpp(ModuleOp moduleOp,
                                            llvm::raw_ostream &os) {
  LogicalResult result = success();
  moduleOp->walk([&](func::FuncOp entry) {
    if (!entry->hasAttr(ThreadTypeAttr::name)) {
      return;
    }

    if (failed(translateKernelFuncToCpp(entry, os))) {
      result = failure();
      return;
    }
  });
  return result;
}

} // namespace mlir::tt::ttkernel
