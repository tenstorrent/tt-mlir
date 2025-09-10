// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "ttmlir/Target/TTKernel/LLKs/experimental_dataflow_api_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_invoke_sfpi_llks_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_matmul_llks_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_tilize_llks_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_untilize_llks_generated.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttkernel {

// Class used to add includes and other boilerplate code to the generated
// kernel.
namespace {
class ScopedModuleHelper {
public:
  ScopedModuleHelper(OpBuilder *builder, Location loc, Region *region,
                     ThreadType threadType, StringRef originalSymbolName = "")
      : builder(builder), loc(loc), region(region), threadType(threadType) {
    if (!originalSymbolName.empty()) {
      emitComment(originalSymbolName);
    }
    builder->create<emitc::IncludeOp>(loc, "cstdint",
                                      /*isStandard=*/true);

    builder->create<emitc::IncludeOp>(loc, "tools/profiler/kernel_profiler.hpp",
                                      /*isStandard=*/false);

    emitDebugPrint();

    if (threadType == ThreadType::Noc) {

      builder->create<emitc::IncludeOp>(loc, "dataflow_api.h",
                                        /*isStandard=*/false);
      emitExperimentalLLKs();
    }
    if (threadType == ThreadType::Compute) {
      builder->create<emitc::IncludeOp>(loc, "llk_defs.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/common.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/matmul.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/tilize.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/untilize.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc,
                                        "compute_kernel_api/transpose_wh.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(loc,
                                        "compute_kernel_api/eltwise_binary.h",
                                        /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_binary_sfpu.h",
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
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/fill.h",
          /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/negative.h",
          /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/sqrt.h",
          /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/rounding.h",
          /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/trigonometry.h",
          /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/logical_not_noti.h",
          /*isStandard=*/false);
      builder->create<emitc::IncludeOp>(
          loc, "compute_kernel_api/eltwise_unary/typecast.h",
          /*isStandard=*/false);
      // Must define macros REDUCE_OP and REDUCE_DIM before including reduce.h
      // because they are default template parameters values in reduce api.
      builder->create<emitc::VerbatimOp>(loc,
                                         "#define REDUCE_OP PoolType::SUM");
      builder->create<emitc::VerbatimOp>(
          loc, "#define REDUCE_DIM ReduceDim::REDUCE_COL");
      builder->create<emitc::IncludeOp>(loc, "compute_kernel_api/reduce.h",
                                        /*isStandard=*/false);
      emitExperimentalLLKs();
      builder->create<emitc::VerbatimOp>(loc, "namespace NAMESPACE {");
    }
  }

  ~ScopedModuleHelper() {
    if (threadType == ThreadType::Compute) {
      builder->create<emitc::VerbatimOp>(loc, "void MAIN { kernel_main(); }");
      builder->create<emitc::VerbatimOp>(loc,
                                         "}"); // close namespace NAMESPACE
    }
  }

  void emitComment(StringRef str) {
    builder->create<emitc::VerbatimOp>(loc, (Twine("// ") + str).str());
  }

  void emitDebugPrint() {
    if (!hasOp<emitc::CallOpaqueOp>([](emitc::CallOpaqueOp op) {
          return op.getCallee() == "ttmlir::dprint";
        })) {
      return;
    }

    builder->create<emitc::IncludeOp>(loc, "debug/dprint.h",
                                      /*isStandard=*/false);

    builder->create<emitc::VerbatimOp>(
        loc, "template <> uint8_t DebugPrintTypeToId<size_t>() { return "
             "DPrintUINT32; }");

    builder->create<emitc::VerbatimOp>(loc, R""""(
namespace ttmlir {
template<typename Arg>
void dprint(Arg &&arg) {
  DPRINT << arg;
}

template<typename Arg, typename... ArgV>
void dprint(Arg &&arg, ArgV&&... argv) {
  DPRINT << arg;
  dprint(argv...);
}
} // namespace ttmlir
)"""");
  }

  void emitExperimentalLLKs() {
    if (hasCall("experimental::tilize")) {
      auto experimentalTilizeLLKs =
          StringRef(experimental_tilize_llks_generated,
                    experimental_tilize_llks_generated_len);
      builder->create<emitc::VerbatimOp>(loc, experimentalTilizeLLKs);
    }

    if (hasCall("experimental::untilize")) {
      auto experimentalUntilizeLLKs =
          StringRef(experimental_untilize_llks_generated,
                    experimental_untilize_llks_generated_len);
      builder->create<emitc::VerbatimOp>(loc, experimentalUntilizeLLKs);
    }

    if (hasCall("experimental::get_noc_multicast_addr")) {
      auto experimentalDataflowLLKs =
          StringRef(experimental_dataflow_api_generated,
                    experimental_dataflow_api_generated_len);
      builder->create<emitc::VerbatimOp>(loc, experimentalDataflowLLKs);
    }

    if (hasCall("experimental::matmul_block")) {
      auto experimentalMatmulLLKs =
          StringRef(experimental_matmul_llks_generated,
                    experimental_matmul_llks_generated_len);
      builder->create<emitc::VerbatimOp>(loc, experimentalMatmulLLKs);
    }

    if (hasVerbatim("experimental::invoke_sfpi")) {
      builder->create<emitc::VerbatimOp>(
          loc, StringRef(experimental_invoke_sfpi_llks_generated,
                         experimental_invoke_sfpi_llks_generated_len));
    }
  }

  bool hasCall(StringRef name) {
    return hasOp<emitc::CallOpaqueOp>([=](emitc::CallOpaqueOp op) {
      return op.getCallee().starts_with(name);
    });
  }

  bool hasVerbatim(StringRef name) {
    return hasOp<emitc::VerbatimOp>(
        [=](emitc::VerbatimOp op) { return op.getValue().starts_with(name); });
  }

  template <typename OpT>
  bool hasOp(llvm::function_ref<bool(OpT)> predicate = [](OpT) {
    return true;
  }) {
    bool found = false;
    region->walk([&](OpT op) {
      if (predicate(op)) {
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found;
  }

private:
  OpBuilder *builder;
  Location loc;
  Region *region;
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
    ScopedModuleHelper threadConfigHelper(&builder, loc, region, threadType,
                                          origEntry.getName());

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
  FailureOr<mlir::ModuleOp> kernelModule =
      cloneEntryIntoStandaloneModule(entry, threadType);
  if (failed(kernelModule)) {
    return failure();
  }
  auto moduleCleanup = llvm::make_scope_exit([&]() { kernelModule->erase(); });
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
