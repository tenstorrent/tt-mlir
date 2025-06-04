// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

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
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir::tt::ttkernel {

static constexpr llvm::StringRef SFPU_INCLUDE =
    "compute_kernel_api/eltwise_unary/sfpu_split_includes.h";
static constexpr llvm::StringRef NOC_INCLUDE = "dataflow_api.h";

static const llvm::StringMap<llvm::StringRef> includeMapping{
    // Register operations
    {"tile_regs_acquire", "compute_kernel_api/reg_api.h"},
    {"tile_regs_commit", "compute_kernel_api/reg_api.h"},
    {"tile_regs_wait", "compute_kernel_api/reg_api.h"},
    {"tile_regs_release", "compute_kernel_api/reg_api.h"},
    {"pack_tile", "compute_kernel_api/pack.h"},
    {"copy_tile_init", "compute_kernel_api/tile_move_copy.h"},
    {"copy_tile", "compute_kernel_api/tile_move_copy.h"},

    // FPU operations
    {"unary_op_init_common",
     "compute_kernel_api/eltwise_unary/eltwise_unary.h"},
    {"binary_op_init_common", "compute_kernel_api/eltwise_binary.h"},
    {"mm_init", "compute_kernel_api/matmul.h"},
    {"mm_init_short", "compute_kernel_api/matmul.h"},
    {"matmul_tiles", "compute_kernel_api/matmul.h"},
    {"add_tiles_init", "compute_kernel_api/eltwise_binary.h"},
    {"add_tiles", "compute_kernel_api/eltwise_binary.h"},
    {"mul_tiles_init", "compute_kernel_api/eltwise_binary.h"},
    {"mul_tiles", "compute_kernel_api/eltwise_binary.h"},
    {"reduce_init", "compute_kernel_api/reduce.h"},
    {"reduce_tile", "compute_kernel_api/reduce.h"},

    // SFPU operations
    {"init_sfpu", SFPU_INCLUDE},
    {"max_tile_init", SFPU_INCLUDE},
    {"max_tile", SFPU_INCLUDE},
    {"div_binary_tile_init", SFPU_INCLUDE},
    {"div_binary_tile", SFPU_INCLUDE},
    {"recip_tile_init", SFPU_INCLUDE},
    {"recip_tile", SFPU_INCLUDE},
    {"exp_tile_init", SFPU_INCLUDE},
    {"exp_tile", SFPU_INCLUDE},
    {"sin_tile_init", SFPU_INCLUDE},
    {"sin_tile", SFPU_INCLUDE},

    // CB operations
    {"cb_push_back", "compute_kernel_api/cb_api.h"},
    {"cb_pop_front", "compute_kernel_api/cb_api.h"},
    {"cb_reserve_back", "compute_kernel_api/cb_api.h"},
    {"cb_wait_front", "compute_kernel_api/cb_api.h"},

    // Tile operations
    {"tilize_block", "compute_kernel_api/tilize.h"},
    {"tilize_init", "compute_kernel_api/tilize.h"},
    {"tilize_init_short", "compute_kernel_api/tilize.h"},
    {"tilize_uninit", "compute_kernel_api/tilize.h"},
    {"untilize_block", "compute_kernel_api/untilize.h"},
    {"untilize_init", "compute_kernel_api/untilize.h"},
    {"untilize_init_short", "compute_kernel_api/untilize.h"},
    {"untilize_uninit", "compute_kernel_api/untilize.h"},

    // NOC operations
    {"get_noc_addr_from_bank_id", NOC_INCLUDE},
    {"noc_async_read", NOC_INCLUDE},
    {"noc_async_read_tile", NOC_INCLUDE},
    {"noc_async_read_one_packet_set_state", NOC_INCLUDE},
    {"noc_async_read_one_packet_with_state", NOC_INCLUDE},
    {"noc_async_read_barrier", NOC_INCLUDE},
    {"noc_async_write", NOC_INCLUDE},
    {"noc_async_write_tile", NOC_INCLUDE},
    {"noc_async_write_barrier", NOC_INCLUDE},
    {"get_semaphore", NOC_INCLUDE},
    {"noc_semaphore_inc", NOC_INCLUDE},
    {"noc_semaphore_set", NOC_INCLUDE},
    {"noc_semaphore_wait", NOC_INCLUDE},
    {"noc_semaphore_wait_min", NOC_INCLUDE},
    {"noc_semaphore_set_multicast", NOC_INCLUDE},
    {"noc_semaphore_set_multicast_loopback_src", NOC_INCLUDE},

    // Compile/runtime arg ops
    {"get_arg_val", "compute_kernel_api/common.h"},
    {"get_compile_time_arg_val",
     "accessor/sharded_accessor.h"},

    // Multicast NoC
    {"get_noc_multicast_addr", NOC_INCLUDE},
    {"noc_async_write_multicast_one_packet", NOC_INCLUDE},
    {"noc_async_write_multicast", NOC_INCLUDE},
    {"noc_async_write_multicast_loopback_src", NOC_INCLUDE},

    // Misc operations
    {"mem_zeros_base", "dev_mem_map.h"},
    {"mem_zeros_size", "dev_mem_map.h"},
    {"get_write_ptr", "dataflow_api.h"},
    {"get_read_ptr", "dataflow_api.h"},
    {"get_tile_size", "dataflow_api.h"},
    {"get_dataformat", "dataflow_api.h"},
};

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

    emitDebugPrint();
    emitIncludes();

    if (threadType == ThreadType::Compute) {
      // Must define macros REDUCE_OP and REDUCE_DIM before including reduce.h
      // because they are default template parameters values in reduce api.
      builder->create<emitc::VerbatimOp>(loc,
                                         "#define REDUCE_OP PoolType::SUM");
      builder->create<emitc::VerbatimOp>(
          loc, "#define REDUCE_DIM ReduceDim::REDUCE_COL");
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

  void emitIncludes() {
    llvm::SmallSet<llvm::StringRef, 8> collectedIncludes;
    region->walk([&](emitc::CallOpaqueOp op) {
      if (auto itr = includeMapping.find(op.getCallee());
          itr != includeMapping.end()) {
        collectedIncludes.insert(itr->getValue());
      }
      return WalkResult::advance();
    });

    llvm::SmallVector<llvm::StringRef, 8> sortedIncludes(
        collectedIncludes.begin(), collectedIncludes.end());
    std::sort(sortedIncludes.begin(), sortedIncludes.end());
    for (auto sr : sortedIncludes) {
      builder->create<emitc::IncludeOp>(loc, sr, /*isStandard=*/false);
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
  }

  bool hasCall(StringRef name) {
    return hasOp<emitc::CallOpaqueOp>([=](emitc::CallOpaqueOp op) {
      return op.getCallee().starts_with(name);
    });
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
