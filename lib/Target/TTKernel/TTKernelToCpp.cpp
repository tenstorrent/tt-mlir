// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "ttmlir/Target/TTKernel/LLKs/experimental_coord_translation_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_dataflow_api_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_fabric_1d_routing_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_fabric_2d_routing_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_fabric_api_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_fabric_topology_info_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_invoke_sfpi_llks_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_matmul_llks_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_pack_untilize_llks_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_padding_llks_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_reg_api_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_semaphore_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_tilize_llks_generated.h"
#include "ttmlir/Target/TTKernel/LLKs/experimental_untilize_llks_generated.h"
#include "ttmlir/Target/TTKernel/TTKernelIncludesMap.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/raw_ostream.h"

#include <set>

namespace mlir::tt::ttkernel {

// Class used to add includes and other boilerplate code to the generated
// kernel.
namespace {

class ScopedModuleHelper {
public:
  ScopedModuleHelper(OpBuilder *builder, Location loc, Region *region,
                     ThreadType threadType, StringRef originalSymbolName = "")
      : builder(builder), loc(loc) {
    if (!originalSymbolName.empty()) {
      emitComment(originalSymbolName);
    }

    std::set<llvm::StringRef> headers;

    // Baseline, always required.
    headers.insert("<cstdint>");
    switch (threadType) {
    case ThreadType::Compute:
      headers.insert("api/compute/compute_kernel_api.h");
      headers.insert("api/compute/common.h");
      break;
    case ThreadType::Noc:
      headers.insert("api/dataflow/dataflow_api.h");
      break;
    case ThreadType::Ethernet:
      break;
    }

    // TODO(wenbinlyuTT): make this conditional, originally from #4680.
    headers.insert("tools/profiler/kernel_profiler.hpp");

    const auto &headerMap = getCalleeToHeadersMap();

    auto insertHeaders = [&](const HeaderRequirement &reqs) {
      switch (threadType) {
      case ThreadType::Compute:
        if (!reqs.computeHeader.empty()) {
          headers.insert(reqs.computeHeader);
        }
        break;
      case ThreadType::Noc:
        if (!reqs.dataflowHeader.empty()) {
          headers.insert(reqs.dataflowHeader);
        }
        break;
      case ThreadType::Ethernet:
        break;
      }
    };

    bool hasDevicePrint = false;
    region->walk([&](emitc::CallOpaqueOp callOp) {
      llvm::StringRef callee = callOp.getCallee();

      auto it = headerMap.find(callee);
      if (it != headerMap.end()) {
        insertHeaders(it->second);
      }

      if (callee.starts_with("ttmlir::dprint")) {
        hasDevicePrint = true;
      }

      // Our experimental kernel code snippets.
      if (callee == "experimental::unpack_stall_on_pack") {
        emitLlk(experimental_reg_api_generated,
                experimental_reg_api_generated_len);
      }
      if (callee == "experimental::tilize_block") {
        emitLlk(experimental_tilize_llks_generated,
                experimental_tilize_llks_generated_len);
      }
      if (callee == "experimental::untilize_block") {
        emitLlk(experimental_untilize_llks_generated,
                experimental_untilize_llks_generated_len);
      }
      if (callee == "experimental::pack_untilize_block") {
        emitLlk(experimental_pack_untilize_llks_generated,
                experimental_pack_untilize_llks_generated_len);
      }
      if (callee == "experimental::get_noc_multicast_addr") {
        emitLlk(experimental_dataflow_api_generated,
                experimental_dataflow_api_generated_len);
      }
      if (callee == "experimental::semaphore_wait" ||
          callee == "experimental::semaphore_wait_min") {
        emitLlk(experimental_semaphore_generated,
                experimental_semaphore_generated_len);
      }
      if (callee == "experimental::convert_logical_x_to_translated" ||
          callee == "experimental::convert_logical_y_to_translated") {
        emitLlk(experimental_coord_translation_generated,
                experimental_coord_translation_generated_len);
      }
      if (callee == "experimental::close_fabric_connections" ||
          callee == "experimental::setup_fabric_connections" ||
          callee == "experimental::get_my_device_id" ||
          callee == "experimental::fabric_fast_write_any_len" ||
          callee == "experimental::fabric_mcast_fast_write_any_len" ||
          callee == "experimental::fabric_sem_inc" ||
          callee == "experimental::fabric_mcast_sem_inc" ||
          callee == "experimental::get_my_logical_mesh_position" ||
          callee == "experimental::get_device_id_from_logical_mesh_position") {
        // Emit in order: topology_info -> routing -> api.
        // 1. Topology info.
        emitLlk(experimental_fabric_topology_info_generated,
                experimental_fabric_topology_info_generated_len);
        // 2. Routing functions.
        emitLlk(experimental_fabric_1d_routing_generated,
                experimental_fabric_1d_routing_generated_len);
        emitLlk(experimental_fabric_2d_routing_generated,
                experimental_fabric_2d_routing_generated_len);
        // 3. Fabric APIs.
        emitLlk(experimental_fabric_api_generated,
                experimental_fabric_api_generated_len);
      }
      if (callee == "experimental::matmul_block") {
        emitLlk(experimental_matmul_llks_generated,
                experimental_matmul_llks_generated_len);
      }
      if (callee == "experimental::write_row_mask_tile" ||
          callee == "experimental::write_col_mask_tile" ||
          callee == "experimental::fill_arange_tile") {
        emitLlk(experimental_padding_llks_generated,
                experimental_padding_llks_generated_len);
      }
    });

    if (hasDevicePrint) {
      headers.insert("api/debug/dprint.h");
      emitDebugPrint(threadType);
    }

    region->walk([&](emitc::VerbatimOp verbatimOp) {
      llvm::StringRef value = verbatimOp.getValue();

      // Some callees are embedded in VerbatimOps.
      for (const auto &[callee, reqs] : headerMap) {
        if (value.starts_with(callee)) {
          insertHeaders(reqs);
        }
      }

      if (value.starts_with("experimental::invoke_sfpi")) {
        emitLlk(experimental_invoke_sfpi_llks_generated,
                experimental_invoke_sfpi_llks_generated_len);
      }
    });

    // Insert default template parameters before "reduce.h" inclusion.
    if (headers.count("api/compute/reduce.h")) {
      builder->create<emitc::VerbatimOp>(loc,
                                         "#define REDUCE_OP PoolType::SUM");
      builder->create<emitc::VerbatimOp>(
          loc, "#define REDUCE_DIM ReduceDim::REDUCE_COL");
    }

    // Emit the headers.
    for (llvm::StringRef header : headers) {
      bool isStandard = false;
      if (header.starts_with("<") && header.ends_with(">")) {
        isStandard = true;
        header = header.drop_front(1).drop_back(1);
      }
      builder->create<emitc::IncludeOp>(loc, header, isStandard);
    }

    if (threadType == ThreadType::Compute) {
      // Helper for float-to-uint32 bit reinterpretation (used by scalar tile
      // ops).
      builder->create<emitc::VerbatimOp>(
          loc, "inline uint32_t float_to_bits(const float f) { "
               "uint32_t r; __builtin_memcpy(&r, &f, sizeof(r)); return r; }");
      // Define INFINITY if not available (needed for OOB masking with inf
      // fill).
      builder->create<emitc::VerbatimOp>(
          loc, "#ifndef INFINITY\n#define INFINITY __builtin_inff()\n#endif");
    }

    // Emit all LLKs and custom functions AFTER the headers.
    for (llvm::StringRef snippet : llksToEmit) {
      builder->create<emitc::VerbatimOp>(loc, snippet);
    }
  }

  ~ScopedModuleHelper() = default;

  void emitComment(StringRef str) {
    builder->create<emitc::VerbatimOp>(loc, (Twine("// ") + str).str());
  }

  void emitLlk(const char *generated, unsigned int len) {
    llvm::StringRef snippet(generated, len);
    // Prevent duplicated emissions.
    if (!emittedLlks.insert(snippet).second) {
      return;
    }
    llksToEmit.push_back(snippet);
  }

  void emitDebugPrint(ThreadType threadType) {
    llksToEmit.push_back(R""""(
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

    if (threadType == ThreadType::Compute) {
      llksToEmit.push_back(R""""(
    namespace ttmlir {
      inline void print_cb_details_(DebugPrinter dp, uint32_t cb_id) {
      dp << "cb_id " << cb_id << ": { ";
      dp << "size: " << get_local_cb_interface(cb_id).fifo_size << ", ";
      dp << "limit: " << get_local_cb_interface(cb_id).fifo_limit << ", ";
      dp << "page_size: " << get_local_cb_interface(cb_id).fifo_page_size << ", ";
      dp << "num_pages: " << get_local_cb_interface(cb_id).fifo_num_pages << ", ";
      dp << "rd_ptr: " << get_local_cb_interface(cb_id).fifo_rd_ptr << ", ";
      dp << "wr_ptr: " << get_local_cb_interface(cb_id).fifo_wr_ptr << ", ";
      dp << "wr_tile_ptr: " << get_local_cb_interface(cb_id).fifo_wr_tile_ptr;
      dp << " }";
    }

    struct CBPrinter {
        uint32_t cb_id;

        constexpr CBPrinter(uint32_t cb_id) : cb_id(cb_id) {}
    };

    DebugPrinter operator<<(DebugPrinter dp, CBPrinter cb) {
        UNPACK((print_cb_details_(dp, cb.cb_id)));
        MATH(DPRINT << cb.cb_id);
        PACK((print_cb_details_(dp, cb.cb_id)));
        return dp;
    }
    } // namespace ttmlir
    )"""");
    }
  }

private:
  OpBuilder *builder;
  Location loc;
  std::set<llvm::StringRef> emittedLlks;
  llvm::SmallVector<llvm::StringRef> llksToEmit;
};
} // namespace

static FailureOr<mlir::ModuleOp>
cloneEntryIntoStandaloneModule(func::FuncOp origEntry, ThreadType threadType) {
  auto *ctx = origEntry.getContext();
  mlir::DialectRegistry registry;
  registry.insert<mlir::emitc::EmitCDialect>();
  ctx->appendDialectRegistry(registry);
  ctx->loadDialect<mlir::emitc::EmitCDialect>();
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
