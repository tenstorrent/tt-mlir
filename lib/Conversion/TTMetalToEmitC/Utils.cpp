// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/TTMetalToEmitC/Utils.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttmetal_to_emitc::utils {

// Returns the closest parent module of the given operation
mlir::ModuleOp getParentModule(mlir::Operation *op) {
  while (op) {
    if (auto moduleOp = llvm::dyn_cast<mlir::ModuleOp>(op)) {
      return moduleOp;
    }
    op = op->getParentOp();
  }
  return nullptr;
}

bool insertRuntimeHelperFunctionsIfNotExists(PatternRewriter &rewriter,
                                              Operation *op) {
  ModuleOp moduleOp = getParentModule(op);
  assert(moduleOp && "Could not find top-level module");

  static constexpr const char *runtimeHelperFunctionsAsStr = R"(
// TTMetal Runtime Helper Functions
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>

class TTMetalRuntimeContext {
private:
  static inline tt::tt_metal::Device* device_ = nullptr;
  static inline tt::tt_metal::CommandQueue* queue_ = nullptr;

public:
  static void initialize() {
    if (!device_) {
      device_ = tt::tt_metal::CreateDevice(0);
      queue_ = &tt::tt_metal::CommandQueue::create_queue(device_->id());
    }
  }
  
  static tt::tt_metal::Device* getDevice() { 
    initialize(); 
    return device_; 
  }
  
  static tt::tt_metal::CommandQueue* getQueue() { 
    initialize(); 
    return queue_; 
  }
};

tt::tt_metal::Buffer* util_create_buffer(size_t size_bytes) {
  auto device = TTMetalRuntimeContext::getDevice();
  return tt::tt_metal::CreateBuffer(
    tt::tt_metal::InterleavedBufferConfig{
      device,
      size_bytes,
      sizeof(uint32_t),
      tt::tt_metal::BufferType::DRAM
    }
  );
}

void util_deallocate_buffer(tt::tt_metal::Buffer* buffer) {
  if (buffer) {
    tt::tt_metal::DeallocateBuffer(*buffer);
  }
}

TTMetalRuntimeContext util_create_runtime_context() {
  return TTMetalRuntimeContext{};
}
)";

  // Check if helper functions already exist
  for (auto &currOp : moduleOp.getOps()) {
    if (auto verbatimOp = dyn_cast<emitc::VerbatimOp>(currOp)) {
      if (verbatimOp.getValue() == runtimeHelperFunctionsAsStr) {
        return false; // Already exists
      }
    }
  }

  // Insert helper functions at the beginning of the module
  rewriter.setInsertionPointToStart(&moduleOp.getBodyRegion().front());
  rewriter.create<emitc::VerbatimOp>(moduleOp.getLoc(), runtimeHelperFunctionsAsStr);

  return true;
}

emitc::OpaqueAttr convertCoreRange(Builder &builder, mlir::Attribute attr) {
  // TODO: Implement proper CoreRange attribute conversion
  return builder.getAttr<emitc::OpaqueAttr>("tt::tt_metal::CoreRange{{0, 0}, {1, 1}}");
}

emitc::OpaqueAttr convertKernelConfig(Builder &builder, mlir::Attribute attr) {
  // TODO: Implement kernel config conversion based on the specific attributes
  // This is a placeholder that would need to be expanded based on the actual
  // kernel configuration structure in TTMetal dialect
  return builder.getAttr<emitc::OpaqueAttr>("tt::tt_metal::ComputeConfig{}");
}

emitc::OpaqueAttr createStdNullopt(Builder &builder) {
  return builder.getAttr<emitc::OpaqueAttr>("::std::nullopt");
}

emitc::CallOpaqueOp createRuntimeContextOp(ConversionPatternRewriter &rewriter,
                                           Location loc) {
  auto contextType = emitc::OpaqueType::get(
      rewriter.getContext(), "TTMetalRuntimeContext");
      
  return rewriter.create<emitc::CallOpaqueOp>(
      loc, contextType, kCreateRuntimeContextFunctionName,
      /*args=*/nullptr, /*template_args=*/nullptr, ValueRange{});
}

} // namespace mlir::tt::ttmetal_to_emitc::utils