// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_cache.h"

#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"
#include <cstddef>
#include <tt_stl/reflection.hpp>
// #include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"
#include "mlir/Pass/PassManager.h"
#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/RegisterAll.h"
#include "ttnn/tensor/tensor.hpp"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn::jit {

JitCache::JitCache(std::size_t cacheSize) {
  cache.reserve(cacheSize);
  mlir::tt::registerAllPasses();
  mlir::tt::registerAllExtensions(registry);
}

std::shared_ptr<void> JitCache::get(Operation *op, const JitCacheKey &key,
                                    const ::ttnn::Tensor &tensor_arg,
                                    std::string options) {
  std::size_t hash = hash_key(key, tensor_arg);

  if (cache.contains(hash)) {
    llvm::outs() << "cache hit \n";
    return cache[hash].flatbuffer_binary;
  }

  llvm::outs() << "cache miss \n";
  mlir::PassManager pm(op->getName());
  mlir::MLIRContext *context = op->getContext();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  // RTTI linking issues if we don't use pipeline lookup.
  // mlir::tt::ttmetal::createTTIRToTTMetalPipeline(pm, metal_options);
  pm.addPass(tt::createConvertTTNNToTTIRPass());
  const auto *pipeline =
      mlir::PassPipelineInfo::lookup("ttir-to-ttmetal-pipeline");
  if (!pipeline) {
    throw std::runtime_error("Failed to find ttir-to-ttmetal-pipeline.");
  }
  std::function<mlir::LogicalResult(const llvm::Twine &)> errHandler =
      [](const llvm::Twine &) { return mlir::failure(); };
  if (mlir::failed(pipeline->addToPipeline(pm, options, errHandler))) {
    throw std::runtime_error("Failed to add pipeline to pass manager");
  }
  if (mlir::failed(pm.run(op))) {
    throw std::runtime_error("Failed to run pass manager");
  }

  std::shared_ptr<void> flatbuffer_binary = ttnnToFlatbuffer(op);
  JitCacheEntry entry = {flatbuffer_binary};
  cache[hash] = entry;
  return flatbuffer_binary;
}

std::size_t JitCache::hash_key(const JitCacheKey &key,
                               const ::ttnn::Tensor &tensor_arg) const {
  return ttsl::hash::hash_objects_with_default_seed(key.func_name, key.backend,
                                                    key.max_grid, tensor_arg);
}

} // namespace mlir::tt::ttnn::jit
