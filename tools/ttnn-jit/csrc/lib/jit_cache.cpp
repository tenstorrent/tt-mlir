// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_cache.h"

#include "mlir/Pass/PassManager.h"
#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"
#include "ttnn/tensor/tensor.hpp"

#include <cstddef>
#include <tt_stl/reflection.hpp>
namespace mlir::tt::ttnn::jit {

JitCache::JitCache(std::size_t cacheSize) {
  cache.reserve(cacheSize);
  mlir::tt::registerAllExtensions(registry);
}

void JitCache::compile(Operation *op, std::string options) {
  mlir::PassManager pm(op->getName());
  mlir::MLIRContext *context = op->getContext();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();
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
}

JitCacheEntry JitCache::get(Operation *op,
                            const std::vector<::ttnn::Tensor> &tensor_args,
                            std::string options) {

  std::size_t hash = hash_key(tensor_args);
  auto it = cache.find(hash);
  if (it != cache.end()) {
    cache_hits++;
    return it->second;
  }
  compile(op, options);
  JitCacheEntry flatbuffer_binary = ttnnToFlatbuffer(op);
  cache.try_emplace(hash, flatbuffer_binary);
  return flatbuffer_binary;
}

std::size_t
JitCache::hash_key(const std::vector<::ttnn::Tensor> &tensor_args) const {
  return ttsl::hash::hash_objects_with_default_seed(tensor_args);
}

} // namespace mlir::tt::ttnn::jit
