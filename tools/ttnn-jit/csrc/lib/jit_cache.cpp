// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_cache.h"

#include "mlir/Pass/PassManager.h"
#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/RegisterAll.h"
#include "ttmlir/Target/TTNN/TTNNToFlatbuffer.h"
#include "ttnn/tensor/tensor.hpp"
// #include "ttmlir/Dialect/TTMetal/Pipelines/TTMetalPipelines.h"

#include <cstddef>
#include <tt_stl/reflection.hpp>
namespace mlir::tt::ttnn::jit {

JitCache::JitCache(std::size_t cacheSize) {
  cache.reserve(cacheSize);
  mlir::tt::registerAllPasses();
  mlir::tt::registerAllExtensions(registry);
}

void JitCache::compile(Operation *op, std::string options) {
  mlir::PassManager pm(op->getName());
  mlir::MLIRContext *context = op->getContext();
  context->appendDialectRegistry(registry);
  context->loadAllAvailableDialects();

  // RTTI linking issues if we don't use pipeline lookup.
  // mlir::tt::ttmetal::createTTIRToTTMetalPipeline(pm,
  // mlir::tt::ttmetal::TTIRToTTMetalPipelineOptions());
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

std::shared_ptr<void> JitCache::get(
    Operation *op, const JitCacheKey &key,
    const std::vector<::ttnn::Tensor> &tensor_args,
    const std::vector<std::variant<int, bool, float, std::string>> &params,
    std::string options) {

  std::size_t hash = hash_key(key, tensor_args, params);
  auto it = cache.find(hash);
  if (it != cache.end()) {
    cache_hits++;
    return it->second.flatbuffer_binary;
  }
  compile(op, options);
  std::shared_ptr<void> flatbuffer_binary = ttnnToFlatbuffer(op);
  cache.try_emplace(hash, JitCacheEntry{flatbuffer_binary});
  return flatbuffer_binary;
}

std::size_t JitCache::hash_key(
    const JitCacheKey &key, const std::vector<::ttnn::Tensor> &tensor_args,
    const std::vector<std::variant<int, bool, float, std::string>> &params)
    const {
  return ttsl::hash::hash_objects_with_default_seed(
      key.func_sig, key.backend, key.max_grid, tensor_args, params);
}

} // namespace mlir::tt::ttnn::jit
