// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_cache.h"

#include "mlir/Pass/PassManager.h"
#include "ttmlir/Conversion/Passes.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
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
  pm.addPass(tt::ttnn::createTTNNDeallocate());
  if (mlir::failed(pm.run(op))) {
    throw std::runtime_error("Failed to run pass manager");
  }
}

bool JitCache::contains(const std::vector<::ttnn::Tensor> &tensorArgs) const {
  std::size_t hash = hashKey(tensorArgs);
  return cache.contains(hash);
}

JitCacheEntry
JitCache::get(const std::vector<::ttnn::Tensor> &tensorArgs) const {
  std::size_t hash = hashKey(tensorArgs);
  auto it = cache.find(hash);
  if (it != cache.end()) {
    return it->second;
  }
  return nullptr;
}

JitCacheEntry
JitCache::compileAndInsert(Operation *op,
                           const std::vector<::ttnn::Tensor> &tensorArgs,
                           std::string options) {
  std::size_t hash = hashKey(tensorArgs);
  if (contains(tensorArgs)) {
    return get(tensorArgs);
  }
  compile(op, options);
  std::shared_ptr<void> flatbufferBytes = ttnnToFlatbuffer(op);
  JitCacheEntry binary = std::make_shared<::tt::runtime::Binary>(
      ::tt::runtime::Flatbuffer(flatbufferBytes));
  cache.try_emplace(hash, binary);
  return binary;
}

std::size_t
JitCache::hashKey(const std::vector<::ttnn::Tensor> &tensorArgs) const {
  return ttsl::hash::hash_objects_with_default_seed(tensorArgs);
}

} // namespace mlir::tt::ttnn::jit
