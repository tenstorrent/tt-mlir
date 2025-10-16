// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef JIT_CACHE_H
#define JIT_CACHE_H

#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"

#include "ttnn/tensor/tensor.hpp"

namespace mlir::tt::ttnn::jit {

using JitCacheEntry = std::shared_ptr<void>;

class JitCache {
public:
  explicit JitCache(std::size_t cacheSize);

  JitCache(const JitCache &) = delete;
  JitCache(JitCache &&) = delete;
  JitCache &operator=(const JitCache &) = delete;
  JitCache &operator=(JitCache &&) = delete;

  JitCacheEntry get(Operation *op,
                    const std::vector<::ttnn::Tensor> &tensor_args,
                    std::string options = "");
  std::size_t num_entries() const { return this->cache.size(); }

private:
  std::size_t hash_key(const std::vector<::ttnn::Tensor> &tensor_args) const;
  void compile(Operation *op, std::string options);
  llvm::DenseMap<std::size_t, JitCacheEntry> cache;
  mlir::DialectRegistry registry;
};

} // namespace mlir::tt::ttnn::jit

#endif
