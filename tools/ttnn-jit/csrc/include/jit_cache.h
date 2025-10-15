// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef JIT_CACHE_H
#define JIT_CACHE_H

#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"

#include "ttnn/tensor/tensor.hpp"

namespace mlir::tt::ttnn::jit {

struct JitCacheEntry {
  std::shared_ptr<void> flatbuffer_binary;
};

struct JitCacheKey {
  std::string func_sig;
  std::string backend;
  std::tuple<uint32_t, uint32_t> max_grid;
};

class JitCache {
public:
  explicit JitCache(std::size_t cacheSize);

  JitCache(const JitCache &) = delete;
  JitCache(JitCache &&) = delete;
  JitCache &operator=(const JitCache &) = delete;
  JitCache &operator=(JitCache &&) = delete;

  std::shared_ptr<void>
  get(Operation *op, const JitCacheKey &key,
      const std::vector<::ttnn::Tensor> &tensor_args,
      const std::vector<std::variant<int, bool, float, std::string>> &params,
      std::string options = "");
  uint32_t get_cache_hits() const { return cache_hits; }

private:
  std::size_t hash_key(
      const JitCacheKey &key, const std::vector<::ttnn::Tensor> &tensor_args,
      const std::vector<std::variant<int, bool, float, std::string>> &params)
      const;

  llvm::DenseMap<std::size_t, JitCacheEntry> cache;
  mlir::DialectRegistry registry;
  uint32_t cache_hits = 0;
};

} // namespace mlir::tt::ttnn::jit

#endif
