// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef JIT_CACHE_H
#define JIT_CACHE_H

#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"
#include "tt/runtime/types.h"

// Forward declarations to avoid including heavy tensor headers
namespace tt::tt_metal {
class Tensor;
} // namespace tt::tt_metal

namespace ttnn {
using Tensor = tt::tt_metal::Tensor;
} // namespace ttnn

namespace mlir::tt::ttnn::jit {

using JitCacheEntry = std::shared_ptr<::tt::runtime::Binary>;

class JitCache {
public:
  explicit JitCache(std::size_t cacheSize);

  JitCache(const JitCache &) = delete;
  JitCache(JitCache &&) = delete;
  JitCache &operator=(const JitCache &) = delete;
  JitCache &operator=(JitCache &&) = delete;

  bool contains(const std::vector<::ttnn::Tensor> &tensorArgs) const;
  JitCacheEntry get(const std::vector<::ttnn::Tensor> &tensorArgs) const;
  JitCacheEntry compileAndInsert(Operation *op,
                                 const std::vector<::ttnn::Tensor> &tensorArgs,
                                 std::string options = "");
  std::size_t numEntries() const { return this->cache.size(); }

private:
  std::size_t hashKey(const std::vector<::ttnn::Tensor> &tensorArgs) const;
  void compile(Operation *op, std::string options);
  llvm::DenseMap<std::size_t, JitCacheEntry> cache;
  mlir::DialectRegistry registry;
};

} // namespace mlir::tt::ttnn::jit

#endif
