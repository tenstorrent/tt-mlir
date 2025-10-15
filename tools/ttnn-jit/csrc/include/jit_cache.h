// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef JIT_CACHE_H
#define JIT_CACHE_H

#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Format.h"

#include "ttnn/tensor/tensor.hpp"

namespace mlir::tt::ttnn::jit {

struct JitCacheEntry {
  std::shared_ptr<void> flatbuffer_binary;
};

struct JitCacheKey {
  std::string func_name;
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

  std::shared_ptr<void> get(Operation *op, const JitCacheKey &key,
                            const ::ttnn::Tensor &tensor_arg,
                            std::string options = "");

private:
  std::size_t hash_key(const JitCacheKey &key,
                       const ::ttnn::Tensor &tensor_arg) const;
  llvm::DenseMap<std::size_t, JitCacheEntry> cache;
  mlir::DialectRegistry registry;
};

} // namespace mlir::tt::ttnn::jit

#endif
