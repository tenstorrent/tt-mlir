// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "Edge.h"
#include "ttmlir/Dialect/TTNN/Analysis/OpConfig.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttnn {

struct MemReconfigEntry {
  // Map of consumer op config bit index to vector of valid producer op configs.
  using ReshardOutputConfigMap =
      llvm::DenseMap<std::size_t, llvm::SmallVector<OpConfig>>;

  ReshardOutputConfigMap reshardOutputConfigMap;
  ReshardOutputConfigMap::const_iterator reshardOutputConfigMapIter =
      reshardOutputConfigMap.end();
};

using MemReconfigEntryMap = llvm::DenseMap<Edge, MemReconfigEntry>;

} // namespace mlir::tt::ttnn
