// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/GenericInterchangeAnalysis.h"

#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir::tt::ttir {

SmallVector<int64_t> calculateOptimalInterchange(GenericOp op) {
  SmallVector<int64_t> interchange;
  interchange = llvm::to_vector(llvm::seq<int64_t>(0, op.getNumLoops()));
  return interchange;
}

} // namespace mlir::tt::ttir
