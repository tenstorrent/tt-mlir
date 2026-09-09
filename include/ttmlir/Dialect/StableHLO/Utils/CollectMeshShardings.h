// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Builds `MeshSharding` instances for a func.func's args/results so consumers
// share one Shardy-vs-GSPMD dispatch and one shard-status deduction.

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_COLLECTMESHSHARDINGS_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_COLLECTMESHSHARDINGS_H

#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace mlir::tt::sharding_utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

// Returned vector is index-aligned with the FuncOp arg/result list;
// boundaries without a sharding annotation come back as Replicate/Unsharded.
llvm::Expected<llvm::SmallVector<MeshSharding, 0>>
collectArgMeshShardings(mlir::func::FuncOp funcOp);

llvm::Expected<llvm::SmallVector<MeshSharding, 0>>
collectResultMeshShardings(mlir::func::FuncOp funcOp);

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::sharding_utils

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_COLLECTMESHSHARDINGS_H
