// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYUTILS_H
#define TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYUTILS_H

#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdefaulted-function-deleted"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#pragma clang diagnostic pop

#include "shardy/dialect/sdy/ir/constants.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "shardy/dialect/sdy/transforms/export/passes.h"
#include "shardy/dialect/sdy/transforms/propagation/op_sharding_rule_builder.h"
#include "shardy/dialect/sdy/transforms/propagation/passes.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::tt::shardy_utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

// We used MapVector because we want the order to be preserved when inserting
// axes into the meshMap.
using MeshMap =
    llvm::MapVector</*axis_name*/ llvm::StringRef, /*axis_size*/ int64_t>;

// Get all the meshOps from the module.
llvm::SmallVector<mlir::sdy::MeshOp> getMeshOps(mlir::ModuleOp &module);

// Remove all meshOps from the module.
void removeMeshOps(mlir::ModuleOp &module);

// Create a meshAttr from a meshMap helper function.
mlir::sdy::MeshAttr createMeshAttrFromMeshMap(MLIRContext *context,
                                              MeshMap &meshMap);

// Create a meshMap from a meshAttr helper function.
MeshMap createMeshMapFromMeshAttr(mlir::sdy::MeshAttr meshAttr);

// Check if the module has any sdy tensor sharding annotations.
bool sdyAnnotationsExist(mlir::ModuleOp &module);

// Check if the module has any gspmd annotations.
bool gspmdAnnotationsExist(mlir::ModuleOp &module);

// Check if any manual computation op exists in the graph.
bool doesManualComputationOpExist(mlir::ModuleOp &module);

// Remove all sdy tensor shardings from the module.
void removeSdyTensorShardings(MLIRContext *context, func::FuncOp &funcOp);

// Get the argument sharding attributes.
llvm::SmallVector<mlir::sdy::TensorShardingAttr>
getInShardingAttrs(MLIRContext *context, func::FuncOp &funcOp,
                   mlir::sdy::MeshOp &globalMeshOp);

// Get the result sharding attributes.
llvm::SmallVector<mlir::sdy::TensorShardingAttr>
getOutShardingAttrs(MLIRContext *context, func::FuncOp &funcOp,
                    mlir::sdy::MeshOp &globalMeshOp);

// Calculate the updated shape based on the tensor sharding annotation.
FailureOr<int64_t>
calculateUpdatedDim(mlir::sdy::MeshAttr meshAttr,
                    mlir::sdy::DimensionShardingAttr dimShardingAttr,
                    int64_t oldShapeDim);

// Calculate the new sharded output based on the sdy tensor sharding attribute.
FailureOr<mlir::RankedTensorType>
populateShardedOutputType(mlir::sdy::MeshAttr meshAttr,
                          mlir::RankedTensorType oldType,
                          mlir::sdy::TensorShardingAttr tensorShardingAttr);

// Loop through all the tensorShardings and apply them to each output.
FailureOr<llvm::SmallVector<mlir::RankedTensorType>> getNewResultTypes(
    mlir::Operation *op, mlir::sdy::MeshOp &globalMeshOp,
    llvm::ArrayRef<mlir::sdy::TensorShardingAttr> tensorShardings);

// Copy nested regions between srcOp and destOp
void copyNestedRegions(mlir::OpBuilder &builder, mlir::Operation *srcOp,
                       mlir::Operation *destOp);

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::shardy_utils

#endif // TTMLIR_DIALECT_STABLEHLO_TRANSFORMS_SHARDYUTILS_H
