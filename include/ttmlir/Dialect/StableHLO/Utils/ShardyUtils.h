// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDYUTILS_H
#define TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDYUTILS_H

#include "ttmlir/Dialect/StableHLO/Transforms/ShardyCCLToStableHLOCCL.h"
#include "ttmlir/Dialect/StableHLO/Utils/ShardingUtils.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdefaulted-function-deleted"
#include "shardy/dialect/sdy/transforms/propagation/aggressive_propagation.h"
#pragma clang diagnostic pop

namespace mlir::tt::shardy_utils {

#ifdef TTMLIR_ENABLE_STABLEHLO

// We used MapVector because we want the order to be preserved when inserting
// axes into the meshMap.
using MeshMap =
    llvm::MapVector</*axis_name*/ std::string, /*axis_size*/ int64_t,
                    std::map<std::string, unsigned>,
                    std::vector<std::pair<std::string, int64_t>>>;

// Get all the meshOps from the module.
llvm::SmallVector<mlir::sdy::MeshOp> getMeshOps(mlir::ModuleOp &module);

// Get all mesh names from a function, returning empty vector if none found.
llvm::SmallVector<std::string> getMeshNames(mlir::func::FuncOp &funcOp);

// Remove all meshOps from the module.
void removeMeshOps(mlir::ModuleOp &module);

// Create a meshAttr from a meshMap helper function.
mlir::sdy::MeshAttr createMeshAttrFromMeshMap(MLIRContext *context,
                                              MeshMap &meshMap);

// Create a meshMap from a meshAttr helper function.
MeshMap createMeshMapFromMeshAttr(mlir::sdy::MeshAttr meshAttr);

// Get mesh shape from meshAttr.
llvm::SmallVector<int64_t>
getMeshShapeFromMeshAttr(mlir::sdy::MeshAttr meshAttr);

// Insert a mesh into the module.
void addMeshToModule(mlir::ModuleOp &module, std::string meshName,
                     std::string firstAxisName, std::string secondAxisName,
                     int64_t firstAxisSize, int64_t secondAxisSize);

// Create a TTMeshAttr from a sdy::meshOp.
mlir::tt::ttcore::MeshAttr
createTTMeshAttrFromSdyMeshOp(mlir::sdy::MeshOp meshOp);

// Check if the module has any sdy tensor sharding annotations.
bool sdyAnnotationsExist(mlir::ModuleOp &module);

// Parse dimension shardings from a string representation.
llvm::SmallVector<mlir::sdy::DimensionShardingAttr>
parseDimensionShardings(const std::string &dimsContent,
                        mlir::MLIRContext *context);

// Convert function argument from mhlo.frontend_attributes to sdy.sharding.
mlir::LogicalResult convertArgumentSharding(mlir::func::FuncOp &funcOp,
                                            mlir::BlockArgument &arg,
                                            mlir::MLIRContext *context);

// Convert dictionary with frontend attributes to dictionary with sdy.sharding.
mlir::DictionaryAttr
convertXlaSdyToSdyDictionary(mlir::MLIRContext *context,
                             mlir::DictionaryAttr currentArgAttrDict);

// Convert all function arguments from frontend attributes format to SDY format.
mlir::LogicalResult convertFrontendAttributesToSDY(mlir::ModuleOp &rootModule,
                                                   mlir::MLIRContext *context);

// Convert all stablehlo.custom_call @Sharding ops to sdy.sharding_constraint
// ops.
mlir::LogicalResult
convertCustomCallToShardingConstraint(mlir::ModuleOp &rootModule,
                                      mlir::MLIRContext *context,
                                      mlir::OpBuilder &builder);

// Check if the graph is solved.
bool isGraphSolved(mlir::ModuleOp &module);

// Create a new DictionaryAttr from an old DictionaryAttr with all sdy.sharding
// annotations removed.
mlir::DictionaryAttr
removeDictionaryAttrSdyShardingAnnotations(MLIRContext *context,
                                           mlir::DictionaryAttr dictAttr);

// Remove all sdy tensor shardings from the module.
void removeSdyTensorShardings(MLIRContext *context, func::FuncOp &funcOp);

// Create a new DictionaryAttr (from an old DictionaryAttr if provided) and add
// a sdy.sharding annotation to it.
mlir::DictionaryAttr addDictionaryAttrSdyShardingAnnotation(
    MLIRContext *context, mlir::sdy::TensorShardingAttr shardingAttr,
    std::optional<mlir::DictionaryAttr> dictAttr = std::nullopt);

// Get a default sdy.sharding annotation (ie all dimensions are open and
// replicated).
mlir::sdy::TensorShardingAttr
getDefaultTensorSdyShardingAttr(MLIRContext *context, llvm::StringRef meshName,
                                mlir::Type type);

// Get the argument sharding attributes.
llvm::SmallVector<mlir::sdy::TensorShardingAttr>
getInShardingAttrs(MLIRContext *context, func::FuncOp &funcOp,
                   mlir::sdy::MeshOp &globalMeshOp);

// Get the result sharding attributes.
llvm::SmallVector<mlir::sdy::TensorShardingAttr>
getOutShardingAttrs(MLIRContext *context, func::FuncOp &funcOp,
                    mlir::sdy::MeshOp &globalMeshOp);

// Get the sharding attribute for an operand.
mlir::sdy::TensorShardingAttr
getOperandShardingAttr(const mlir::OpOperand &operand,
                       mlir::sdy::MeshOp globalMeshOp);

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

class ShardyMeshSharding : public sharding_utils::MeshSharding {
public:
  // Static factory methods.
  static llvm::Expected<ShardyMeshSharding> generateDefault();
  static llvm::Expected<ShardyMeshSharding>
  generate(sdy::MeshAttr meshAttr, sdy::TensorShardingAttr sdySharding,
           mlir::tt::ttcore::ShardStatus shardStatus,
           ttcore::MeshShardDirection shardDirection);
  ShardyMeshSharding(mlir::tt::ttcore::MeshShardDirection shardDirection,
                     mlir::tt::ttcore::MeshShardType shardType,
                     const llvm::SmallVector<int64_t> &shardShape,
                     const llvm::SmallVector<int64_t> &shardDims,
                     const llvm::SmallVector<int64_t> &meshShape,
                     const llvm::SmallVector<int64_t> &deviceIds,
                     mlir::tt::ttcore::ShardStatus shardStatus,
                     sdy::MeshAttr meshAttr,
                     sdy::TensorShardingAttr sdySharding)
      : MeshSharding(shardDirection, shardType, shardShape, shardDims,
                     meshShape, deviceIds, shardStatus),
        meshAttr(meshAttr), sdySharding(sdySharding) {}

  // Getters
  mlir::sdy::MeshAttr getMeshAttr() const { return meshAttr; }
  mlir::sdy::TensorShardingAttr getSdySharding() const { return sdySharding; }

private:
  // Member variables
  mlir::sdy::MeshAttr meshAttr;
  mlir::sdy::TensorShardingAttr sdySharding;
};

// Return true if every dimension has no axes -> replicated.
bool isFullyReplicatedTensor(mlir::sdy::TensorShardingAttr tsh);

#endif // #ifdef TTMLIR_ENABLE_STABLEHLO

} // namespace mlir::tt::shardy_utils

#endif // TTMLIR_DIALECT_STABLEHLO_UTILS_SHARDYUTILS_H
