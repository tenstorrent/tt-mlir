// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
#define TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::sharding_utils {

#if defined(TTMLIR_ENABLE_STABLEHLO) && (TTMLIR_ENABLE_STABLEHLO != 0)

class MeshSharding {
public:
  MeshSharding() {};
  ~MeshSharding() {};

  // Convert mhlo.sharding string to meshSharding.
  llvm::Expected<bool>
  convertGSPMDShardingToMeshSharding(StringRef shardingStr);

  // Check and update arg sharding attribute and determine if mesh_shard op
  // needs to be created or not.
  bool checkAndUpdateGSPMDArgSharding(mlir::PatternRewriter &rewriter,
                                      mlir::stablehlo::CustomCallOp srcOp,
                                      mlir::StringAttr shardingAttr);

  // Check and update ret sharding attribute and determine if mesh_shard op
  // needs to be created or not.
  bool checkAndUpdateGSPMDRetSharding(mlir::PatternRewriter &rewriter,
                                      mlir::stablehlo::CustomCallOp srcOp,
                                      mlir::StringAttr shardingAttr);

  // Convert sdy.sharding to meshSharding.
  llvm::Expected<bool>
  convertSdyShardingToMeshSharding(mlir::sdy::TensorShardingAttr sdySharding,
                                   mlir::sdy::MeshAttr mesh,
                                   mlir::tt::MeshShardDirection direction);

  // Check and update arg sharding attribute and determine if
  // mesh_shard op needs to be created or not.
  bool checkAndUpdateShardyArgSharding(
      mlir::PatternRewriter &rewriter, mlir::func::FuncOp funcOp,
      mlir::Value argOperand, mlir::sdy::TensorShardingAttr shardingAttr);

  // Check and update ret sharding attribute and determine if mesh_shard op
  // needs to be created or not.
  bool
  checkAndUpdateShardyRetSharding(mlir::PatternRewriter &rewriter,
                                  mlir::func::FuncOp funcOp, uint64_t retIdx,
                                  mlir::sdy::TensorShardingAttr shardingAttr);

  // Parse Shardy sharing attribuite. Make this as public for
  // potential frontend use.
  llvm::Expected<bool>
  parseSdySharding(mlir::sdy::TensorShardingAttr sdySharding,
                   mlir::sdy::MeshAttr meshAttr);

  // Given parsed MeshSharding info, get TensorMeshShardingAttr.
  mlir::tt::TensorMeshShardingAttr
  getTensorMeshShardingAttr(mlir::PatternRewriter &rewriter);

  // Given TensorMeshShardingAttr, extract MeshSharding info.
  void extractMeshShardingFromTensorMeshShardingAttr(
      mlir::tt::MeshAttr mesh,
      mlir::tt::TensorMeshShardingAttr tensorMeshShardingAttr,
      mlir::tt::MeshShardDirection direction);

  // Getter functions.
  mlir::tt::MeshShardDirection getShardDirection() const {
    return shardDirection;
  }

  // Needed by the frontend (tt-xla) to create a 'strategy' map for creating
  // multiDevice tensors based on the MeshSharding object attached to it. This
  // is needed fren the frontend generates a multiDevice tensor as input.
  static FailureOr<std::unordered_map<std::string, std::string>>
  fillStrategyMapFromSharding(
      const mlir::tt::sharding_utils::MeshSharding &meshSharding,
      size_t num_devices);

  mlir::tt::MeshShardType getShardType() const { return shardType; }
  llvm::ArrayRef<int64_t> getShardShape() const { return shardShape; }
  llvm::ArrayRef<int64_t> getShardDims() const { return shardDims; }
  llvm::ArrayRef<int64_t> getMeshShape() const { return meshShape; }

private:
  // Parse GSPMD devices string and fill out MeshSharding info.
  llvm::Expected<bool> parseGSPMDDevicesStr(StringRef devicesStr);

  // Based on current MeshSharding info, finalize sharding dimensions.
  llvm::Expected<bool> determineGSPMDShardingDims();

  // Set sharyType other than devices and reset values.
  void setNonDevicesShardType(tt::MeshShardType targetShardType) {
    assert(targetShardType != tt::MeshShardType::Devices);
    shardType = targetShardType;
    // Specific values are required to fill corresponding attributes in
    // mesh_shard operation.
    shardShape = llvm::SmallVector<int64_t>{1};
    shardDims = llvm::SmallVector<int64_t>{-1};
    meshShape = llvm::SmallVector<int64_t>{-1};
  }

  // Force dummy sharding op by setting shard_type to identity. The mesh_shard
  // op will be ignored at runtime by simply copying input tensor to output.
  void setDummyShardingOp() { shardType = mlir::tt::MeshShardType::Identity; }

  // Decide wheter to create mesh_shard op and shard_type. Detailed
  // description can be found in function body.
  bool determineMeshShardOpCreationAndShardType(bool foundArgSharding);

private:
  mlir::tt::MeshShardDirection shardDirection =
      mlir::tt::MeshShardDirection::ShardToFull;
  mlir::tt::MeshShardType shardType = mlir::tt::MeshShardType::Identity;
  llvm::SmallVector<int64_t> shardShape{-1};
  llvm::SmallVector<int64_t> shardDims{-1};
  llvm::SmallVector<int64_t> meshShape{-1};
  llvm::SmallVector<int64_t> deviceIds{-1};
  std::string meshName;
  bool lastTileDimReplicate = false;
};

inline mlir::sdy::MeshAttr adjustSdyMeshAttr(mlir::Operation *srcOp,
                                             mlir::sdy::MeshAttr sdyMesh) {
  if (sdyMesh.getAxes().size() == 1) {
    // 1D (N) mesh is always extended to 2D (1, N) mesh.
    sdyMesh = mlir::sdy::MeshAttr::get(
        srcOp->getContext(),
        llvm::ArrayRef<mlir::sdy::MeshAxisAttr>(
            {mlir::sdy::MeshAxisAttr::get(srcOp->getContext(), "unit", 1),
             sdyMesh.getAxes().front()}));
  }
  return sdyMesh;
}

inline mlir::RankedTensorType addTensorMeshShardingAttrToRankedTensorType(
    mlir::RankedTensorType type,
    mlir::tt::TensorMeshShardingAttr tensorMeshShardingAttr) {
  mlir::Type elementType = type.getElementType();
  llvm::ArrayRef<int64_t> shape = type.getShape();
  return RankedTensorType::get(shape, elementType, tensorMeshShardingAttr);
}

inline mlir::Type addTensorMeshShardingAttrToValue(
    mlir::Value value,
    mlir::tt::TensorMeshShardingAttr tensorMeshShardingAttr) {
  auto updatedType = addTensorMeshShardingAttrToRankedTensorType(
      mlir::cast<RankedTensorType>(value.getType()), tensorMeshShardingAttr);
  value.setType(updatedType);
  return updatedType;
}

inline void addTensorMeshShardingAttrToFunctionArg(
    mlir::func::FuncOp funcOp, int64_t argIdx,
    mlir::tt::TensorMeshShardingAttr tensorMeshShardingAttr) {
  // Add TensorMeshShardingAttr to function arg.
  addTensorMeshShardingAttrToValue(funcOp.getArgument(argIdx),
                                   tensorMeshShardingAttr);
  // Update function signature.
  auto funcOpType = funcOp.getFunctionType();
  auto inputTypes = llvm::SmallVector<Type>(funcOpType.getInputs());
  inputTypes[argIdx] = addTensorMeshShardingAttrToRankedTensorType(
      mlir::cast<RankedTensorType>(inputTypes[argIdx]), tensorMeshShardingAttr);
  funcOp.setType(FunctionType::get(funcOpType.getContext(), inputTypes,
                                   funcOpType.getResults()));
}

inline void addTensorMeshShardingAttrToFunctionRet(
    mlir::func::FuncOp funcOp, int64_t retIdx,
    mlir::tt::TensorMeshShardingAttr tensorMeshShardingAttr) {
  // Add TensorMeshShardingAttr to function return.
  auto *funcReturnOp = funcOp.getBody().front().getTerminator();
  addTensorMeshShardingAttrToValue(funcReturnOp->getOperand(retIdx),
                                   tensorMeshShardingAttr);
  // Update function signature.
  auto funcOpType = funcOp.getFunctionType();
  auto resultTypes = llvm::SmallVector<Type>(funcOpType.getResults());
  resultTypes[retIdx] = addTensorMeshShardingAttrToRankedTensorType(
      mlir::cast<RankedTensorType>(resultTypes[retIdx]),
      tensorMeshShardingAttr);
  funcOp.setType(FunctionType::get(funcOpType.getContext(),
                                   funcOpType.getInputs(), resultTypes));
}

// Remove arg sharding and return true if it is found, otherwise return false.
template <typename AttrType>
bool checkAndRemoveFuncArgSharding(
    mlir::PatternRewriter &rewriter, mlir::func::FuncOp funcOp, uint64_t argIdx,
    AttrType shardingAttr,
    mlir::tt::TensorMeshShardingAttr tensorMeshShardingAttr,
    llvm::StringRef argShardingStrRef) {
  auto argShardingAttr =
      funcOp.getArgAttrOfType<AttrType>(argIdx, argShardingStrRef);
  if (!argShardingAttr) {
    return false;
  }
  if (argShardingAttr != shardingAttr) {
    llvm_unreachable("MeshSharding operation and function argument shardings "
                     "are different.");
  }
  rewriter.modifyOpInPlace(funcOp, [&]() {
    funcOp.removeArgAttr(argIdx, argShardingStrRef);
    if (tensorMeshShardingAttr) {
      addTensorMeshShardingAttrToFunctionArg(funcOp, argIdx,
                                             tensorMeshShardingAttr);
    }
  });

  return true;
}

// Remove ret sharding and return true if it is found, otherwise return false.
template <typename AttrType>
bool checkAndRemoveFuncReturnSharding(
    mlir::PatternRewriter &rewriter, mlir::func::FuncOp funcOp, uint64_t retIdx,
    AttrType shardingAttr,
    mlir::tt::TensorMeshShardingAttr tensorMeshShardingAttr,
    llvm::StringRef retShardingStrRef) {
  auto retShardingAttr =
      funcOp.getResultAttrOfType<AttrType>(retIdx, retShardingStrRef);
  if (!retShardingAttr) {
    return false;
  }
  if (retShardingAttr != shardingAttr) {
    llvm_unreachable("MeshSharding operation and function return shardings "
                     "are different.");
  }
  rewriter.modifyOpInPlace(funcOp, [&]() {
    funcOp.removeResultAttr(
        retIdx, mlir::StringAttr::get(funcOp->getContext(), retShardingStrRef));
    if (tensorMeshShardingAttr) {
      addTensorMeshShardingAttrToFunctionRet(funcOp, retIdx,
                                             tensorMeshShardingAttr);
    }
  });
  return true;
}

// Sharding related string definitions from open-xla
// https://github.com/openxla/xla/blob/main/xla/service/spmd/shardy/constants.h

inline constexpr llvm::StringRef kShardingCustomCallTargetName = "Sharding";
inline constexpr llvm::StringRef kSPMDFullToShardShapeCallTargetName =
    "SPMDFullToShardShape";
inline constexpr llvm::StringRef kSPMDShardToFullShapeCallTargetName =
    "SPMDShardToFullShape";
inline constexpr llvm::StringRef kXlaShardingAttr = "mhlo.sharding";

#endif

} // namespace mlir::tt::sharding_utils

#endif // TTMLIR_CONVERSION_STABLEHLOTOTTIR_SHARDINGUTILS_H
