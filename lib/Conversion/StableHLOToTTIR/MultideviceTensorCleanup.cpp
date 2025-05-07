// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Conversion/StableHLOToTTIR/MultideviceTensorCleanup.h"

#include "ttmlir/Conversion/StableHLOToTTIR/ShardingUtils.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt {

static bool checkIfValueIsFuncArg(mlir::func::FuncOp funcOp, mlir::Value arg) {
  if (!arg) {
    return false;
  }
  for (auto funcArg : funcOp.getArguments()) {
    if (arg == funcArg) {
      return true;
    }
  }
  return false;
}

llvm::LogicalResult insertMissingMeshShardOps(mlir::OpBuilder &builder,
                                              mlir::func::FuncOp &funcOp,
                                              mlir::tt::MeshesAttr meshes,
                                              bool multiDeviceFix = false) {

  mlir::Operation *funcReturnOp = funcOp.getBody().front().getTerminator();
  builder.setInsertionPoint(funcReturnOp);

  for (auto [retIdx, ret] : llvm::enumerate(funcReturnOp->getOperands())) {
    auto retType = mlir::cast<mlir::RankedTensorType>(ret.getType());
    auto retTensorMeshShardingAttr =
        mlir::dyn_cast_if_present<mlir::tt::TensorMeshShardingAttr>(
            retType.getEncoding());
    if (!retTensorMeshShardingAttr) {
      continue;
    }
    if (mlir::isa_and_present<mlir::tt::ttir::MeshShardOp>(
            ret.getDefiningOp()) ||
        checkIfValueIsFuncArg(funcOp, ret)) {
      continue;
    }

    mlir::tt::sharding_utils::MeshSharding meshSharding;
    meshSharding.extractMeshShardingFromTensorMeshShardingAttr(
        meshes.getMesh(retTensorMeshShardingAttr.getName().str()),
        retTensorMeshShardingAttr, mlir::tt::MeshShardDirection::ShardToFull);

    if (meshSharding.getShardType() != mlir::tt::MeshShardType::Devices) {
      continue;
    }

    auto meshShardOp =
        mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::MeshShardOp>(
            builder, funcReturnOp->getLoc(), retType, ret,
            mlir::tt::MeshShardType::Identity, meshSharding.getShardDirection(),
            meshSharding.getShardShape(), meshSharding.getShardDims());

    funcReturnOp->replaceUsesOfWith(ret, meshShardOp.getResult());
  }

  auto isMultiDeviceResult = [&](mlir::Operation *srcOp) -> bool {
    if (srcOp->getNumResults()) {
      auto resultType =
          mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType());
      auto tensorMeshShardingAttr =
          mlir::dyn_cast_if_present<mlir::tt::TensorMeshShardingAttr>(
              resultType.getEncoding());
      if (tensorMeshShardingAttr) {
        return true;
      }
    }
    return false;
  };

  for (auto arg : funcOp.getArguments()) {
    auto argType = mlir::cast<mlir::RankedTensorType>(arg.getType());
    auto argTensorMeshShardingAttr =
        mlir::dyn_cast_if_present<mlir::tt::TensorMeshShardingAttr>(
            argType.getEncoding());
    if (!argTensorMeshShardingAttr) {
      continue;
    }
    if (llvm::any_of(arg.getUsers(), [&](mlir::Operation *user) {
          return mlir::isa<mlir::tt::ttir::MeshShardOp>(*user);
        })) {
      continue;
    }

    auto userOp = arg.getUsers().begin();
    builder.setInsertionPoint(*userOp);

    mlir::tt::sharding_utils::MeshSharding meshSharding;
    meshSharding.extractMeshShardingFromTensorMeshShardingAttr(
        meshes.getMesh(argTensorMeshShardingAttr.getName().str()),
        argTensorMeshShardingAttr, mlir::tt::MeshShardDirection::FullToShard);
    // replicate mesh sharding doesn't require conversion or shape change
    if (meshSharding.getShardType() != mlir::tt::MeshShardType::Devices) {
      continue;
    }

    if (multiDeviceFix && !isMultiDeviceResult(*userOp)) {
      // Insert the conversion of multi- to single-device tensor.
      auto newResultType = mlir::RankedTensorType::get(
          argType.getShape(), argType.getElementType());
      auto meshShardOp =
          mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::MeshShardOp>(
              builder, userOp->getLoc(), newResultType, arg,
              meshSharding.getShardType(),
              mlir::tt::MeshShardDirection::ShardToFull,
              meshSharding.getShardShape(), meshSharding.getShardDims());
      userOp->replaceUsesOfWith(arg, meshShardOp.getResult());
    } else {
      // Insert identity mesh shard op.
      auto meshShardOp =
          mlir::tt::ttir::utils::createDPSOp<mlir::tt::ttir::MeshShardOp>(
              builder, userOp->getLoc(), argType, arg,
              mlir::tt::MeshShardType::Identity,
              meshSharding.getShardDirection(), meshSharding.getShardShape(),
              meshSharding.getShardDims());
      userOp->replaceUsesOfWith(arg, meshShardOp.getResult());
    }
  }

  return llvm::success();
}

// Adjust TensorMeshShardingAttr for the non-elementwise, positional operations
// that requires non-linear shape-to-shape relationship between the output and
// the inputs.
mlir::tt::TensorMeshShardingAttr adjustOpSepcificTensorMeshShardingAttr(
    mlir::OpBuilder &builder, mlir::Operation *srcOp, const size_t argIdx,
    const mlir::RankedTensorType argType,
    mlir::tt::TensorMeshShardingAttr argTensorMeshShardingAttr,
    const mlir::RankedTensorType resultType,
    mlir::tt::TensorMeshShardingAttr tensorMeshShardingAttr) {

  auto tensorMeshShardingAttrAxes =
      tensorMeshShardingAttr.getTensorMeshShardingAxis();
  // Propagate current result tensorMeshShardingAttr if there is no
  // TensorMeshShardingAtteAxes or Op is DestinationPassingStyle and argIdx is
  // the last one.
  if (tensorMeshShardingAttrAxes.empty() ||
      (mlir::isa<mlir::DestinationStyleOpInterface>(srcOp) &&
       argIdx == srcOp->getNumOperands() - 1)) {
    return tensorMeshShardingAttr;
  }

  llvm::ArrayRef<int64_t> inputShape = argType.getShape();
  llvm::ArrayRef<int64_t> resultShape = resultType.getShape();
  auto unitTensorMeshShardingAxisAttr = TensorMeshShardingAxisAttr::get(
      srcOp->getContext(), 1, llvm::ArrayRef<int64_t>());
  llvm::SmallVector<TensorMeshShardingAxisAttr>
      updatedTensorMeshShardingAttrAxes;

  if (mlir::isa<mlir::tt::ttir::ArgMaxOp>(srcOp) ||
      mlir::isa<mlir::tt::ttir::SumOp>(srcOp) ||
      mlir::isa<mlir::tt::ttir::MeanOp>(srcOp) ||
      mlir::isa<mlir::tt::ttir::MaxOp>(srcOp) ||
      mlir::isa<mlir::tt::ttir::MinOp>(srcOp) ||
      mlir::isa<mlir::tt::ttir::ProdOp>(srcOp) ||
      mlir::isa<mlir::tt::ttir::ReduceAndOp>(srcOp) ||
      mlir::isa<mlir::tt::ttir::ReduceOrOp>(srcOp)) {
    for (auto [idx, is] : llvm::enumerate(inputShape)) {
      if (idx < resultShape.size() && is == resultShape[idx]) {
        updatedTensorMeshShardingAttrAxes.push_back(
            tensorMeshShardingAttrAxes[idx]);
      } else {
        updatedTensorMeshShardingAttrAxes.push_back(
            unitTensorMeshShardingAxisAttr);
      }
    }
  } else if (auto reshapeOp =
                 mlir::dyn_cast<mlir::tt::ttir::ReshapeOp>(srcOp)) {
    if (argTensorMeshShardingAttr) {
      return argTensorMeshShardingAttr;
    }
    for (auto [idx, is] : llvm::enumerate(inputShape)) {
      if (idx < resultShape.size() && is == resultShape[idx]) {
        updatedTensorMeshShardingAttrAxes.push_back(
            tensorMeshShardingAttrAxes[idx]);
      } else {
        updatedTensorMeshShardingAttrAxes.push_back(
            unitTensorMeshShardingAxisAttr);
      }
    }
  } else if (auto broadcastOp =
                 mlir::dyn_cast<mlir::tt::ttir::BroadcastOp>(srcOp)) {
    llvm::ArrayRef<int64_t> broadcastDimensions =
        broadcastOp.getBroadcastDimensions();
    for (auto [idx, args] : llvm::enumerate(llvm::zip_equal(
             inputShape, broadcastDimensions, tensorMeshShardingAttrAxes))) {
      auto [s, d, axis] = args;
      if (d == 1 && s != 1) {
        updatedTensorMeshShardingAttrAxes.push_back(axis);
      } else {
        updatedTensorMeshShardingAttrAxes.push_back(
            unitTensorMeshShardingAxisAttr);
      }
    }
  } else if (auto permuteOp =
                 mlir::dyn_cast<mlir::tt::ttir::PermuteOp>(srcOp)) {
    llvm::ArrayRef<int64_t> permutation = permuteOp.getPermutation();
    updatedTensorMeshShardingAttrAxes.reserve(permutation.size());
    for (auto p : permutation) {
      updatedTensorMeshShardingAttrAxes.push_back(
          tensorMeshShardingAttrAxes[p]);
    }
  } else if (auto dotGeneralOp =
                 mlir::dyn_cast<mlir::tt::ttir::DotGeneralOp>(srcOp)) {
    auto contractingDimensions = (argIdx == 0)
                                     ? dotGeneralOp.getContractDimsLhs()
                                     : dotGeneralOp.getContractDimsRhs();
    updatedTensorMeshShardingAttrAxes.assign(tensorMeshShardingAttrAxes.begin(),
                                             tensorMeshShardingAttrAxes.end());
    for (auto d : contractingDimensions) {
      updatedTensorMeshShardingAttrAxes[d] = unitTensorMeshShardingAxisAttr;
    }
  } else if (mlir::isa<mlir::tt::ttir::ConvolutionOp, mlir::tt::ttir::Conv2dOp,
                       mlir::tt::ttir::ConvTranspose2dOp>(srcOp)) {
    // weight has no direct relationship with output
    if (argIdx == 1) {
      return argTensorMeshShardingAttr;
    }
  }

  if (!updatedTensorMeshShardingAttrAxes.empty()) {
    if (std::all_of(updatedTensorMeshShardingAttrAxes.begin(),
                    updatedTensorMeshShardingAttrAxes.end(),
                    [](TensorMeshShardingAxisAttr axis) {
                      return axis.getShardShape() == 1;
                    })) {
      updatedTensorMeshShardingAttrAxes.clear();
    }
    tensorMeshShardingAttr = TensorMeshShardingAttr::get(
        srcOp->getContext(), tensorMeshShardingAttr.getName(),
        updatedTensorMeshShardingAttrAxes);
  }

  return tensorMeshShardingAttr;
}

// Adjust tensor annotations of args given a multi-device result tensor
// annotation.
llvm::LogicalResult
adjustMultiDeviceTensorAnnotations(mlir::OpBuilder &builder,
                                   mlir::func::FuncOp &funcOp) {
  auto adjustPerOpMultiDeviceTensorAnnotations =
      [&](mlir::OpBuilder &builder,
          mlir::Operation *srcOp) -> llvm::LogicalResult {
    if (srcOp->getNumOperands() == 0 || srcOp->getNumResults() == 0) {
      return llvm::success();
    }

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType());
    auto resultTensorMeshShardingAttr =
        mlir::dyn_cast_if_present<mlir::tt::TensorMeshShardingAttr>(
            resultType.getEncoding());
    for (auto [argIdx, arg] : llvm::enumerate(srcOp->getOperands())) {
      auto argType = mlir::cast<mlir::RankedTensorType>(arg.getType());
      auto argTensorMeshShardingAttr =
          mlir::dyn_cast_if_present<mlir::tt::TensorMeshShardingAttr>(
              argType.getEncoding());

      if (!resultTensorMeshShardingAttr) {
        // Fail if arg is multi device tensor while result is single device
        // tensor. This should not happen.
        if (argTensorMeshShardingAttr) {
          srcOp->emitError()
              << "Inconsistency found in TensorMeshShardingAttr : "
                 "result is single device tensor while arg is "
                 "multi device tensor.";
          return llvm::failure();
        }
        // Pass this arg if both arg and result are single device tensors.
        continue;
      }

      // Check matching of mesh if both arg and result are multi-device tensors.
      if (argTensorMeshShardingAttr && resultTensorMeshShardingAttr) {
        if (argTensorMeshShardingAttr.getName() !=
            resultTensorMeshShardingAttr.getName()) {
          srcOp->emitError()
              << "Inconsistency found in TensorMeshShardingAttr : expected ("
              << resultTensorMeshShardingAttr.getName() << ") and current ("
              << argTensorMeshShardingAttr.getName() << ") are different.";
          return llvm::failure();
        }
      }

      if (checkIfValueIsFuncArg(funcOp, arg)) {
        continue;
      }

      arg.setType(mlir::RankedTensorType::get(
          argType.getShape(), argType.getElementType(),
          adjustOpSepcificTensorMeshShardingAttr(
              builder, srcOp, argIdx, argType, argTensorMeshShardingAttr,
              resultType, resultTensorMeshShardingAttr)));
    }

    return llvm::success();
  };

  funcOp->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      [&](mlir::Operation *op) {
        // Skip ReturnOp, FuncOp, and MeshShardOp and visit the other ops
        // to adjust per-op tensor annotations from back to front.
        if (mlir::isa<mlir::func::ReturnOp, mlir::func::FuncOp,
                      mlir::tt::ttir::MeshShardOp>(op)) {
          return mlir::WalkResult::skip();
        }
        if (failed(adjustPerOpMultiDeviceTensorAnnotations(builder, op))) {
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });

  return llvm::success();
}

// Last step is to clear all TensorMeshShardingAttr axes from the tensors
// in the graph and adjust the shape accordingly. This step allows us to
// maintain the SPMD style shape of all tensors. All tensors except args and
// returns of functions/meshShardOps are not supposed to have
// TensorMeshShardingAttrAxes.
llvm::LogicalResult
removeMultideviceTensorMeshShardingAttrAxes(mlir::OpBuilder &builder,
                                            mlir::func::FuncOp &funcOp) {
  auto updateShapeAndClearTensorShardingAttrAxes =
      [](mlir::Operation *srcOp, mlir::RankedTensorType argType,
         bool includeEncoding = true) -> mlir::RankedTensorType {
    auto argTensorMeshShardingAttr =
        mlir::dyn_cast_if_present<mlir::tt::TensorMeshShardingAttr>(
            argType.getEncoding());
    if (!argTensorMeshShardingAttr) {
      return argType;
    }
    auto argTensorMeshShardingAttrAxes =
        argTensorMeshShardingAttr.getTensorMeshShardingAxis();
    if (argTensorMeshShardingAttrAxes.empty()) {
      return argType;
    }

    // This multi-device tensor has TensorMeshShardingAttrAxes, so adjust shape.
    llvm::SmallVector<int64_t> argShape(argType.getShape().begin(),
                                        argType.getShape().end());
    for (auto [idx, axis] : llvm::enumerate(argTensorMeshShardingAttrAxes)) {
      if (argShape[idx] != 1) {
        argShape[idx] = argShape[idx] / axis.getShardShape();
      }
    }
    // New argType with adjusted shape and without TensorMeshShardingAttrAxes.
    return mlir::RankedTensorType::get(
        argShape, argType.getElementType(),
        ((includeEncoding)
             ? TensorMeshShardingAttr::get(srcOp->getContext(),
                                           argTensorMeshShardingAttr.getName())
             : nullptr));
  };

  // Some ops needs to adjust attirbutes if the output size has changed.
  // Current ops are BroadcastOp and ReshapeOp, but it can be extended as we
  // test more graphs with various ops.
  auto adjustOpAttribute = [](mlir::OpBuilder &builder,
                              mlir::Operation *srcOp) {
    if (auto broadcastOp = mlir::dyn_cast<mlir::tt::ttir::BroadcastOp>(srcOp)) {
      llvm::ArrayRef<int64_t> broadcastDimensions =
          broadcastOp.getBroadcastDimensions();
      auto resultShape =
          mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType())
              .getShape();
      llvm::SmallVector<int64_t> updatedBroadcastDimensions;
      for (auto [d, s] : llvm::zip_equal(broadcastDimensions, resultShape)) {
        if (d == 1) {
          updatedBroadcastDimensions.push_back(1);
        } else {
          updatedBroadcastDimensions.push_back(s);
        }
      }
      broadcastOp.setBroadcastDimensions(updatedBroadcastDimensions);
    } else if (auto reshapeOp =
                   mlir::dyn_cast<mlir::tt::ttir::ReshapeOp>(srcOp)) {
      auto resultShape =
          mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType())
              .getShape();
      llvm::SmallVector<int32_t> reshape(resultShape.begin(),
                                         resultShape.end());
      auto reshapeDimAttr = builder.getI32ArrayAttr(reshape);
      reshapeOp.setShapeAttr(reshapeDimAttr);
    }
  };

  funcOp->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
      [&](mlir::Operation *srcOp) {
        // Skip ReturnOp as we are not supposed to change the shape of
        // inputs/outputs of a graph.
        if (mlir::isa<mlir::func::ReturnOp, mlir::func::FuncOp>(srcOp)) {
          return mlir::WalkResult::skip();
        }
        for (auto [argIdx, arg] : llvm::enumerate(srcOp->getOperands())) {
          if (auto meshShardOp =
                  mlir::dyn_cast<mlir::tt::ttir::MeshShardOp>(srcOp)) {
            auto shardDirection = meshShardOp.getShardDirection();
            if ((shardDirection == mlir::tt::MeshShardDirection::ShardToFull &&
                 argIdx == 1) ||
                (shardDirection == mlir::tt::MeshShardDirection::FullToShard &&
                 argIdx == 0)) {
              continue;
            }
          }
          if (checkIfValueIsFuncArg(funcOp, arg)) {
            continue;
          }
          auto updatedType = updateShapeAndClearTensorShardingAttrAxes(
              srcOp, mlir::cast<mlir::RankedTensorType>(arg.getType()));
          if (arg.getType() != updatedType) {
            // update arg shape and clear TensorMeshShardingAttr axes.
            arg.setType(updatedType);
            // adjust op attributes if needed
            adjustOpAttribute(builder, srcOp);
          }
        }
        // During the conversion of stablehlo to ttir, some constants may have
        // sharding axes with full shape. Thus, we need to remove the sharding
        // axes and adjust the shape.
        if (auto constantOp =
                mlir::dyn_cast<mlir::tt::ttir::ConstantOp>(srcOp)) {
          auto constantElementsAttr = constantOp.getValue();
          if (!constantElementsAttr.isSplat()) {
            // Unable to apply sharding to non-splat constants.
            // Do not throw error here as this will cause some issue when
            // validating the graph.
            return mlir::WalkResult::advance();
          }
          auto constantElementsAttrType = mlir::cast<mlir::RankedTensorType>(
              constantElementsAttr.getType());
          Attribute constantValue =
              constantElementsAttr.getSplatValue<Attribute>();
          auto updatedType = updateShapeAndClearTensorShardingAttrAxes(
              srcOp, constantElementsAttrType, false);
          if (constantElementsAttrType != updatedType) {
            constantOp.setValueAttr(
                mlir::SplatElementsAttr::get(updatedType, constantValue));
          }
        }
        return mlir::WalkResult::advance();
      });

  return llvm::success();
}

// This pass is to clean up the multidevice tensor annotations in the module.
// 1. Insert missing MeshShardOp for args/rets.
// 2. Propagate tensor shardings from back to front and fix unmatched
// TensorMeshShardingAttr between args and output in a single op.
// 3. Remove TensorMeshShardingAttrAxes and adjust shape of all intermediate
// tensors.
class TTIRMultideviceTensorAnnotationCleanupPass
    : public mlir::PassWrapper<TTIRMultideviceTensorAnnotationCleanupPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      TTIRMultideviceTensorAnnotationCleanupPass)

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *context = moduleOp.getContext();
    auto builder = mlir::OpBuilder(context);
    auto meshes = moduleOp->getAttrOfType<mlir::tt::MeshesAttr>(
        mlir::tt::MeshesAttr::name);

    auto checkIfSingleDeviceMeshes = [](mlir::tt::MeshesAttr meshes) -> bool {
      // Single device mesh is determined if the module has no meshes or has
      // only a single device mesh shape (e.g., 1x1).
      if (!meshes) {
        return true;
      }
      for (auto mesh : meshes.getMeshes()) {
        auto meshShape = mesh.getShape();
        if (std::any_of(meshShape.begin(), meshShape.end(),
                        [](int64_t dim) { return dim != 1; })) {
          return false;
        }
      }
      return true;
    };

    if (checkIfSingleDeviceMeshes(meshes)) {
      return;
    }

    auto checkIfMeshShardOpExists = [](mlir::func::FuncOp funcOp) -> bool {
      bool foundMeshShardOp = false;
      funcOp.walk([&](mlir::tt::ttir::MeshShardOp op) {
        foundMeshShardOp = true;
        return mlir::WalkResult::interrupt();
      });
      return foundMeshShardOp;
    };

    // Visit all functions (pretty much single function due to inliner pass)
    // and performs either (1) adding missing meshShardOp for formally sharded
    // graph or (2) performing multidevice tensor sharding propagation for
    // non-sharded graph with automatic parallelization.
    for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
      bool isPreShardedWithShardMap = checkIfMeshShardOpExists(funcOp);
      if (failed(insertMissingMeshShardOps(builder, funcOp, meshes,
                                           isPreShardedWithShardMap))) {
        signalPassFailure();
      }
      if (isPreShardedWithShardMap) {
        // (1) Skip remaining steps if formally sharded with shardmap.
        continue;
      }

      // (2) Propagate tensor shardings from back to front and fix unmatched
      // TensorMeshShardingAttr of args given the output tensor shardings in a
      // single op.
      if (failed(adjustMultiDeviceTensorAnnotations(builder, funcOp))) {
        signalPassFailure();
      }

      // Lastly, adjust shape and clear up tensor annotations in a graph.
      // e.g., A full shape with TensorMeshShardingAttrAxes,
      // tensor<1024x1024xf32, #tt.mesh_sharding<"mesh", [ 8(1), 1]>>,
      // becomes a sharded shape without TensorMeshShardingAttrAxes,
      // tensor<128x1024xf32, #tt.mesh_sharding<"mesh">>.
      // Note that we allows a full shape with TensorMeshShardingAttrAxes in
      // both args/rets of functions and meshShardOps.
      if (failed(
              removeMultideviceTensorMeshShardingAttrAxes(builder, funcOp))) {
        signalPassFailure();
      }
    }
  }

  llvm::StringRef getArgument() const override {
    return "ttir-multidevice-tensor-annotation-cleanup";
  }

  llvm::StringRef getDescription() const override {
    return "Cleanup pass that fixes the multi-device tensor annotations.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const final {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

std::unique_ptr<Pass> createTTIRMultideviceTensorAnnotationCleanupPass() {
  return std::make_unique<TTIRMultideviceTensorAnnotationCleanupPass>();
}

} // namespace mlir::tt
