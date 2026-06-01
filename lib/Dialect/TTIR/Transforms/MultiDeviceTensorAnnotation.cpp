// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"

namespace {

static void
annotateMeshToValue(mlir::Value value,
                    mlir::tt::ttcore::TensorMeshAttr tensorMeshAttr) {
  auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
  if (auto valueTensorMeshAttr =
          mlir::dyn_cast_if_present<mlir::tt::ttcore::TensorMeshAttr>(
              type.getEncoding())) {
    assert(valueTensorMeshAttr == tensorMeshAttr && "MeshAttr mismatch");
    return;
  }
  mlir::Type elementType = type.getElementType();
  llvm::ArrayRef<int64_t> shape = type.getShape();
  value.setType(
      mlir::RankedTensorType::get(shape, elementType, tensorMeshAttr));
  return;
}

static void annotateMeshToValue(mlir::Value value,
                                mlir::tt::ttcore::MeshAttr mesh) {
  auto type = mlir::cast<mlir::RankedTensorType>(value.getType());
  auto tensorMeshAttr =
      mlir::tt::ttcore::TensorMeshAttr::get(type.getContext(), mesh.getName());
  annotateMeshToValue(value, tensorMeshAttr);
  return;
}

// Check op and add multi-device tensor annotations if necessary.
static void
handleMeshShardOpAnnotation(mlir::OpBuilder &builder,
                            mlir::func::FuncOp &funcOp,
                            mlir::tt::ttir::MeshShardOp &meshShardOp,
                            mlir::tt::ttcore::MeshAttr targetMesh) {
  if (meshShardOp.getShardDirection() ==
      mlir::tt::ttcore::MeshShardDirection::FullToShard) {
    // FullToShard: result is multi-device tensors while input is single device
    // tensor.
    annotateMeshToValue(meshShardOp.getResult(), targetMesh);
  } else {
    // ShardToFull: result is single device tensors while input is multi-device
    // tensor.
    annotateMeshToValue(meshShardOp.getInput(), targetMesh);
  }
  return;
}

// Add multi-device tensor annotations to arguments if result is multi-device
// tensor.
static void
handleRestOpsForMultiDeviceTensorAnnotation(mlir::OpBuilder &builder,
                                            mlir::Operation *srcOp) {
  if (srcOp->getNumResults() == 0 || srcOp->getNumOperands() == 0) {
    return;
  }

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(srcOp->getResult(0).getType());
  if (auto tensorMeshAttr =
          mlir::dyn_cast_if_present<mlir::tt::ttcore::TensorMeshAttr>(
              resultType.getEncoding())) {
    for (auto arg : srcOp->getOperands()) {
      annotateMeshToValue(arg, tensorMeshAttr);
    }
  }
  return;
}

} // namespace

namespace mlir::tt::ttir {

// This pass is to add multi-device tensor annotations in the module.
class TTIRMultiDeviceTensorAnnotation
    : public mlir::PassWrapper<TTIRMultiDeviceTensorAnnotation,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTIRMultiDeviceTensorAnnotation)

  void runOnOperation() final {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *context = moduleOp.getContext();
    auto builder = mlir::OpBuilder(context);
    auto meshes = moduleOp->getAttrOfType<mlir::tt::ttcore::MeshesAttr>(
        mlir::tt::ttcore::MeshesAttr::name);
    // We regard a module without mesh or mesh with size of 1x1 as a single
    // device module and skip the pass.
    if (!meshes || meshes.getMeshes().empty()) {
      return;
    }
    assert(meshes.getMeshes().size() == 1 &&
           "Only one mesh in a module is supported for now.");
    auto meshAttr = meshes.getMeshes()[0];
    if (std::accumulate(meshAttr.getShape().begin(), meshAttr.getShape().end(),
                        uint64_t{1}, std::multiplies<uint64_t>()) == 1) {
      return;
    }

    // Visit all functions and add multi-device tensor annotations to each op if
    // necessary.
    for (auto funcOp : moduleOp.getOps<mlir::func::FuncOp>()) {
      funcOp->walk<mlir::WalkOrder::PostOrder, mlir::ReverseIterator>(
          [&](mlir::Operation *op) {
            if (mlir::isa<mlir::func::ReturnOp>(op)) {
              return mlir::WalkResult::skip();
            }
            if (auto meshShardOp =
                    mlir::dyn_cast_if_present<mlir::tt::ttir::MeshShardOp>(
                        op)) {
              handleMeshShardOpAnnotation(builder, funcOp, meshShardOp,
                                          meshAttr);
            } else {
              handleRestOpsForMultiDeviceTensorAnnotation(builder, op);
            }
            return mlir::WalkResult::advance();
          });

      return;
    }
  }

  llvm::StringRef getArgument() const override {
    return "ttir-multi-device-tensor-annotation";
  }

  llvm::StringRef getDescription() const override {
    return "Add multi-device tensor annotations.";
  }
};

std::unique_ptr<Pass> createTTIRMultiDeviceTensorAnnotation() {
  return std::make_unique<TTIRMultiDeviceTensorAnnotation>();
}

} // namespace mlir::tt::ttir
