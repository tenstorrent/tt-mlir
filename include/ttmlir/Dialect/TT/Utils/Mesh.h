// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TT_UTILS_MESH_H
#define TTMLIR_DIALECT_TT_UTILS_MESH_H

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::tt::utils {

// Add a new mesh info to module attribute.
inline void addMeshToModuleAttribute(MLIRContext *context,
                                     mlir::ModuleOp module, StringAttr meshName,
                                     llvm::SmallVector<int64_t> &meshShape) {
  llvm::SmallVector<tt::MeshAttr> systemMesh;
  if (module->hasAttr(tt::SystemMeshAttr::name)) {
    mlir::Attribute systemMeshAttr = module->getAttr(tt::SystemMeshAttr::name);
    auto existingSystemMesh =
        mlir::cast<tt::SystemMeshAttr>(systemMeshAttr).getSystemMesh();
    if (!existingSystemMesh.empty()) {
      systemMesh = llvm::SmallVector<tt::MeshAttr>(existingSystemMesh.begin(),
                                                   existingSystemMesh.end());
    }
  }
  // Avoid adding multiple meshes with the same name and shape as GSPMD may try
  // to add the same meshes.
  if (llvm::all_of(systemMesh,
                   [&](tt::MeshAttr m) { return m.getName() != meshName; })) {
    systemMesh.push_back(mlir::tt::MeshAttr::get(context, meshName, meshShape));
    module->setAttr(tt::SystemMeshAttr::name,
                    tt::SystemMeshAttr::get(context, systemMesh));
  }
}

// Determine hardware mesh config for DeviceAttr.
// If none exists, the empty meshShape leads to single device config.
// If either option.meshShape or system mesh exists, use one of them.
// If both exist, compare mesh and throw error if they are different.
inline llvm::Expected<llvm::SmallVector<int64_t>>
determineMeshShape(mlir::ModuleOp module,
                   const ArrayRef<int64_t> orgMeshShape) {
  llvm::SmallVector<int64_t> meshShape(orgMeshShape.begin(),
                                       orgMeshShape.end());
  if (module->hasAttr(tt::SystemMeshAttr::name)) {
    mlir::Attribute systemMesh = module->getAttr(tt::SystemMeshAttr::name);
    const llvm::ArrayRef<MeshAttr> meshAttr =
        mlir::cast<tt::SystemMeshAttr>(systemMesh).getSystemMesh();
    if (!meshAttr.empty()) {
      // For now, use the first meshShape.
      const llvm::ArrayRef<int64_t> meshFromSystemMesh = meshAttr[0].getShape();
      // If both meshes exist, they should be identical. Otherwise, throw error.
      if (!meshShape.empty() && !llvm::equal(meshShape, meshFromSystemMesh)) {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Option.meshShape and mesh info from graph should be identical.");
      }
      meshShape = llvm::SmallVector<int64_t>(meshFromSystemMesh.begin(),
                                             meshFromSystemMesh.end());
    }
  } else if (meshShape.empty()) {
    // If none of mesh info exists, but there are mesh_shard ops in the module,
    // throw error because multi-chip is required.
    bool foundMeshShardOp = false;
    module->walk([&](ttir::MeshShardOp srcOp) { foundMeshShardOp = true; });
    if (foundMeshShardOp) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Either option.meshShape or mesh info from graph should exists due "
          "to mesh_shard op.");
    }
  }
  return meshShape;
}

} // namespace mlir::tt::utils

#endif // TTMLIR_DIALECT_TT_UTILS_MESH_H
