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
inline void addMeshToModuleAttribute(PatternRewriter &rewriter,
                                     mlir::ModuleOp module, StringAttr meshName,
                                     llvm::ArrayRef<int64_t> meshShape) {
  MLIRContext *context = rewriter.getContext();
  llvm::SmallVector<tt::MeshAttr> meshes;
  if (auto meshesAttr =
          module->getAttrOfType<tt::MeshesAttr>(tt::MeshesAttr::name)) {
    meshes = llvm::SmallVector<tt::MeshAttr>(meshesAttr.getMeshes());
  }
  // Avoid adding multiple meshes with the same name and shape as GSPMD may try
  // to add the same meshes.
  if (llvm::all_of(meshes,
                   [&](tt::MeshAttr m) { return m.getName() != meshName; })) {
    meshes.push_back(mlir::tt::MeshAttr::get(context, meshName, meshShape));
    rewriter.modifyOpInPlace(module, [&]() {
      module->setAttr(tt::MeshesAttr::name,
                      tt::MeshesAttr::get(context, meshes));
    });
  }
}

// Determine hardware mesh config for DeviceAttr.
// By default, meshAttr will be compiled for single device config.
// If either user provided mesh information through options or meshes exists
// from graph, use one of them. If both exist, compare mesh and throw error if
// they are different.
inline llvm::Expected<tt::MeshAttr> determineMeshAttr(mlir::ModuleOp module,
                                                      tt::MeshAttr &meshAttr) {
  if (auto meshesAttr =
          module->getAttrOfType<tt::MeshesAttr>(tt::MeshesAttr::name)) {
    llvm::ArrayRef<MeshAttr> meshAttrList = meshesAttr.getMeshes();
    if (meshAttrList.empty()) {
      return meshAttr;
    }
    // For now, use the first meshAttr.
    tt::MeshAttr meshFromMeshes = meshAttrList[0];
    // If both meshes exist, they should be identical. Otherwise, throw error.
    if (meshAttr != meshFromMeshes) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Mesh attr information queried from options and mesh info from graph "
          "should be identical.");
    }
    return meshAttr;
  }
  return meshAttr;
}

} // namespace mlir::tt::utils

#endif // TTMLIR_DIALECT_TT_UTILS_MESH_H
