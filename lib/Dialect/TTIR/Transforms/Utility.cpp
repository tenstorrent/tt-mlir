// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/Utils/Mesh.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIRIMPLICITDEVICE
#define GEN_PASS_DEF_TTIRLOADSYSTEMDESC
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Implicit device pass
//===----------------------------------------------------------------------===//

class TTIRImplicitDevice
    : public impl::TTIRImplicitDeviceBase<TTIRImplicitDevice> {
public:
  using impl::TTIRImplicitDeviceBase<
      TTIRImplicitDevice>::TTIRImplicitDeviceBase;
  void runOnOperation() final {
    ModuleOp module = getOperation();

    if (not module->hasAttr(tt::DeviceAttr::name)) {
      assert(module->hasAttr(tt::SystemDescAttr::name));
      auto systemDesc = module->getAttr(tt::SystemDescAttr::name);
      auto finalMeshShape = tt::utils::determineMeshShape(module, *meshShape);
      if (auto err = finalMeshShape.takeError()) {
        return;
      }
      module->setAttr(
          tt::DeviceAttr::name,
          tt::DeviceAttr::get(&getContext(),
                              mlir::cast<tt::SystemDescAttr>(systemDesc),
                              *finalMeshShape));
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
  }
};

//===----------------------------------------------------------------------===//
// Load system descriptor pass
//===----------------------------------------------------------------------===//

class TTIRLoadSystemDesc
    : public impl::TTIRLoadSystemDescBase<TTIRLoadSystemDesc> {
public:
  using impl::TTIRLoadSystemDescBase<
      TTIRLoadSystemDesc>::TTIRLoadSystemDescBase;

  void runOnOperation() final {
    ModuleOp module = getOperation();

    if (not path.empty()) {
      module->setAttr(tt::SystemDescAttr::name,
                      tt::SystemDescAttr::getFromPath(&getContext(), path));
    } else if (not module->hasAttr(tt::SystemDescAttr::name)) {
      module->setAttr(tt::SystemDescAttr::name,
                      tt::SystemDescAttr::getDefault(&getContext()));
    }
  }
};

} // namespace mlir::tt::ttir
