// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TTOps.h"
#include "ttmlir/Dialect/TT/IR/Utils.h"
#include "ttmlir/Dialect/TT/Transforms/Passes.h"
#include "ttmlir/Dialect/TT/Utils/Mesh.h"

namespace mlir::tt {
#define GEN_PASS_DEF_TTREGISTERDEVICEPASS
#define GEN_PASS_DEF_TTIRDEPRECATEDLOADSYSTEMDESC
#include "ttmlir/Dialect/TT/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Register device pass
//===----------------------------------------------------------------------===//

void registerDevice(ModuleOp module, std::string path,
                    ArrayRef<int64_t> meshShape, StringRef meshName) {
  MLIRContext *context = module.getContext();

  // Create MeshAttr
  auto meshAttr = tt::MeshAttr::get(context, meshName, meshShape);

  if (!path.empty()) {
    module->setAttr(tt::SystemDescAttr::name,
                    tt::SystemDescAttr::getFromPath(context, path));
  } else if (!module->hasAttr(tt::SystemDescAttr::name)) {
    module->setAttr(tt::SystemDescAttr::name,
                    tt::SystemDescAttr::getDefault(context, meshAttr));
  }

  SymbolTable symbolTable(module);
  if (!symbolTable.lookup(tt::getDefaultDeviceName())) {
    auto systemDesc =
        module->getAttrOfType<tt::SystemDescAttr>(tt::SystemDescAttr::name);
    assert(systemDesc && "expected system desc to be present on the module");
    auto finalMeshAttr = tt::utils::determineMeshAttr(module, meshAttr);
    if (auto err = finalMeshAttr.takeError()) {
      emitError(module.getLoc()) << "Error determining mesh attr\n";
      assert(false && "Error determining mesh attr");
      return;
    }
    OpBuilder builder(module.getBodyRegion());
    symbolTable.insert(builder.create<tt::DeviceOp>(
        module.getLoc(), tt::getDefaultDeviceName(),
        tt::DeviceAttr::get(context, systemDesc, *finalMeshAttr)));
  }
}

namespace {
class TTRegisterDevicePass
    : public impl::TTRegisterDevicePassBase<TTRegisterDevicePass> {
public:
  using impl::TTRegisterDevicePassBase<
      TTRegisterDevicePass>::TTRegisterDevicePassBase;

  void runOnOperation() final {
    registerDevice(getOperation(), systemDescPath, *meshShape, meshName);
  }
};
} // namespace

} // namespace mlir::tt
