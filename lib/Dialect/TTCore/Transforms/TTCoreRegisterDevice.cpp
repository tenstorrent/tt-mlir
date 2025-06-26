// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h"
#include "ttmlir/Dialect/TTCore/Utils/Mesh.h"

namespace mlir::tt {
#define GEN_PASS_DEF_TTCOREREGISTERDEVICEPASS
#define GEN_PASS_DEF_TTIRDEPRECATEDLOADSYSTEMDESC
#include "ttmlir/Dialect/TTCore/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Register device pass
//===----------------------------------------------------------------------===//

static LogicalResult registerDeviceInSymbolTable(ModuleOp moduleOp,
                                                 ArrayRef<int64_t> meshShape) {
  MLIRContext *context = moduleOp.getContext();

  SymbolTable symbolTable(moduleOp);
  if (!symbolTable.lookup(tt::getDefaultDeviceName())) {
    auto systemDesc =
        moduleOp->getAttrOfType<tt::SystemDescAttr>(tt::SystemDescAttr::name);
    assert(systemDesc && "expected system desc to be present on the moduleOp");
    auto finalMeshShape = tt::utils::determineMeshShape(moduleOp, meshShape);
    if (auto err = finalMeshShape.takeError()) {
      emitError(moduleOp.getLoc()) << "Error determining mesh shape\n";
      assert(false && "Error determining mesh shape");
      return failure();
    }
    OpBuilder builder(moduleOp.getBodyRegion());
    symbolTable.insert(builder.create<tt::DeviceOp>(
        moduleOp.getLoc(), tt::getDefaultDeviceName(),
        tt::DeviceAttr::get(context, systemDesc, *finalMeshShape)));
  }
  return success();
}

LogicalResult registerDevice(ModuleOp moduleOp,
                             tt::Arch mockSystemDescArch = tt::Arch::WormholeB0,
                             ArrayRef<int64_t> meshShape = {}) {
  MLIRContext *context = moduleOp.getContext();

  if (!moduleOp->hasAttr(tt::SystemDescAttr::name)) {
    moduleOp->setAttr(
        tt::SystemDescAttr::name,
        tt::SystemDescAttr::getDefault(context, mockSystemDescArch,
                                       llvm::to_vector(meshShape)));
  }

  return registerDeviceInSymbolTable(moduleOp, meshShape);
}

LogicalResult registerDevice(ModuleOp moduleOp,
                             const std::string &systemDescPath,
                             ArrayRef<int64_t> meshShape = {}) {
  MLIRContext *context = moduleOp.getContext();
  assert(!systemDescPath.empty() && "path must be set");
  FailureOr<tt::SystemDescAttr> systemDesc = tt::SystemDescAttr::getFromPath(
      context, systemDescPath,
      [&]() -> InFlightDiagnostic { return moduleOp->emitOpError(); });
  if (failed(systemDesc)) {
    return systemDesc;
  }
  moduleOp->setAttr(tt::SystemDescAttr::name, *systemDesc);
  return registerDeviceInSymbolTable(moduleOp, meshShape);
}

namespace {
class TTCoreRegisterDevicePass
    : public impl::TTCoreRegisterDevicePassBase<TTCoreRegisterDevicePass> {
public:
  using impl::TTCoreRegisterDevicePassBase<
      TTCoreRegisterDevicePass>::TTCoreRegisterDevicePassBase;

  void runOnOperation() final {
    LogicalResult registered =
        systemDescPath.empty()
            ? registerDevice(getOperation(), mockSystemDescArch, *meshShape)
            : registerDevice(getOperation(), systemDescPath, *meshShape);
    if (failed(registered)) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::tt
