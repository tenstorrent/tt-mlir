// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// CaptureMeshShape pass.
//
// A pre-pass for the ConvertTTIRCPUToEmitPy conversion. It runs before the
// TTNN->EmitPy conversion erases the `ttcore.device` op, reads the device mesh
// shape, and stamps it as the `ttcore.device_mesh_shape` attribute on each
// `forward_cpu` function. The emitted CPU-hoisted execution helper
// (`utils.execute_cpu_hoisted_function`) needs the mesh shape to split and
// reassemble multi-device tensors, but by the time the CPU module is lowered
// the device op is gone; capturing it here as metadata makes it available
// independent of pass ordering, without polluting the flatbuffer path. No-op
// for single-chip (empty mesh shape).
//

#include "ttmlir/Conversion/TTIRToEmitPy/TTIRToEmitPy.h"

#include "ttmlir/Asserts.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/FunctionTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_CAPTUREMESHSHAPE
#include "ttmlir/Conversion/Passes.h.inc"
} // namespace mlir::tt::ttir

namespace {

// Returns the inner ModuleOp wrapped by the WrapperOp (a
// ttcore.device_module or ttcore.cpu_module) in `rootModule`, or nullptr if no
// such wrapper is present.
template <typename WrapperOp>
static ModuleOp getInnerModule(ModuleOp rootModule) {
  for (Operation &op : rootModule.getBodyRegion().front()) {
    if (auto wrapper = dyn_cast<WrapperOp>(op)) {
      ModuleOp inner = dyn_cast_if_present<ModuleOp>(
          wrapper.getBodyRegion().front().front());
      TT_assertv(inner, "module wrapper must have a single ModuleOp child");
      return inner;
    }
  }
  return nullptr;
}

struct CaptureMeshShape
    : public ::mlir::tt::ttir::impl::CaptureMeshShapeBase<CaptureMeshShape> {
  using ::mlir::tt::ttir::impl::CaptureMeshShapeBase<
      CaptureMeshShape>::CaptureMeshShapeBase;

  void runOnOperation() final {
    ModuleOp rootModule = getOperation();

    // Nothing to do if there are no CPU-hoisted functions.
    ModuleOp cpuInnerModule = getInnerModule<ttcore::CPUModuleOp>(rootModule);
    if (!cpuInnerModule) {
      return;
    }

    // Past this point CPU-hoisted functions exist, so they were hoisted out of
    // a device module whose device op is still present.
    ModuleOp deviceInnerModule =
        getInnerModule<ttcore::DeviceModuleOp>(rootModule);
    TT_assertv(deviceInnerModule,
               "CPU module present but device module missing. Run "
               "tt::WrapDeviceModulePass before this pass.");

    ttcore::DeviceOp deviceOp =
        deviceInnerModule.lookupSymbol<ttcore::DeviceOp>(
            ttcore::getDefaultDeviceName());
    TT_assertv(deviceOp,
               "Device module has no device op. Run TTCoreRegisterDevicePass "
               "before this pass.");

    llvm::ArrayRef<int64_t> meshShape = deviceOp.getDeviceAttr().getMeshShape();
    if (meshShape.empty()) {
      // Single-chip: no mesh to shard over, so no mesh_shape is needed.
      return;
    }

    auto meshShapeAttr = OpBuilder(&getContext()).getI64ArrayAttr(meshShape);
    for (func::FuncOp func : cpuInnerModule.getOps<func::FuncOp>()) {
      if (ttmlir::utils::isForwardCPUFunc(func)) {
        func->setAttr(ttcore::g_deviceMeshShapeAttrName, meshShapeAttr);
      }
    }
  }
};

} // namespace

namespace mlir::tt {

std::unique_ptr<OperationPass<ModuleOp>> createCaptureMeshShapePass() {
  return std::make_unique<CaptureMeshShape>();
}

} // namespace mlir::tt
