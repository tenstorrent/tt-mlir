// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include <optional>

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNCONFIGURECCLOPS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNConfigureCCLOps
    : public impl::TTNNConfigureCCLOpsBase<TTNNConfigureCCLOps> {

public:
  TTNNConfigureCCLOps() = default;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Look up the device to get mesh topology info.
    ttcore::DeviceOp deviceOp = ttcore::lookupDeviceOp(moduleOp);
    if (!deviceOp) {
      return;
    }

    ttcore::DeviceAttr deviceAttr = deviceOp.getDeviceAttr();
    if (deviceAttr.getMeshTopology().empty()) {
      return;
    }

    MLIRContext *context = &getContext();

    moduleOp.walk(
        [&](AllGatherOp op) { setCCLTopology(op, deviceAttr, context); });
    moduleOp.walk(
        [&](AllReduceOp op) { setCCLTopology(op, deviceAttr, context); });
    moduleOp.walk(
        [&](ReduceScatterOp op) { setCCLTopology(op, deviceAttr, context); });
    moduleOp.walk(
        [&](MoeComputeOp op) { setCCLTopology(op, deviceAttr, context); });
    // Always rewrite fused ring-joint topologyfabric: a Ring op on a
    // Linear axis waits for a wrap link that does not exist and wedges the
    // board. The pipeline runs this pass again after ttnn-fusing.
    moduleOp.walk([&](RingJointScaledDotProductAttentionOp op) {
      setCCLTopology(op, deviceAttr, context, /*overwrite=*/true);
    });
    moduleOp.walk([&](ExpRingJointScaledDotProductAttentionOp op) {
      setCCLTopology(op, deviceAttr, context, /*overwrite=*/true);
    });
  }

private:
  template <typename OpTy>
  void setCCLTopology(OpTy op, ttcore::DeviceAttr deviceAttr,
                      MLIRContext *context, bool overwrite = false) {
    if (op.getTopology() && !overwrite) {
      return;
    }

    std::optional<ttcore::Topology> axisTopology =
        ttcore::getMeshTopologyForClusterAxis(deviceAttr, op.getClusterAxis());
    if (!axisTopology || *axisTopology == ttcore::Topology::Disabled) {
      return;
    }

    op.setTopologyAttr(ttcore::TopologyAttr::get(context, *axisTopology));
  }
};

} // namespace

} // namespace mlir::tt::ttnn
