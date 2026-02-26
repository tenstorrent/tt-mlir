// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

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
    ArrayRef<ttcore::Topology> meshTopology = deviceAttr.getMeshTopology();
    if (meshTopology.empty()) {
      return;
    }

    MLIRContext *context = &getContext();

    moduleOp.walk(
        [&](AllGatherOp op) { setCCLTopology(op, meshTopology, context); });
    moduleOp.walk(
        [&](AllReduceOp op) { setCCLTopology(op, meshTopology, context); });
    moduleOp.walk(
        [&](ReduceScatterOp op) { setCCLTopology(op, meshTopology, context); });
  }

private:
  template <typename OpTy>
  void setCCLTopology(OpTy op, ArrayRef<ttcore::Topology> meshTopology,
                      MLIRContext *context) {
    if (op.getTopology()) {
      return;
    }

    uint32_t clusterAxis = op.getClusterAxis();

    // meshTopology follows meshShape indexing:
    //   meshTopology[0] = row-axis (horizontal) connectivity
    //   meshTopology[1] = col-axis (vertical) connectivity
    // cluster_axis follows tt-metal convention:
    //   cluster_axis=0 = vertical movement (devices in same column)
    //   cluster_axis=1 = horizontal movement (devices in same row)
    // Map between the two by reversing the index.
    uint32_t topologyIdx = meshTopology.size() - 1 - clusterAxis;
    if (topologyIdx >= meshTopology.size()) {
      return;
    }

    ttcore::Topology axisTopology = meshTopology[topologyIdx];
    if (axisTopology == ttcore::Topology::Disabled) {
      return;
    }

    op.setTopologyAttr(ttcore::TopologyAttr::get(context, axisTopology));
  }
};

} // namespace

} // namespace mlir::tt::ttnn
