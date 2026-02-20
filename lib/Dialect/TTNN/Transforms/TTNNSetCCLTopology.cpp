// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/Utils.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace mlir::tt::ttnn {

#define GEN_PASS_DEF_TTNNSETCCLTOPOLOGY
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

namespace {

class TTNNSetCCLTopology
    : public impl::TTNNSetCCLTopologyBase<TTNNSetCCLTopology> {

public:
  TTNNSetCCLTopology() = default;

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
    if (clusterAxis >= meshTopology.size()) {
      return;
    }

    ttcore::Topology axisTopology = meshTopology[clusterAxis];
    if (axisTopology == ttcore::Topology::Disabled) {
      return;
    }

    op.setTopologyAttr(ttcore::TopologyAttr::get(context, axisTopology));
  }
};

} // namespace

} // namespace mlir::tt::ttnn
