// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsInterfaces.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNLEGALIZEELTWISEBINARYNGOPSBROADCASTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNLegalizeEltwiseBinaryNgOpsBroadcasts
    : public impl::TTNNLegalizeEltwiseBinaryNgOpsBroadcastsBase<
          TTNNLegalizeEltwiseBinaryNgOpsBroadcasts> {

public:
  using impl::TTNNLegalizeEltwiseBinaryNgOpsBroadcastsBase<
      TTNNLegalizeEltwiseBinaryNgOpsBroadcasts>::
      TTNNLegalizeEltwiseBinaryNgOpsBroadcastsBase;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();
    IRRewriter rewriter(&getContext());

    moduleOp.walk([&](ttnn::ElementwiseBinary op) {
      auto resultShape =
          mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
              .getShape();

      mlir::Value lhs = op->getOperand(0);
      mlir::Value rhs = op->getOperand(1);

      auto lhsShape =
          mlir::cast<mlir::RankedTensorType>(lhs.getType()).getShape();
      auto rhsShape =
          mlir::cast<mlir::RankedTensorType>(rhs.getType()).getShape();

      bool shouldExplicate = false;
      if (lhsShape[-2] != resultShape[-2] || rhsShape[-2] != resultShape[-2]) {
        shouldExplicate = true;
      }

      auto lhsLayout = mlir::cast<ttnn::TTNNLayoutAttr>(
          mlir::cast<mlir::RankedTensorType>(lhs.getType()).getEncoding());
      auto rhsLayout = mlir::cast<ttnn::TTNNLayoutAttr>(
          mlir::cast<mlir::RankedTensorType>(rhs.getType()).getEncoding());
      auto resultLayout = mlir::cast<ttnn::TTNNLayoutAttr>(
          mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType())
              .getEncoding());

      if ((lhsLayout.hasShardedTensorMemoryLayout() &&
           lhsLayout.getMemLayout().getValue() !=
               ttnn::TensorMemoryLayout::HeightSharded) ||
          (rhsLayout.hasShardedTensorMemoryLayout() &&
           rhsLayout.getMemLayout().getValue() !=
               ttnn::TensorMemoryLayout::HeightSharded)) {
        shouldExplicate = true;
      }

      if (resultLayout.hasShardedTensorMemoryLayout() &&
          resultLayout.getMemLayout().getValue() !=
              ttnn::TensorMemoryLayout::HeightSharded) {
        shouldExplicate = true;
      }
    });
  }
};
} // namespace mlir::tt::ttnn
