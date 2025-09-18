// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace mlir::tt::ttir {

#define GEN_PASS_DEF_TTIRLOWERTTNNLAYOUTS
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

namespace {
static ttnn::TTNNLayoutAttr getTTNNLayout(RankedTensorType tensor) {
  auto enc = tensor.getEncoding();
  assert(enc && "RankedTensorType must have an encoding");
  return mlir::cast<ttnn::TTNNLayoutAttr>(enc);
}

struct TTIRLowerTTNNLayouts
    : public impl::TTIRLowerTTNNLayoutsBase<TTIRLowerTTNNLayouts> {
  using Base::Base;

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    auto buildMetalTensor =
        [&](RankedTensorType tensor,
            ttnn::TTNNLayoutAttr ttnnLayout) -> RankedTensorType {
      SmallVector<int64_t> logicalShape(tensor.getShape().begin(),
                                        tensor.getShape().end());
      SmallVector<int64_t> gridShape(
          llvm::to_vector(ttnnLayout.getGrid().getShape()));

      // Get memory space from the TTNN memref
      assert(ttnnLayout.isDeviceBufferType() && "Must be a device tensor");
      ttcore::MemorySpace memSpace =
          ttnnLayout.getBufferType() == ttnn::BufferType::DRAM
              ? ttcore::MemorySpace::DeviceDRAM
              : ttcore::MemorySpace::DeviceL1;

      // With these assumptions we can use the default alignment and dim
      // collapsing behaviour in the MetalLayoutAttr
      assert(ttnnLayout.isTiled() &&
             "Row major TTNN layouts are not supported yet");
      assert(mlir::cast<ttcore::TileType>(ttnnLayout.getElementType())
                     .getHeight() == ttcore::TileType::getDefaultShape()[0] &&
             mlir::cast<ttcore::TileType>(ttnnLayout.getElementType())
                     .getWidth() == ttcore::TileType::getDefaultShape()[1] &&
             "Only default tile shape is supported");
      assert(ttnnLayout.hasL1BufferType() &&
             ttnnLayout.getMemLayout().getValue() ==
                 ttnn::TensorMemoryLayout::BlockSharded &&
             "Only block sharded L1 tensor memory layout is supported");

      ttnnLayout.dump();
      // The index map in TTNNLayoutAttr is for collapsing an N-D tensor on to
      // the grid. It has no relevance to the index map in MetalLayoutAttr.
      // Hardcode collapse intervals to [[0, -1]].
      auto i64Ty = IntegerType::get(&getContext(), 64);
      auto intervalTy = RankedTensorType::get({1, 2}, i64Ty);
      DenseIntElementsAttr collapsedIntervals = DenseIntElementsAttr::get(
          intervalTy, llvm::ArrayRef<int64_t>({0, -1}));
      auto metalLayout = ttcore::MetalLayoutAttr::get(
          &getContext(), logicalShape, gridShape, ttcore::OOBVal::Undef,
          memSpace, collapsedIntervals);
      auto metalTensor = RankedTensorType::get(
          logicalShape, tensor.getElementType(), metalLayout);
      metalLayout.getMemRefType(metalTensor).dump();
      return metalTensor;
    };

    // Handle function arguments: insert a representational cast to metal
    // layout after entry start and rewrite uses.
    module.walk([&](func::FuncOp func) {
      if (func.isDeclaration()) {
        return WalkResult::advance();
      }
      Block &entry = func.getBody().front();
      rewriter.setInsertionPointToStart(&entry);
      for (BlockArgument arg : entry.getArguments()) {
        RankedTensorType tensor =
            mlir::dyn_cast<RankedTensorType>(arg.getType());
        if (!tensor) {
          continue;
        }
        ttnn::TTNNLayoutAttr ttnnLayout = getTTNNLayout(tensor);
        RankedTensorType metalTensor = buildMetalTensor(tensor, ttnnLayout);
        auto castOp = rewriter.create<ttir::TTNNMetalLayoutCastOp>(
            func.getLoc(), metalTensor, arg);
        arg.replaceAllUsesExcept(castOp.getResult(), castOp.getOperation());
      }
      return WalkResult::advance();
    });

    // // Handle op results: insert view after the defining op and rewrite uses.
    // module.walk([&](Operation *op) {
    //   if (isa<func::FuncOp>(op)) {
    //     return WalkResult::advance();
    //   }
    //   for (OpResult res : op->getResults()) {
    //     RankedTensorType tensor =
    //         mlir::dyn_cast<RankedTensorType>(res.getType());
    //     if (!tensor) {
    //       continue;
    //     }
    //     tensor.dump();
    //     ttnn::TTNNLayoutAttr ttnnLayout = getTTNNLayout(tensor);
    //     RankedTensorType metalTensor = buildMetalTensor(tensor, ttnnLayout);
    //     rewriter.setInsertionPointAfter(op);
    //     auto view =
    //         rewriter.create<ttir::ViewLayoutOp>(op->getLoc(), metalTensor,
    //         res,
    //                                             /*reinterpretLayout=*/false);
    //     res.replaceAllUsesExcept(view.getResult(), view.getOperation());
    //   }
    //   return WalkResult::advance();
    // });
  }
};
} // namespace

} // namespace mlir::tt::ttir
