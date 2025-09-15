// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

      // Derive memory space from the TTNN memref
      auto memrefTy = ttnnLayout.getMemref();
      auto msAttr =
          dyn_cast_or_null<ttcore::MemorySpaceAttr>(memrefTy.getMemorySpace());
      assert(msAttr && msAttr.getValue() != ttcore::MemorySpace::System &&
             "Must be a device tensor");
      auto memSpace = msAttr.getValue();

      // Prefer TTNN linear map if provided; otherwise identity of rank
      AffineMap ttnnIndexMap = ttnnLayout.getLinear();
      if (!ttnnIndexMap) {
        ttnnIndexMap =
            AffineMap::getMultiDimIdentityMap(tensor.getRank(), &getContext());
      }

      // The affine map from TTNNLayoutAttr is a mapping of logical tensor
      // dimensions to the physical grid. The MetalLayout needs an affine map
      // that maps (logical grid, grid divided logical dims) to physical grid
      // TODO: Implement this

      auto metal = ttcore::MetalLayoutAttr::get(
          &getContext(), logicalShape, dimAlignments, gridShape, ttcore::OOBVal::Undef,
          memSpace, ttnnIndexMap);
      return RankedTensorType::get(logicalShape, tensor.getElementType(),
                                   metal);
    };

    // Handle function arguments: insert a view after entry start and rewrite
    // uses.
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
        auto view =
            rewriter.create<ttir::ViewLayoutOp>(func.getLoc(), metalTensor, arg,
                                                /*reinterpretLayout=*/false);
        arg.replaceAllUsesExcept(view.getResult(), view.getOperation());
      }
      return WalkResult::advance();
    });

    // Handle op results: insert view after the defining op and rewrite uses.
    module.walk([&](Operation *op) {
      if (isa<func::FuncOp>(op)) {
        return WalkResult::advance();
      }
      for (OpResult res : op->getResults()) {
        RankedTensorType tensor =
            mlir::dyn_cast<RankedTensorType>(res.getType());
        if (!tensor) {
          continue;
        }

        ttnn::TTNNLayoutAttr ttnnLayout = getTTNNLayout(tensor);
        RankedTensorType metalTensor = buildMetalTensor(tensor, ttnnLayout);
        rewriter.setInsertionPointAfter(op);
        auto view =
            rewriter.create<ttir::ViewLayoutOp>(op->getLoc(), metalTensor, res,
                                                /*reinterpretLayout=*/false);
        res.replaceAllUsesExcept(view.getResult(), view.getOperation());
      }
      return WalkResult::advance();
    });
  }
};
} // namespace

} // namespace mlir::tt::ttir
