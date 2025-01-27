// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Location/PassOpLoc.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace tt {
namespace ttir {
#define GEN_PASS_DEF_TTIRLAYOUT
#define GEN_PASS_DEF_TTIRSPLITCOMPOUNDLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

class TTIRLayout : public impl::TTIRLayoutBase<TTIRLayout> {
public:
  using impl::TTIRLayoutBase<TTIRLayout>::TTIRLayoutBase;

  void runOnOperation() final;
  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  inline static mlir::ttmlir::PassOpLocFrom loc =
      mlir::ttmlir::PassOpLocFrom(TTIRLayout::getArgumentName());
};

class TTIRSplitCompoundLayout
    : public impl::TTIRSplitCompoundLayoutBase<TTIRSplitCompoundLayout> {
public:
  using impl::TTIRSplitCompoundLayoutBase<
      TTIRSplitCompoundLayout>::TTIRSplitCompoundLayoutBase;

  void runOnOperation() final;

  void getDependentDialects(mlir::DialectRegistry &registry) const override;

  inline static mlir::ttmlir::PassOpLocFrom loc =
      mlir::ttmlir::PassOpLocFrom(TTIRLayout::getArgumentName());
};
} // namespace ttir
} // namespace tt
} // namespace mlir
