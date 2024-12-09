// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNWorkarounds.h"
#include "ttmlir/Dialect/TTNN/Types/Types.h"
#include "ttmlir/Dialect/TTNN/Utils/TransformUtils.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttnn {

// Fold the BroadcastOp if Operand 0 of the instruction requires broadcast.
// Otherwise add a dummy instruction to apply the broadcast for other operands.
// TODO(uazizTT): Canonicalize the instructions such that broadcast operand is
// moved to operand 0.

class TTNNBroadcastWorkaround : public OpRewritePattern<ttnn::BroadcastOp> {
public:
  using OpRewritePattern<ttnn::BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttnn::BroadcastOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace mlir::tt::ttnn
