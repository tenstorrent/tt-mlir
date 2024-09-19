// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttmlir/Conversion/TTIRToTTNN/TTIRToTTNN.h"
#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TT/IR/TTOpsTypes.h"
#include "ttmlir/Dialect/TTIR/IR/TTIR.h"
#include "ttmlir/Dialect/TTIR/IR/TTIROps.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"
#include <llvm/Support/LogicalResult.h>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNMEMCONFIGOUTPUTS
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

// template <typename T>
// class OpOutputMemConfigInserter : public OpRewritePattern<T> {
// public:
//   using OpRewritePattern<T>::OpRewritePattern;

//   ToMemoryConfigOp createToMemConfigOp(PatternRewriter &rewriter, Location
//   loc, Value &input, DeviceAttr device) {
//     return rewriter.create<ToMemoryConfigOp>(loc, device);
//   }

//   LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final
//   {
//     ::llvm::ArrayRef<int64_t> input_shape =
//         mlir::cast<mlir::RankedTensorType>(op.getInput().getType()).getShape();

//     if (input_shape.size() != 4) {
//       return failure();
//     }

//     if (input_shape[0] == 1 && input_shape[1] == 1) {
//       return failure();
//     }

//     if (!llvm::isa<ToMemoryConfigOp>(op)) {
//       return failure();
//     }

//     Operation &operation = mlir::cast<Operation&>(op);

//     std::vector<Operation *> users;
//     for (auto user :operation.getUsers()) {
//         if (isa<func::ReturnOp>(user)) {
//             return failure();
//         }
//         users.push_back(user);
//     }

//     rewriter.create<T>(operation.getLoc(),
//     operation.getPropertiesAsAttribute());

//     return success();
//   }
// };

class TTNNMemconfigOutputs
    : public impl::TTNNMemconfigOutputsBase<TTNNMemconfigOutputs> {
public:
  using impl::TTNNMemconfigOutputsBase<
      TTNNMemconfigOutputs>::TTNNMemconfigOutputsBase;

  void runOnOperation() final {
    // {
    //   auto device = getCurrentScopeDevice(getOperation());
    //   assert(device && "Device not found");

    //   RewritePatternSet patterns(&getContext());

    //   patterns.add<OpOutputMemConfigInserter<Conv2dOp>>(&getContext());

    //   FrozenRewritePatternSet patternSet(std::move(patterns));
    //   if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet))) {
    //     getOperation().dump();
    //     signalPassFailure();
    //     return;
    //   }
    //   getOperation().dump();
    // }
  }

  //   void getDependentDialects(mlir::DialectRegistry &registry) const override
  //   {
  //     registry.insert<mlir::tt::ttnn::TTNNDialect>();
  //     registry.insert<mlir::tt::TTDialect>();
  //     registry.insert<mlir::func::FuncDialect>();
  //   }
};

} // namespace mlir::tt::ttnn
