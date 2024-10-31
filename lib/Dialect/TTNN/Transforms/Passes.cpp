// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsTypes.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <llvm/Support/raw_ostream.h>
#include <system_error>

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNDEALLOCATE
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNDeallocate : public impl::TTNNDeallocateBase<TTNNDeallocate> {

public:
  using impl::TTNNDeallocateBase<TTNNDeallocate>::TTNNDeallocateBase;

  Operation *getLastValueUsageOp(const LivenessBlockInfo *livenessInfo,
                                 Value value) {
    Operation *startOp = livenessInfo->getStartOperation(value);
    Operation *endOp = livenessInfo->getEndOperation(value, startOp);
    auto *opOperandIter =
        llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
          return opOperand.is(value);
        });

    // In case of DPS op keep going until we find the last usage of the tensor.
    //
    while (
        opOperandIter != endOp->getOpOperands().end() &&
        isa<DestinationStyleOpInterface>(endOp) &&
        cast<DestinationStyleOpInterface>(endOp).isDpsInit(&(*opOperandIter))) {
      OpResult result =
          cast<DestinationStyleOpInterface>(endOp).getTiedOpResult(
              &(*opOperandIter));
      endOp = livenessInfo->getEndOperation(result, endOp);
      opOperandIter =
          llvm::find_if(endOp->getOpOperands(), [&](OpOperand &opOperand) {
            return opOperand.is(result);
          });
    }

    return endOp;
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    module->walk([&](func::FuncOp func) {
      assert(func.getBody().hasOneBlock());
      Liveness liveness(func.getOperation());
      const LivenessBlockInfo *livenessInfo =
          liveness.getLiveness(&func.getBody().front());

      // Handle non DPS ops which do not store function result and are used to
      // allocate tensors. DPS ops are handled via ttnn::EmptyOp.
      //
      func->walk([&](Operation *op) {
        if (isa<DestinationStyleOpInterface>(op)) {
          return;
        }

        // Skip ops which do not have results.
        //
        if (op->getNumResults() == 0) {
          return;
        }

        // Iterate over all results of the op.
        //
        for (OpResult result : op->getResults()) {
          // Check if result is ranked tensor type.
          //
          if (!isa<RankedTensorType>(result.getType())) {
            continue;
          }

          RankedTensorType resultTy =
              mlir::cast<RankedTensorType>(result.getType());
          assert(resultTy.getEncoding());

          Operation *lastOp = getLastValueUsageOp(livenessInfo, result);

          if (isa<func::ReturnOp>(lastOp)) {
            continue;
          }

          rewriter.setInsertionPointAfter(lastOp);
          rewriter.create<DeallocOp>(lastOp->getLoc(), result);
        }
      });
    });
    // std::string tmp_name = std::tmpnam(nullptr);
    {
      std::cout << "THIS IS MLIR OUTPUT:" << std::endl;
      std::error_code ec;
      auto fs = llvm::raw_fd_stream("temp_debug.mlir", ec);
      module.print(fs);
      std::cout << "THIS IS THE END OF MLIR OUTPUT" << std::endl;
    }

    {
      std::cout << "EXECUTING FLAT BUFFER COMMAND" << std::endl;
      system("./build/bin/ttmlir-translate --ttnn-to-flatbuffer "
             "temp_debug.mlir -o temp_debug.ttnn");
      std::cout << "FINISHED FLAT BUFFER DUMP" << std::endl;
    }

    {
      std::cout << "RUNNING RUNTIME" << std::endl;
      system("ttrt run temp_debug.ttnn");
      std::cout << "FINISHED RUNNING RUNTIME" << std::endl;
    }
  }
};

} // namespace mlir::tt::ttnn
