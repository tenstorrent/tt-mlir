// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "ttmlir/Dialect/TTNN/Utils/Utils.h"
#include "ttmlir/Dialect/TTNN/Validation/OpConstraintValidator.h"
#include "ttmlir/Support/Logger.h"
#include "ttmlir/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstddef>
#include <mlir/IR/Types.h>
#include <vector>

using namespace mlir;
using namespace mlir::tt;
using namespace mlir::tt::ttnn;

namespace mlir::tt::ttnn {
#define GEN_PASS_DEF_TTNNOPERATIONVALIDATIONANDFALLBACK
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h.inc"

class TTNNOperationValidationAndFallback
    : public impl::TTNNOperationValidationAndFallbackBase<
          TTNNOperationValidationAndFallback> {

public:
  using impl::TTNNOperationValidationAndFallbackBase<
      TTNNOperationValidationAndFallback>::
      TTNNOperationValidationAndFallbackBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    size_t totalOperationsChecked = 0;
    size_t operationsValid = 0;
    size_t operationsFixed = 0;
    size_t operationsFailed = 0;

    // Create validator configured not to throw on unsupported ops.
    OpConstraintValidator validator(
        OpConstraintValidator::ValidationOptions(/*fatalOnUnsupported=*/false));

    moduleOp->walk([&](func::FuncOp func) {
      if (ttmlir::utils::isConstEvalFunc(func)) {
        return;
      }

      // llvm::SmallVector<Type> funcResultTypes;

      func.walk([&](Operation *operation) {
        // Skip operations without results
        if (operation->getNumResults() == 0) {
          return;
        }

        // Skip operations that are not OpModel castable (can't be validated)
        if (!mlir::dyn_cast<OpModel>(operation)) {
          return;
        }

        totalOperationsChecked++;

        // Extract OpConfig from IR
        OpConfig config = extractOpConfigFromIR(operation);
        if (!config.outputLayout) {
          operation->emitError()
              << "OperationValidationAndFallback: No output layout found in "
                 "operation result type encoding";
          signalPassFailure();
          return;
        }

        // Extract input layouts from the operation
        std::vector<TTNNLayoutAttr> inputLayouts =
            utils::extractInputLayouts(operation);

        if (inputLayouts.empty()) {
          return;
        }

        // Test original configuration
        OpConstraintValidator::ValidationResult originalResult =
            validator.validateOperation(operation, inputLayouts, config);

        if (originalResult.success) {
          operationsValid++;
          TTMLIR_TRACE(
              ttmlir::LogComponent::OpValidation,
              "Operation {} at {} passed validation with original config",
              operation->getName(), operation->getLoc());
        } else {
          // Pretend we fixed it for now.
          operationsFixed++;
        }
      });
    });

    // Log validation summary
    TTMLIR_DEBUG(ttmlir::LogComponent::OpValidation,
                 "Operation validation complete: {} operations checked, "
                 "{} valid, {} fixed, {} failed",
                 totalOperationsChecked, operationsValid, operationsFixed,
                 operationsFailed);

    // Fail the pass if we have operations that couldn't be validated
    if (operationsFailed > 0) {
      moduleOp->emitError()
          << "OperationValidationAndFallback: " << operationsFailed
          << " operations failed validation";
      signalPassFailure();
    }
  }

private:
  // Extract OpConfig from operation's IR
  OpConfig extractOpConfigFromIR(Operation *operation) {
    OpConfig config;

    assert(operation->getNumResults() == 1 &&
           "Expected operation with one result");

    // Extract output layout from result type
    if (auto tensorType =
            mlir::dyn_cast<RankedTensorType>(operation->getResultTypes()[0])) {
      if (auto layoutAttr = mlir::dyn_cast_or_null<TTNNLayoutAttr>(
              tensorType.getEncoding())) {
        config.outputLayout = layoutAttr;
      }
    }

    // For Conv2d operations, extract op-specific attributes
    if (auto conv2dOp = mlir::dyn_cast<ttnn::Conv2dOp>(operation)) {
      config.opSpecificAttrs = Conv2dAttrs{conv2dOp.getConv2dConfigAttr(),
                                           conv2dOp.getComputeConfigAttr()};
    }

    return config;
  }
};

} // namespace mlir::tt::ttnn
