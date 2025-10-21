// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/Passes.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tt::d2m {
#define GEN_PASS_DEF_D2MVALIDATELAYOUTS
#include "ttmlir/Dialect/D2M/Analysis/Passes.h.inc"

namespace {

struct D2MValidateLayoutsPass
    : public impl::D2MValidateLayoutsBase<D2MValidateLayoutsPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    bool hasErrors = false;

    module.walk([&](mlir::Operation *op) {
      if (auto genericOp = mlir::dyn_cast<d2m::GenericOp>(op)) {
        if (mlir::failed(validateGenericOp(genericOp))) {
          hasErrors = true;
        }
      }
    });

    if (hasErrors) {
      signalPassFailure();
    }
  }

private:
  mlir::LogicalResult validateGenericOp(d2m::GenericOp op) {
    // Check that all input operands have MetalLayoutAttr
    for (auto [idx, operand] : llvm::enumerate(op.getInputs())) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
        if (!mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
                tensorType.getEncoding())) {
          auto diag =
              op.emitError("operand #")
              << idx << " of operation 'd2m.generic' must have MetalLayoutAttr";
          diag.attachNote(operand.getLoc())
              << "operand defined here without layout information";
          return mlir::failure();
        }
      }
    }

    // Check that all output operands have MetalLayoutAttr
    for (auto [idx, operand] : llvm::enumerate(op.getOutputs())) {
      if (auto tensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(operand.getType())) {
        if (!mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
                tensorType.getEncoding())) {
          auto diag =
              op.emitError("output operand #")
              << idx << " of operation 'd2m.generic' must have MetalLayoutAttr";
          diag.attachNote(operand.getLoc())
              << "output operand defined here without layout information";
          return mlir::failure();
        }
      }
    }

    // Validate layout consistency between inputs and outputs
    if (mlir::failed(validateLayoutConsistency(op))) {
      return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult validateLayoutConsistency(d2m::GenericOp op) {
    if (op.getInputs().empty() || op.getOutputs().empty()) {
      return mlir::success();
    }
    auto context = op.getContext();

    // Get the first input layout as reference
    auto firstInput = op.getInputs().front();
    auto firstInputType =
        mlir::dyn_cast<mlir::RankedTensorType>(firstInput.getType());
    if (!firstInputType) {
      return mlir::success(); // Already validated above
    }

    auto referenceLayout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
        firstInputType.getEncoding());
    if (!referenceLayout) {
      return mlir::success(); // Already validated above
    }
    auto referenceGrid = referenceLayout.getGridShape(firstInputType);
    auto referenceMemorySpace = referenceLayout.getMemorySpace();

    // Check that all inputs have consistent grid and memory space
    for (auto [idx, input] : llvm::enumerate(op.getInputs())) {
      auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
      if (!inputType) {
        continue; // Already validated above
      }

      auto layout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
          inputType.getEncoding());
      if (!layout) {
        continue; // Already validated above
      }
      if (layout.getGridShape(inputType) != referenceGrid) {
        auto diag = op.emitError("inconsistent worker grid in input operand #")
                    << idx << " of operation 'd2m.generic'";
        diag.attachNote(input.getLoc())
            << "expected grid " << referenceGrid << " but got "
            << layout.getGridShape(inputType);
        return mlir::failure();
      }

      if (layout.getMemorySpace() != referenceMemorySpace) {
        auto diag = op.emitError("inconsistent memory space in input operand #")
                    << idx << " of operation 'd2m.generic'";
        diag.attachNote(input.getLoc())
            << "expected "
            << ttcore::MemorySpaceAttr::get(inputType.getContext(),
                                            referenceMemorySpace)
            << " but got "
            << ttcore::MemorySpaceAttr::get(context, layout.getMemorySpace());
        return mlir::failure();

        // Check that all outputs have consistent grid and memory space
        for (auto [idx, output] : llvm::enumerate(op.getOutputs())) {
          auto outputType =
              mlir::dyn_cast<mlir::RankedTensorType>(output.getType());
          if (!outputType) {
            continue; // Already validated above
          }

          auto layout = mlir::dyn_cast_if_present<ttcore::MetalLayoutAttr>(
              outputType.getEncoding());
          if (!layout) {
            continue; // Already validated above
          }
          if (layout.getGridShape(outputType) != referenceGrid) {
            auto diag =
                op.emitError("inconsistent worker grid in output operand #")
                << idx << " of operation 'd2m.generic'";
            diag.attachNote(output.getLoc())
                << "expected grid " << referenceGrid << " but got "
                << layout.getGridShape(outputType);
            return mlir::failure();
          }

          if (layout.getMemorySpace() != referenceMemorySpace) {
            auto diag =
                op.emitError("inconsistent memory space in output operand #")
                << idx << " of operation 'd2m.generic'";
            diag.attachNote(output.getLoc())
                << "expected "
                << ttcore::MemorySpaceAttr::get(context, referenceMemorySpace)
                << " but got "
                << ttcore::MemorySpaceAttr::get(context,
                                                layout.getMemorySpace());
            return mlir::failure();
          }
        }

        return mlir::success();
      }
    }
    return mlir::success();
  }
};

} // namespace
} // namespace mlir::tt::d2m
