#ifndef D2M_ANALYSIS_PASSES
#define D2M_ANALYSIS_PASSES

include "mlir/Pass/PassBase.td"

def D2MValidateLayouts: Pass<"d2m-validate-layouts", "::mlir::ModuleOp"> {
  let summary = "Validate that D2M operations have proper MetalLayoutAttr encoding.";
  let description = [{
    This pass validates that all d2m.generic operations have proper MetalLayoutAttr
    encoding on their tensor operands and results. It checks for:

    1. Presence of MetalLayoutAttr on all input and output tensor operands
    2. Consistency of worker grid across all operands in the same operation
    3. Consistency of memory space across all operands in the same operation

    This pass is useful for catching layout-related errors early in the compilation
    pipeline.

    Example error messages:
    - "operand #0 of operation 'd2m.generic' must have MetalLayoutAttr"
    - "inconsistent worker grid in input operand #1 of operation 'd2m.generic'"
    - "inconsistent memory space in output operand #0 of operation 'd2m.generic'"
  }];
  let dependentDialects = ["::mlir::tt::d2m::D2MDialect"];
}

#endif
