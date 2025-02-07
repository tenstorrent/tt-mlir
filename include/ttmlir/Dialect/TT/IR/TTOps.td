// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPS_TD
#define TTMLIR_TTMLIR_TTOPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TT_GetTupleElementOp: TT_Op<"get_tuple_element", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "GetTupleElement operation";
    let description = [{
      Extracts element at `index` position of the `operand` tuple and produces a `result`.

      Example:
      ```mlir
      %result = tt.get_tuple_element %operand[0] : (tuple<tensor<32x32xbf16>, tensor<1x32xf32>>) -> tensor<32x32xbf16>
      ```
    }];

    let arguments = (ins TT_Tuple:$operand,
                         ConfinedAttr<I32Attr, [IntNonNegative]>:$index
    );

    let results = (outs TT_TupleMemberType:$result);

    let assemblyFormat = [{
      $operand `[` $index `]` attr-dict `:` functional-type(operands, results)
    }];
}

def TT_TupleOp : TT_Op<"tuple", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Tuple operation";
  let description = [{
    Produces a `result` tuple from operands `operands`.

    Example:
    ```mlir
    %result = tt.tuple %operand0, %operand1 : tuple<tensor<32xbf16, tensor<1x32xf32>>
    ```
  }];

  let arguments = (ins Variadic<TT_TupleMemberType>:$operands);
  let results = (outs TT_Tuple:$result);

  let assemblyFormat = [{
    $operands attr-dict `:` custom<TupleOpType>(type($operands), type($result))
  }];
}

def TT_DeviceModuleOp : TT_Op<"device_module", [
    IsolatedFromAbove,
    NoRegionArguments,
    NoTerminator,
    SingleBlock,
    SymbolTable,
]> {
  let summary = "Module-wrapper operation for device ops";
  let description = [{
    Custom module operation that can a single ModuleOp, which should contain all funcs which should be run on device.

    Example:
    ```mlir
    tt.device_module {
      module {
        func.func foo() { ... }
      }
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "attr-dict-with-keyword $bodyRegion";

  let builders = [OpBuilder<(ins)>];


  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;

  let hasVerifier = 1;
}

def TT_CPUModuleOp : TT_Op<"cpu_module", [
    IsolatedFromAbove,
    NoTerminator,
    NoRegionArguments,
    SingleBlock,
    SymbolTable,
]> {
  let summary = "Module-wrapper operation for CPU ops";
  let description = [{
    Custom module operation that can a single ModuleOp, which should contain all funcs which should be run on CPU.

    Example:
    ```mlir
    tt.cpu_module {
      module {
        func.func foo() { ... }
      }
    }
    ```
  }];

  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = "attr-dict-with-keyword regions";

  let builders = [OpBuilder<(ins)>];

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;

  let hasVerifier = 1;
}

#endif
