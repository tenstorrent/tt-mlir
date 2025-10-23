// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTCORE_IR_TTCOREOPS_TD
#define TTMLIR_DIALECT_TTCORE_IR_TTCOREOPS_TD

include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TTCore_GetTupleElementOp: TTCore_Op<"get_tuple_element", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "GetTupleElement operation";
    let description = [{
      Extracts element at `index` position of the `operand` tuple and produces a `result`.

      Example:
      ```mlir
      %result = ttcore.get_tuple_element %operand[0] : (tuple<tensor<32x32xbf16>, tensor<1x32xf32>>) -> tensor<32x32xbf16>
      ```
    }];

    let arguments = (ins TTCore_Tuple:$operand,
                         ConfinedAttr<I32Attr, [IntNonNegative]>:$index
    );

    let results = (outs TTCore_TupleMemberType:$result);

    let assemblyFormat = [{
      $operand `[` $index `]` attr-dict `:` functional-type(operands, results)
    }];
}

def TTCore_TupleOp : TTCore_Op<"tuple", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Tuple operation";
  let description = [{
    Produces a `result` tuple from operands `operands`.

    Example:
    ```mlir
    %result = ttcore.tuple %operand0, %operand1 : tuple<tensor<32xbf16, tensor<1x32xf32>>
    ```
  }];

  let arguments = (ins Variadic<TTCore_TupleMemberType>:$operands);
  let results = (outs TTCore_Tuple:$result);

  let assemblyFormat = [{
    $operands attr-dict `:` custom<TupleOpType>(type($operands), type($result))
  }];
}

def TTCore_DeviceModuleOp : TTCore_Op<"device_module", [
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
    ttcore.device_module {
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

def TTCore_CPUModuleOp : TTCore_Op<"cpu_module", [
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
    ttcore.cpu_module {
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

def TTCore_DeviceOp : TTCore_Op<"device", [Symbol]> {
  let summary = "Named device";
  let description = [{
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TTCore_DeviceAttr:$device_attr);

  let assemblyFormat = "$sym_name `=` $device_attr attr-dict";
}

def TTCore_LoadCachedOp : TTCore_Op<"load_cached", []> {
  let summary = "Load cached results from a previously computed function";
  let description = [{
    The load_cached operation calls a precomputed function with given arguments
    and returns its results. This is typically used to load constant or hoisted
    computation results.

    Example:
    ```mlir
    %0, %1, %2 = "ttcore.load_cached"(@forward_const_eval_1, [%arg0, %arg2])
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyRankedTensor>:$inputs
  );

  let results = (outs
    Variadic<AnyRankedTensor>:$results
  );

  let assemblyFormat = [{
    `(` $callee `,` `[` $inputs `]` `)` attr-dict `:` functional-type($inputs, $results)
  }];

  let hasVerifier = 1;
}

def TTCore_OptimizationBarrierOp : TTCore_Op<"optimization_barrier", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Optimization barrier operation.";
  let description = [{
    The `optimization_barrier` operation prevents compiler optimizations from reordering or eliminating
    the values passed through it. It acts as a barrier for optimization passes.

    Inputs:
    - `inputs` (Variadic): Values of tensor type.

    Outputs:
    - `results` (Variadic): Same values as inputs, passed through unchanged.
  }];

  let arguments = (ins Variadic<AnyRankedTensor>:$inputs);

  let results = (outs Variadic<AnyRankedTensor>:$results);
}

def TTCore_GlobalOp : TTCore_Op<"global", [Symbol]> {
  let summary = "Named global";
  let description = [{
    Declares a global variable with an optional index.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type, OptionalAttr<SI32Attr>:$index);

  let assemblyFormat = "$sym_name `=` $type (` ` `[` $index^ `]`)? attr-dict";
}

def TTCore_GetGlobalOp : TTCore_Op<"get_global", [Pure]> {
  let summary = "Named global";
  let description = [{
    Retrieves a named global value declared with `ttcore.global`
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$sym_name attr-dict `:` type($result)";
}

#endif // TTMLIR_DIALECT_TTCORE_IR_TTCOREOPS_TD
