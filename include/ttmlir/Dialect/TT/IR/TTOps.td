// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_TTOPS_TD
#define TTMLIR_TTMLIR_TTOPS_TD

include "ttmlir/Dialect/TT/IR/TTOpsTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

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

#endif
