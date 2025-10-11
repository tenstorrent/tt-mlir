// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPS_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIROPS_TD

include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"
include "ttmlir/Dialect/TTIR/IR/TTIRBase.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsAttrs.td"
include "ttmlir/Dialect/TTIR/IR/TTIROpsInterfaces.td"

include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

class TTIR_DPSOp<string mnemonic, list<Trait> traits = []> :
    TTIR_Op<mnemonic, [TTIROpInterface, DestinationStyleOpInterface] # traits> {
    let extraClassDeclaration = [{
      // base implementation that will detect getOutputMutable() or getOutputsMutable()
      MutableOperandRange getDpsInitsMutable() { return ttir::getDpsOutputs(this); }
    }];
}

//===----------------------------------------------------------------------===//
// TTIR top level, bufferizable, ops
//===----------------------------------------------------------------------===//

class TTIR_BufferizableOp<string mnemonic, list<Trait> traits = []> :
    TTIR_DPSOp<mnemonic, !listconcat(
      [ DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                           , "bufferizesToMemoryWrite"
                                                           , "bufferize"
                                                           , "getAliasingValues"
                                                           , "getBufferType"
                                                           , "isWritable"
                                                           ]>
      , DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ], traits)> {

    let extraClassDefinition = [{
      bool $cppClass::bufferizesToMemoryRead(
          mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
        // If the operand is an input, it is a bufferized to a memory read.
        return isDpsInput(&operand);
      }

      bool $cppClass::bufferizesToMemoryWrite(
          mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
        // If the operand is an output, it is a bufferized to a memory write.
        return isDpsInit(&operand);
      }

      bufferization::AliasingValueList $cppClass::getAliasingValues(
          OpOperand &, const bufferization::AnalysisState &) {
        bufferization::AliasingValueList result;
        return result;
      }

      void $cppClass::getEffects(
          SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
              &effects) {
        ttir::getDpsEffects(*this, effects);
      }

      bool $cppClass::isWritable(
          mlir::Value value, const mlir::bufferization::AnalysisState &) {
        return mlir::isa<mlir::OpResult>(value) ||
               mlir::isa<mlir::BlockArgument>(value);
      }
    }];
}

def TTIR_ToLayoutOp : TTIR_BufferizableOp<"to_layout"> {
    let summary = "Layout op.";
    let description = [{
      ToLayout operation, transition tensors from one layout to another.  Some examples include:
        - Transitioning between different memory spaces, e.g. DRAM to L1.
        - Transitioning between different data types, e.g. f32 to f16.
        - Transitioning between different tile sizes, e.g. 1x16 to 32x32
        - Transitioning between different tensor sharding
        - Some combination of the above

      ```llvm
      #layout = #ttcore.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #system>>
      #layout1 = #ttcore.metal_layout<8192x128x1, undef, <1x1>, memref<64x128xf32, #l1_>>
      %1 = "ttir.to_layout"(%arg0, %0) : (tensor<64x128xf32, #layout>, tensor<64x128xf32, #layout1>) -> tensor<64x128xf32, #layout1>
      ```
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$input,
                         AnyRankedTensorOrMemRef:$output,
                         OptionalAttr<TTCore_MetalLayoutAttr>:$layout);
    let results = (outs Variadic<AnyRankedTensor>:$results);

    let assemblyFormat = [{ $input `,` $output `:` type($input) `into` type($output) (`hostInfo` `=` $layout^)? attr-dict (`->` type($results)^)? }];

    let builders =
    [
      OpBuilder<(ins "Value": $input, "Value": $output, CArg<"ttcore::MetalLayoutAttr", "nullptr">: $layout),
      [{
        build($_builder, $_state, {output.getType()}, input, output, layout);
      }]>,
    ];

    let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return ttir::getDpsOutputs(this); }

      ttcore::MetalLayoutAttr getOrCreateInputLayout();
      ttcore::MetalLayoutAttr getOrCreateOutputLayout();

      bool isHostToDevice();
      bool isDeviceToHost();
    }];

    let hasVerifier = 1;

    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def TTIR_TTNNMetalLayoutCastOp : TTIR_Op<"ttnn_metal_layout_cast",
  [ Pure
  , TTIROpInterface
  , DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  ]> {
  let summary = "Cast TTNN layout-encoded tensor to/from TTCore metal layout-encoded tensor";
  let description = [{
    Purely representational op that reinterprets a tensor's layout encoding from `#ttnn.ttnn_layout<...>`
    to/from `#ttcore.metal_layout<...>` without modifying the underlying data.

    Pre-bufferization, the input and output must be a RankedTensorType where one holds a
    `ttnn::TTNNLayoutAttr` and the other, a `ttcore::MetalLayoutAttr`. Post bufferization,
    the tensor encoded with a `ttcore::MetalLayoutAttr` is bufferized to a memref.

    Examples:
    ```
    %cast_to_metal = ttir.ttnn_metal_layout_cast %arg0
                      : tensor<32x32xf32, #ttnn.ttnn_layout<...>>
                    -> tensor<32x32xf32, #ttcore.metal_layout<...>>

    %cast_to_metal_bufferized = ttir.ttnn_metal_layout_cast %arg0
                            : tensor<32x32xf32, #ttnn.ttnn_layout<...>>
                          -> memref<32x32xf32, ...>

    %cast_to_ttnn = ttir.ttnn_metal_layout_cast %arg0
                    : tensor<32x32xf32, #ttcore.metal_layout<...>>
                  -> tensor<32x32xf32, #ttnn.ttnn_layout<...>>

    %cast_to_ttnn_bufferized = ttir.ttnn_metal_layout_cast %arg0
                            : memref<32x32xf32, ...>
                          -> tensor<32x32xf32, #ttnn.ttnn_layout<...>>
    ```
  }];

  let arguments = (ins AnyRankedTensorOrMemRef:$input);
  let results   = (outs AnyRankedTensorOrMemRef:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// TTIR top level ops
//===----------------------------------------------------------------------===//

def TTIR_GetDimensionSizeOp : TTIR_Op<"get_dimension_size"> {
  let summary = "GetDimensionSize op.";
  let description = [{
      Produces the size of the given `dimension` of the `operand`.

      Example:
        %operand: [[3, 2, 7], [1, 4, 4]]
        "ttir.get_dimension_size"(%operand, value = dense<0>, %out) -> %out: [[3]]
  }];

  let arguments = (ins AnyRankedTensor:$operand,
                       I32Attr:$dimension);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TTIR_AllocOp : TTIR_Op<"alloc"> {
    let summary = "Alloc op.";
    let description = [{
      Tensor Alloc operation
    }];

    let arguments = (ins I64Attr:$address, I64Attr:$size, TTCore_MemorySpaceAttr:$memory_space);
    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_DotGeneralOp : TTIR_Op<"dot_general"> {
    let summary = "Dot general operation.";
    let description = [{
      Flexible tensor operation that generalizes matrix multiplication by allowing user to specify which
      dimensions of two tensors to contract. Matrix multiplication is a special case of this operation,
      where the contraction happens along the last axis of the first tensor and the second-to-last axis of the second tensor.
      From StableHLO DotGeneral Op https://openxla.org/stablehlo/spec#dot_general
    }];

    let arguments = (ins AnyRankedTensor:$lhs,
                         AnyRankedTensor:$rhs,
                         DenseI64ArrayAttr:$batch_dims_lhs,
                         DenseI64ArrayAttr:$contract_dims_lhs,
                         DenseI64ArrayAttr:$batch_dims_rhs,
                         DenseI64ArrayAttr:$contract_dims_rhs);

    let results = (outs AnyRankedTensor:$result);
}


def TTIR_DeallocOp : TTIR_Op<"dealloc"> {
    let summary = "Dealloc op.";
    let description = [{
      Tensor Dealloc operation
    }];

    let arguments = (ins AnyRankedTensor:$result);
}

//===----------------------------------------------------------------------===//
// TTIR top level named ops
//===----------------------------------------------------------------------===//

class TTIR_NamedOp<string mnemonic, list<Trait> traits = []> :
    TTIR_DPSOp<mnemonic, [Pure] # traits>;

class TTIR_ElementwiseOp<string mnemonic, list<Trait> traits = []> :
    TTIR_NamedOp<mnemonic, [TTIR_Broadcastable] # traits> {

    let description = [{
      Base class for elementwise operations. Elementwise operations can take inputs with different shape,
      as long as the input tensors are broadcastable to the same shape.
    }];
}

class TTIR_ElementwiseTernaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, [TTIR_ElementwiseTernary] # traits> {
    let summary = "Base class for elementwise operations with three input tensors.";
    let description = [{
      A base class for elementwise operations that apply a ternary function to corresponding
      elements from three input tensors. The operation produces a result tensor of the same
      shape as the inputs (or the broadcasted shape if broadcasting is supported).

      Each derived operation implements a specific ternary function (e.g., where/select) that
      is applied independently to each element position across the three input tensor operands.

      Inputs:
      - `first` (Tensor): The first input tensor.
      - `second` (Tensor): The second input tensor.
      - `third` (Tensor): The third input tensor.

      Outputs:
      - `output` (Tensor): The output tensor after applying the ternary operation.
    }];

    let arguments = (ins AnyRankedTensor:$first,
                         AnyRankedTensor:$second,
                         AnyRankedTensor:$third,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "Value": $first, "Value": $second, "Value": $third, "Value": $output),
      [{
        build($_builder, $_state, output.getType(), first, second, third, output);
      }]>,
    ];
}

def TTIR_WhereOp: TTIR_ElementwiseTernaryOp<"where"> {
    let summary = "Elementwise conditional selection operation based on a predicate.";
    let description = [{
      The `where` operation performs element-wise conditional selection based on a predicate.

      For each element position, it selects between two values based on a boolean condition in first tensor:
      - If the condition is true (non-zero), it selects the corresponding element from the second tensor
      - If the condition is false (zero), it selects the corresponding element from the third tensor

      This operation supports broadcasting, allowing inputs of different shapes to be combined
      according to standard broadcasting rules.

      Example:
      ```mlir
      // Select elements from %true_values where %condition is true,
      // otherwise select from %false_values
      %result = ttir.where(%condition, %true_values, %false_values, %output) : tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>

      // With broadcasting (condition is a scalar)
      %result = ttir.where(%scalar_condition, %true_values, %false_values, %output) : tensor<1xi1>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      ```

      This operation is equivalent to the ternary conditional operator (`condition ? true_value : false_value`)
      in many programming languages, applied elementwise across tensors.
    }];
}

class TTIR_ElementwiseUnaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, [TwoOperands, TTIR_ElementwiseUnary, TTIR_QuantizableOpInterface] # traits> {
    let summary = "Base class for elementwise operations with one input tensor.";
    let description = [{
      A base class for elementwise operations that apply a unary function to a single input tensor.

      Each derived operation implements a specific unary function (e.g., abs, exp, log) that is applied
      independently to each element of the input tensor.

      Inputs:
      - `input` (Tensor): The input tensor.

      Outputs:
      - `output` (Tensor): The output tensor after applying the unary operation.
    }];

    dag commonArgs = (ins AnyRankedTensor:$input,
                          AnyRankedTensor:$output);

    let arguments = commonArgs;

    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "Value": $input, "Value": $output),
      [{
        build($_builder, $_state, output.getType(), input, output);
      }]>,
    ];
}

def TTIR_AbsOp: TTIR_ElementwiseUnaryOp<"abs", [TTIR_Idempotence]> {
    let summary = "Elementwise absolute value operation.";
    let description = [{
      The `abs` operation computes the absolute value of each element in the input tensor.

      For each element, it returns the magnitude of the value without regard to its sign:
      - For real numbers, it returns |x| (the non-negative value without sign)

      This operation has the idempotence property, meaning that applying it multiple times
      produces the same result as applying it once: abs(abs(x)) = abs(x). The operation
      preserves the data type of the input.

      Example:
      ```mlir
      // Compute absolute values of all elements in %input
      %result = ttir.abs(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[-2.5,  3.7,  0.0,  1.2], ... ]
      // Output tensor:
      // [[2.5, 3.7, 0.0, 1.2], ... ]

      // Example with integer tensor
      %result = ttir.abs(%int_input, %int_output) : tensor<10xi32>, tensor<10xi32> -> tensor<10xi32>
      // Input tensor:
      // [-5, 0, 3, -2, ...]
      // Output tensor:
      // [5, 0, 3, 2, ...]
      ```

      Mathematical definition: abs(x) = |x| = {
        x  if x ≥ 0
        -x if x < 0
      }
    }];
}

def TTIR_CbrtOp: TTIR_ElementwiseUnaryOp<"cbrt"> {
    let summary = "Elementwise cubic root operation.";
    let description = [{
      The `cbrt` operation computes the cubic root (∛) of each element in the input tensor.

      For each element, it returns the real-valued number that, when cubed, equals the input value.
      Unlike square root, cubic root is defined for negative numbers as well as positive numbers.

      Example:
      ```mlir
      // Compute cubic root of all elements in %input
      %result = ttir.cbrt(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[8.0, 27.0, -8.0, 1.0], ... ]
      // Output tensor:
      // [[2.0, 3.0, -2.0, 1.0], ... ]

      // Example with different values
      %result = ttir.cbrt(%float_input, %float_output) : tensor<3x2xf32>, tensor<3x2xf32> -> tensor<3x2xf32>
      // Input tensor:
      // [[125.0, -27.0],
      //  [0.0, 0.001],
      //  [1000.0, -1.0]]
      // Output tensor:
      // [[5.0, -3.0],
      //  [0.0, 0.1],
      //  [10.0, -1.0]]
      ```

      Mathematical definition: cbrt(x) = ∛x = x^(1/3)
    }];
}

def TTIR_CeilOp: TTIR_ElementwiseUnaryOp<"ceil", [TTIR_Idempotence]> {
    let summary = "Elementwise ceiling operation.";
    let description = [{
      The `ceil` operation computes the ceiling (smallest integer greater than or equal to x)
      of each element in the input tensor.

      For each element, it rounds the value up to the nearest integer. The operation preserves
      the data type of the input.

      This operation has the idempotence property, meaning that applying it multiple times
      produces the same result as applying it once: ceil(ceil(x)) = ceil(x).

      Example:
      ```mlir
      // Compute ceiling of all elements in %input
      %result = ttir.ceil(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[2.0, 2.0, 0.0, 5.0], ... ]

      // Example with different values
      %result = ttir.ceil(%float_input, %float_output) : tensor<3x2xf32>, tensor<3x2xf32> -> tensor<3x2xf32>
      // Input tensor:
      // [[3.14, -2.5],
      //  [0.0, 0.001],
      //  [9.999, -0.0]]
      // Output tensor:
      // [[4.0, -2.0],
      //  [0.0, 1.0],
      //  [10.0, 0.0]]
      ```

      Mathematical definition: ceil(x) = ⌈x⌉ = min{n ∈ ℤ | n ≥ x}
    }];
}

def TTIR_CosOp: TTIR_ElementwiseUnaryOp<"cos"> {
    let summary = "Elementwise cosine operation.";
    let description = [{
      The `cos` operation computes the cosine of each element in the input tensor.

      For each element, it returns the cosine of the angle in radians.

      Example:
      ```mlir
      // Compute cosine of all elements in %input
      %result = ttir.cos(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[0.9601, 0.5403, -0.9553, -0.1365], ... ]
      ```
    }];
}

def TTIR_FloorOp: TTIR_ElementwiseUnaryOp<"floor", [TTIR_Idempotence]> {
    let summary = "Elementwise floor operation.";
    let description = [{
      The `floor` operation computes the floor (greatest integer less than or equal to x)
      of each element in the input tensor.

      For each element, it rounds the value down to the nearest integer. The operation preserves
      the data type of the input.

      This operation has the idempotence property, meaning that applying it multiple times
      produces the same result as applying it once: floor(floor(x)) = floor(x).

      Example:
      ```mlir
      // Compute floor of all elements in %input
      %result = ttir.floor(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[1.0, 2.0, -1.0, 4.0], ... ]
      ```

      Mathematical definition: floor(x) = ⌊x⌋ = max{n ∈ ℤ | n ≤ x}
    }];
}

def TTIR_GeluOp: TTIR_ElementwiseUnaryOp<"gelu"> {
  let summary = "Elementwise GELU operation.";
  let description = [{
    The `gelu` operation computes the GELU (Gaussian Error Linear Unit) of each element in the input tensor.

    For each element, it returns the GELU value, which is a smooth, non-monotonic function that approximates the
    cumulative distribution function of a standard normal distribution. The operation preserves the data type of the
    input.

    Example:
    ```mlir
    // Compute GELU of all elements in %input
    %result = ttir.gelu(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    // Input tensor:
    // [[1.7, 2.0, -0.3, 4.5], ... ]
    // Output tensor:
    // [[0.9601, 0.5403, -0.3, 4.5], ... ]
    ```

    Mathematical definition: gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
  }];
}

def TTIR_IsFiniteOp: TTIR_ElementwiseUnaryOp<"isfinite"> {
    let summary = "Elementwise isfinite operation.";
    let description = [{
      The `isfinite` operation checks if each element in the input tensor is finite (neither infinite nor NaN).

      For each element, it returns a boolean value indicating whether the element is finite.

      Example:
      ```mlir
      // Check if all elements in %input are finite
      %result = ttir.isfinite(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, Inf, 4.5], ... ]
      // Output tensor:
      // [[true, true, false, true], ... ]
      ```

      Mathematical definition: isfinite(x) = x ∈ ℝ
    }];
}

def TTIR_LogicalNotOp: TTIR_ElementwiseUnaryOp<"logical_not"> {
    let summary = "Elementwise logical not operation.";
    let description = [{
      The `logical_not` operation computes the logical negation of each element in the input tensor.

      For each element, it returns a boolean value indicating whether the element is false (zero) or true (non-zero).

      Example:
      ```mlir
      // Compute logical negation of all elements in %input
      %result = ttir.logical_not(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.0, 4.5], ... ]
      // Output tensor:
      // [[false, false, true, false], ... ]
      ```

      Mathematical definition: logical_not(x) = !x
    }];
}

def TTIR_BitwiseNotOp : TTIR_ElementwiseUnaryOp<"bitwise_not", [TTIR_Involution]> {
    let summary = "Elementwise bitwise NOT.";
    let description = [{
      The `bitwise_not` operation computes the bitwise NOT (one's complement) of each element in the input tensor.

      For each element, it flips all the bits in the binary representation of the value. This operation
      is typically used with integer data types and has the involution property, meaning that
      applying it twice returns the original value: bitwise_not(bitwise_not(x)) = x.

      Example:
      ```mlir
      // Bitwise operation with with integer tensors
      %result = "ttir.bitwise_not"(%operand, %result) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
      // %operand: [[1, 2], [3, 4]]
      // %result: [[-2, -3], [-4, -5]]

      // Example with binary representation (for 8-bit integers)
      %result = ttir.bitwise_not(%int8_input, %int8_output) : tensor<3xi8>, tensor<3xi8> -> tensor<3xi8>
      // Input %int8_input:
      // [0, 5, 255]  (binary: [00000000, 00000101, 11111111])
      // Output %int8_output:
      // [255, 250, 0]  (binary: [11111111, 11111010, 00000000])
      ```

      Mathematical definition: bitwise_not(x) = ~x
    }];
}

def TTIR_NegOp: TTIR_ElementwiseUnaryOp<"neg", [TTIR_Involution]> {
    let summary = "Elementwise negate operation.";
    let description = [{
      The `neg` operation negates each element in the input tensor.

      For each element, it returns the negation of the value. The operation preserves
      the data type of the input.

      Example:
      ```mlir
      // Compute negation of all elements in %input
      %result = ttir.neg(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[-1.7, -2.0, 0.3, -4.5], ... ]
      ```

      Mathematical definition: neg(x) = -x
    }];
}

def TTIR_TanOp: TTIR_ElementwiseUnaryOp<"tan"> {
    let summary = "Elementwise tan operation.";
    let description = [{
      The `tan` operation computes the tangent of each element in the input tensor.

      For each element, it returns the tangent of the angle in radians.

      Example:
      ```mlir
      // Compute tangent of all elements in %input
      %result = ttir.tan(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[0.9601, 0.5403, -0.3, 4.5], ... ]
      ```

      Mathematical definition: tan(x) = sin(x) / cos(x)
    }];
}

def TTIR_AtanOp: TTIR_ElementwiseUnaryOp<"atan"> {
    let summary = "Eltwise arctangent op.";
    let description = [{
      The `atan` operation computes the arctangent (inverse tangent) of each element in the input tensor.

      For each element, it returns the angle in radians whose tangent is the input value. The operation
      returns values in the range [-π/2, π/2].

      Example:
      ```mlir
      // Compute arctangent of all elements in %input
      %result = ttir.atan(%input, %output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
      // Input tensor:
      // [1.0, 0.5, 0.0, -1.0]
      // Output tensor:
      // [0.785, 0.464, 0.0, -0.785]  // values in radians

      // Example with different values
      %result = ttir.atan(%float_input, %float_output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensor:
      // [0.0, 1.0, 1000.0]
      // Output tensor:
      // [0.0, 0.785, 1.571]  // values approach π/2 as input grows
      ```

      Mathematical definition: atan(x) = tan⁻¹(x), where the result is in the range [-π/2, π/2]
    }];
}

def TTIR_TanhOp: TTIR_ElementwiseUnaryOp<"tanh"> {
    let summary = "Elementwise hyperbolic tangent operation.";
    let description = [{
      The `tanh` operation computes the hyperbolic tangent of each element in the input tensor.

      For each element, it returns the hyperbolic tangent of the value.

      Example:
      ```mlir
      // Compute hyperbolic tangent of all elements in %input
      %result = ttir.tanh(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[0.9601, 0.5403, -0.3, 4.5], ... ]
      ```

      Mathematical definition: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    }];
}

// TODO (azecevic): What should we do with 0.0 case?
// 1/0.0 = inf, 1/inf = 0.0, but TTNN isn't IEEE754 compliant.
// https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/Handling_Special_Value/special_values.md
def TTIR_ReciprocalOp : TTIR_ElementwiseUnaryOp<"reciprocal", [TTIR_Involution]> {
    let summary = "Eltwise reciprocal.";
    let description = [{
      The `reciprocal` operation computes the reciprocal (1/x) of each element in the input tensor.

      For each element, it returns the reciprocal of the value.

      Example:
      ```mlir
      // Compute reciprocal of all elements in %input
      %result = ttir.reciprocal(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[0.5882, 0.5, -3.3333, 0.2173], ... ]
      ```

      Mathematical definition: reciprocal(x) = 1 / x
    }];
}

def TTIR_ReluOp : TTIR_ElementwiseUnaryOp<"relu", [TTIR_Idempotence]> {
    let summary = "Eltwise ReLU.";
    let description = [{
      The `relu` operation computes the rectified linear unit (ReLU) of each element in the input tensor.

      For each element, it returns the maximum of 0 and the value. The operation preserves
      the data type of the input.

      Example:
      ```mlir
      // Compute ReLU of all elements in %input
      %result = ttir.relu(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[1.7, 2.0, 0.0, 4.5], ... ]
      ```

      Mathematical definition: relu(x) = max(0, x)
    }];
}
def TTIR_Relu6Op : TTIR_ElementwiseUnaryOp<"relu6", [TTIR_Idempotence]> {
  let summary = "Eltwise ReLU6.";
  let description = [{
    The `relu6` operation computes the ReLU6 activation function of each element in the input tensor.

    For each element, it returns the minimum of 6 and the maximum of 0 and the value. The operation preserves
    the data type of the input.

    This operation has the idempotence property, meaning that applying it multiple times
    produces the same result as applying it once: relu6(relu6(x)) = relu6(x).

    Example:
    ```mlir
    // Compute ReLU6 of all elements in %input
    %result = ttir.relu6(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    // Input tensor:
    // [[1.7, 2.0, -0.3, 7.0], ... ]
    // Output tensor:
    // [[1.7, 2.0, 0.0, 6.0], ... ]
    ```

    Mathematical definition: relu6(x) = min(6, max(0, x))
  }];
}

def TTIR_RsqrtOp : TTIR_ElementwiseUnaryOp<"rsqrt"> {
    let summary = "Eltwise reciprocal square root.";
    let description = [{
      The `rsqrt` operation computes the reciprocal square root of each element in the input tensor.

      For each element, it returns the reciprocal of the square root of the value.

      Example:
      ```mlir
      // Compute reciprocal square root of all elements in %input
      %result = ttir.rsqrt(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[0.5882, 0.5, -3.3333, 0.2173], ... ]
      ```

      Mathematical definition: rsqrt(x) = 1 / sqrt(x)
    }];
}

def TTIR_SigmoidOp: TTIR_ElementwiseUnaryOp<"sigmoid"> {
    let summary = "Eltwise sigmoid.";
    let description = [{
      The `sigmoid` operation computes the sigmoid of each element in the input tensor.

      For each element, it returns the sigmoid of the value.

      Example:
      ```mlir
      // Compute sigmoid of all elements in %input
      %result = ttir.sigmoid(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[0.8391, 0.9641, 0.5793, 0.9899], ... ]
      ```

      Mathematical definition: sigmoid(x) = 1 / (1 + exp(-x))
    }];
}
def TTIR_SiluOp: TTIR_ElementwiseUnaryOp<"silu"> {
  let summary = "Elementwise SiLU (Sigmoid Linear Unit) operation.";
  let description = [{
    The `silu` operation computes the SiLU (Sigmoid Linear Unit) activation function of each element in the input tensor.

    SiLU, also known as Swish, is defined as x * sigmoid(x). It combines the properties of both
    linear and sigmoid functions, providing smooth, non-monotonic activation that has shown good
    performance in deep learning applications.

    Example:
    ```mlir
    // Compute SiLU of all elements in %input
    %result = ttir.silu(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    // Input tensor:
    // [[-2.0, -1.0, 0.0, 1.0], ... ]
    // Output tensor:
    // [[-0.238, -0.269, 0.0, 0.731], ... ]
    ```

    Mathematical definition: silu(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
  }];
}

def TTIR_SignOp: TTIR_ElementwiseUnaryOp<"sign", [TTIR_Idempotence]> {
    let summary = "Eltwise sign operation.";
    let description = [{
      The `sign` operation computes the sign of each element in the input tensor.

      For each element, it returns:
      - 1 if the value is positive
      - 0 if the value is zero
      - -1 if the value is negative

      This operation has the idempotence property, meaning that applying it multiple times
      produces the same result as applying it once: sign(sign(x)) = sign(x).

      Example:
      ```mlir
      // Compute sign of all elements in %input
      %result = ttir.sign(%input, %output) : tensor<2x3xi32>, tensor<2x3xi32> -> tensor<2x3xi32>
      // Input tensor:
      // [[3, -2, 0],
      //  [1, -4, 4]]
      // Output tensor:
      // [[1, -1, 0],
      //  [1, -1, 1]]

      // Example with floating-point values
      %result = ttir.sign(%float_input, %float_output) : tensor<4xf32>, tensor<4xf32> -> tensor<4xf32>
      // Input tensor:
      // [5.7, -0.0, 0.001, -3.14]
      // Output tensor:
      // [1.0, 0.0, 1.0, -1.0]
      ```

      Mathematical definition: sign(x) = {
        1  if x > 0
        0  if x = 0
        -1 if x < 0
      }
    }];
}

def TTIR_SinOp: TTIR_ElementwiseUnaryOp<"sin"> {
    let summary = "Eltwise sin operation.";
    let description = [{
      The `sin` operation computes the sine of each element in the input tensor.

      For each element, it returns the sine of the angle in radians.

      Example:
      ```mlir
      // Compute sine of all elements in %input
      %result = ttir.sin(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[0.9601, 0.5403, -0.3, 4.5], ... ]
      ```
    }];
}

def TTIR_SqrtOp : TTIR_ElementwiseUnaryOp<"sqrt"> {
    let summary = "Eltwise square root.";
    let description = [{
      The `sqrt` operation computes the square root of each element in the input tensor.

      For each element, it returns the square root of the value.

      Example:
      ```mlir
      // Compute square root of all elements in %input
      %result = ttir.sqrt(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.7, 2.0, -0.3, 4.5], ... ]
      // Output tensor:
      // [[0.5882, 0.5, -3.3333, 0.2173], ... ]
      ```

      Mathematical definition: sqrt(x) = √x
    }];
}

def TTIR_TypecastOp: TTIR_ElementwiseUnaryOp<"typecast"> {
    let summary = "Elementwise type casting operation.";
    let description = [{
      The `typecast` operation converts each element in the input tensor to a different data type.

      This operation performs element-wise type conversion, such as converting from integers to
      floating-point values or between different floating-point precisions. The conversion follows
      the standard type conversion rules for the target platform.

      Example:
      ```mlir
      // Cast from int32 to float32
      %result = ttir.typecast(%input, %output) : tensor<4x4xi32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1, 2, -3, 4], ... ]
      // Output tensor:
      // [[1.0, 2.0, -3.0, 4.0], ... ]

      // Cast from float32 to int32
      %result = ttir.typecast(%float_input, %int_output) : tensor<3xf32>, tensor<3xi32> -> tensor<3xi32>
      // Input tensor:
      // [1.7, -2.3, 3.0]
      // Output tensor:
      // [1, -2, 3]  // Note: truncation, not rounding

      // Cast from float32 to float64 (higher precision)
      %result = ttir.typecast(%f32_input, %f64_output) : tensor<2xf32>, tensor<2xf64> -> tensor<2xf64>
      // Input tensor:
      // [3.14159, 2.71828]
      // Output tensor:
      // [3.14159, 2.71828]  // Same values but with higher precision
      ```
    }];

    let arguments = !con(commonArgs,
                         (ins DefaultValuedAttr<BoolAttr, "false">:$conservative_folding));

    let hasFolder = 1;
    let hasCanonicalizeMethod = 1;
}

def TTIR_LogOp: TTIR_ElementwiseUnaryOp<"log"> {
    let summary = "Elementwise natural logarithm operation.";
    let description = [{
      The `log` operation computes the natural logarithm of each element in the input tensor.

      For each element, it returns the natural logarithm (base e) of the value. This operation
      is defined only for positive values; the behavior for zero or negative inputs
      depends on the implementation (may return NaN, infinity, or other special values).

      Example:
      ```mlir
      // Compute natural logarithm of all elements in %input
      %result = ttir.log(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.0, 2.718, 7.389, 20.086], ... ]
      // Output tensor:
      // [[0.0, 1.0, 2.0, 3.0], ... ]

      // Example with different values
      %result = ttir.log(%float_input, %float_output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensor:
      // [10.0, 100.0, 1000.0]
      // Output tensor:
      // [2.303, 4.605, 6.908]  // ln(10), ln(100), ln(1000)
      ```

      Mathematical definition: log(x) = ln(x), where ln is the natural logarithm
    }];
}

def TTIR_Log1pOp: TTIR_ElementwiseUnaryOp<"log1p"> {
    let summary = "Elementwise natural logarithm of one plus input operation.";
    let description = [{
      The `log1p` operation computes the natural logarithm of one plus each element in the input tensor.

      For each element x, it returns ln(1 + x). This operation is more accurate than computing
      log(1 + x) directly for x values close to zero, and it is defined for x > -1. For values
      less than or equal to -1, the behavior depends on the implementation (may return NaN or
      negative infinity).

      Example:
      ```mlir
      // Compute log1p of all elements in %input
      %result = ttir.log1p(%input, %output) : tensor<5xf32>, tensor<5xf32> -> tensor<5xf32>
      // Input tensor:
      // [0.0, -0.999, 7.0, 6.38905621, 15.0]
      // Output tensor:
      // [0.0, -6.90776825, 2.07944155, 2.0, 2.77258873]

      // Example with small values where log1p is more accurate than log(1+x)
      %result = ttir.log1p(%small_input, %small_output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensor:
      // [1e-10, 1e-7, 1e-5]
      // Output tensor:
      // [1e-10, 1e-7, 1e-5]  // Approximately equal to the input for very small values
      ```

      Mathematical definition: log1p(x) = ln(1 + x)
      }];
}

def TTIR_Expm1Op: TTIR_ElementwiseUnaryOp<"expm1"> {
  let summary = "Elementwise exponential minus one operation.";
  let description = [{
    The `expm1` operation computes the exponential of each element in the input tensor and subtracts one.

    For each element x, it returns e^x - 1. This operation is more accurate than computing
    exp(x) - 1 directly for x values close to zero, where catastrophic cancellation can occur
    in the subtraction.

    Example:
    ```mlir
    // Compute expm1 of all elements in %input
    %result = ttir.expm1(%input, %output) : tensor<2x2xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
    // Input tensor:
    // [[0.0, 1.0],
    //  [0.0, 0.0]]
    // Output tensor:
    // [[0.0, 1.71828],
    //  [0.0, 0.0]]

    // Example with small values where expm1 is more accurate than exp(x)-1
    %result = ttir.expm1(%small_input, %small_output) : tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
    // Input tensor:
    // [1e-10, 1e-7, 1e-5]
    // Output tensor:
    // [1e-10, 1e-7, 1e-5]  // Approximately equal to the input for very small values
    ```

    Mathematical definition: expm1(x) = e^x - 1
  }];
}

def TTIR_ExpOp: TTIR_ElementwiseUnaryOp<"exp"> {
    let summary = "Elementwise exponential op.";
    let description = [{
    The `exp` operation computes the exponential of each element in the input tensor.

      For each element, it returns e^x, where e is the base of natural logarithms (approximately 2.71828).

      Example:
      ```mlir
      // Compute exponential of all elements in %input
      %result = ttir.exp(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
      // Input tensor:
      // [[1.0, 2.0, -3.0, 4.0], ... ]
      // Output tensor:
      // [[2.71828, 7.389056, 0.090031, 54.59815], ... ]
      ```

      Mathematical definition: exp(x) = e^x
    }];
}

def TTIR_ErfOp: TTIR_ElementwiseUnaryOp<"erf"> {
  let summary = "Element-wise error function operation.";
  let description = [{
    Element-wise error function (erf) operation. Calculates erf(x) for each element of the input tensor.

    Example:
    ```mlir
    // Compute error function for all elements in %input
    %result = ttir.erf(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    // Input tensor with values [0.0, 1.0, -1.0, 2.0]
    // Output tensor with values [0.0, 0.8427, -0.8427, 0.9953]
    ```

    Mathematical definition: erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
  }];
}

def TTIR_ErfcOp: TTIR_ElementwiseUnaryOp<"erfc"> {
  let summary = "Element-wise complementary error function operation.";
  let description = [{
    Element-wise complementary error function (erfc) operation. Calculates erfc(x) = 1 - erf(x) for each element of the input tensor.

    Example:
    ```mlir
    // Compute complementary error function for all elements in %input
    %result = ttir.erfc(%input, %output) : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    // Input tensor with values [0.0, 1.0, -1.0, 2.0]
    // Output tensor with values [1.0, 0.1573, 1.8427, 0.0047]
    ```

    Mathematical definition: erfc(x) = 1 - erf(x) = 1 - (2/√π) ∫ₓ^∞ e^(-t²) dt
  }];
}

class TTIR_ElementwiseUnaryWithFloatParameterOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseUnaryOp<mnemonic, traits> {
    let summary = "Base class for elementwise operations with one input tensor and a floating-point parameter.";
    let description = [{
      A base class for elementwise operations that apply a unary function to a single input tensor,
      with the function behavior modified by a floating-point parameter.

      This class extends the basic elementwise unary operations by adding a floating-point attribute
      that can be used to parameterize the operation's behavior. This is useful for operations like
      leaky ReLU, where the slope for negative values is configurable.

      Derived operations will implement specific functions that use the parameter in different ways,
      such as scaling factors, thresholds, or other numerical constants that affect the operation's
      behavior.

      The operation takes an input tensor, an output tensor, and a floating-point parameter attribute.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         F32Attr:$parameter);

    let builders =
    [
      OpBuilder<(ins "Value": $input, "Value": $output, "FloatAttr": $parameter),
      [{
        build($_builder, $_state, output.getType(), input, output, parameter);
      }]>,
    ];
}

def TTIR_LeakyReluOp : TTIR_ElementwiseUnaryWithFloatParameterOp<"leaky_relu"> {
    let summary = "Eltwise leaky relu operation.";
    let description = [{
      The Leaky ReLU (Rectified Linear Unit) operation computes an element-wise
      activation function over its input tensor. It is defined as:

      y = x if x > 0
      y = parameter * x if x <= 0

      where `parameter` is a small, user-defined constant that determines the slope for
      negative inputs.

      Inputs:
      - `input` (Tensor): The input tensor to be activated.

      Outputs:
      - `output` (Tensor): The tensor after applying the Leaky ReLU activation.

      Attributes:
      - `parameter` (float): The slope for negative values.
    }];
}

class TTIR_ElementwiseBinaryOp<string mnemonic, list<Trait> traits = []> :
    TTIR_ElementwiseOp<mnemonic, [ThreeOperands, TTIR_ElementwiseBinary, TTIR_QuantizableOpInterface] #  traits> {
    let summary = "Base class for elementwise operations with two input tensors.";
    let description = [{
      A base class for elementwise operations that apply a binary function to corresponding
      elements from two input tensors. The operation produces a result tensor of the same
      shape as the inputs (or the broadcasted shape).

      Each derived operation implements a specific binary function (e.g., add, multiply,
      compare, logical operations) that is applied independently to each element position
      across the two input tensors.

      Inputs:
      - `lhs` (Tensor): The left-hand side input tensor.
      - `rhs` (Tensor): The right-hand side input tensor.

      Outputs:
      - `output` (Tensor): The output tensor after applying the binary operation.
    }];

    let arguments = (ins AnyRankedTensor:$lhs,
                         AnyRankedTensor:$rhs,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "Value": $lhs, "Value": $rhs, "Value": $output),
      [{
        build($_builder, $_state, output.getType(), lhs, rhs, output);
      }]>,
    ];
}

// TODO (azecevic): NaN != NaN, otherwise eq(x, x) == 1.
def TTIR_EqualOp : TTIR_ElementwiseBinaryOp<"eq"> {
    let summary = "Elementwise equality comparison operation.";
    let description = [{
      The `eq` operation performs an elementwise equality comparison between two tensors.

      For each pair of corresponding elements, it returns:
      - 1 (true) if the elements are equal
      - 0 (false) if the elements are not equal

      Note that special handling may be required for floating-point NaN values, as NaN is not
      equal to any value, including itself.

      Example:
      ```mlir
      // Compare elements for equality
      %result = ttir.eq(%lhs, %rhs, %output) : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1> -> tensor<4x4xi1>
      // Input tensors:
      // %lhs: [[1.0, 2.0, 3.0, 2.0], ... ]
      // %rhs: [[1.0, 2.0, 4.0, 5.0], ... ]
      // Output tensor:
      // [[1, 1, 0, 0], ... ]  // 1 where equal, 0 where not equal

      // Example with integer tensors
      %result = ttir.eq(%int_lhs, %int_rhs, %int_output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi1> -> tensor<3xi1>
      // Input tensors:
      // %int_lhs: [10, -5, 0]
      // %int_rhs: [10, 5, 1]
      // Output tensor:
      // [1, 0, 0]  // Only the first elements are equal
      ```

      Mathematical definition: equal(x, y) = x == y
    }];
}

// TODO (azecevic): NaN != NaN, otherwise ne(x, x) == 0.
def TTIR_NotEqualOp : TTIR_ElementwiseBinaryOp<"ne"> {
    let summary = "Elementwise inequality comparison operation.";
    let description = [{
      The `ne` operation performs an elementwise inequality comparison between two tensors.

      For each pair of corresponding elements, it returns:
      - 1 (true) if the elements are not equal
      - 0 (false) if the elements are equal

      Note that special handling may be required for floating-point NaN values, as NaN is not
      equal to any value, including itself. This means ne(NaN, NaN) should return true.

      Example:
      ```mlir
      // Compare elements for inequality
      %result = ttir.ne(%lhs, %rhs, %output) : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1> -> tensor<4x4xi1>
      // Input tensors:
      // %lhs: [[1.0, 2.0, 3.0, 2.0], ... ]
      // %rhs: [[1.0, 2.0, 4.0, 5.0], ... ]
      // Output tensor:
      // [[0, 0, 1, 1], ... ]  // 0 where equal, 1 where not equal

      // Example with integer tensors
      %result = ttir.ne(%int_lhs, %int_rhs, %int_output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi1> -> tensor<3xi1>
      // Input tensors:
      // %int_lhs: [10, -5, 0]
      // %int_rhs: [10, 5, 1]
      // Output tensor:
      // [0, 1, 1]  // Only the first elements are equal, so their result is 0
      ```

      Mathematical definition: not_equal(x, y) = x != y
    }];
}

// TODO (azecevic): NaN != NaN, otherwise ge(x, x) == 1.
def TTIR_GreaterEqualOp : TTIR_ElementwiseBinaryOp<"ge"> {
    let summary = "Elementwise greater than or equal to.";
    let description = [{
      The `ge` operation performs an elementwise greater than or equal to comparison between two tensors.

      For each pair of corresponding elements, it returns:
      - 1 (true) if the left element is greater than or equal to the right element
      - 0 (false) if the left element is less than the right element

      Example:
      ```mlir
      // Compare elements for greater than or equal to
      %result = ttir.ge(%lhs, %rhs, %output) : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1> -> tensor<4x4xi1>
      // Input tensors:
      // %lhs: [[1.0, 2.0, 3.0, 2.0], ... ]
      // %rhs: [[1.0, 2.0, 4.0, 5.0], ... ]
      // Output tensor:
      // [[1, 1, 0, 0], ... ]  // 1 where greater or equal, 0 where less

      // Example with integer tensors
      %result = ttir.ge(%int_lhs, %int_rhs, %int_output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi1> -> tensor<3xi1>
      // Input tensors:
      // %int_lhs: [10, -5, 0]
      // %int_rhs: [10, 5, 1]
      // Output tensor:
      // [1, 0, 0]  // Only the first elements are greater or equal
      ```

      Mathematical definition: greater_equal(x, y) = x >= y
    }];
}

// TODO (azecevic): NaN != NaN, otherwise gt(x, x) == 0.
def TTIR_GreaterThanOp : TTIR_ElementwiseBinaryOp<"gt"> {
    let summary = "Elementwise greater than.";
    let description = [{
      The `gt` operation performs an elementwise greater than comparison between two tensors.

      For each pair of corresponding elements, it returns:
      - 1 (true) if the left element is greater than the right element
      - 0 (false) if the left element is less than or equal to the right element

      Example:
      ```mlir
      // Compare elements for greater than
      %result = ttir.gt(%lhs, %rhs, %output) : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1> -> tensor<4x4xi1>
      // Input tensors:
      // %lhs: [[1.0, 2.0, 3.0, 2.0], ... ]
      // %rhs: [[1.0, 2.0, 4.0, 5.0], ... ]
      // Output tensor:
      // [[0, 0, 0, 1], ... ]  // 1 where greater, 0 where less or equal

      // Example with integer tensors
      %result = ttir.gt(%int_lhs, %int_rhs, %int_output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi1> -> tensor<3xi1>
      // Input tensors:
      // %int_lhs: [10, -5, 0]
      // %int_rhs: [10, 5, 1]
      // Output tensor:
      // [0, 0, 0]  // Only the last element is greater
      ```

      Mathematical definition: greater_than(x, y) = x > y
    }];
}

// TODO (azecevic): NaN != NaN, otherwise le(x, x) == 1.
def TTIR_LessEqualOp : TTIR_ElementwiseBinaryOp<"le"> {
    let summary = "Elementwise less than or equal to.";
    let description = [{
      The `le` operation performs an elementwise less than or equal to comparison between two tensors.

      For each pair of corresponding elements, it returns:
      - 1 (true) if the left element is less than or equal to the right element
      - 0 (false) if the left element is greater than the right element

      Example:
      ```mlir
      // Compare elements for less than or equal to
      %result = ttir.le(%lhs, %rhs, %output) : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1> -> tensor<4x4xi1>
      // Input tensors:
      // %lhs: [[1.0, 2.0, 3.0, 2.0], ... ]
      // %rhs: [[1.0, 2.0, 4.0, 5.0], ... ]
      // Output tensor:
      // [[1, 1, 1, 0], ... ]  // 1 where less or equal, 0 where greater

      // Example with integer tensors
      %result = ttir.le(%int_lhs, %int_rhs, %int_output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi1> -> tensor<3xi1>
      // Input tensors:
      // %int_lhs: [10, -5, 0]
      // %int_rhs: [10, 5, 1]
      // Output tensor:
      // [1, 1, 1]  // All elements are less or equal
      ```

      Mathematical definition: less_equal(x, y) = x <= y
    }];
}

// TODO (azecevic): NaN != NaN, otherwise lt(x, x) == 0.
def TTIR_LessThanOp : TTIR_ElementwiseBinaryOp<"lt"> {
    let summary = "Elementwise less than.";
    let description = [{
      The `lt` operation performs an elementwise less than comparison between two tensors.

      For each pair of corresponding elements, it returns:
      - 1 (true) if the left element is less than the right element
      - 0 (false) if the left element is greater than or equal to the right element

      Example:
      ```mlir
      // Compare elements for less than
      %result = ttir.lt(%lhs, %rhs, %output) : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xi1> -> tensor<4x4xi1>
      // Input tensors:
      // %lhs: [[1.0, 2.0, 3.0, 2.0], ... ]
      // %rhs: [[1.0, 2.0, 4.0, 5.0], ... ]
      // Output tensor:
      // [[0, 0, 0, 1], ... ]  // 1 where less, 0 where greater or equal

      // Example with integer tensors
      %result = ttir.lt(%int_lhs, %int_rhs, %int_output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi1> -> tensor<3xi1>
      // Input tensors:
      // %int_lhs: [10, -5, 0]
      // %int_rhs: [10, 5, 1]
      // Output tensor:
      // [0, 0, 0]  // Only the last element is less
      ```

      Mathematical definition: less_than(x, y) = x < y
    }];
}

def TTIR_LogicalAndOp : TTIR_ElementwiseBinaryOp<"logical_and", [TTIR_BinaryIdempotence]> {
    let summary = "Elementwise logical and.";
    let description = [{
      The `logical_and` operation performs an elementwise logical AND operation between two tensors.

      For each pair of corresponding elements, it returns:
      - 1 (true) if both elements are 1 (true)
      - 0 (false) if at least one element is 0 (false)

      Example:
      ```mlir
      // Logical AND operation
      %result = ttir.logical_and(%lhs, %rhs, %output) : tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1> -> tensor<4x4xi1>
      // Input tensors:
      // %lhs: [[1, 0, 1, 0], ... ]
      // %rhs: [[1, 1, 0, 1], ... ]
      // Output tensor:
      // [[1, 0, 0, 0], ... ]  // 1 where both are 1, 0 otherwise

      // Example with integer tensors
      %result = ttir.logical_and(%int_lhs, %int_rhs, %int_output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi1> -> tensor<3xi1>
      // Input tensors:
      // %int_lhs: [10, 0, 0]
      // %int_rhs: [10, 5, 1]
      // Output tensor:
      // [1, 0, 0]  // Only the first element is true
      ```

      Mathematical definition: logical_and(x, y) = x && y
    }];
}

def TTIR_LogicalOrOp : TTIR_ElementwiseBinaryOp<"logical_or", [TTIR_BinaryIdempotence]> {
    let summary = "Elementwise logical or.";
    let description = [{
      The `logical_or` operation performs an elementwise logical OR operation between two tensors.

      For each pair of corresponding elements, it returns:
      - 1 (true) if at least one element is 1 (true)
      - 0 (false) if both elements are 0 (false)

      Example:
      ```mlir
      // Logical OR operation
      %result = ttir.logical_or(%lhs, %rhs, %output) : tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1> -> tensor<4x4xi1>
      // Input tensors:
      // %lhs: [[1, 0, 1, 0], ... ]
      // %rhs: [[1, 1, 0, 1], ... ]
      // Output tensor:
      // [[1, 1, 1, 1], ... ]  // 1 where at least one is 1, 0 otherwise

      // Example with integer tensors
      %result = ttir.logical_or(%int_lhs, %int_rhs, %int_output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi1> -> tensor<3xi1>
      // Input tensors:
      // %int_lhs: [10, 0, 0]
      // %int_rhs: [10, 5, 1]
      // Output tensor:
      // [1, 1, 1]  // All elements are true
      ```

      Mathematical definition: logical_or(x, y) = x || y
    }];
}

def TTIR_LogicalXorOp : TTIR_ElementwiseBinaryOp<"logical_xor"> {
    let summary = "Elementwise logical xor.";
    let description = [{
      The `logical_xor` operation performs an elementwise logical XOR operation between two tensors.

      For each pair of corresponding elements, it returns:
      - 1 (true) if exactly one element is 1 (true)
      - 0 (false) if both elements are 0 (false) or both are 1 (true)

      Example:
      ```mlir
      // Logical XOR operation
      %result = ttir.logical_xor(%lhs, %rhs, %output) : tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1> -> tensor<4x4xi1>
      // Input tensors:
      // %lhs: [[1, 0, 1, 0], ... ]
      // %rhs: [[1, 1, 0, 1], ... ]
      // Output tensor:
      // [[0, 1, 1, 1], ... ]  // 1 where exactly one is 1, 0 otherwise

      // Example with integer tensors
      %result = ttir.logical_xor(%int_lhs, %int_rhs, %int_output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi1> -> tensor<3xi1>
      // Input tensors:
      // %int_lhs: [10, 0, 0]
      // %int_rhs: [10, 5, 1]
      // Output tensor:
      // [0, 1, 1]  // Only the last element is true
      ```

      Mathematical definition: logical_xor(x, y) = x ^^ y
    }];
}

def TTIR_BitwiseAndOp : TTIR_ElementwiseBinaryOp<"bitwise_and", [TTIR_BinaryIdempotence]> {
    let summary = "Elementwise bitwise AND.";
    let description = [{
      The `bitwise_and` operation performs an elementwise bitwise AND operation between two tensors.

      For each pair of corresponding elements, it computes the bitwise AND of their binary
      representations. This operation is typically used with integer data types and has the
      idempotence property, meaning that applying it twice with the same second operand
      returns the original result: bitwise_and(bitwise_and(x, y), y) = bitwise_and(x, y).

      Example:
      ```mlir
      // Bitwise AND operation
      %result = ttir.bitwise_and(%lhs, %rhs, %output) : tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
      // Input tensors:
      // %lhs: [[1, 2], [3, 4]]
      // %rhs: [[5, 6], [7, 8]]
      // Output tensor:
      // [[1, 2], [3, 0]]

      // Example with binary representation (for 8-bit integers)
      %result = ttir.bitwise_and(%int8_lhs, %int8_rhs, %int8_output) : tensor<4xi8>, tensor<4xi8>, tensor<4xi8> -> tensor<4xi8>
      // Input tensors:
      // %int8_lhs: [0x0F, 0xAA, 0xFF, 0x00]  (binary: [00001111, 10101010, 11111111, 00000000])
      // %int8_rhs: [0xF0, 0x55, 0xFF, 0x00]  (binary: [11110000, 01010101, 11111111, 00000000])
      // Output tensor:
      // [0x00, 0x00, 0xFF, 0x00]  (binary: [00000000, 00000000, 11111111, 00000000])
      ```

      Mathematical definition: bitwise_and(x, y) = x & y
    }];
}

def TTIR_BitwiseOrOp : TTIR_ElementwiseBinaryOp<"bitwise_or", [TTIR_BinaryIdempotence]> {
    let summary = "Elementwise bitwise OR operation.";
    let description = [{
      The `bitwise_or` operation performs an elementwise bitwise OR operation between two tensors.

      For each pair of corresponding elements, it computes the bitwise OR of their binary
      representations. This operation is typically used with integer data types and has the
      idempotence property, meaning that applying it twice with the same second operand
      returns the original result: bitwise_or(bitwise_or(x, y), y) = bitwise_or(x, y).

      Example:
      ```mlir
      // Bitwise OR operation
      %result = ttir.bitwise_or(%lhs, %rhs, %output) : tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
      // Input tensors:
      // %lhs: [[1, 2], [3, 4]]
      // %rhs: [[5, 6], [7, 8]]
      // Output tensor:
      // [[5, 6], [7, 12]]

      // Example with binary representation (for 8-bit integers)
      %result = ttir.bitwise_or(%int8_lhs, %int8_rhs, %int8_output) : tensor<4xi8>, tensor<4xi8>, tensor<4xi8> -> tensor<4xi8>
      // Input tensors:
      // %int8_lhs: [0x0F, 0xAA, 0x00, 0x55]  (binary: [00001111, 10101010, 00000000, 01010101])
      // %int8_rhs: [0xF0, 0x55, 0x00, 0xAA]  (binary: [11110000, 01010101, 00000000, 10101010])
      // Output tensor:
      // [0xFF, 0xFF, 0x00, 0xFF]  (binary: [11111111, 11111111, 00000000, 11111111])
      ```

      Mathematical definition: bitwise_or(x, y) = x | y
    }];
}

def TTIR_BitwiseXorOp : TTIR_ElementwiseBinaryOp<"bitwise_xor"> {
    let summary = "Elementwise bitwise XOR operation.";
    let description = [{
    The `bitwise_xor` operation performs an elementwise bitwise XOR (exclusive OR) operation between two tensors.

      For each pair of corresponding elements, it computes the bitwise XOR of their binary
      representations. This operation is typically used with integer data types and has the
      property that when applied twice with the same second operand, it returns the original input:
      bitwise_xor(bitwise_xor(x, y), y) = x.

      Example:
      ```mlir
      // Bitwise XOR operation
      %result = ttir.bitwise_xor(%lhs, %rhs, %output) : tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32> -> tensor<2x2xi32>
      // Input tensors:
      // %lhs: [[1, 2], [3, 4]]
      // %rhs: [[5, 6], [7, 8]]
      // Output tensor:
      // [[4, 4], [4, 12]]

      // Example with binary representation (for 8-bit integers)
      %result = ttir.bitwise_xor(%int8_lhs, %int8_rhs, %int8_output) : tensor<4xi8>, tensor<4xi8>, tensor<4xi8> -> tensor<4xi8>
      // Input tensors:
      // %int8_lhs: [0x0F, 0xAA, 0xFF, 0x00]  (binary: [00001111, 10101010, 11111111, 00000000])
      // %int8_rhs: [0xF0, 0x55, 0xFF, 0x00]  (binary: [11110000, 01010101, 11111111, 00000000])
      // Output tensor:
      // [0xFF, 0xFF, 0x00, 0x00]  (binary: [11111111, 11111111, 00000000, 00000000])
      ```

      Mathematical definition: bitwise_xor(x, y) = x ^ y
    }];

    let hasCanonicalizer = 1;
}

def TTIR_MinimumOp :  TTIR_ElementwiseBinaryOp<"minimum", [TTIR_BinaryIdempotence]> {
    let summary = "Elementwise minimum operation.";
    let description = [{
      The `minimum` operation computes the elementwise minimum between two tensors.

      For each pair of corresponding elements, it selects the smaller value and places it in the
      output tensor. This operation has the idempotence property, meaning that applying it twice
      with the same second operand returns the original result: minimum(minimum(x, y), y) = minimum(x, y).

      Example:
      ```mlir
      // Minimum operation
      %result = ttir.minimum(%lhs, %rhs, %output) : tensor<2x3xi32>, tensor<2x3xi32>, tensor<2x3xi32> -> tensor<2x3xi32>
      // Input tensors:
      // %lhs: [[3, 2, 7], [1, 4, 4]]
      // %rhs: [[1, 4, 2], [1, 2, 3]]
      // Output tensor:
      // [[1, 2, 2], [1, 2, 3]]

      // Example with floating point values
      %result = ttir.minimum(%float_lhs, %float_rhs, %float_output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensors:
      // %float_lhs: [3.5, -2.1, 0.0]
      // %float_rhs: [1.2, -5.0, 0.0]
      // Output tensor:
      // [1.2, -5.0, 0.0]
      ```

      Note: When comparing with NaN values, NaN is typically not selected as the minimum value.

      Mathematical definition: minimum(x, y) = min(x, y)
    }];
}

def TTIR_LogicalRightShiftOp : TTIR_ElementwiseBinaryOp<"logical_right_shift"> {
    let summary = "Eltwise Logical Right Shift operation";
    let description = [{
      The `logical_right_shift` operation performs an elementwise logical right shift
      on the elements of the first tensor by the corresponding shift amounts in the
      second tensor.
    }];
    let hasVerifier = 1;
}

def TTIR_LogicalLeftShiftOp : TTIR_ElementwiseBinaryOp<"logical_left_shift"> {
    let summary = "Eltwise Logical Left Shift operation";
    let description = [{
      The `logical_left_shift` operation performs an elementwise logical left shift
      on the elements of the first tensor by the corresponding shift amounts in the
      second tensor.
    }];
    let hasVerifier = 1;
}

def TTIR_SubtractOp : TTIR_ElementwiseBinaryOp<"subtract"> {
    let summary = "Elementwise subtract operation.";
    let description = [{
      The `subtract` operation performs an elementwise subtraction between two tensors.

      For each pair of corresponding elements, it subtracts the element in the second tensor from
      the element in the first tensor and places the result in the output tensor.

      Example:
      ```mlir
      // Subtraction operation
      %result = ttir.subtract(%lhs, %rhs, %output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>
      // Input tensors:
      // %lhs: [10, 20, 30]
      // %rhs: [1, 2, 3]
      // Output tensor:
      // [9, 18, 27]

      // Example with floating point values
      %result = ttir.subtract(%float_lhs, %float_rhs, %float_output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensors:
      // %float_lhs: [3.5, 0.0, -1.2]
      // %float_rhs: [1.5, 2.0, -3.2]
      // Output tensor:
      // [2.0, -2.0, 2.0]
      ```

      Note: The data type of the output tensor matches the data type of the input tensors.

      Mathematical definition: subtract(x, y) = x - y
    }];
}

def TTIR_RemainderOp : TTIR_ElementwiseBinaryOp<"remainder"> {
    let summary = "Elementwise remainder operation.";
    let description = [{
      The `remainder` operation performs an elementwise remainder (modulo) operation between two tensors.

      For each pair of corresponding elements, it computes the remainder when dividing the element
      in the first tensor (dividend) by the element in the second tensor (divisor) and places the
      result in the output tensor.

      Example:
      ```mlir
      // Remainder operation
      %result = ttir.remainder(%lhs, %rhs, %output) : tensor<4xi64>, tensor<4xi64>, tensor<4xi64> -> tensor<4xi64>
      // Input tensors:
      // %lhs: [17, -17, 17, -17]  // Dividends
      // %rhs: [3, 3, -3, -3]      // Divisors
      // Output tensor:
      // [2, -2, 2, -2]

      // Example with floating point values
      %result = ttir.remainder(%float_lhs, %float_rhs, %float_output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensors:
      // %float_lhs: [10.5, -10.5, 3.0]
      // %float_rhs: [3.0, 3.0, 2.0]
      // Output tensor:
      // [1.5, -1.5, 1.0]
      ```

      Note: Division by zero typically results in undefined behavior or NaN for floating-point types.

      Mathematical definition: remainder(x, y) = x % y (where % is the remainder operator)
    }];
}

// TODO We need to add a broadcastable trait to this op.
// See issue https://github.com/tenstorrent/tt-mlir/issues/2015
def TTIR_PowOp : TTIR_ElementwiseBinaryOp<"pow"> {
    let summary = "Elementwise power operation.";
    let description = [{
      The `pow` operation performs an elementwise exponentiation between two tensors.

      For each pair of corresponding elements, it raises the element in the first tensor (base)
      to the power of the element in the second tensor (exponent) and places the result in the
      output tensor.

      Example:
      ```mlir
      // Power operation
      %result = ttir.pow(%lhs, %rhs, %output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensors:
      // %lhs: [2.0, 3.0, 4.0]  // Bases
      // %rhs: [2.0, 2.0, 0.5]  // Exponents
      // Output tensor:
      // [4.0, 9.0, 2.0]

      // Example with integer values
      %result = ttir.pow(%int_lhs, %int_rhs, %int_output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>
      // Input tensors:
      // %int_lhs: [2, 3, 5]
      // %int_rhs: [3, 2, 1]
      // Output tensor:
      // [8, 9, 5]
      ```

      Special cases:
      - 0^0 is typically defined as 1
      - For integer types, negative bases with non-integer exponents may result in complex numbers,
        which are typically not supported and may result in undefined behavior

      Mathematical definition: pow(x, y) = x^y
    }];
}

def TTIR_Atan2Op :  TTIR_ElementwiseBinaryOp<"atan2"> {
    let summary = "Elementwise atan2 operation.";
    let description = [{
      The `atan2` operation performs an elementwise arc tangent (inverse tangent) operation between two tensors.

      For each pair of corresponding elements, it computes the angle in radians between the positive x-axis
      and the vector from the origin to the point (x, y) in the Cartesian plane. This operation is
      typically used in trigonometric calculations and supports partial broadcasting, allowing operands
      of different shapes to be combined.

      Example:
      ```mlir
      // %lhs: [0.0, 1.0, -1.0]
      // %rhs: [1.0, 0.0, 0.0]
      %result = ttir.atan2(%lhs, %rhs, %output) : tensor<3xf64>, tensor<3xf64>, tensor<3xf64> -> tensor<3xf64>
      // %result: [0.0, 1.57079637, -1.57079637] // [0.0, pi/2, -pi/2]
      ```
      Mathematical definition: atan2(x, y) = arctan(y / x)
    }];
}

def TTIR_AddOp : TTIR_ElementwiseBinaryOp<"add", [DeclareOpInterfaceMethods<TTIR_QuantizableOpInterface, ["isQuantizedRewriteFavorable", "rewriteWithQuantizedInputs"]>]> {
    let summary = "Elementwise addition operation.";
    let description = [{
      The `add` operation performs an elementwise addition between two tensors.

      For each pair of corresponding elements, it adds the elements and places the result in the output tensor.

      Example:
      ```mlir
      // Addition operation
      %result = ttir.add(%lhs, %rhs, %output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>
      // Input tensors:
      // %lhs: [10, 20, 30]
      // %rhs: [1, 2, 3]
      // Output tensor:
      // [11, 22, 33]

      // Example with floating point values
      %result = ttir.add(%float_lhs, %float_rhs, %float_output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensors:
      // %float_lhs: [3.5, 0.0, -1.2]
      // %float_rhs: [1.5, 2.0, -3.2]
      // Output tensor:
      // [5.0, 2.0, -2.0]
      ```

      Note: The data type of the output tensor matches the data type of the input tensors.

      Mathematical definition: add(x, y) = x + y
    }];
}

def TTIR_MultiplyOp : TTIR_ElementwiseBinaryOp<"multiply"> {
    let summary = "Elementwise multiplication operation.";
    let description = [{
      The `multiply` operation performs an elementwise multiplication between two tensors.

      For each pair of corresponding elements, it multiplies the elements and places the result in the output tensor.

      Example:
      ```mlir
      // Multiplication operation
      %result = ttir.multiply(%lhs, %rhs, %output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>
      // Input tensors:
      // %lhs: [10, 20, 30]
      // %rhs: [1, 2, 3]
      // Output tensor:
      // [10, 40, 90]

      // Example with floating point values
      %result = ttir.multiply(%float_lhs, %float_rhs, %float_output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensors:
      // %float_lhs: [3.5, 0.0, -1.2]
      // %float_rhs: [1.5, 2.0, -3.2]
      // Output tensor:
      // [5.25, 0.0, -3.84]
      ```

      Note: The data type of the output tensor matches the data type of the input tensors.

      Mathematical definition: multiply(x, y) = x * y
    }];
}

def TTIR_DivOp : TTIR_ElementwiseBinaryOp<"div"> {
    let summary = "Elementwise division operation.";
    let description = [{
      The `div` operation performs an elementwise division between two tensors.

      For each pair of corresponding elements, it divides the element in the first tensor (dividend) by
      the element in the second tensor (divisor) and places the result in the output tensor.

      Example:
      ```mlir
      // Division operation
      %result = ttir.div(%lhs, %rhs, %output) : tensor<3xi32>, tensor<3xi32>, tensor<3xi32> -> tensor<3xi32>
      // Input tensors:
      // %lhs: [10, 20, 20]
      // %rhs: [1, 2, 3]
      // Output tensor:
      // [10, 10, 6]

      // Example with floating point values
      %result = ttir.div(%float_lhs, %float_rhs, %float_output) : tensor<3xf32>, tensor<3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensors:
      // %float_lhs: [3.5, 0.0, -1.2]
      // %float_rhs: [1.5, 2.0, -3.2]
      // Output tensor:
      // [2.333333333, 0.0, -0.375]
      ```

      Note: Division by zero typically results in undefined behavior or NaN for floating-point types.

      Mathematical definition: div(x, y) = x / y
    }];
}

def TTIR_MaximumOp : TTIR_ElementwiseBinaryOp<"maximum"> {
    let summary = "Elementwise maximum operation.";
    let description = [{
      The `maximum` operation calculates the elementwise maximum between two tensors.

      For each pair of corresponding elements, it selects the larger value and places it in the
      output tensor. This operation has the idempotence property, meaning that applying it twice
      with the same second operand returns the original result: maximum(maximum(x, y), y) = maximum(x, y).

      Example:
      ```mlir
      // Maximum operation
      %result = ttir.maximum(%lhs, %rhs, %output) : tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32> -> tensor<3x3xi32>
      // Input tensors:
      // %lhs: [[3, 2, 7], [1, 4, 4]]
      // %rhs: [[1, 4, 2], [1, 2, 3]]
      // Output tensor:
      // [[3, 4, 7], [1, 4, 4]]
      ```

      Note: When comparing with NaN values, NaN is typically not selected as the maximum value.

      Mathematical definition: maximum(x, y) = max(x, y)
    }];
}

def TTIR_BatchNormOp : TTIR_DPSOp<"batch_norm"> {
  let summary = "BatchNormInference operation";
  let description = [{
    Performs batch normalization on the input tensor. Normalizes the `operand` tensor
    across all dimensions except for the specified `dimension` (feature dimension) and
    produces the normalized result.

    Inputs:
    - `operand` (Tensor): The input tensor to be normalized.
    - `scale` (Tensor): The scale parameter (gamma).
    - `offset` (Tensor): The offset parameter (beta).
    - `mean` (Tensor): The pre-computed mean of the input.
    - `variance` (Tensor): The pre-computed variance of the input.

    Attributes:
    - `epsilon` is a small constant added to variance for numerical stability.
    - `dimension` specifies which dimension represents the features/channels.
    - `training` (Bool): Whether the operation is in training mode.

    Output:
    - `result` (Tensor): The normalized output tensor.

    Example:
    ```mlir
      // Normalize a batch of activations
      %result = ttir.batch_norm(%operand, %scale, %offset, %mean, %variance, %output,
                              epsilon = 0.001, dimension = 1, training = false) :
            (tensor<8x16x32x32xf32>, tensor<16xf32>, tensor<16xf32>,
              tensor<16xf32>, tensor<16xf32>, tensor<8x16x32x32xf32>) -> tensor<8x16x32x32xf32>
    ```

    Mathematical definition: batch_norm(x, scale, offset, mean, variance, epsilon, dimension) =
      (x - mean) / sqrt(variance + epsilon) * scale + offset
  }];

  let arguments = (ins AnyRankedTensor:$operand,
                       AnyRankedTensor:$scale,
                       AnyRankedTensor:$offset,
                       AnyRankedTensor:$mean,
                       AnyRankedTensor:$variance,
                       AnyRankedTensor:$output,
                       F32Attr:$epsilon,
                       I32Attr:$dimension,
                       BoolAttr:$training);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_RMSNormOp : TTIR_DPSOp<"rms_norm", [AttrSizedOperandSegments]> {
  let summary = "RMS normalization operation";
  let description = [{
    Performs RMS (Root Mean Square) normalization on the input tensor. This operation
    normalizes the input tensor by computing the root mean square of elements across
    the specified dimensions and dividing by that value, optionally scaling and
    shifting the result.

    Inputs:
    - `input` (Tensor): The input tensor to be normalized.
    - `weight` (Optional Tensor): The scale parameter (gamma). If provided, the normalized
      result is element-wise multiplied by this weight.
    - `bias` (Optional Tensor): The shift parameter (beta). If provided, this bias is
      added to the scaled result.

    Attributes:
    - `normalized_shape` specifies the dimensions over which to normalize. Typically
      the last few dimensions of the input tensor.
    - `epsilon` is a small constant added for numerical stability (default: 1e-05).

    Output:
    - `result` (Tensor): The RMS normalized output tensor.

    Example:
    ```mlir
      // RMS normalization over last dimension (shape: [2, 4, 8] -> normalize over [8])
      %result = ttir.rms_norm(%input, %weight, %bias, %output,
                             normalized_shape = [8], epsilon = 1e-05) :
            (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<2x4x8xf32>

      // RMS normalization over last two dimensions (shape: [2, 4, 8] -> normalize over [4, 8])
      %result = ttir.rms_norm(%input, %weight, %bias, %output,
                             normalized_shape = [4, 8], epsilon = 1e-05) :
            (tensor<2x4x8xf32>, tensor<2x4x8xf32>, tensor<4x8xf32>) -> tensor<2x4x8xf32>
    ```

    Mathematical definition: rms_norm(x, weight, bias, epsilon) =
      (x / sqrt(mean(x^2, dims=normalized_dims) + epsilon)) * weight + bias
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       Optional<AnyRankedTensor>:$weight,
                       Optional<AnyRankedTensor>:$bias,
                       AnyRankedTensor:$output,
                       DenseI64ArrayAttr:$normalized_shape,
                       DefaultValuedAttr<F32Attr, "1e-05">:$epsilon);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

class TTIR_ReductionOp<string mnemonic, list<Trait> traits = []> :
    TTIR_NamedOp<mnemonic, traits> {

    let summary = "Base class for reduction operations.";
    let description = [{
      This serves as a base class for operations that reduce tensor values along specified dimensions.

      Reduction operations take an input tensor and apply a specific reduction function (such as sum, mean, max, etc.)
      along one or more dimensions. The result is a tensor with reduced dimensionality unless `keep_dim` is set to true,
      in which case the reduced dimensions are retained with a size of 1.

      The `dim_arg` attribute specifies which dimensions to reduce along. If not provided, the reduction is performed
      over all dimensions, resulting in a scalar value (or a tensor with all dimensions of size 1 if `keep_dim` is true).

      Derived operations will implement specific reduction functions such as sum, mean, max, min, etc.

      Inputs:
      - `input` (Tensor): The input tensor.

      Attributes:
      - `keep_dim` (Bool): Whether to keep the reduced dimensions or not.
      - `dim_arg` (Array of Int32): Dimensions to reduce along.

      Outputs:
      - `output` (Tensor): The result tensor after applying the reduction.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         BoolAttr:$keep_dim,
                         OptionalAttr<I32ArrayAttr>:$dim_arg);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_ArgMaxOp : TTIR_ReductionOp<"argmax"> {
  let summary = "Argmax reduction op.";
  let description = [{
    Determine the indices of the maximum values along a specified dimension of a tensor or over all elements
    in a tensor.

    This operation reduces the input tensor by finding the index of the maximum value along the dimensions
    specified in `dim_arg`. If `dim_arg` is not provided, the argmax is computed over all dimensions,
    resulting in a scalar index. If `keep_dim` is set to true, the reduced dimensions are retained
    with a size of 1.

    ### Example IR Usage:
    ```mlir
    // Argmax along dimension 1
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<2xi32>
    %result = ttir.argmax(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : tensor<2x3xf32>, tensor<2xi32> -> tensor<2xi32>
    // Input tensor:
    // [[1.0, 5.0, 3.0],
    //  [2.0, 4.0, 6.0]]
    // Output tensor:
    // [1, 2]  // Index of maximum value in each row (5.0 in first row, 6.0 in second row)

    // Argmax along dimension 0
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<3xi32>
    %result = ttir.argmax(%input, %output) {keep_dim = false, dim_arg = [0: i32]} : tensor<2x3xf32>, tensor<3xi32> -> tensor<3xi32>
    // Input tensor:
    // [[1.0, 5.0, 3.0],
    //  [2.0, 4.0, 6.0]]
    // Output tensor:
    // [1, 0, 1]  // Index of maximum value in each column

    // Argmax over all dimensions
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<i32>
    %result = ttir.argmax(%input, %output) {keep_dim = false} : tensor<2x3xf32>, tensor<i32> -> tensor<i32>
    // Input tensor:
    // [[1.0, 5.0, 3.0],
    //  [2.0, 4.0, 6.0]]
    // Output tensor:
    // 5  // Flattened index of the maximum value (6.0)
    ```

    Inputs:
      - `input` (Tensor): The input tensor.

      Attributes:
      - `keep_dim` (Bool): Whether to keep the reduced dimensions or not.
      - `dim_arg` (Array of Int32): Dimensions to reduce along.

      Outputs:
      - `output` (Tensor): The result tensor after applying the reduction.
  }];
}

def TTIR_SumOp : TTIR_ReductionOp<"sum"> {
    let summary = "Sum reduction operation.";
    let description = [{
      The `sum` operation computes the sum of elements along specified dimensions of the input tensor.

      This operation reduces the input tensor by computing the sum of all elements along the dimensions
      specified in `dim_arg`. If `dim_arg` is not provided, the sum is computed over all dimensions,
      resulting in a scalar value. If `keep_dim` is set to true, the reduced dimensions are retained
      with a size of 1.

      Example:
      ```mlir
      // Sum along dimension 1
      %input = ... : tensor<2x3xf32>
      %output = ttir.empty() : tensor<2xf32>
      %result = ttir.sum(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : tensor<2x3xf32>, tensor<2xf32> -> tensor<2xf32>
      // Input tensor:
      // [[1.0, 2.0, 3.0],
      //  [4.0, 5.0, 6.0]]
      // Output tensor:
      // [6.0, 15.0]  // Sum of each row

      // Sum along dimension 0
      %input = ... : tensor<2x3xf32>
      %output = ttir.empty() : tensor<3xf32>
      %result = ttir.sum(%input, %output) {keep_dim = false, dim_arg = [0: i32]} : tensor<2x3xf32>, tensor<3xf32> -> tensor<3xf32>
      // Input tensor:
      // [[1.0, 2.0, 3.0],
      //  [4.0, 5.0, 6.0]]
      // Output tensor:
      // [5.0, 7.0, 9.0]  // Sum of each column

      // Sum over all dimensions
      %input = ... : tensor<2x3xf32>
      %output = ttir.empty() : tensor<f32>
      %result = ttir.sum(%input, %output) {keep_dim = false} : tensor<2x3xf32>, tensor<f32> -> tensor<f32>
      // Input tensor:
      // [[1.0, 2.0, 3.0],
      //  [4.0, 5.0, 6.0]]
      // Output tensor:
      // 21.0  // Sum of all elements
      ```

      Mathematical definition: sum(x, dim) = ∑ x[i] for all i in dimension dim

      Inputs:
      - `input` (Tensor): The input tensor.

      Attributes:
      - `keep_dim` (Bool): Whether to keep the reduced dimensions or not.
      - `dim_arg` (Array of Int32): Dimensions to reduce along.

      Outputs:
      - `output` (Tensor): The result tensor after applying the reduction.
    }];
}

def TTIR_MeanOp : TTIR_ReductionOp<"mean"> {
  let summary = "Mean reduction op.";
  let description = [{
    The `mean` operation computes the arithmetic mean of elements along specified dimensions of the input tensor.

    This operation reduces the input tensor by computing the average of all elements along the dimensions
    specified in `dim_arg`. If `dim_arg` is not provided, the mean is computed over all dimensions,
    resulting in a scalar value. If `keep_dim` is set to true, the reduced dimensions are retained
    with a size of 1.

    Example:
    ```mlir
    // Mean along dimension 1
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<2xf32>
    %result = ttir.mean(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : tensor<2x3xf32>, tensor<2xf32> -> tensor<2xf32>
    // Input tensor:
    // [[1.0, 2.0, 3.0],
    //  [4.0, 5.0, 6.0]]
    // Output tensor:
    // [2.0, 5.0]  // Mean of each row

    // Mean along dimension 0
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<3xf32>
    %result = ttir.mean(%input, %output) {keep_dim = false, dim_arg = [0: i32]} : tensor<2x3xf32>, tensor<3xf32> -> tensor<3xf32>
    // Input tensor:
    // [[1.0, 2.0, 3.0],
    //  [4.0, 5.0, 6.0]]
    // Output tensor:
    // [2.5, 3.5, 4.5]  // Mean of each column

    // Mean over all dimensions
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<f32>
    %result = ttir.mean(%input, %output) {keep_dim = false} : tensor<2x3xf32>, tensor<f32> -> tensor<f32>
    // Input tensor:
    // [[1.0, 2.0, 3.0],
    //  [4.0, 5.0, 6.0]]
    // Output tensor:
    // 3.5  // Mean of all elements
    ```

    Note: For integer input tensors, the result is typically rounded to the nearest integer according to the rounding mode.

    Mathematical definition: mean(x, dim) = (∑ x[i]) / n for all i in dimension dim, where n is the number of elements in dimension dim

    Inputs:
    - `input` (Tensor): The input tensor.

    Attributes:
    - `keep_dim` (Bool): Whether to keep the reduced dimensions or not.
    - `dim_arg` (Array of Int32): Dimensions to reduce along.

    Outputs:
    - `output` (Tensor): The result tensor after applying the reduction.
  }];
}

def TTIR_MaxOp : TTIR_ReductionOp<"max"> {
  let summary = "Maximum reduction operation.";
  let description = [{
    The `max` operation computes the maximum value of elements along specified dimensions of the input tensor.

    This operation reduces the input tensor by finding the maximum value of all elements along the dimensions
    specified in `dim_arg`. If `dim_arg` is not provided, the maximum is computed over all dimensions,
    resulting in a scalar value. If `keep_dim` is set to true, the reduced dimensions are retained
    with a size of 1.

    Example:
    ```mlir
    // Maximum along dimension 1
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<2xf32>
    %result = ttir.max(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : tensor<2x3xf32>, tensor<2xf32> -> tensor<2xf32>
    // Input tensor:
    // [[1.0, 5.0, 3.0],
    //  [4.0, 2.0, 6.0]]
    // Output tensor:
    // [5.0, 6.0]  // Maximum of each row

    // Maximum along dimension 0
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<3xf32>
    %result = ttir.max(%input, %output) {keep_dim = false, dim_arg = [0: i32]} : tensor<2x3xf32>, tensor<3xf32> -> tensor<3xf32>
    // Input tensor:
    // [[1.0, 5.0, 3.0],
    //  [4.0, 2.0, 6.0]]
    // Output tensor:
    // [4.0, 5.0, 6.0]  // Maximum of each column

    // Maximum over all dimensions
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<f32>
    %result = ttir.max(%input, %output) {keep_dim = false} : tensor<2x3xf32>, tensor<f32> -> tensor<f32>
    // Input tensor:
    // [[1.0, 5.0, 3.0],
    //  [4.0, 2.0, 6.0]]
    // Output tensor:
    // 6.0  // Maximum of all elements
    ```

    Note: When comparing with NaN values, NaN is typically not selected as the maximum value.

    Mathematical definition: max(x, dim) = max(x[i]) for all i in dimension dim

    Inputs:
    - `input` (Tensor): The input tensor.

    Attributes:
    - `keep_dim` (Bool): Whether to keep the reduced dimensions or not.
    - `dim_arg` (Array of Int32): Dimensions to reduce along.

    Outputs:
    - `output` (Tensor): The result tensor after applying the reduction.
  }];
}

def TTIR_MinOp : TTIR_ReductionOp<"min"> {
  let summary = "Minimum reduction operation.";
  let description = [{
    The `min` operation computes the minimum value of elements along specified dimensions of the input tensor.

    This operation reduces the input tensor by finding the minimum value of all elements along the dimensions
    specified in `dim_arg`. If `dim_arg` is not provided, the minimum is computed over all dimensions,
    resulting in a scalar value. If `keep_dim` is set to true, the reduced dimensions are retained
    with a size of 1.

    Example:
    ```mlir
    // Minimum along dimension 1
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<2xf32>
    %result = ttir.min(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : tensor<2x3xf32>, tensor<2xf32> -> tensor<2xf32>
    // Input tensor:
    // [[1.0, 5.0, 3.0],
    //  [4.0, 2.0, 6.0]]
    // Output tensor:
    // [1.0, 2.0]  // Minimum of each row

    // Minimum along dimension 0
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<3xf32>
    %result = ttir.min(%input, %output) {keep_dim = false, dim_arg = [0: i32]} : tensor<2x3xf32>, tensor<3xf32> -> tensor<3xf32>
    // Input tensor:
    // [[1.0, 5.0, 3.0],
    //  [4.0, 2.0, 6.0]]
    // Output tensor:
    // [1.0, 2.0, 3.0]  // Minimum of each column

    // Minimum over all dimensions
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<f32>
    %result = ttir.min(%input, %output) {keep_dim = false} : tensor<2x3xf32>, tensor<f32> -> tensor<f32>
    // Input tensor:
    // [[1.0, 5.0, 3.0],
    //  [4.0, 2.0, 6.0]]
    // Output tensor:
    // 1.0  // Minimum of all elements
    ```

    Note: When comparing with NaN values, NaN is typically not selected as the minimum value.

    Mathematical definition: min(x, dim) = min(x[i]) for all i in dimension dim

    Inputs:
    - `input` (Tensor): The input tensor.

    Attributes:
    - `keep_dim` (Bool): Whether to keep the reduced dimensions or not.
    - `dim_arg` (Array of Int32): Dimensions to reduce along.

    Outputs:
    - `output` (Tensor): The result tensor after applying the reduction.
  }];
}

def TTIR_ReduceAndOp : TTIR_ReductionOp<"reduce_and"> {
    let summary = "Logical AND reduction operation.";
    let description = [{
      The `reduce_and` operation performs a logical AND reduction along specified dimensions of the input tensor.

      This operation reduces the input tensor by applying a logical AND operation to all elements along the dimensions
      specified in `dim_arg`. If `dim_arg` is not provided, the reduction is computed over all dimensions,
      resulting in a scalar value. If `keep_dim` is set to true, the reduced dimensions are retained
      with a size of 1.

      The operation treats non-zero values as True and zero values as False when performing the logical AND.

      Example:
      ```mlir
      // Logical AND reduction along dimension 0
      %input = ... : tensor<4x4xi1>
      %output = ttir.empty() : tensor<4xi1>
      %result = ttir.reduce_and(%input, %output) {keep_dim = false, dim_arg = [0: i32]} : tensor<4x4xi1>, tensor<4xi1> -> tensor<4xi1>
      // Input tensor (where 1 represents True and 0 represents False):
      // [[1, 0, 1, 0],
      //  [1, 1, 1, 1],
      //  [0, 0, 1, 1],
      //  [0, 1, 1, 0]]
      // Output tensor:
      // [0, 0, 1, 0]  // Logical AND of each column

      // Logical AND reduction along dimension 1
      %input = ... : tensor<4x4xi1>
      %output = ttir.empty() : tensor<4xi1>
      %result = ttir.reduce_and(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : tensor<4x4xi1>, tensor<4xi1> -> tensor<4xi1>
      // Input tensor:
      // [[1, 0, 1, 0],
      //  [1, 1, 1, 1],
      //  [0, 0, 1, 1],
      //  [0, 1, 1, 0]]
      // Output tensor:
      // [0, 1, 0, 0]  // Logical AND of each row

      // Logical AND reduction over all dimensions
      %input = ... : tensor<4x4xi1>
      %output = ttir.empty() : tensor<i1>
      %result = ttir.reduce_and(%input, %output) {keep_dim = false} : tensor<4x4xi1>, tensor<i1> -> tensor<i1>
      // Input tensor:
      // [[1, 0, 1, 0],
      //  [1, 1, 1, 1],
      //  [0, 0, 1, 1],
      //  [0, 1, 1, 0]]
      // Output tensor:
      // 0  // Logical AND of all elements
      ```

      Mathematical definition: reduce_and(x, dim) = AND(x[i]) for all i in dimension dim

      Inputs:
      - `input` (Tensor): The input tensor.

      Attributes:
      - `keep_dim` (Bool): Whether to keep the reduced dimensions or not.
      - `dim_arg` (Array of Int32): Dimensions to reduce along.

      Outputs:
      - `output` (Tensor): The result tensor after applying the reduction.
    }];
}

def TTIR_ReduceOrOp : TTIR_ReductionOp<"reduce_or"> {
    let summary = "Logical OR reduction operation.";
    let description = [{
      The `reduce_or` operation performs a logical OR reduction along specified dimensions of the input tensor.

      This operation reduces the input tensor by applying a logical OR operation to all elements along the dimensions
      specified in `dim_arg`. If `dim_arg` is not provided, the reduction is computed over all dimensions,
      resulting in a scalar value. If `keep_dim` is set to true, the reduced dimensions are retained
      with a size of 1.

      The operation treats non-zero values as True and zero values as False when performing the logical OR.

      Example:
      ```mlir
      // Logical OR reduction along dimension 0
      %input = ... : tensor<4x4xi1>
      %output = ttir.empty() : tensor<4xi1>
      %result = ttir.reduce_or(%input, %output) {keep_dim = false, dim_arg = [0: i32]} : tensor<4x4xi1>, tensor<4xi1> -> tensor<4xi1>
      // Input tensor (where 1 represents True and 0 represents False):
      // [[1, 0, 0, 0],
      //  [1, 1, 0, 1],
      //  [0, 0, 0, 1],
      //  [0, 0, 0, 0]]
      // Output tensor:
      // [1, 1, 0, 1]  // Logical OR of each column

      // Logical OR reduction along dimension 1
      %input = ... : tensor<4x4xi1>
      %output = ttir.empty() : tensor<4xi1>
      %result = ttir.reduce_or(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : tensor<4x4xi1>, tensor<4xi1> -> tensor<4xi1>
      // Input tensor:
      // [[1, 0, 0, 0],
      //  [1, 1, 0, 1],
      //  [0, 0, 0, 1],
      //  [0, 0, 0, 0]]
      // Output tensor:
      // [1, 1, 1, 0]  // Logical OR of each row

      // Logical OR reduction over all dimensions
      %input = ... : tensor<4x4xi1>
      %output = ttir.empty() : tensor<i1>
      %result = ttir.reduce_or(%input, %output) {keep_dim = false} : tensor<4x4xi1>, tensor<i1> -> tensor<i1>
      // Input tensor:
      // [[1, 0, 0, 0],
      //  [1, 1, 0, 1],
      //  [0, 0, 0, 1],
      //  [0, 0, 0, 0]]
      // Output tensor:
      // 1  // Logical OR of all elements
      ```

      Mathematical definition: reduce_or(x, dim) = OR(x[i]) for all i in dimension dim

      Inputs:
      - `input` (Tensor): The input tensor.

      Attributes:
      - `keep_dim` (Bool): Whether to keep the reduced dimensions or not.
      - `dim_arg` (Array of Int32): Dimensions to reduce along.

      Outputs:
      - `output` (Tensor): The result tensor after applying the reduction.
    }];
}

def TTIR_ProdOp : TTIR_ReductionOp<"prod"> {
  let summary = "Product reduction op.";
  let description = [{
        The `prod` operation computes the product of elements along specified dimensions of the input tensor.

    This operation reduces the input tensor by multiplying all elements along the dimensions
    specified in `dim_arg`. If `dim_arg` is not provided, the product is computed over all dimensions,
    resulting in a scalar value. If `keep_dim` is set to true, the reduced dimensions are retained
    with a size of 1.

    Example:
    ```mlir
    // Product along dimension 0
    %input = ... : tensor<2x3xi32>
    %output = ttir.empty() : tensor<3xi32>
    %result = ttir.prod(%input, %output) {keep_dim = false, dim_arg = [0: i32]} : tensor<2x3xi32>, tensor<3xi32> -> tensor<3xi32>
    // Input tensor:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // Output tensor:
    // [4, 10, 18]  // Product of each column

    // Product along dimension 1
    %input = ... : tensor<2x3xi32>
    %output = ttir.empty() : tensor<2xi32>
    %result = ttir.prod(%input, %output) {keep_dim = false, dim_arg = [1: i32]} : tensor<2x3xi32>, tensor<2xi32> -> tensor<2xi32>
    // Input tensor:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // Output tensor:
    // [6, 120]  // Product of each row

    // Product over all dimensions
    %input = ... : tensor<2x3xi32>
    %output = ttir.empty() : tensor<i32>
    %result = ttir.prod(%input, %output) {keep_dim = false} : tensor<2x3xi32>, tensor<i32> -> tensor<i32>
    // Input tensor:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // Output tensor:
    // 720  // Product of all elements
    ```

    Note: For floating-point inputs, the order of multiplication may affect the result due to floating-point precision issues.

    Mathematical definition: prod(x, dim) = ∏ x[i] for all i in dimension dim

    Inputs:
    - `input` (Tensor): The input tensor.

    Attributes:
    - `keep_dim` (Bool): Whether to keep the reduced dimensions or not.
    - `dim_arg` (Array of Int32): Dimensions to reduce along.

    Outputs:
    - `output` (Tensor): The result tensor after applying the reduction.
  }];
}

def TTIR_EmbeddingOp : TTIR_NamedOp<"embedding"> {
    let summary = "Embedding lookup operation.";
    let description = [{
      The `embedding` operation performs a lookup in an embedding table (weight matrix) using integer indices.

      This operation takes an input tensor of indices and a weight tensor representing the embedding table.
      For each index in the input tensor, it retrieves the corresponding row from the weight tensor.
      The result is a tensor where each input index is replaced by its corresponding embedding vector.

      Example:
      ```mlir
      // Embedding lookup
      %input = ... : tensor<2x3xi32>  // Batch of indices
      %weight = ... : tensor<10x4xf32>  // Embedding table with 10 entries of dimension 4
      %output = ttir.empty() : tensor<2x3x4xf32>
      %result = ttir.embedding(%input, %weight, %output) : tensor<2x3xi32>, tensor<10x4xf32>, tensor<2x3x4xf32> -> tensor<2x3x4xf32>

      // Input tensor (indices):
      // [[0, 2, 5],
      //  [7, 1, 9]]

      // Weight tensor (embedding table):
      // [[0.1, 0.2, 0.3, 0.4],  // embedding vector for index 0
      //  [0.5, 0.6, 0.7, 0.8],  // embedding vector for index 1
      //  [0.9, 1.0, 1.1, 1.2],  // embedding vector for index 2
      //  ...
      //  [1.7, 1.8, 1.9, 2.0]]  // embedding vector for index 9

      // Output tensor:
      // [[[0.1, 0.2, 0.3, 0.4],  // embedding for index 0
      //   [0.9, 1.0, 1.1, 1.2],  // embedding for index 2
      //   [...]],                 // embedding for index 5
      //  [[...],                  // embedding for index 7
      //   [0.5, 0.6, 0.7, 0.8],  // embedding for index 1
      //   [...]]]                 // embedding for index 9
      ```

      Note: The indices in the input tensor must be valid indices into the first dimension of the weight tensor.

      Inputs:
      - `input` (Tensor): The input tensor containing indices.
      - `weight` (Tensor): The embedding table tensor.

      Outputs:
      - `result` (Tensor): The resulting tensor containing the embeddings.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_EmbeddingBackwardOp : TTIR_NamedOp<"embedding_backward"> {
    let summary = "Embedding backward operation.";
    let description = [{
      The `embedding_backward` operation computes the gradient of the embedding operation with respect to the weight tensor.

      This operation takes an input tensor of indices, the original weight tensor, and the gradient tensor from the forward pass.
      It computes how the embedding weights should be updated during backpropagation by accumulating gradients at the appropriate
      indices in the weight tensor.

      Example:
      ```mlir
      // Embedding backward
      %input = ... : tensor<2x3xi32>  // Original indices used in the forward pass
      %weight = ... : tensor<10x4xf32>  // Original embedding table
      %in_gradient = ... : tensor<2x3x4xf32>  // Gradient from the forward pass
      %output = ttir.empty() : tensor<10x4xf32>  // Gradient for the embedding table
      %result = ttir.embedding_backward(%input, %weight, %in_gradient, %output) :
          tensor<2x3xi32>, tensor<10x4xf32>, tensor<2x3x4xf32>, tensor<10x4xf32> -> tensor<10x4xf32>

      // Input tensor (indices):
      // [[0, 2, 5],
      //  [7, 1, 9]]

      // Input gradient tensor (from forward pass):
      // [[[0.1, 0.2, 0.3, 0.4],  // gradient for embedding of index 0
      //   [0.5, 0.6, 0.7, 0.8],  // gradient for embedding of index 2
      //   [...]],                 // gradient for embedding of index 5
      //  [[...],                  // gradient for embedding of index 7
      //   [0.9, 1.0, 1.1, 1.2],  // gradient for embedding of index 1
      //   [...]]]                 // gradient for embedding of index 9

      // Output tensor (gradient for the embedding table):
      // The gradients are accumulated at the corresponding indices in the weight tensor.
      // For example, at index 0, the gradient is [0.1, 0.2, 0.3, 0.4]
      ```

      Note: If the same index appears multiple times in the input tensor, the gradients are accumulated (added) at that index in the output tensor.

      Inputs:
      - `input` (Tensor): The original input tensor containing indices used in the forward pass.
      - `weight` (Tensor): The original embedding table tensor.
      - `in_gradient` (Tensor): The gradient tensor from the forward pass.

      Outputs:
      - `result` (Tensor): The gradient tensor for the embedding table.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         AnyRankedTensor:$in_gradient,
                         AnyRankedTensor:$output);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_CumSumOp : TTIR_NamedOp<"cumsum"> {
  let summary = "Cumulative sum operation.";
  let description = [{
    The `cumsum` operation computes the cumulative sum of elements along a specified dimension of the input tensor.

    For each position in the output tensor, this operation computes the sum of all elements in the input tensor
    along the specified dimension up to and including that position. The shape of the output tensor matches
    the shape of the input tensor.

    Example:
    ```mlir
    // Cumulative sum along dimension 0
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<2x3xf32>
    %result = ttir.cumsum(%input, %output) {dim = 0 : i64} : tensor<2x3xf32>, tensor<2x3xf32> -> tensor<2x3xf32>
    // Input tensor:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // Output tensor:
    // [[1, 2, 3],   // first row remains the same
    //  [5, 7, 9]]   // each element is the sum of the corresponding column up to this point

    // Cumulative sum along dimension 1
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<2x3xf32>
    %result = ttir.cumsum(%input, %output) {dim = 1 : i64} : tensor<2x3xf32>, tensor<2x3xf32> -> tensor<2x3xf32>
    // Input tensor:
    // [[1, 2, 3],
    //  [4, 5, 6]]
    // Output tensor:
    // [[1, 3, 6],   // each element is the sum of the corresponding row up to this point
    //  [4, 9, 15]]
    ```

    Inputs:
    - `input` (Tensor): The input tensor.

    Attributes:
    - `dim` (Integer): The dimension along which to compute the cumulative sum.

    Outputs:
    - `result` (Tensor): The tensor containing the cumulative sums.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output,
                       I64Attr:$dim);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TTIR_SoftmaxOp : TTIR_NamedOp<"softmax"> {
    let summary = "Softmax normalization operation.";
    let description = [{
      The `softmax` operation applies the softmax function along a specified dimension of the input tensor.

      The softmax function transforms each element of the input tensor to a value between 0 and 1, such that
      the sum of all elements along the specified dimension equals 1. This is commonly used to convert a vector
      of real numbers into a probability distribution.

      The softmax function is defined as:
      softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j in the specified dimension

      Example:
      ```mlir
      // Softmax along dimension 1
      %input = ... : tensor<2x3xf32>
      %output = ttir.empty() : tensor<2x3xf32>
      %result = ttir.softmax(%input, %output) {dimension = 1 : i32} : tensor<2x3xf32>, tensor<2x3xf32> -> tensor<2x3xf32>
      // Input tensor:
      // [[1.0, 2.0, 3.0],
      //  [4.0, 1.0, 2.0]]
      // Output tensor (approximate values):
      // [[0.09, 0.24, 0.67],  // sum = 1.0
      //  [0.71, 0.09, 0.20]]  // sum = 1.0
      ```

      Note: For numerical stability, the implementation typically subtracts the maximum value in each slice
      before applying the exponential function.

      Inputs:
      - `input` (Tensor): The input tensor.

      Attributes:
      - `dimension` (Integer): The dimension along which to apply the softmax function.
      - `numericStable` (Boolean, default=false): Whether to use numerically stable computation.

      Outputs:
      - `result` (Tensor): The tensor after applying the softmax function.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dimension,
                         DefaultValuedAttr<BoolAttr, "false">:$numericStable);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_SortOp : TTIR_NamedOp<"sort"> {
  let summary = "Sort operation.";
  let description = [{
    Sorts elements of a tensor along a given dimension.

    Input:
      - input: AnyRankedTensor

    Attributes:
      - dim (int32): The dimension to sort along (default: -1, the last dim).
      - descending (bool): If True, sort in descending order (default: False).
      - stable (bool): If True, ensures stable sort (equal elements keep order).

    Returns a tuple:
      - values: the sorted tensor.
      - indices: the original indices of the sorted values.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       Variadic<AnyRankedTensor>:$output,
                       DefaultValuedAttr<SI32Attr, "-1">:$dim,
                       DefaultValuedAttr<BoolAttr, "false">:$descending,
                       DefaultValuedAttr<BoolAttr, "false">:$stable);

  let results = (outs AnyRankedTensor:$values,
                      AnyRankedTensor:$indices);

  let hasVerifier = 1;
}

def TTIR_TransposeOp : TTIR_NamedOp<"transpose", [TTIR_TensorManipulation]> {
    let summary = "Tensor transpose operation.";
    let description = [{
      The `transpose` operation swaps two dimensions of a tensor.

      This operation exchanges the positions of two specified dimensions in the input tensor,
      effectively transposing those dimensions. The shape of the output tensor is the same as
      the input tensor, except that the dimensions specified by `dim0` and `dim1` are swapped.

      Example:
      ```mlir
      // Transpose dimensions 0 and 1
      %input = ... : tensor<2x3x4xf32>
      %output = ttir.empty() : tensor<3x2x4xf32>
      %result = ttir.transpose(%input, %output) {dim0 = 0 : i32, dim1 = 1 : i32} :
          tensor<2x3x4xf32>, tensor<3x2x4xf32> -> tensor<3x2x4xf32>
      // Input tensor shape: [2, 3, 4]
      // Output tensor shape: [3, 2, 4]

      // Transpose dimensions 1 and 2
      %input = ... : tensor<2x3x4xf32>
      %output = ttir.empty() : tensor<2x4x3xf32>
      %result = ttir.transpose(%input, %output) {dim0 = 1 : i32, dim1 = 2 : i32} :
          tensor<2x3x4xf32>, tensor<2x4x3xf32> -> tensor<2x4x3xf32>
      // Input tensor shape: [2, 3, 4]
      // Output tensor shape: [2, 4, 3]
      ```

      Inputs:
      - `input` (Tensor): The input tensor.

      Attributes:
      - `dim0` (Integer): The first dimension to swap.
      - `dim1` (Integer): The second dimension to swap.

      Outputs:
      - `result` (Tensor): The transposed tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim0,
                         SI32Attr:$dim1);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

def TTIR_ConcatOp : TTIR_NamedOp<"concat"> {
    let summary = "Tensor concatenation operation.";
    let description = [{
      The `concat` operation joins multiple tensors along a specified dimension.

      This operation concatenates a list of tensors along the dimension specified by `dim`.
      All input tensors must have the same shape except for the dimension being concatenated,
      and the output tensor's shape will match the input tensors except for the concatenated
      dimension, which will be the sum of the input dimensions.

      Example:
      ```mlir
      // Concatenate along dimension 0
      %input1 = ... : tensor<2x3xf32>
      %input2 = ... : tensor<3x3xf32>
      %output = ttir.empty() : tensor<5x3xf32>
      %result = ttir.concat(%input1, %input2, %output) {dim = 0 : i32} :
          tensor<2x3xf32>, tensor<3x3xf32>, tensor<5x3xf32> -> tensor<5x3xf32>
      // Input1 shape: [2, 3]
      // Input2 shape: [3, 3]
      // Output shape: [5, 3]

      // Concatenate along dimension 1
      %input1 = ... : tensor<2x3xf32>
      %input2 = ... : tensor<2x2xf32>
      %output = ttir.empty() : tensor<2x5xf32>
      %result = ttir.concat(%input1, %input2, %output) {dim = 1 : i32} :
          tensor<2x3xf32>, tensor<2x2xf32>, tensor<2x5xf32> -> tensor<2x5xf32>
      // Input1 shape: [2, 3]
      // Input2 shape: [2, 2]
      // Output shape: [2, 5]
      ```

      Inputs:
      - `inputs` (Variadic Tensor): A list of input tensors to concatenate.

      Attributes:
      - `dim` (Integer): The dimension along which to concatenate the tensors.

      Outputs:
      - `result` (Tensor): The concatenated tensor.
    }];

    let arguments = (ins Variadic<AnyRankedTensor>:$inputs,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTIR_RepeatOp : TTIR_NamedOp<"repeat"> {
  let summary = "Repeat operation.";
  let description = [{
    The `repeat` operation creates a new tensor by replicating the input tensor's elements
    along specified dimensions.

    This operation repeats the entire input tensor along each dimension according to the
    values specified in the `repeat_dimensions` attribute. The resulting tensor's shape
    is the product of the input tensor's shape and the corresponding repeat values.

    Example:
    ```mlir
    // Repeat a 2x3 tensor with repeat dimensions [2, 2]
    %input = ... : tensor<2x3xf32>
    %output = ttir.empty() : tensor<4x6xf32>
    %result = ttir.repeat(%input, %output) {repeat_dimensions = [2, 2]} :
        tensor<2x3xf32>, tensor<4x6xf32> -> tensor<4x6xf32>
    // Input tensor:
    // [[1.0, 2.0, 3.0],
    //  [4.0, 5.0, 6.0]]
    // Output tensor:
    // [[1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    //  [4.0, 5.0, 6.0, 4.0, 5.0, 6.0],
    //  [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
    //  [4.0, 5.0, 6.0, 4.0, 5.0, 6.0]]

    // Repeat a 2x2 tensor with repeat dimensions [1, 3]
    %input = ... : tensor<2x2xf32>
    %output = ttir.empty() : tensor<2x6xf32>
    %result = ttir.repeat(%input, %output) {repeat_dimensions = [1, 3]} :
        tensor<2x2xf32>, tensor<2x6xf32> -> tensor<2x6xf32>
    // Input tensor:
    // [[1.0, 2.0],
    //  [3.0, 4.0]]
    // Output tensor:
    // [[1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    //  [3.0, 4.0, 3.0, 4.0, 3.0, 4.0]]
    ```

    Inputs:
    - `input` (Tensor): The input tensor to repeat.

    Attributes:
    - `repeat_dimensions` (Array of Integer): The number of times to repeat the tensor along each dimension.

    Outputs:
    - `result` (Tensor): The repeated tensor.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output,
                       DenseI64ArrayAttr:$repeat_dimensions);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_RepeatInterleaveOp : TTIR_NamedOp<"repeat_interleave"> {
    let summary = "Tensor repeat interleave operation.";
    let description = [{
      The `repeat_interleave` operation repeats elements of a tensor along a specified dimension.

      Unlike the `repeat` operation which repeats the entire tensor, this operation repeats each individual
      element of the input tensor the specified number of times along the given dimension. This creates an
      interleaved pattern of repeated values.

      Example:
      ```mlir
      // Repeat interleave along dimension 0 with repeats=2
      %input = ... : tensor<2x3xf32>
      %output = ttir.empty() : tensor<4x3xf32>
      %result = ttir.repeat_interleave(%input, %output) {repeats = 2 : ui32, dim = 0 : i32} :
          tensor<2x3xf32>, tensor<4x3xf32> -> tensor<4x3xf32>
      // Input tensor:
      // [[1.0, 2.0, 3.0],
      //  [4.0, 5.0, 6.0]]
      // Output tensor:
      // [[1.0, 2.0, 3.0],  // First row repeated
      //  [1.0, 2.0, 3.0],
      //  [4.0, 5.0, 6.0],  // Second row repeated
      //  [4.0, 5.0, 6.0]]

      // Repeat interleave along dimension 1 with repeats=3
      %input = ... : tensor<2x2xf32>
      %output = ttir.empty() : tensor<2x6xf32>
      %result = ttir.repeat_interleave(%input, %output) {repeats = 3 : ui32, dim = 1 : i32} :
          tensor<2x2xf32>, tensor<2x6xf32> -> tensor<2x6xf32>
      // Input tensor:
      // [[1.0, 2.0],
      //  [3.0, 4.0]]
      // Output tensor:
      // [[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],  // Each element repeated 3 times
      //  [3.0, 3.0, 3.0, 4.0, 4.0, 4.0]]
      ```

      Inputs:
      - `input` (Tensor): The input tensor.

      Attributes:
      - `repeats` (Integer): The number of times to repeat each element.
      - `dim` (Integer): The dimension along which to repeat elements.

      Outputs:
      - `result` (Tensor): The tensor with repeated elements.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         UI32Attr:$repeats,
                         SI32Attr:$dim);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_UpdateCacheOp : TTIR_NamedOp<"update_cache"> {
  let summary = "Cache update operation.";
  let description = [{
    The `update_cache` operation updates a cache tensor with values from an input tensor at specific indices.

    This operation is commonly used in sequence models like transformers to update a key-value cache
    with new token information. It takes a cache tensor, an input tensor, and update indices, and
    updates the cache at the specified positions.

    Example:
    ```mlir
    // Update cache at specific indices
    %cache = ... : tensor<2x16x64xf32>  // Batch size 2, sequence length 16, hidden dim 64
    %input = ... : tensor<2x1x64xf32>   // New token embeddings
    %update_index = ... : tensor<1xi32> // Update at position [15]
    %result = ttir.update_cache(%cache, %input, %update_index) {batch_offset = 0 : i32} :
        tensor<2x16x64xf32>, tensor<2x1x64xf32>, tensor<1xi32> -> tensor<2x16x64xf32>
    // The cache tensor is updated at position 15 for both batches with the values from input
    ```

    Inputs:
    - `cache` (Tensor): The cache tensor to be updated.
    - `input` (Tensor): The input tensor containing new values.
    - `update_index` (Tensor): Indices specifying where to update the cache.

    Attributes:
    - `batch_offset` (Integer): Offset in the batch dimension.

    Outputs:
    - `result` (Tensor): The updated cache tensor.
  }];

  let arguments = (ins AnyRankedTensor:$cache,
                       AnyRankedTensor:$input,
                       AnyRankedTensor:$update_index,
                       I32Attr:$batch_offset);

  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getCacheMutable(); }
  }];

  let hasVerifier = 1;
}

def TTIR_FillCacheOp : TTIR_NamedOp<"fill_cache"> {
  let summary = "Cache filling operation.";
  let description = [{
      The `fill_cache` operation fills a cache tensor with values from an input tensor.

    Unlike `update_cache` which updates specific positions, this operation fills the entire cache
    or a contiguous section of it with values from the input tensor. This is commonly used to
    initialize a cache in sequence models.

    Example:
    ```mlir
    // Fill cache with input values
    %cache = ... : tensor<2x16x64xf32>  // Batch size 2, sequence length 16, hidden dim 64
    %input = ... : tensor<2x16x64xf32>  // Initial values for the entire cache
    %result = ttir.fill_cache(%cache, %input) {batch_offset = 0 : i32} :
        tensor<2x16x64xf32>, tensor<2x16x64xf32> -> tensor<2x16x64xf32>
    // The entire cache tensor is filled with values from input

    // Fill a portion of the cache
    %cache = ... : tensor<2x16x64xf32>  // Batch size 2, sequence length 16, hidden dim 64
    %input = ... : tensor<2x8x64xf32>   // Values for half of the cache
    %result = ttir.fill_cache(%cache, %input) {batch_offset = 0 : i32} :
        tensor<2x16x64xf32>, tensor<2x8x64xf32> -> tensor<2x16x64xf32>
    // The first 8 positions of the cache are filled with values from input
    ```

    Inputs:
    - `cache` (Tensor): The cache tensor to be filled.
    - `input` (Tensor): The input tensor containing the values to fill the cache with.

    Attributes:
    - `batch_offset` (Integer): Offset in the batch dimension.

    Outputs:
    - `result` (Tensor): The filled cache tensor.
  }];

  let arguments = (ins AnyRankedTensor:$cache,
                       AnyRankedTensor:$input,
                       I32Attr:$batch_offset);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
      MutableOperandRange getDpsInitsMutable() { return getCacheMutable(); }
  }];
}

def TTIR_BroadcastOp : TTIR_NamedOp<"broadcast"> {
    let summary = "Broadcast operation.";
    let description = [{
      The `broadcast` operation expands the dimensions of an input tensor according to specified broadcast dimensions.

      This operation takes an input tensor and broadcasts it to a larger shape by repeating elements along
      dimensions where the input has size 1 and the output has a larger size. This is commonly used to
      make tensors compatible for elementwise operations.

      Example:
      ```mlir
      // Broadcast a tensor from shape [1, 1, 32] to [1, 16, 32]
      %input = ... : tensor<1x1x32xf32>
      %output = ttir.empty() : tensor<1x16x32xf32>
      %result = ttir.broadcast(%input, %output) {broadcast_dimensions = [1, 16, 1]} :
          tensor<1x1x32xf32>, tensor<1x16x32xf32> -> tensor<1x16x32xf32>
      // The input tensor is repeated 16 times along the second dimension

      // Broadcast a tensor from shape [1, 3] to [2, 3]
      %input = ... : tensor<1x3xf32>
      %output = ttir.empty() : tensor<2x3xf32>
      %result = ttir.broadcast(%input, %output) {broadcast_dimensions = [2, 1]} :
          tensor<1x3xf32>, tensor<2x3xf32> -> tensor<2x3xf32>
      // The input tensor is repeated 2 times along the first dimension
      ```

      Note: Currently, when generating a TTNN executable, the broadcast and repeat operations share the same
      semantics due to the lack of tensor view support in TTNN. As a result, the broadcast operation is
      lowered to a repeat operation in the TTNN compilation pipeline.

      Inputs:
      - `input` (Tensor): The input tensor to broadcast.

      Attributes:
      - `broadcast_dimensions` (Array of Integer): The number of times to broadcast the tensor along each dimension.

      Outputs:
      - `result` (Tensor): The broadcasted tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         DenseI64ArrayAttr:$broadcast_dimensions);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTIR_Conv2dOp : TTIR_NamedOp<"conv2d"> {
    let summary = "Conv2d operation.";
    let description = [{
      Applies a 2D convolution over an input image composed of several input planes.

      This operation performs a 2D convolution on the input tensor using the provided weight tensor
      and optional bias. It supports configurable stride, padding, dilation, and grouping parameters
      to control the convolution behavior.

      Example:
      ```mlir
      // Basic 2D convolution
      %input = ... : tensor<1x28x28x3xf32>    // Batch size 1, 28x28 image, 3 channels
      %weight = ... : tensor<16x3x3x3xf32>    // 16 output channels, 3 input channels, 3x3 kernel
      %bias = ... : tensor<1x1x1x16xf32>      // Bias for 16 output channels
      %output = ttir.empty() : tensor<1x26x26x16xf32>  // Output shape with no padding
      %result = ttir.conv2d(%input, %weight, %bias, %output) {
          stride = [1, 1],
          padding = [0, 0, 0, 0],
          dilation = [1, 1],
          groups = 1
      } : tensor<1x28x28x3xf32>, tensor<16x3x3x3xf32>, tensor<1x1x1x16xf32>, tensor<1x26x26x16xf32> -> tensor<1x26x26x16xf32>

      // Convolution with stride 2 and padding
      %input = ... : tensor<1x28x28x3xf32>    // Batch size 1, 28x28 image, 3 channels
      %weight = ... : tensor<16x3x3x3xf32>    // 16 output channels, 3 input channels, 3x3 kernel
      %bias = ... : tensor<1x1x1x16xf32>      // Bias for 16 output channels
      %output = ttir.empty() : tensor<1x14x14x16xf32>  // Output shape with stride 2
      %result = ttir.conv2d(%input, %weight, %bias, %output) {
          stride = [2, 2],
          padding = [1, 1, 1, 1],
          dilation = [1, 1],
          groups = 1
      } : tensor<1x28x28x3xf32>, tensor<16x3x3x3xf32>, tensor<1x1x1x16xf32>, tensor<1x14x14x16xf32> -> tensor<1x14x14x16xf32>
      ```

      Inputs:
      - `input` (AnyRankedTensor): expected in the following format (N, H_in, W_in, C) where:
        - N is the batch size
        - H_in is the height of the input planes
        - W_in is the width of the input planes
        - C is the number of channels
      - `weight` (AnyRankedTensor): expected in the following format (O, C/G, K_H, K_W) where:
        - C is the number of input channels
        - O is the number of output channels
        - G is the number of groups
        - K_H is the height of the kernel
        - K_W is the width of the kernel
      - `bias` Optional<AnyRankedTensor>: expected in the following format (1, 1, 1, O).

      Attributes:
      - `stride` (i32 | array<2xi32>):
        - i32: Same stride for height and width dimensions (sH = sW = value).
        - array<2xi32>: [sH, sW] where sH is stride for height and sW is stride for width.
      - `padding` (i32 | array<2xi32> | array<4xi32>):
        - i32: Same padding for all sides (pT = pL = pB = pR = value).
        - array<2xi32>: [pH, pW] where pH is padding for height (top/bottom) and pW is padding for width (left/right).
        - array<4xi32>: [pT, pL, pB, pR] for top, left, bottom, and right padding respectively.
      - `dilation` (i32 | array<2xi32>): Spacing between kernel elements.
        - i32: Same dilation for height and width dimensions (dH = dW = value).
        - array<2xi32>: [dH, dW] where dH is dilation for height and dW is dilation for width.
      - `groups` (i32): Number of blocked connections from input channels to output channels. Input and output channels must both be divisible by groups.

      Outputs:
      - `result` AnyRankedTensor: expected in the following format (N, H_out, W_out, O) where:
        - `H_out = (H_in + pT + pB - dH * (K_H - 1) - 1) / sH + 1`
        - `W_out = (W_in + pL + pR - dW * (K_W - 1) - 1) / sW + 1`
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$stride,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$padding,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$dilation,
                         I32Attr:$groups,
                         DefaultValuedAttr<TTIR_FlattenedCompatInfoAttr, "nullptr">:$flattened_compat_info);

    let extraClassDeclaration = [{
      // Get number of output channels
      int64_t getOutputChannelSize();
      bool isBiasCompatible(llvm::ArrayRef<int64_t> biasDims);
      MutableOperandRange getDpsInitsMutable() { return ttir::getDpsOutputs(this); }
    }];

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_ConvTranspose2dOp : TTIR_NamedOp<"conv_transpose2d"> {
    let summary = "ConvTranspose2d operation.";
    let description = [{
      Applies a 2D transposed convolution operator over an input image composed of several input planes.

      This operation performs the gradient of a 2D convolution with respect to the input, which is useful
      for tasks like upsampling feature maps in neural networks. It supports configurable stride, padding,
      dilation, output padding, and grouping parameters.

      Example:
      ```mlir
      // Basic 2D transposed convolution
      %input = ... : tensor<1x14x14x16xf32>   // Batch size 1, 14x14 feature map, 16 channels
      %weight = ... : tensor<16x8x3x3xf32>    // 16 input channels, 8 output channels, 3x3 kernel
      %bias = ... : tensor<1x1x1x8xf32>       // Bias for 8 output channels
      %output = ttir.empty() : tensor<1x28x28x8xf32>  // Output shape with stride 2
      %result = ttir.conv_transpose2d(%input, %weight, %bias, %output) {
          stride = [2, 2],
          padding = [0, 0, 0, 0],
          dilation = [1, 1],
          output_padding = [0, 0],
          groups = 1
      } : tensor<1x14x14x16xf32>, tensor<16x8x3x3xf32>, tensor<1x1x1x8xf32>, tensor<1x28x28x8xf32> -> tensor<1x28x28x8xf32>

      // Transposed convolution with padding and output padding
      %input = ... : tensor<1x14x14x16xf32>   // Batch size 1, 14x14 feature map, 16 channels
      %weight = ... : tensor<16x8x4x4xf32>    // 16 input channels, 8 output channels, 4x4 kernel
      %bias = ... : tensor<1x1x1x8xf32>       // Bias for 8 output channels
      %output = ttir.empty() : tensor<1x29x29x8xf32>  // Output shape with output padding
      %result = ttir.conv_transpose2d(%input, %weight, %bias, %output) {
          stride = [2, 2],
          padding = [1, 1, 1, 1],
          dilation = [1, 1],
          output_padding = [1, 1],
          groups = 1
      } : tensor<1x14x14x16xf32>, tensor<16x8x4x4xf32>, tensor<1x1x1x8xf32>, tensor<1x29x29x8xf32> -> tensor<1x29x29x8xf32>
      ```

      Inputs:
      - `input` AnyRankedTensor: expected in the following format (N, H_in, W_in, C) where:
        - N is the batch size
        - H_in is the height of the input planes
        - W_in is the width of the input planes
        - C is the number of channels
      - `weight` (AnyRankedTensor): expected in the following format (C, O/G, K_H, K_W) where:
        - C is the number of input channels
        - O is the number of output channels
        - G is the number of groups
        - K_H is the height of the kernel
        - K_W is the width of the kernel
      - `bias` Optional<AnyRankedTensor>: expected in the following format (1, 1, 1, O).

      Attributes:
      - `stride` (i32 | array<2xi32>): Controls the stride for the cross-correlation.
      - `padding` (i32 | array<2xi32> | array<4xi32>): Controls the amount of implicit zero padding on both sides for dilation * (kernel_size - 1) - padding number of points.
      - `output_padding` (i32 | array<2xi32>): Controls the additional size added to one side of the output shape.
      - `dilation` (i32 | array<2xi32>): Controls the spacing between the kernel points
      - `groups` i32: Controls the connections between inputs and outputs. Must be divisible by input and output channels.

      Outputs:
      - `result` AnyRankedTensor: expected in the following format (N, H_out, W_out, O) where:
        - H_out = (H_in - 1) * stride[0] - (padding_top + padding_bottom) + dilation[0] * (K_H - 1) + output_padding[0] + 1
        - W_out = (W_in - 1) * stride[1] - (padding_left + padding_right) + dilation[1] * (K_W - 1) + output_padding[1] + 1
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$stride,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$padding,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$output_padding,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$dilation,
                         I32Attr:$groups,
                         DefaultValuedAttr<TTIR_FlattenedCompatInfoAttr, "nullptr">:$flattened_compat_info);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_ConvolutionOp : TTIR_NamedOp<"convolution", [DeclareOpInterfaceMethods<TTIR_QuantizableOpInterface, ["isQuantizedRewriteFavorable", "rewriteWithQuantizedInputs"]>]> {
  let summary = "Generalized convolution operation.";
  let description = [{
    This operation is a more flexible form of convolution that can handle arbitrary dimensionality
    and supports various configuration options. It's designed to be a generalization of specific
    convolution operations like `conv2d` and `conv_transpose2d`.

    Example:
    ```mlir
    // 2D convolution using the generalized convolution operation
    %lhs = ... : tensor<1x32x32x3xf32>     // Input tensor: batch size 1, 32x32 image, 3 channels
    %rhs = ... : tensor<5x5x3x16xf32>      // Filter tensor: 5x5 kernel, 3 input channels, 16 output channels
    %output = ttir.empty() : tensor<1x28x28x16xf32>  // Output tensor
    %result = ttir.convolution(%lhs, %rhs, %output) {
        window_strides = [1, 1],
        padding = [[0, 0], [0, 0]],
        lhs_dilation = [1, 1],
        rhs_dilation = [1, 1],
        window_reversal = [false, false],
        dimension_numbers = {
            input_batch_dimension = 0,
            input_feature_dimension = 3,
            input_spatial_dimensions = [1, 2],
            kernel_input_feature_dimension = 2,
            kernel_output_feature_dimension = 3,
            kernel_spatial_dimensions = [0, 1],
            output_batch_dimension = 0,
            output_feature_dimension = 3,
            output_spatial_dimensions = [1, 2]
        },
        feature_group_count = 1,
        batch_group_count = 1
    } : tensor<1x32x32x3xf32>, tensor<5x5x3x16xf32>, tensor<1x28x28x16xf32> -> tensor<1x28x28x16xf32>
    ```

    Inputs:
    - `input` - The input tensor.
    - `weight` - The filter/kernel tensor.
    - `bias` - The bias tensor.

    Attributes:
    - `window_strides` (Array): Stride of the sliding window for each spatial dimension.
    - `padding` (Array): Padding applied to the input in each spatial dimension.
    - `input_dilation` (Array): Dilation factor for the input in each spatial dimension.
    - `weight_dilation` (Array): Dilation factor for the filter in each spatial dimension.
    - `window_reversal` (Array): Whether to reverse the window in each spatial dimension.
    - `convolution_layout` (Struct): Specifies the dimension numbering in the inputs and outputs.
    - `feature_group_count` (Integer): Number of feature groups for grouped convolution.
    - `batch_group_count` (Integer): Number of batch groups for grouped convolution.

    Outputs:
    - `result` (Tensor): Output tensor containing the result of the convolution.
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias,
    AnyRankedTensor:$output,
    DenseI64ArrayAttr:$window_strides,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$input_dilation,
    DenseI64ArrayAttr:$weight_dilation,
    DenseBoolArrayAttr:$window_reversal,
    TTIR_ConvolutionLayoutAttr:$convolution_layout,
    ConfinedAttr<I64Attr, [IntPositive]>:$feature_group_count,
    ConfinedAttr<I64Attr, [IntPositive]>:$batch_group_count
  );

  let results = (outs AnyRankedTensor);
  let hasVerifier = 1;
}

def TTIR_GatherOp: TTIR_NamedOp<"gather"> {
    let summary = "Gather operation.";
    let description = [{
      The `gather` operation collects slices from an input tensor at positions specified by start indices.

      This operation is based on the StableHLO Gather operation (https://openxla.org/stablehlo/spec#gather) and
      allows for flexible slicing and indexing of tensors. It can be used to implement operations like array
      indexing, slicing, dynamic indexing, and more complex gathering patterns.

      Example:
      ```mlir
      // Basic gather example: gather elements from a 2D tensor using indices
      %input = ... : tensor<5x3xf32>         // Input tensor with shape [5,3]
      %indices = ... : tensor<2xi64>         // Indices tensor with values [2, 1]
      %output = ttir.empty() : tensor<3xf32> // Output tensor
      %result = ttir.gather(%input, %indices, %output) {
          offset_dims = [0],                 // Output dimensions that are gathered from input
          collapsed_slice_dims = [0],        // Input dimensions that are collapsed
          operand_batching_dims = [],        // Batch dimensions of the input
          start_indices_batching_dims = [],  // Batch dimensions of the indices
          start_index_map = [0],             // Maps indices to input dimensions
          index_vector_dim = 0,              // Which dimension of indices contains the index vector
          slice_sizes = [1, 3],              // Size of the slice to extract from each position
          indices_are_sorted = false         // Whether indices are sorted
      } : tensor<5x3xf32>, tensor<2xi64>, tensor<3xf32> -> tensor<3xf32>

      // This gathers a slice of size [1,3] starting at position [2,0] from the input tensor,
      // which results in the values from the third row of the input tensor.
      ```

      Inputs:
      - `input` (Tensor): The tensor from which to gather values.
      - `start_indices` (Tensor): Tensor containing the starting indices for slices.

      Attributes:
      - `offset_dims` (Array of Integer): Output dimensions that correspond to dimensions of the gathered slice.
      - `collapsed_slice_dims` (Array of Integer): Input dimensions that are collapsed when gathering.
      - `operand_batching_dims` (Array of Integer): Batch dimensions of the input tensor.
      - `start_indices_batching_dims` (Array of Integer): Batch dimensions of the indices tensor.
      - `start_index_map` (Array of Integer): Maps index values to input dimensions.
      - `index_vector_dim` (Integer): Which dimension of indices contains the index vector.
      - `slice_sizes` (Array of Integer): Size of the slice to extract from each position.
      - `indices_are_sorted` (Boolean): Whether indices are sorted (for optimization).

      Outputs:
      - `result` (Tensor): The gathered tensor.
    }];
    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$start_indices,
                         AnyRankedTensor:$output,
                         DenseI64ArrayAttr:$offset_dims,
                         DenseI64ArrayAttr:$collapsed_slice_dims,
                         DenseI64ArrayAttr:$operand_batching_dims,
                         DenseI64ArrayAttr:$start_indices_batching_dims,
                         DenseI64ArrayAttr:$start_index_map,
                         SI64Attr:$index_vector_dim,
                         DenseI64ArrayAttr:$slice_sizes,
                         BoolAttr:$indices_are_sorted);
    let results = (outs AnyRankedTensor:$result);
}

def TTIR_PoolingOp : TTIR_NamedOp<"pooling", [AttrSizedOperandSegments, DeclareOpInterfaceMethods<TTIR_QuantizableOpInterface, ["rewriteWithQuantizedInputs"]>]> {
  let summary = "General pooling operation.";
  let description = [{
    The `pooling` operation is a generalized pooling operation that can implement various pooling methods
    such as max pooling, average pooling, and sum pooling.

    Pooling operations are commonly used in neural networks to reduce the spatial dimensions of feature maps
    by applying a specific function (like maximum or average) over local regions of the input tensor.

    Example:
    ```mlir
    // Max pooling with 2x2 window and stride 2
    %input = ... : tensor<1x32x32x16xf32>    // Batch size 1, 32x32 feature map, 16 channels
    %output = ttir.empty() : tensor<1x16x16x16xf32>  // Output tensor
    %result = ttir.pooling(%input, %output) {
        pooling_method = "MAX",
        window_dimensions = [1, 2, 2, 1],
        window_strides = [1, 2, 2, 1],
        base_dilations = [1, 1, 1, 1],
        window_dilations = [1, 1, 1, 1],
        padding = [0, 0, 0, 0, 0, 0, 0, 0]
    } : tensor<1x32x32x16xf32>, tensor<1x16x16x16xf32> -> tensor<1x16x16x16xf32>

    // Average pooling with 3x3 window and stride 2
    %input = ... : tensor<1x32x32x16xf32>    // Batch size 1, 32x32 feature map, 16 channels
    %output = ttir.empty() : tensor<1x15x15x16xf32>  // Output tensor
    %result = ttir.pooling(%input, %output) {
        pooling_method = "AVG",
        window_dimensions = [1, 3, 3, 1],
        window_strides = [1, 2, 2, 1],
        base_dilations = [1, 1, 1, 1],
        window_dilations = [1, 1, 1, 1],
        padding = [0, 0, 0, 0, 0, 0, 0, 0]
    } : tensor<1x32x32x16xf32>, tensor<1x15x15x16xf32> -> tensor<1x15x15x16xf32>
    ```

    Inputs:
    - `inputs` (Variadic Tensor): Input tensors to be pooled.

    Attributes:
    - `pooling_method` (Enum): The pooling method to use (MAX, AVG, SUM).
    - `window_dimensions` (Array of Integer): Dimensions of the pooling window. The rank of this array is the same as the rank of the input tensors.
    - `window_strides` (Array of Integer): Stride of the pooling window. The rank of this array is the same as the rank of the input tensors.
    - `base_dilations` (Array of Integer): Dilation factors for the input. The rank of this array is the same as the rank of the input tensors.
    - `window_dilations` (Array of Integer): Dilation factors for the pooling window. The rank of this array is the same as the rank of the input tensors.
    - `padding` (Array of Integer): Padding to apply to the input. The rank of this array is twice the rank of the input tensors, representing padding for each dimension.

    Outputs:
    - `results` (Variadic Tensor): Output tensors after pooling.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    Variadic<AnyRankedTensor>:$outputs,
    TTIR_PoolingMethodAttr:$pooling_method,
    DenseI64ArrayAttr:$window_dimensions,
    DenseI64ArrayAttr:$window_strides,
    DenseI64ArrayAttr:$base_dilations,
    DenseI64ArrayAttr:$window_dilations,
    DenseI64ArrayAttr:$padding
  );

  let results = (outs Variadic<AnyRankedTensor>);

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TTIR_AvgPool2dOp : TTIR_NamedOp<"avg_pool2d"> {
  let summary = "2D average pooling operation.";
  let description = [{
    The `avg_pool2d` operation applies a 2D average pooling over an input tensor composed of several input planes.

    This operation performs downsampling by dividing the input into local regions and computing the average value
    of each region. It reduces the spatial dimensions (height and width) of an input tensor while preserving the
    batch and channel dimensions. This is commonly used in neural networks to reduce the spatial size of feature maps.

    Example:
    ```mlir
    // Basic 2D average pooling with a 2x2 kernel and stride 1
    %input = ... : tensor<1x3x3x1xf32>  // 3x3 input tensor with values:
                                        // [[[1, 2, 3],
                                        //   [4, 5, 6],
                                        //   [7, 8, 9]]]]
    %output = ttir.empty() : tensor<1x2x2x1xf32>
    %result = ttir.avg_pool2d(%input, %output) {
        kernel = [2, 2],
        stride = [1, 1],
        dilation = [1, 1],
        padding = [0, 0, 0, 0],
        ceil_mode = false
    } : tensor<1x3x3x1xf32>, tensor<1x2x2x1xf32> -> tensor<1x2x2x1xf32>
    // Result: [[[3, 4],
    //           [6, 7]]]]
    // Where: 3 = (1+2+4+5)/4, 4 = (2+3+5+6)/4, 6 = (4+5+7+8)/4, 7 = (5+6+8+9)/4
    ```

    Inputs:
    - `input` (Tensor): Input tensor in NHWC format (batch, height, width, channels).

    Attributes:
    - `kernel` (i32 | array<2xi32>):
        - i32: Same kernel size for height and width dimensions (kH = kW = value).
        - array<2xi32>: [kH, kW] where kH is kernel size for height and kW is kernel size for width.
    - `stride` (i32 | array<2xi32>):
        - i32: Same stride for height and width dimensions (sH = sW = value).
        - array<2xi32>: [sH, sW] where sH is stride for height and sW is stride for width.
    - `dilation` (i32 | array<2xi32>):
        - i32: Same dilation for height and width dimensions (dH = dW = value).
        - array<2xi32>: [dH, dW] where dH is dilation for height and dW is dilation for width.
    - `padding` (i32 | array<2xi32> | array<4xi32>):
        - i32: Same padding for all sides (pT = pL = pB = pR = value).
        - array<2xi32>: [pH, pW] where pH is padding for height (top/bottom) and pW is padding for width (left/right).
        - array<4xi32>: [pT, pL, pB, pR] for top, left, bottom, and right padding respectively.
    - `ceil_mode` (Boolean): When true, uses ceil instead of floor for output shape calculation.
    - `count_include_pad` (Boolean): When true, include padding in the average calculation (default: True)

    Outputs:
    - `result` (Tensor): Output tensor after average pooling.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output,
                       AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$kernel,
                       AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$stride,
                       AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$dilation,
                       AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$padding,
                       BoolAttr:$ceil_mode,
                       DefaultValuedAttr<BoolAttr, "true">:$count_include_pad,
                       DefaultValuedAttr<TTIR_FlattenedCompatInfoAttr, "nullptr">:$flattened_compat_info);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
  let hasFolder = 1;
}

def TTIR_MaxPool2dOp : TTIR_NamedOp<"max_pool2d"> {
    let summary = "2D maximum pooling operation.";
    let description = [{
      The `max_pool2d` operation applies a 2D maximum pooling over an input tensor composed of several input planes.

    This operation performs downsampling by dividing the input into local regions and computing the maximum value
    of each region. It reduces the spatial dimensions (height and width) of an input tensor while preserving the
    batch and channel dimensions. This is commonly used in neural networks to reduce the spatial size of feature
    maps while retaining the most important features.

    Example:
    ```mlir
    // Basic 2D max pooling with a 2x2 kernel and stride 1
    %input = ... : tensor<1x3x3x1xf32>  // 3x3 input tensor with values:
                                        // [[[1, 2, 3],
                                        //   [4, 5, 6],
                                        //   [7, 8, 9]]]]
    %output = ttir.empty() : tensor<1x2x2x1xf32>
    %result = ttir.max_pool2d(%input, %output) {
        kernel = [2, 2],
        stride = [1, 1],
        dilation = [1, 1],
        padding = [0, 0, 0, 0],
        ceil_mode = false
    } : tensor<1x3x3x1xf32>, tensor<1x2x2x1xf32> -> tensor<1x2x2x1xf32>
    // Result: [[[5, 6],
    //           [8, 9]]]]
    // Where: 5 = max(1,2,4,5), 6 = max(2,3,5,6), 8 = max(4,5,7,8), 9 = max(5,6,8,9)
    ```

    Inputs:
    - `input` (Tensor): Input tensor in NHWC format (batch, height, width, channels).

    Attributes:
    - `kernel` (i32 | array<2xi32>):
        - i32: Same kernel size for height and width dimensions (kH = kW = value).
        - array<2xi32>: [kH, kW] where kH is kernel size for height and kW is kernel size for width.
    - `stride` (i32 | array<2xi32>):
        - i32: Same stride for height and width dimensions (sH = sW = value).
        - array<2xi32>: [sH, sW] where sH is stride for height and sW is stride for width.
    - `dilation` (i32 | array<2xi32>):
        - i32: Same dilation for height and width dimensions (dH = dW = value).
        - array<2xi32>: [dH, dW] where dH is dilation for height and dW is dilation for width.
    - `padding` (i32 | array<2xi32> | array<4xi32>):
        - i32: Same padding for all sides (pT = pL = pB = pR = value).
        - array<2xi32>: [pH, pW] where pH is padding for height (top/bottom) and pW is padding for width (left/right).
        - array<4xi32>: [pT, pL, pB, pR] for top, left, bottom, and right padding respectively.
    - `ceil_mode` (Boolean): When true, uses ceil instead of floor for output shape calculation.

    Outputs:
    - `result` (Tensor): Output tensor after maximum pooling.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$kernel,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$stride,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$dilation,
                         AnyAttrOf<[I32Attr, DenseI32ArrayAttr]>:$padding,
                         BoolAttr:$ceil_mode,
                         DefaultValuedAttr<TTIR_FlattenedCompatInfoAttr, "nullptr">:$flattened_compat_info);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTIR_ReshapeOp: TTIR_NamedOp<"reshape", [TTIR_TensorManipulation]> {
    let summary = "Tensor reshape operation.";
    let description = [{
      The `reshape` operation changes the shape of a tensor without changing the data or number of elements.

      This operation takes an input tensor and reshapes it to a new shape specified by the shape attribute.
      The total number of elements in the tensor must remain the same after reshaping. This operation is
      commonly used in neural networks to change the dimensionality of tensors between layers.

      Example:
      ```mlir
      // Reshape a 2x3 tensor to a 1x6 tensor
      %input = ... : tensor<2x3xf32>  // Input tensor with shape [2,3]
      %output = ttir.empty() : tensor<1x6xf32>  // Output tensor with shape [1,6]
      %result = ttir.reshape(%input, %output) {shape = [1, 6]} :
          tensor<2x3xf32>, tensor<1x6xf32> -> tensor<1x6xf32>

      // Reshape a 3D tensor to a 2D tensor
      %input = ... : tensor<2x3x4xf32>  // Input tensor with shape [2,3,4]
      %output = ttir.empty() : tensor<6x4xf32>  // Output tensor with shape [6,4]
      %result = ttir.reshape(%input, %output) {shape = [6, 4]} :
          tensor<2x3x4xf32>, tensor<6x4xf32> -> tensor<6x4xf32>
      ```

      Inputs:
      - `input` (Tensor): The input tensor to reshape.

      Attributes:
      - `shape` (Array of Integer): The new shape for the tensor.

      Outputs:
      - `result` (Tensor): The reshaped tensor.

      Note: The total number of elements in the input tensor must equal the total number of elements
      in the output tensor. For example, a tensor of shape [2,3] (6 elements) can be reshaped to [1,6],
      [6,1], [2,1,3], etc., but not to [2,4] (8 elements).
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I32ArrayAttr:$shape);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTIR_PadOp: TTIR_NamedOp<"pad"> {
    let summary = "Tensor padding operation.";
    let description = [{
      The `pad` operation adds padding to the edges of an input tensor with a specified constant value.

      This operation extends the dimensions of the input tensor by adding padding elements with a constant value.
      The padding is specified for each dimension as the number of elements to add at the beginning (low) and
      end (high) of that dimension.

      The `padding` attribute must be a sequence of integers that is twice the size as the rank of the input.
      Each pair of integers in the padding attribute represents the amount of padding to add to the low and high
      of that dimension. For example, for a 2D tensor, the padding attribute would have 4 values: [dim0_low, dim0_high, dim1_low, dim1_high].

      Example:
      ```mlir
      // Pad a 2x3 tensor with different padding on each dimension
      %input = ... : tensor<2x3xf32>  // Input tensor with values:
                                      // [[1, 2, 3],
                                      //  [4, 5, 6]]
      %output = ttir.empty() : tensor<3x5xf32>  // Output tensor shape
      %result = ttir.pad(%input, %output) {
          padding = [1, 0, 1, 1],  // Format: [dim0_low, dim0_high, dim1_low, dim1_high]
          value = 0.0 : f32
      } : tensor<2x3xf32>, tensor<3x5xf32> -> tensor<3x5xf32>
      // Result:
      // [[0, 0, 0, 0, 0],
      //  [0, 1, 2, 3, 0],
      //  [0, 4, 5, 6, 0]]
      ```

      Inputs:
      - `input` (Tensor): The input tensor to pad.

      Attributes:
      - `padding` (Array of Integer): The padding values for each dimension, specified as [dim0_low, dim0_high, dim1_low, dim1_high, ...].
      - `value` (Float): The constant value to use for the padding elements.

      Outputs:
      - `result` (Tensor): The padded tensor.

      Note: The shape of the output tensor must match the shape of the input tensor plus the padding specified
      in the padding attribute. For example, if the input shape is [2,3] and the padding is [1,0,1,1], then
      the output shape must be [3,5].
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         DenseI32ArrayAttr:$padding,
                         F32Attr:$value);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_SliceStaticOp: TTIR_NamedOp<"slice_static"> {
    let summary = "Tensor slice operation with constant parameters.";
    let description = [{
      The `slice_static` operation extracts a sub-tensor (slice) from the input tensor across one or more dimensions.

      This operation selects a subset of elements from the input tensor based on the specified begin, end, and
      step indices for each dimension. It's similar to Python's slicing notation `tensor[begin:end:step]` but
      extended to multiple dimensions. The `begins` and `ends` parameters are attributes with fixed values.

      Example:
      ```mlir
      // Extract a 2x2 slice from a 4x4 tensor
      %input = ... : tensor<4x4xf32>  // Input tensor with values:
                                      // [[1,  2,  3,  4],
                                      //  [5,  6,  7,  8],
                                      //  [9,  10, 11, 12],
                                      //  [13, 14, 15, 16]]
      %output = ttir.empty() : tensor<2x2xf32>  // Output tensor shape
      %result = ttir.slice_static(%input, %output) {
          begins = [1, 1],  // Start indices for each dimension
          ends = [3, 3],    // End indices for each dimension (exclusive)
          step = [1, 1]     // Step size for each dimension
      } : tensor<4x4xf32>, tensor<2x2xf32> -> tensor<2x2xf32>
      // Result:
      // [[6,  7],
      //  [10, 11]]

      // Extract elements with a step of 2
      %input = ... : tensor<5xf32>  // Input tensor with values: [1, 2, 3, 4, 5]
      %output = ttir.empty() : tensor<3xf32>  // Output tensor shape
      %result = ttir.slice_static(%input, %output) {
          begins = [0],  // Start index
          ends = [5],    // End index (exclusive)
          step = [2]     // Step size
      } : tensor<5xf32>, tensor<3xf32> -> tensor<3xf32>
      // Result: [1, 3, 5]
      ```

      Inputs:
      - `input` (Tensor): The input tensor to slice.

      Attributes:
      - `begins` (Array of Integer): The starting indices for the slice in each dimension.
      - `ends` (Array of Integer): The ending indices (exclusive) for the slice in each dimension.
      - `step` (Array of Integer): The step sizes for the slice in each dimension.

      Outputs:
      - `result` (Tensor): The sliced tensor.

      Note: The shape of the output tensor is determined by the slice parameters. For each dimension i,
      the output size is calculated as `ceil((ends[i] - begins[i]) / step[i])`. The `begins`, `ends`, and
      `step` arrays must have the same length as the rank of the input tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I32ArrayAttr:$begins,
                         I32ArrayAttr:$ends,
                         I32ArrayAttr:$step);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_SliceDynamicOp: TTIR_NamedOp<"slice_dynamic"> {
    let summary = "Tensor slice operation with dynamic parameters.";
    let description = [{
      The `slice_dynamic` operation extracts a sub-tensor (slice) from the input tensor across one or more dimensions.

      This operation selects a subset of elements from the input tensor based on the specified begin, end, and
      step indices for each dimension. It's similar to Python's slicing notation `tensor[begin:end:step]` but
      extended to multiple dimensions. The `begins` and `ends` parameters are tensor inputs determined at runtime.

      Example:
      ```mlir
      // Extract a 2x2 slice from a 4x4 tensor with dynamic begin/end indices
      %input = ... : tensor<4x4xf32>  // Input tensor with values:
                                      // [[1,  2,  3,  4],
                                      //  [5,  6,  7,  8],
                                      //  [9,  10, 11, 12],
                                      //  [13, 14, 15, 16]]
      %begins = ... : tensor<2xi32>   // Tensor with values [1, 1]
      %ends = ... : tensor<2xi32>     // Tensor with values [3, 3]
      %output = ttir.empty() : tensor<2x2xf32>  // Output tensor shape
      %result = ttir.slice_dynamic(%input, %begins, %ends, %output) {
          step = [1, 1]     // Step size for each dimension
      } : tensor<4x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2x2xf32> -> tensor<2x2xf32>
      // Result:
      // [[6,  7],
      //  [10, 11]]

      // Extract elements with a step of 2 using dynamic indices
      %input = ... : tensor<5xf32>    // Input tensor with values: [1, 2, 3, 4, 5]
      %begins = ... : tensor<1xi32>   // Tensor with values [0]
      %ends = ... : tensor<1xi32>     // Tensor with values [5]
      %output = ttir.empty() : tensor<3xf32>  // Output tensor shape
      %result = ttir.slice_dynamic(%input, %begins, %ends, %output) {
          step = [2]        // Step size
      } : tensor<5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<3xf32> -> tensor<3xf32>
      // Result: [1, 3, 5]
      ```

      Inputs:
      - `input` (Tensor): The input tensor to slice.
      - `begins` (Tensor): The starting indices for the slice in each dimension.
      - `ends` (Tensor): The ending indices (exclusive) for the slice in each dimension.

      Attributes:
      - `step` (Array of Integer): The step sizes for the slice in each dimension.

      Outputs:
      - `result` (Tensor): The sliced tensor.

      Note: The `begins` and `ends` tensors must have the same length as the rank of the input tensor.
      The output tensor shape may contain dynamic dimensions when slice parameters are runtime-determined.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$begins,
                         AnyRankedTensor:$ends,
                         AnyRankedTensor:$output,
                         OptionalAttr<I32ArrayAttr>:$step);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_IndexSelectOp: TTIR_NamedOp<"index_select"> {
    let summary = "Tensor selection operation.";
    let description = [{
      The `index_select` operation extracts a sub-tensor (slice) from the input tensor along a specified dimension.

      Unlike the more general `slice` operation, `index_select` operates on a single dimension with a specified
      starting index, length, and optional stride. This is useful for extracting specific segments of a tensor
      along a particular axis.

      Example:
      ```mlir
      // Select elements 2, 3, 4 from a 1D tensor along dimension 0
      %input = ... : tensor<6xf32>  // Input tensor with values: [1, 2, 3, 4, 5, 6]
      %output = ttir.empty() : tensor<3xf32>  // Output tensor shape
      %result = ttir.index_select(%input, %output) {
          dim = 0 : i32,     // Dimension to select from
          begin = 2 : i32,   // Start index
          length = 3 : i32,  // Number of elements to select
          stride = 0 : i32   // No stride (consecutive elements)
      } : tensor<6xf32>, tensor<3xf32> -> tensor<3xf32>
      // Result: [3, 4, 5]

      // Select every other row from a 2D tensor
      %input = ... : tensor<4x3xf32>  // Input tensor with values:
                                      // [[1, 2, 3],
                                      //  [4, 5, 6],
                                      //  [7, 8, 9],
                                      //  [10, 11, 12]]
      %output = ttir.empty() : tensor<2x3xf32>  // Output tensor shape
      %result = ttir.index_select(%input, %output) {
          dim = 0 : i32,     // Select along rows
          begin = 0 : i32,   // Start from the first row
          length = 2 : i32,  // Select 2 rows
          stride = 2 : i32   // Select every other row
      } : tensor<4x3xf32>, tensor<2x3xf32> -> tensor<2x3xf32>
      // Result:
      // [[1, 2, 3],
      //  [7, 8, 9]]
      ```

      Inputs:
      - `input` (Tensor): The input tensor to select from.

      Attributes:
      - `dim` (Integer): The dimension along which to select elements.
      - `begin` (Integer): The starting index for selection.
      - `length` (Integer): The number of elements to select.
      - `stride` (Integer, default=0): The step size for selection. A value of 0 means no stride (consecutive elements).

      Outputs:
      - `result` (Tensor): The selected tensor.

      Note: The shape of the output tensor is the same as the input tensor except for the selected dimension,
      which will have size `length`. If `stride` is non-zero, the elements selected will be at indices
      `begin`, `begin + stride`, `begin + 2*stride`, etc., up to `length` elements.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim,
                         SI32Attr:$begin,
                         SI32Attr:$length,
                         DefaultValuedOptionalAttr<SI32Attr, "0">:$stride);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

// ANCHOR: decomposing_an_op_index_ttir
def TTIR_IndexOp: TTIR_NamedOp<"index"> {
    let summary = "Tensor indexing operation.";
    let description = [{
      The `index` operation extracts a sub-tensor (slice) from the input tensor along a specified dimension.

      This operation selects elements from the input tensor along a single dimension based on the specified
      begin, end, and step indices. It's similar to Python's slicing notation `tensor[:, begin:end:step, :]`
      where the slicing is applied only to the specified dimension.

      Example:
      ```mlir
      // Extract elements with indices 1, 3, 5 from dimension 0 of a 1D tensor
      %input = ... : tensor<6xf32>  // Input tensor with values: [1, 2, 3, 4, 5, 6]
      %output = ttir.empty() : tensor<3xf32>  // Output tensor shape
      %result = ttir.index(%input, %output) {
          dim = 0 : i32,    // Dimension to index
          begin = 1 : i32,  // Start index
          end = 6 : i32,    // End index (exclusive)
          step = 2 : i32    // Step size
      } : tensor<6xf32>, tensor<3xf32> -> tensor<3xf32>
      // Result: [2, 4, 6]

      // Extract columns 0 and 2 from a 2D tensor
      %input = ... : tensor<3x4xf32>  // Input tensor with values:
                                      // [[1, 2, 3, 4],
                                      //  [5, 6, 7, 8],
                                      //  [9, 10, 11, 12]]
      %output = ttir.empty() : tensor<3x2xf32>  // Output tensor shape
      %result = ttir.index(%input, %output) {
          dim = 1 : i32,    // Index along columns (dimension 1)
          begin = 0 : i32,  // Start from first column
          end = 3 : i32,    // End at third column (exclusive)
          step = 2 : i32    // Take every other column
      } : tensor<3x4xf32>, tensor<3x2xf32> -> tensor<3x2xf32>
      // Result:
      // [[1, 3],
      //  [5, 7],
      //  [9, 11]]
      ```

      Inputs:
      - `input` (Tensor): The input tensor to index.

      Attributes:
      - `dim` (Integer): The dimension along which to index.
      - `begin` (Integer): The starting index.
      - `end` (Integer): The ending index (exclusive).
      - `step` (Integer): The step size between indices.

      Outputs:
      - `result` (Tensor): The indexed tensor.

      Note: The shape of the output tensor is the same as the input tensor except for the indexed dimension,
      which will have size `ceil((end - begin) / step)`. The indices selected will be `begin`, `begin + step`,
      `begin + 2*step`, etc., up to but not including `end`.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I32Attr:$dim,
                         I32Attr:$begin,
                         I32Attr:$end,
                         I32Attr:$step);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}
// ANCHOR_END: decomposing_an_op_index_ttir

def TTIR_SqueezeOp : TTIR_NamedOp<"squeeze"> {
    let summary = "Tensor dimension squeezing operation.";
    let description = [{
      The `squeeze` operation removes a dimension of size 1 from the shape of a tensor.

      This operation is commonly used to eliminate unnecessary singleton dimensions from a tensor's shape.
      It specifies which dimension to remove using the `dim` attribute. The specified dimension must have size 1.

      Example:
      ```mlir
      // Squeeze dimension 0 from a tensor of shape [1, 3, 4]
      %input = ... : tensor<1x3x4xf32>  // Input tensor with shape [1, 3, 4]
      %output = ttir.empty() : tensor<3x4xf32>  // Output tensor shape
      %result = ttir.squeeze(%input, %output) {
          dim = 0 : i32  // Dimension to squeeze
      } : tensor<1x3x4xf32>, tensor<3x4xf32> -> tensor<3x4xf32>
      // Result: tensor with shape [3, 4]

      // Squeeze dimension 1 from a tensor of shape [2, 1, 3]
      %input = ... : tensor<2x1x3xf32>  // Input tensor with shape [2, 1, 3]
      %output = ttir.empty() : tensor<2x3xf32>  // Output tensor shape
      %result = ttir.squeeze(%input, %output) {
          dim = 1 : i32  // Dimension to squeeze
      } : tensor<2x1x3xf32>, tensor<2x3xf32> -> tensor<2x3xf32>
      // Result: tensor with shape [2, 3]
      ```

      Inputs:
      - `input` (Tensor): The input tensor to squeeze.

      Attributes:
      - `dim` (Integer): The dimension to squeeze.

      Outputs:
      - `result` (Tensor): The squeezed tensor.

      Note: The specified dimension must have size 1. The shape of the output tensor is the same as the
      input tensor with the specified dimension removed. For example, squeezing dimension 1 of a tensor
      with shape [2, 1, 3] results in a tensor with shape [2, 3].
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_UnsqueezeOp : TTIR_NamedOp<"unsqueeze"> {
    let summary = "Tensor dimension insertion operation.";
    let description = [{
      The `unsqueeze` operation inserts a dimension of size 1 into the shape of a tensor.

      This operation is the inverse of the `squeeze` operation and is commonly used to add a singleton dimension
      to a tensor's shape. It specifies which position to insert the new dimension using the `dim` attribute.

      Example:
      ```mlir
      // Insert a dimension at position 0 of a tensor with shape [3, 4]
      %input = ... : tensor<3x4xf32>  // Input tensor with shape [3, 4]
      %output = ttir.empty() : tensor<1x3x4xf32>  // Output tensor shape
      %result = ttir.unsqueeze(%input, %output) {
          dim = 0 : i32  // Position to insert the new dimension
      } : tensor<3x4xf32>, tensor<1x3x4xf32> -> tensor<1x3x4xf32>
      // Result: tensor with shape [1, 3, 4]

      // Insert a dimension at position 1 of a tensor with shape [2, 3]
      %input = ... : tensor<2x3xf32>  // Input tensor with shape [2, 3]
      %output = ttir.empty() : tensor<2x1x3xf32>  // Output tensor shape
      %result = ttir.unsqueeze(%input, %output) {
          dim = 1 : i32  // Position to insert the new dimension
      } : tensor<2x3xf32>, tensor<2x1x3xf32> -> tensor<2x1x3xf32>
      // Result: tensor with shape [2, 1, 3]
      ```

      Inputs:
      - `input` (Tensor): The input tensor to unsqueeze.

      Attributes:
      - `dim` (Integer): The position to insert the new dimension.

      Outputs:
      - `result` (Tensor): The unsqueezed tensor.

      Note: The shape of the output tensor is the same as the input tensor with a new dimension of size 1
      inserted at the specified position. For example, unsqueezing at position 1 of a tensor with shape [2, 3]
      results in a tensor with shape [2, 1, 3].
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$dim);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_ClampScalarOp : TTIR_NamedOp<"clamp_scalar", [TTIR_ElementwiseUnary]> {
    let summary = "Scalar value clamping operation.";
    let description = [{
      The `clamp_scalar` operation constrains all elements of a tensor to be within a specified range.

      This operation applies element-wise clamping to the input tensor, ensuring that all values fall within
      the range [min, max]. Values less than `min` are set to `min`, and values greater than `max` are set to `max`.
      This is commonly used to ensure that tensor values stay within a valid range.

      Example:
      ```mlir
      // Clamp values to the range [2.0, 5.0]
      %input = ... : tensor<1x8xf32>  // Input tensor with values:
                                      // [[0, 1, 2, 3, 4, 5, 6, 7]]
      %output = ttir.empty() : tensor<1x8xf32>  // Output tensor shape
      %result = ttir.clamp_scalar(%input, %output) {
          min = 2.0 : f32,  // Minimum value
          max = 5.0 : f32   // Maximum value
      } : tensor<1x8xf32>, tensor<1x8xf32> -> tensor<1x8xf32>
      // Result: [[2, 2, 2, 3, 4, 5, 5, 5]]
      // Values < 2.0 are clamped to 2.0, values > 5.0 are clamped to 5.0
      ```

      Inputs:
      - `input` (Tensor): The input tensor to clamp.

      Attributes:
      - `min` (Float): The minimum value for clamping.
      - `max` (Float): The maximum value for clamping.

      Outputs:
      - `result` (Tensor): The clamped tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         F32Attr:$min,
                         F32Attr:$max);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_ClampTensorOp : TTIR_NamedOp<"clamp_tensor"> {
  let summary = "Tensor value clamping operation.";
  let description = [{
    The `clamp_tensor` operation constrains elements of a tensor to be within ranges specified by min and max tensors.

    Unlike `clamp_scalar`, which uses scalar values for min and max, this operation uses tensor values for
    element-wise clamping. Each element in the input tensor is clamped between the corresponding elements
    in the min and max tensors. This allows for different clamping ranges for different elements.

    Example:
    ```mlir
    // Clamp values using min and max tensors
    %input = ... : tensor<1x8xf32>  // Input tensor with values:
                                    // [[0, 1, 2, 3, 4, 5, 6, 7]]
    %min = ... : tensor<1x8xf32>    // Min tensor with values:
                                    // [[2, 2, 2, 3, 3, 3, 0, 0]]
    %max = ... : tensor<1x8xf32>    // Max tensor with values:
                                    // [[5, 5, 5, 9, 9, 9, 6, 6]]
    %output = ttir.empty() : tensor<1x8xf32>  // Output tensor shape
    %result = ttir.clamp_tensor(%input, %min, %max, %output) :
        tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32>, tensor<1x8xf32> -> tensor<1x8xf32>
    // Result: [[2, 2, 2, 3, 4, 5, 6, 6]]
    // Each element is clamped between its corresponding min and max values
    ```

    Inputs:
    - `input` (Tensor): The input tensor to clamp.
    - `min` (Tensor): The tensor containing minimum values for clamping.
    - `max` (Tensor): The tensor containing maximum values for clamping.

    Outputs:
    - `result` (Tensor): The clamped tensor.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$min,
                       AnyRankedTensor:$max,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

def TTIR_ReverseOp : TTIR_NamedOp<"reverse", [AllShapesMatch<["input", "result"]>]> {
  let summary = "Tensor reversal operation.";

  let description = [{
    The `reverse` operation reverses the order of elements in the input tensor along the specified dimensions.

    This operation flips the elements of a tensor along one or more axes, which is useful for operations
    like sequence reversal, matrix transposition with reversal, and other tensor manipulations that require
    changing the order of elements.

    Example:
    ```mlir
    // Reverse a 3x2 tensor along dimension 1 (columns)
    %input = ... : tensor<3x2xi32>  // Input tensor with values:
                                    // [[1, 2],
                                    //  [3, 4],
                                    //  [5, 6]]
    %output = ttir.empty() : tensor<3x2xi32>  // Output tensor shape
    %result = ttir.reverse(%input, %output) {
        dimensions = [1]  // Reverse along columns
    } : tensor<3x2xi32>, tensor<3x2xi32> -> tensor<3x2xi32>
    // Result:
    // [[2, 1],
    //  [4, 3],
    //  [6, 5]]

    // Reverse a 3x2 tensor along both dimensions
    %input = ... : tensor<3x2xi64>  // Input tensor with values:
                                    // [[1, 2],
                                    //  [3, 4],
                                    //  [5, 6]]
    %output = ttir.empty() : tensor<3x2xi64>  // Output tensor shape
    %result = ttir.reverse(%input, %output) {
        dimensions = [0, 1]  // Reverse along both rows and columns
    } : tensor<3x2xi64>, tensor<3x2xi64> -> tensor<3x2xi64>
    // Result:
    // [[6, 5],
    //  [4, 3],
    //  [2, 1]]
    ```

    Inputs:
    - `input` (Tensor): The input tensor to reverse.

    Attributes:
    - `dimensions` (Array of Integer): The dimensions along which to reverse the tensor.

    Outputs:
    - `result` (Tensor): The reversed tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         DenseI64ArrayAttr:$dimensions);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

def TTIR_LinearOp : TTIR_NamedOp<"linear"> {
    let summary = "Linear transformation operation.";
    let description = [{
      The `linear` operation performs a linear transformation by computing the matrix multiplication
      of tensors `a` and `b` with an optional addition of a `bias` tensor.

      This operation is commonly used in neural networks to implement fully connected layers. It computes
      the matrix multiplication of the input tensor with a weight tensor and adds an optional bias.

      Example:
      ```mlir
      // Linear transformation with bias
      %a = ... : tensor<10x64x32xbf16>  // Input tensor: batch_size=10, sequence_length=64, input_dim=32
      %b = ... : tensor<32x128xbf16>    // Weight tensor: input_dim=32, output_dim=128
      %bias = ... : tensor<128xbf16>    // Bias tensor: output_dim=128
      %output = ttir.empty() : tensor<10x64x128xbf16>  // Output tensor shape
      %result = ttir.linear(%a, %b, %bias, %output) :
          tensor<10x64x32xbf16>, tensor<32x128xbf16>, tensor<128xbf16>, tensor<10x64x128xbf16> -> tensor<10x64x128xbf16>

      // Linear transformation without bias
      %a = ... : tensor<10x64x32xf32>  // Input tensor
      %b = ... : tensor<32x128xf32>    // Weight tensor
      %output = ttir.empty() : tensor<10x64x128xf32>  // Output tensor shape
      %result = ttir.linear(%a, %b, %output) :
          tensor<10x64x32xf32>, tensor<32x128xf32>, tensor<10x64x128xf32> -> tensor<10x64x128xf32>
      ```

      Inputs:
      - `a` (Tensor): The input tensor.
      - `b` (Tensor): The weight tensor.
      - `bias` (Optional Tensor): The bias tensor to add to the result of the matrix multiplication.

      Attributes:
      - `transpose_a` (Boolean, default=false): Whether to transpose tensor `a` before multiplication.
      - `transpose_b` (Boolean, default=false): Whether to transpose tensor `b` before multiplication.

      Outputs:
      - `result` (Tensor): The result of the linear transformation.

      The operation computes: result = matmul(a, b) + bias

      Note: The shapes of the tensors must be compatible for matrix multiplication. For a 3D input tensor
      with shape [batch_size, sequence_length, input_dim], the weight tensor should have shape [input_dim, output_dim],
      and the bias tensor should have shape [output_dim]. The resulting tensor will have shape
      [batch_size, sequence_length, output_dim].
    }];

    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         Optional<AnyRankedTensor>:$bias,
                         AnyRankedTensor:$output,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_b);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}

// ANCHOR: adding_an_op_matmul_ttir
def TTIR_MatmulOp : TTIR_NamedOp<"matmul"> {
    let summary = "Matrix multiplication operation.";
    let description = [{
      The `matmul` operation computes the matrix multiplication of two tensors.

      This operation performs matrix multiplication between tensors `a` and `b`. It supports optional
      transposition of either input tensor before multiplication. For 2D tensors, this computes the standard
      matrix product. For tensors with more dimensions, it applies batched matrix multiplication.

      Example:
      ```mlir
      // Basic matrix multiplication of 2D tensors
      %a = ... : tensor<3x4xf32>  // Matrix A with shape [3,4]
      %b = ... : tensor<4x5xf32>  // Matrix B with shape [4,5]
      %output = ttir.empty() : tensor<3x5xf32>  // Output matrix shape
      %result = ttir.matmul(%a, %b, %output) :
          tensor<3x4xf32>, tensor<4x5xf32>, tensor<3x5xf32> -> tensor<3x5xf32>

      // Batched matrix multiplication with transposition
      %a = ... : tensor<2x3x4xf32>  // Batch of 2 matrices with shape [3,4]
      %b = ... : tensor<2x5x4xf32>  // Batch of 2 matrices with shape [5,4]
      %output = ttir.empty() : tensor<2x3x5xf32>  // Output shape
      %result = ttir.matmul(%a, %b, %output) {
          transpose_a = false,  // Don't transpose A
          transpose_b = true    // Transpose B before multiplication
      } : tensor<2x3x4xf32>, tensor<2x5x4xf32>, tensor<2x3x5xf32> -> tensor<2x3x5xf32>
      ```

      Inputs:
      - `a` (Tensor): The first input tensor.
      - `b` (Tensor): The second input tensor.

      Attributes:
      - `transpose_a` (Boolean, default=false): Whether to transpose tensor `a` before multiplication.
      - `transpose_b` (Boolean, default=false): Whether to transpose tensor `b` before multiplication.

      Outputs:
      - `result` (Tensor): The result of the matrix multiplication.

      Note: The inner dimensions of the input tensors must be compatible for matrix multiplication.
      If `a` has shape [..., m, k] and `b` has shape [..., k, n], then the result will have shape [..., m, n].
      If `transpose_a` is true, then `a` is treated as having shape [..., k, m].
      If `transpose_b` is true, then `b` is treated as having shape [..., n, k].
    }];

    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         AnyRankedTensor:$output,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
                         DefaultValuedAttr<BoolAttr, "false">:$transpose_b);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasCanonicalizer = 1;
}
// ANCHOR_END: adding_an_op_matmul_ttir

def TTIR_PermuteOp : TTIR_NamedOp<"permute", [TTIR_TensorManipulation]> {
    let summary = "Tensor dimension permutation operation.";
    let description = [{
      The `permute` operation reorders the dimensions of the input tensor according to the specified permutation.

      This operation is similar to transpose but generalizes to tensors of any rank. It rearranges the
      dimensions of the input tensor based on the permutation attribute, which specifies the new order
      of dimensions.

      Example:
      ```mlir
      // Transpose a 2D tensor (swap dimensions 0 and 1)
      %input = ... : tensor<3x4xf32>  // Input tensor with shape [3,4]
      %output = ttir.empty() : tensor<4x3xf32>  // Output tensor shape
      %result = ttir.permute(%input, %output) {
          permutation = [1, 0]  // Swap dimensions 0 and 1
      } : tensor<3x4xf32>, tensor<4x3xf32> -> tensor<4x3xf32>
      // Result: tensor with shape [4,3], equivalent to transposing the input

      // Permute a 3D tensor
      %input = ... : tensor<2x3x4xf32>  // Input tensor with shape [2,3,4]
      %output = ttir.empty() : tensor<3x4x2xf32>  // Output tensor shape
      %result = ttir.permute(%input, %output) {
          permutation = [1, 2, 0]  // Reorder dimensions to [1,2,0]
      } : tensor<2x3x4xf32>, tensor<3x4x2xf32> -> tensor<3x4x2xf32>
      // Result: tensor with shape [3,4,2]
      ```

      Inputs:
      - `input` (Tensor): The input tensor to permute.

      Attributes:
      - `permutation` (Array of Integer): The permutation of the input tensor dimensions.
        This must be a valid permutation of the indices [0, 1, ..., rank-1].

      Outputs:
      - `result` (Tensor): The permuted tensor.

      Note: The permutation attribute must contain exactly one occurrence of each integer in the range
      [0, rank-1], where rank is the number of dimensions in the input tensor. The shape of the output
      tensor is determined by permuting the dimensions of the input tensor according to the permutation.
      For example, if the input shape is [2,3,4] and the permutation is [1,2,0], then the output shape
      will be [3,4,2].
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         DenseI64ArrayAttr:$permutation);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;

    let hasFolder = 1;
}

def TTIR_Upsample2dOp : TTIR_NamedOp<"upsample2d"> {
    let summary = "Upsample 2D operation.";

    let description = [{
      The `upsample2d` operation increases the spatial dimensions (height and width) of an input tensor.

      This operation is commonly used in neural networks to increase the spatial resolution of feature maps.
      It supports different upsampling algorithms such as "nearest" and "bilinear" interpolation.
      The input tensor is assumed to be in NHWC format (batch, height, width, channels).

      Example:
      ```mlir
      // Upsample a tensor with different scale factors for height and width
      %input = ... : tensor<10x64x32x3xbf16>  // Input tensor: [batch=10, height=64, width=32, channels=3]
      %output = ttir.empty() : tensor<10x128x128x3xbf16>  // Output tensor shape
      %result = ttir.upsample2d(%input, %output) {
          scale_factor = [2, 4],  // Scale height by 2, width by 4
          mode = "bilinear"       // Use bilinear interpolation
      } : tensor<10x64x32x3xbf16>, tensor<10x128x128x3xbf16> -> tensor<10x128x128x3xbf16>
      // Result: tensor with shape [10,128,128,3]

      // Upsample with the same scale factor for both dimensions
      %input = ... : tensor<1x32x32x16xf32>  // Input tensor
      %output = ttir.empty() : tensor<1x64x64x16xf32>  // Output tensor shape
      %result = ttir.upsample2d(%input, %output) {
          scale_factor = 2,     // Scale both height and width by 2
          mode = "nearest"      // Use nearest neighbor interpolation
      } : tensor<1x32x32x16xf32>, tensor<1x64x64x16xf32> -> tensor<1x64x64x16xf32>
      // Result: tensor with shape [1,64,64,16]
      ```

      Inputs:
      - `input` (Tensor): The input tensor to upsample, in NHWC format.

      Attributes:
      - `scale_factor` (Integer or Array of Integer): The scale factor for upsampling in height and width dimensions.
        If a single integer is provided, it's used for both dimensions. If an array is provided, the first value
        is used for height and the second for width.
      - `mode` (String, default="nearest"): The upsampling algorithm to use. Currently supported values are
        "nearest" for nearest neighbor interpolation and "bilinear" for bilinear interpolation.

      Outputs:
      - `result` (Tensor): The upsampled tensor.

      Note: The output height is calculated as input_height * scale_factor[0] and the output width as
      input_width * scale_factor[1]. The batch and channel dimensions remain unchanged.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         AnyAttrOf<[SI32Attr, DenseI32ArrayAttr]>:$scale_factor,
                         DefaultValuedAttr<StrAttr, "\"nearest\"">:$mode);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_ScatterOp: TTIR_NamedOp<"scatter"> {
  let summary = "Scatter operation";
  let description = [{
    The `scatter` operation updates slices of an input tensor at indices specified by scatter_indices
    with values from the update tensor.

    This operation is the inverse of the `gather` operation. It allows for updating specific slices of a
    tensor at locations determined by indices. The operation is highly configurable through various
    dimension attributes that control how the indices and updates are interpreted.

    Example:
    ```mlir
    // Basic scatter example: update values at specific indices in a 1D tensor
    %input = ... : tensor<8xf32>        // Input tensor with values: [0, 0, 0, 0, 0, 0, 0, 0]
    %indices = ... : tensor<3xi32>      // Indices tensor with values: [1, 3, 5]
    %update = ... : tensor<3xf32>       // Update tensor with values: [10, 30, 50]
    %output = ttir.empty() : tensor<8xf32>  // Output tensor shape
    %result = ttir.scatter(%input, %indices, %update, %output) {
        update_window_dims = [],        // No window dimensions in update tensor
        inserted_window_dims = [0],     // Insert window dimension 0
        input_batching_dims = [],       // No batching dimensions in input
        scatter_indices_batching_dims = [], // No batching dimensions in indices
        scatter_dims_to_operand_dims = [0], // Map scatter dimension 0 to operand dimension 0
        index_vector_dim = 0,           // Indices are in dimension 0
        indices_are_sorted = true,      // Indices are sorted
        unique_indices = true           // Indices are unique
    } : tensor<8xf32>, tensor<3xi32>, tensor<3xf32>, tensor<8xf32> -> tensor<8xf32>
    // Result: [0, 10, 0, 30, 0, 50, 0, 0]

    // Scatter to update a 2D tensor
    %input = ... : tensor<4x4xf32>      // Input tensor (4x4 matrix of zeros)
    %indices = ... : tensor<2x2xi32>    // Indices tensor with values: [[0, 1], [2, 3]]
    %update = ... : tensor<2xf32>       // Update tensor with values: [100, 200]
    %output = ttir.empty() : tensor<4x4xf32>  // Output tensor shape
    %result = ttir.scatter(%input, %indices, %update, %output) {
        update_window_dims = [],
        inserted_window_dims = [0, 1],
        input_batching_dims = [],
        scatter_indices_batching_dims = [0],
        scatter_dims_to_operand_dims = [0, 1],
        index_vector_dim = 1,
        indices_are_sorted = false,
        unique_indices = true
    } : tensor<4x4xf32>, tensor<2x2xi32>, tensor<2xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    // Result: A 4x4 tensor with 100 at position [0,1] and 200 at position [2,3]
    ```

    Inputs:
    - `input` (Tensor): The tensor to be updated.
    - `scatter_indices` (Tensor): Tensor containing the starting indices for slices to update.
    - `update` (Tensor): Tensor containing values to scatter into the input tensor.

    Attributes:
    - `update_window_dims` (Array of Integer): Dimensions in update that are window dimensions.
    - `inserted_window_dims` (Array of Integer): Dimensions in the output that are not present in update.
    - `input_batching_dims` (Array of Integer): Batch dimensions in the input tensor.
    - `scatter_indices_batching_dims` (Array of Integer): Batch dimensions in the scatter indices tensor.
    - `scatter_dims_to_operand_dims` (Array of Integer): Maps dimensions in scatter indices to dimensions in operand.
    - `index_vector_dim` (Integer): The dimension in scatter indices that contains the index vector.
    - `indices_are_sorted` (Boolean): Whether indices are sorted lexicographically.
    - `unique_indices` (Boolean): Whether indices are guaranteed to be unique.

    Outputs:
    - `result` (Tensor): The updated tensor.

    Note: The semantics of this operation are complex and based on the StableHLO scatter operation.
    The configuration of the various dimension attributes determines exactly how the scatter indices
    are interpreted and how the update values are applied to the input tensor.
    }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$scatter_indices,
                       AnyRankedTensor:$update,
                       AnyRankedTensor:$output,
                       DenseI32ArrayAttr:$update_window_dims,
                       DenseI32ArrayAttr:$inserted_window_dims,
                       DenseI32ArrayAttr:$input_batching_dims,
                       DenseI32ArrayAttr:$scatter_indices_batching_dims,
                       DenseI32ArrayAttr:$scatter_dims_to_operand_dims,
                       I32Attr:$index_vector_dim,
                       BoolAttr:$indices_are_sorted,
                       BoolAttr:$unique_indices);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TTIR creation ops
//===----------------------------------------------------------------------===//

class TTIR_CreationOp<string mnemonic, list<Trait> traits = []> :
    TTIR_Op<mnemonic, [Pure, TTCore_CreationOpTrait] # traits>;


def TTIR_ArangeOp : TTIR_CreationOp<"arange"> {
  let summary = "Tensor range generation operation.";
  let description = [{
    The `arange` operation generates a tensor with evenly spaced values within a given interval.

    This operation creates a tensor with values from `start` to `end` (exclusive) with a step size of `step`,
    along the dimension specified by `arange_dimension`. It's similar to NumPy's arange function and is useful
    for creating tensors with regular sequences of values.

    Example:
    ```mlir
    // Generate a 1D tensor with values [0, 1, 2, 3, 4]
    %result = ttir.arange() {
        start = 0 : si64,
        end = 5 : si64,
        step = 1 : si64,
        arange_dimension = 0 : i64
    } : () -> tensor<5xi64>

    // Generate a 1D tensor with values [0.0, 2.0, 4.0, 6.0, 8.0]
    %result = ttir.arange() {
        start = 0 : si64,
        end = 10 : si64,
        step = 2 : si64,
        arange_dimension = 0 : i64
    } : () -> tensor<5xf32>

    // Generate a 2D tensor with the sequence along dimension 0
    %result = ttir.arange() {
        start = 0 : si64,
        end = 5 : si64,
        step = 1 : si64,
        arange_dimension = 0 : i64
    } : () -> tensor<5x3xi64>
    // Result:
    // [[0, 0, 0],
    //  [1, 1, 1],
    //  [2, 2, 2],
    //  [3, 3, 3],
    //  [4, 4, 4]]

    // Generate a 2D tensor with the sequence along dimension 1
    %result = ttir.arange() {
        start = 0 : si64,
        end = 3 : si64,
        step = 1 : si64,
        arange_dimension = 1 : i64
    } : () -> tensor<5x3xi64>
    // Result:
    // [[0, 1, 2],
    //  [0, 1, 2],
    //  [0, 1, 2],
    //  [0, 1, 2],
    //  [0, 1, 2]]
    ```

    Attributes:
    - `start` (Integer): The start value of the sequence.
    - `end` (Integer): The end value of the sequence (exclusive).
    - `step` (Integer): The step size between values in the sequence.
    - `arange_dimension` (Integer): The dimension along which to generate the sequence.

    Outputs:
    - `result` (Tensor): The generated tensor containing the sequence.
  }];

  let arguments = (ins SI64Attr:$start,
                       SI64Attr:$end,
                       SI64Attr:$step,
                       I64Attr:$arange_dimension,
                       TypeAttr:$dtype);

  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
}

def TTIR_FullOp : TTIR_CreationOp<"full",
  [ DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  ]> {
  let summary = "Creates a tensor filled with the specified value";
  let description = [{
    Tensor operation to create a tensor filled with a specified value.

    Given a `shape` and a `fill_value`, produces a tensor with the shape, filled with the specified value.

    Example:
      %0 = "ttir.full"() <{shape = array<i32: 64, 32, 32>, fill_value = 7 : i32}> : () -> tensor<64x32x32xi32>
      // %0: [[[7, 7, 7, ..., 7], [7, 7, 7, ..., 7], ..., [7, 7, 7, ..., 7]]]
  }];

  let arguments = (ins DenseI32ArrayAttr:$shape,
                       AnyAttrOf<[F32Attr, I32Attr]>:$fill_value);

  let results = (outs AnyRankedTensor:$result);

  let builders =
  [
    OpBuilder<(ins "Type": $resultType, "Attribute": $fillValue),
    [{
      RankedTensorType tensorType = cast<RankedTensorType>(resultType);
      build($_builder, $_state, tensorType, llvm::to_vector_of<int32_t>(tensorType.getShape()), fillValue);
    }]>
  ];

  let hasVerifier = 1;
}

def TTIR_RandOp : TTIR_CreationOp<"rand"> {
  let summary = "Random number generation operation.";
  let description = [{
    Returns a tensor filled with random numbers drawn from a uniform distribution over given interval [low, high) [Default: [0, 1)].

    Example:
      %0 = "ttir.rand"() <{dtype = bf16, high = 1.000000e+00 : f32, low = 0.000000e+00 : f32, seed = 0 : ui32, size = [32 : i32, 32 : i32]}> : () -> tensor<32x32xbf16>

    Attributes:
      - `size` (Array of Integer): The shape of the tensor to create.
      - `dtype` (mlir:Type): Data type of the returned tensor.
      - `low` (Float): The lower bound of the range (inclusive) [Default: 0.0].
      - `high` (Float): The upper bound of the range (exclusive) [Default: 1.0].
      - `seed` (Integer): Value to initialize the random number generator for reproducible results [Default: 0].

    Outputs:
      - `result` (Tensor): The generated tensor containing the random values.
  }];

  let arguments = (ins I32ArrayAttr:$size,
                       TypeAttr:$dtype,
                       DefaultValuedAttr<F32Attr, "0.0">:$low,
                       DefaultValuedAttr<F32Attr, "1.0">:$high,
                       DefaultValuedAttr<UI32Attr, "0">:$seed);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

class TTIR_NamedFullOp<string mnemonic, list<Trait> traits = []> :
  TTIR_CreationOp<mnemonic, traits> {
  let summary = "Base class for tensor creation operations with a specified shape.";
  let description = [{
    The `TTIR_NamedFullOp` class serves as a base class for tensor creation operations that
    generate tensors with a specified shape and fill them with predefined values.

    This class is used by operations like `zeros` and `ones` that create tensors with a
    specific shape and initialize all elements to the same value. The shape is specified
    as a dense array of 32-bit integers.

    Attributes:
    - `shape` (Array of Integer): The shape of the tensor to create.

    Outputs:
    - `result` (Tensor): The created tensor with the specified shape.

    Note: The element type of the result tensor is determined by the return type specified
    in the operation. This class simplifies the implementation of tensor creation operations
    by providing common argument and result definitions.
  }];

  let arguments = (ins DenseI32ArrayAttr:$shape, TypeAttr:$dtype);

  let results = (outs AnyRankedTensor:$result);
}

def TTIR_ZerosOp : TTIR_NamedFullOp<"zeros"> {
  let summary = "Creates a tensor filled with zeros.";
  let description = [{
    The `zeros` operation creates a tensor filled with zeros of the specified shape.

    This operation is commonly used to initialize tensors with zero values. It takes a shape attribute
    and produces a tensor of that shape with all elements set to zero.

    Example:
    ```mlir
    // Create a 3D tensor of zeros with shape [64, 28, 28]
    %result = ttir.zeros() {
        shape = [64, 28, 28]
    } : () -> tensor<64x28x28xbf16>
    // Result: A tensor of shape [64, 28, 28] filled with zeros

    // Create a 2D tensor of zeros with shape [3, 4]
    %result = ttir.zeros() {
        shape = [3, 4]
    } : () -> tensor<3x4xf32>
    // Result: [[0.0, 0.0, 0.0, 0.0],
    //          [0.0, 0.0, 0.0, 0.0],
    //          [0.0, 0.0, 0.0, 0.0]]
    ```

    Attributes:
    - `shape` (Array of Integer): The shape of the tensor to create.

    Outputs:
    - `result` (Tensor): The tensor filled with zeros.

    Note: The element type of the result tensor is determined by the return type specified in the operation.
    This operation is useful for initializing tensors before filling them with computed values or as a
    starting point for accumulation operations.
  }];
  let arguments = (ins DenseI32ArrayAttr:$shape);

}

def TTIR_OnesOp : TTIR_NamedFullOp<"ones"> {
  let summary = "Creates a tensor filled with ones.";
  let description = [{
    The `ones` operation creates a tensor filled with ones of the specified shape.

    This operation is commonly used to initialize tensors with one values. It takes a shape attribute
    and produces a tensor of that shape with all elements set to one.

    Example:
    ```mlir
    // Create a 3D tensor of ones with shape [64, 28, 28]
    %result = ttir.ones() {
        shape = [64, 28, 28]
    } : () -> tensor<64x28x28xbf16>
    // Result: A tensor of shape [64, 28, 28] filled with ones

    // Create a 2D tensor of ones with shape [3, 4]
    %result = ttir.ones() {
        shape = [3, 4]
    } : () -> tensor<3x4xf32>
    // Result: [[1.0, 1.0, 1.0, 1.0],
    //          [1.0, 1.0, 1.0, 1.0],
    //          [1.0, 1.0, 1.0, 1.0]]
    ```

    Attributes:
    - `shape` (Array of Integer): The shape of the tensor to create.

    Outputs:
    - `result` (Tensor): The tensor filled with ones.

    Note: The element type of the result tensor is determined by the return type specified in the operation.
    This operation is useful for initializing tensors before scaling them or as a starting point for
    operations that require tensors filled with ones, such as creating masks or constant multipliers.
  }];
}

def TTIR_EmptyOp : TTIR_CreationOp<"empty",
  [DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                      , "bufferizesToMemoryWrite"
                                                      , "bufferizesToAllocation"
                                                      , "bufferize"
                                                      , "getAliasingValues"
                                                      , "getBufferType"
                                                      ]>
  ]> {
    let summary = "Empty tensor allocation operation.";
    let description = [{
      The `empty` operation creates an uninitialized tensor with the specified shape and element type.

      This operation allocates memory for a tensor but does not initialize its values. It's commonly used
      as a first step before filling the tensor with computed values. The shape and element type of the
      tensor are determined by the return type.

      Example:
      ```mlir
      // Create an uninitialized 2D tensor with shape [3, 4]
      %result = ttir.empty() : tensor<3x4xf32>

      // Create an uninitialized 3D tensor with shape [2, 3, 4]
      %result = ttir.empty() : tensor<2x3x4xi32>

      // Use empty to create a tensor for storing computation results
      %input = ... : tensor<10x20xf32>
      %output = ttir.empty() : tensor<10x20xf32>
      %result = ttir.some_computation(%input, %output) : tensor<10x20xf32>, tensor<10x20xf32> -> tensor<10x20xf32>
      ```

      Outputs:
      - `result` (Tensor): The uninitialized tensor.

      Note: Since the tensor is uninitialized, reading from it before writing may yield undefined values.
      This operation is typically used in conjunction with other operations that will fill the tensor with
      meaningful values. The `empty` operation is more efficient than `zeros` or `ones` when the tensor
      will be completely overwritten, as it avoids the initialization step.
    }];

    let results = (outs AnyRankedTensor:$result);

    let builders =
    [
      OpBuilder<(ins "ArrayRef<int64_t>": $shape, "Type": $elementType, "Attribute": $encoding),
      [{
        build($_builder, $_state, RankedTensorType::get(shape, elementType, encoding));
      }]>,
      OpBuilder<(ins "ArrayRef<int64_t>": $shape, "Type": $elementType),
      [{
        build($_builder, $_state, shape, elementType, nullptr);
      }]>
    ];

    let assemblyFormat = "`(` `)` attr-dict `:` type($result)";
}

def TTIR_ConstantOp : TTIR_CreationOp<"constant",
  [ ConstantLike
  , AllShapesMatch<["value", "result"]>
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  ]> {
    let summary = "Tensor constant creation operation.";
    let description = [{
      The `constant` operation creates a tensor with values specified by a constant attribute.

      This operation is used to create tensors with predefined values that remain constant
      throughout program execution. It's commonly used for initializing model weights, biases,
      and other fixed parameters in neural networks.

      Example:
      ```mlir
      // Create a 2D tensor of zeros
      %result = ttir.constant() {
          value = dense<0> : tensor<2x3xi32>
      } : () -> tensor<2x3xi32>
      // Result: [[0, 0, 0], [0, 0, 0]]

      // Create a 1D tensor with specific floating-point values
      %result = ttir.constant() {
          value = dense<[0.2, 1.3]> : tensor<2xf32>
      } : () -> tensor<2xf32>
      // Result: [0.2, 1.3]

      // Create a scalar constant
      %result = ttir.constant() {
          value = dense<5.0> : tensor<f32>
      } : () -> tensor<f32>
      // Result: 5.0

      // Create a 2D tensor with different values
      %result = ttir.constant() {
          value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>
      } : () -> tensor<2x3xi32>
      // Result: [[1, 2, 3], [4, 5, 6]]
      ```

      Attributes:
      - `value` (DenseElementsAttr): The constant value of the tensor.

      Outputs:
      - `result` (Tensor): The tensor with the specified constant values.

      Note: The shape and element type of the result tensor are determined by the `value` attribute.
      The `constant` operation is typically folded during compilation, allowing for optimizations
      such as constant propagation.
    }];

    let arguments = (ins ElementsAttr:$value);

    let results = (outs AnyRankedTensor:$result);

    let hasFolder = 1;

    let hasCanonicalizer = 1;

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TTIR ccl ops
//===----------------------------------------------------------------------===//

def TTIR_AllGatherOp : TTIR_NamedOp<"all_gather"> {
    let summary = "All gather operation.";
    let description = [{
      All gather op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$all_gather_dim,
                         UI32Attr:$cluster_axis);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_AllReduceOp : TTIR_NamedOp<"all_reduce"> {
    let summary = "AllReduce operation.";
    let description = [{
      AllReduce op.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         TTCore_ReduceTypeAttr:$reduce_type,
                         UI32Attr:$cluster_axis);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}

def TTIR_ReduceScatterOp : TTIR_NamedOp<"reduce_scatter"> {
    let summary = "Reduce scatter operation.";
    let description = [{
      Reduce scatter op.
    }];
    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         TTCore_ReduceTypeAttr:$reduce_type,
                         SI32Attr:$scatter_dim,
                         UI32Attr:$cluster_axis);

    let results = (outs AnyRankedTensor:$result);
    let hasVerifier = 1;
}

def TTIR_CollectivePermuteOp : TTIR_NamedOp<"collective_permute"> {
    let summary = "Collective permute operation.";
    let description = [{
      Collective permute op. This operation ingests a multi-device tensor spread across multi-devices and will shuffle the data according to source_target_pairs [['src', 'dest']].

      Example:
        For a 1x2 mesh, the following will take the device shard living in device 0 and move it to device 1. The device shard living in device 1 will move to device 0.
        %source_target_pairs: [[0, 1], [1, 0]]

        In the case of missing 'dest', the device shard living on that device will contain values of 0. For example, device shard living in device 0 will contain 0 values.
        %source_target_pairs: [[0, 1]]
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         I64ElementsAttr:$source_target_pairs);

    let results = (outs AnyRankedTensor:$result);
    let hasVerifier = 1;
}

def TTIR_MeshShardOp: TTIR_Op<"mesh_shard",
  [ Pure
  , TTIROpInterface
  , DeclareOpInterfaceMethods<BufferizableOpInterface, [ "bufferizesToMemoryRead"
                                                       , "bufferizesToMemoryWrite"
                                                       , "bufferize"
                                                       , "getAliasingValues"
                                                       , "getBufferType"
                                                       ]>
  ]> {
    let summary = "Mesh shard operation.";
    let description = [{
      MeshShard op shards the inputs (FullToShard) or concatnates the outputs (ShardToFull) for ccl ops.

      shard_direction attribute determines whether to shard or concat.

      shard_type attribute determines how to shard or concat.
        manual: no sharding
        replicate: all devices have identical data
        maximal: only one device contains full data
        devices: shard_shape/shard_dims determine particular sharding

      shard_dims attribute determines row and column sharding dimension of input tensor

      For example, on 2x4 mesh hardware, following op shards arg0 to 8 slices, row divided by 2
      and col divided by 4.

        %1 = "ttir.mesh_shard"(%arg0) <
          {... shard_direction = #ttcore.shard_direction<full_to_shard>,
               shard_shape = array<i64: 2, 4>,
               shard_dims = array<i64: 0, 1>,
               shard_type = #ttcore.shard_type<devices>}> :  (tensor<8192x784xf32>, ...) -> tensor<4096x196xf32>

      On the other hand, this op concatnates %4 to single tensor by concatnating
      one of the top row tensor with one of the bottom row tensor.

        %6 = "ttir.mesh_shard"(%4) <
          {..., shard_direction = #ttcore.shard_direction<shard_to_full>,
                shard_shape = array<i64: 2, 1>,
                shard_dims = arrray<i64: 1, -1>,
                shard_type = #ttcore.shard_type<devices>}> : (tensor<4096x16384xf32>, ...) -> tensor<8192x16384xf32>
    }];

    let arguments = (ins AnyRankedTensorOrMemRef:$input,
                         TTCore_MeshShardTypeAttr:$shard_type,
                         TTCore_MeshShardDirectionAttr:$shard_direction,
                         DenseI64ArrayAttr:$shard_shape,
                         DenseI64ArrayAttr:$shard_dims
    );
    let results = (outs AnyRankedTensorOrMemRef:$result);

    let extraClassDefinition = [{
      bool $cppClass::bufferizesToMemoryRead(
          mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
        return true;
      }
      bool $cppClass::bufferizesToMemoryWrite(
          mlir::OpOperand &operand, const mlir::bufferization::AnalysisState &) {
        return false;
      }
      bufferization::AliasingValueList $cppClass::getAliasingValues(
          OpOperand &, const bufferization::AnalysisState &) {
        bufferization::AliasingValueList result;
        return result;
      }
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
}

def TTIR_QuantizeOp : TTIR_NamedOp<"quantize"> {
  let summary = "Quantize operation.";
  let description = [{
      The Quantize operation converts a tensor into a quantized tensor using the `quant.uniform` type from the MLIR Quant dialect.
      This type encapsulates the scale and zero-point metadata directly within the tensor type.
      The output tensor will be of type 'quant.uniform', where each element is computed as:
      ```
      output[i] = (input[i] / scale) + zero_point
      ```

      Example:
      ```mlir
      %input = ttir.empty() : () -> tensor<64x128xf32>
      %output = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
      %quantized = "ttir.quantize"(%input, %output) : (tensor<64x128xf32>, tensor<64x128x!quant.uniform<i32:f32, 0.1>>) -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>

      // In this example:
      // - The input is a 64x128 tensor of 32-bit floating-point values
      // - The output is a 64x128 tensor of 32-bit quantized values
      // - The scale is 0.1 (each step represents 0.1 in the original scale)
      // - The zero point is 128 (the value 128 in the quantized space represents 0.0 in the original space)
      ```

      Inputs:
      - `input` (Tensor): Input tensor to be quantized.

      Results:
      - `result` (Quantized Tensor): The quantized tensor with type `quant.uniform`.

      Note: The quantization parameters (scale and zero point) are specified in the result type.
      Quantization helps reduce model size and computational requirements by representing
      floating-point values with lower-precision integers, which is particularly useful for
      deployment on resource-constrained devices.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_DequantizeOp : TTIR_NamedOp<"dequantize"> {
  let summary = "Dequantize operation.";
  let description = [{
    The Dequantize operation converts a quantized tensor back into a floating-point tensor using the `quant.uniform` type from the MLIR Quant dialect.
    The input tensor is expected to be of type `quant.uniform.`
    The output tensor will be a floating-point tensor, where each element is computed as:
    ```
    output[i] = (input[i] - zero_point) * scale
    ```
    Example:
    ```mlir
    %input = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
    %output = ttir.empty() : () -> tensor<64x128xf32>
    %dequantized = "ttir.dequantize"(%input, %output) : (tensor<64x128x!quant.uniform<i32:f32, 0.1>, tensor<64x128xf32>) -> tensor<64x128xf32>

    // In this example:
    // - The input is a 64x128 tensor of 32-bit quantized values
    // - The output is a 64x128 tensor of 32-bit floating-point values
    // - The scale is 0.1 (each step represents 0.1 in the original scale)
    // - The zero point is 128 (the value 128 in the quantized space represents 0.0 in the original space)
    ```

    Inputs:
    - `input` (Quantized Tensor): The quantized tensor to be dequantized.

    Results:
    - `result` (Tensor): The floating-point tensor after dequantization.

    Note: The quantization parameters (scale and zero point) are specified in the input tensor type.
    Dequantization is the reverse process of quantization, converting quantized values back to floating-point values.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_QuantizeUnrolledOp : TTIR_NamedOp<"quantize_unrolled"> {
  let summary = "Quantize operation unrolled (scale and zero point as input operands).";
  let description = [{
    The QuantizeUnrolledOp quantizes a tensor using the scale and zero point provided as input operands.

    Inputs:
      - `input` AnyRankedTensor: The input tensor to be quantized. Must have floating-point element type.
      - `scale` AnyRankedTensor: The scale factor (or factors for per-axis quantization). Must be either a scalar (for per-tensor quantization) or a 1D tensor with size matching the dimension of the specified axis (for per-axis quantization).
      - `zero_point` AnyRankedTensor: The zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
      - `axis` Optional<i32>: The axis along which quantization is applied. Must be in range [0, rank) where rank is the rank of the input tensor.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$scale,
                       AnyRankedTensor:$zero_point,
                       OptionalAttr<I32Attr>:$axis,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_DequantizeUnrolledOp : TTIR_NamedOp<"dequantize_unrolled"> {
  let summary = "Dequantize operation unrolled (scale and zero point as input operands).";
  let description = [{
    The DequantizeUnrolledOp dequantizes a tensor using the scale and zero point provided as input operands.

    Inputs:
      - `input` AnyRankedTensor: The input tensor to be dequantized. Must have quantized element type.
      - `scale` AnyRankedTensor: The scale factor (or factors for per-axis quantization).
      - `zero_point` AnyRankedTensor: The zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
      - `axis` Optional<i32>: The axis along which quantization is applied. Must be in range [0, rank) where rank is the rank of the input tensor.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$scale,
                       AnyRankedTensor:$zero_point,
                       OptionalAttr<I32Attr>:$axis,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_RequantizeOp : TTIR_NamedOp<"requantize"> {
  let summary = "Requantize operation.";
  let description = [{
    The Requantize operation converts a quantized tensor from one scale and zero-point to another, using the `quant.uniform` type from the MLIR Quant dialect.
    The input tensor is expected to be of type `quant.uniform.`
    The output tensor will also be of type `quant.uniform.`
    Each element in the output tensor is computed as:
    ```
    output[i] = round((input[i] - input_zero_point) * (input_scale / output_scale)) + output_zero_point
    ```
    Example:
    ```mlir
    %input = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.1>>
    %output = ttir.empty() : () -> tensor<64x128x!quant.uniform<i32:f32, 0.2>>
    %requantized = "ttir.requantize"(%input, %output) : (tensor<64x128x!quant.uniform<i32:f32, 0.1>, tensor<64x128x!quant.uniform<i32:f32, 0.2>>) -> tensor<64x128x!quant.uniform<i32:f32, 0.2>>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;

  let hasFolder = 1;
}

def TTIR_RequantizeUnrolledOp : TTIR_NamedOp<"requantize_unrolled"> {
  let summary = "Requantize operation unrolled (scale and zero point as input operands).";
  let description = [{
    The RequantizeUnrolledOp requantizes a tensor using the scale and zero point provided as input operands.

    Inputs:
      - `input` AnyRankedTensor: The input tensor to be requantized. Must have quantized element type.
      - `in_scale` AnyRankedTensor: The input scale factor (or factors for per-axis quantization). Must be either a scalar (for per-tensor quantization) or a 1D tensor with size matching the dimension of the specified axis (for per-axis quantization).
      - `in_zero_point` AnyRankedTensor: The input zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
      - `out_scale` AnyRankedTensor: The output scale factor (or factors for per-axis quantization). Must be either a scalar (for per-tensor quantization) or a 1D tensor with size matching the dimension of the specified axis (for per-axis quantization).
      - `out_zero_point` AnyRankedTensor: The output zero point value (or values for per-axis quantization). Must be in range of the quantized storage type.
      - `axis` Optional<i32>: The axis along which quantization is applied. Must be in range [0, rank) where rank is the rank of the input tensor.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$in_scale,
                       AnyRankedTensor:$in_zero_point,
                       AnyRankedTensor:$out_scale,
                       AnyRankedTensor:$out_zero_point,
                       OptionalAttr<I32Attr>:$axis,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_AllToAllOp : TTIR_NamedOp<"all_to_all"> {
    let summary = "All to All operation.";
    let description = [{
      The all_to_all operation redistributes slices of a tensor across a cluster of devices. It splits each local tensor along split_dimension, sends
      the resulting slices to other devices along cluster_axis, and then concatenates the received slices along concat_dimension.

      Example:
        For a 1x2 mesh and a local input of shape [8, 4]:
          - split_dimension = 1
          - concat_dimension = 0
          - split_count = 2
          - cluster_axis = 1

        Each device splits its [8, 4] tensor into two [8, 2] slices. After the exchange, each device concatenates the two received [8, 2] slices
        into a [16, 2] output tensor.
    }];

    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$output,
                         SI32Attr:$split_dim,
                         SI32Attr:$concat_dim,
                         SI32Attr:$split_count,
                         I64ElementsAttr:$replica_groups);

    let results = (outs AnyRankedTensor:$result);
    let hasVerifier = 1;
}


def TTIR_CollectiveBroadcastOp : TTIR_NamedOp<"collective_broadcast"> {
  let summary = "Collective Broadcast op";
  let description = [{
    The collective_broadcast operation distributes a tensor from a single source device to all
    other devices within each replica group. Each replica group defines a subset of devices that
    participate in the broadcast, and the operation is applied independently within each group.

    By convention, the first device listed in each replica group is treated as the broadcast source.
    The value of the `input` tensor on that source device is sent to all other devices in the same
    group. The `input` tensor values on non-source devices are ignored and will be overwritten
    during the operation.

    Inputs:
      - input: The tensor to broadcast. Only the value on the first device of each replica group
              (the source) is used; values on other devices are ignored.
      - output: An allocated tensor that will be populated with the broadcasted value on all devices.
      - replica_groups: A list of replica groups. Each group is a list of device IDs, and the first
                        ID in each group is treated as the broadcast source for that group.

    Result:
      - result: The output tensor containing the broadcasted value, identical across all devices
                in the same replica group.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                        AnyRankedTensor:$output,
                        I64ElementsAttr:$replica_groups);

  let results = (outs AnyRankedTensor:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
}


def TTIR_ConcatenateHeadsOp: TTIR_NamedOp<"concatenate_heads"> {
  let summary = "Concatenate heads operation.";
  let description = [{
    The `concatenate_heads` operation concatenates multiple heads of a multi-head attention tensor into a single tensor.

    This operation is typically used in transformer models where the attention mechanism is split into multiple heads.
    It combines the outputs of these heads into a single tensor, allowing further processing.

    It takes an input tensor with shape `[batch_size, num_heads, sequence_size, head_size]`
    and produces an output tensor with shape `[batch_size, sequence_size, num_heads * head_size]`.

    It corresponds to a sequence of permute and reshape operations.

    Example:
    ```mlir
    // Concatenate heads from a multi-head attention output
    %input = ... : tensor<1x24x32x128xbf16> // batch_size: 1, num_heads: 24, sequence_size: 32, head_size: 128
    %output = ttir.empty() : tensor<1x32x3072xbf16>  // Concatenated output of shape [batch_size, sequence_size, num_heads * head_size]
    %result = ttir.concatenate_heads(%input, %output) : tensor<1x24x32x128xbf16>, tensor<1x32x3072xbf16> -> tensor<1x32x3072xbf16>
    // Input tensor shape: [1, 24, 32, 128]
    // Output tensor shape: [1, 32, 3072]
    ```

    Inputs:
    - `input` (Tensor): The input tensor containing multiple heads.

    Outputs:
    - `result` (Tensor): The concatenated output tensor.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}



def TTIR_ScaledDotProductAttentionDecodeOp : TTIR_NamedOp<"scaled_dot_product_attention_decode", [AttrSizedOperandSegments]> {
    let summary = "A version of scaled dot product attention specifically for decode.";
    let description = [{
      A version of scaled dot product attention specifically for decode.
        The implementation is Flash-Decode and it currently only supports MQA on decoding single token.

        Args:
            query (AnyRankedTensor): The query tensor [1 x batch x num_heads x head_size]. Note that there is no sequence length dimension as this op is intended for processing a single query token.
            key (AnyRankedTensor): The key tensor [batch x num_kv_heads x seq_len x head_size].
            value (AnyRankedTensor): The value tensor [batch x num_kv_heads x seq_len x head_size].
            is_causal (bool, optional): Whether the attention is causal. Defaults to `true`.
            attention_mask (AnyRankedTensor, optional): The attention mask [batch x 1 x num_heads x seq_len].
            cur_pos_tensor (AnyRankedTensor): [batch] Tensor of integers of length batch.
            attention_sink (AnyRankedTensor, optional): The attention sink [num_heads, 32] (must be a single tile wide).
            output (AnyRankedTensor): The output DPS operand [1 x batch x num_heads x head_size].
            scale (float, optional): Defaults to `None`.

        Returns:
            AnyRankedTensor: The output tensor [1 x batch x num_heads x head_size].
    }];

    let arguments = (ins AnyRankedTensor:$query,
                         AnyRankedTensor:$key,
                         AnyRankedTensor:$value,
                         DefaultValuedAttr<BoolAttr, "true">:$is_causal,
                         Optional<AnyRankedTensor>:$attention_mask,
                         AnyRankedTensor:$cur_pos_tensor,
                         Optional<AnyRankedTensor>:$attention_sink,
                         AnyRankedTensor:$output,
                         OptionalAttr<F32Attr>:$scale);

    let results = (outs AnyRankedTensor:$result);

    let hasVerifier = 1;
}


def TTIR_ScaledDotProductAttentionOp : TTIR_NamedOp<"scaled_dot_product_attention"> {
  let summary = "Scaled dot product attention operation.";
  let description = [{
    Scaled dot product attention.
    The implementation is FlashAttention-2.

    Args:
        query (AnyRankedTensor): The query tensor.          [batch x num_heads x query_seq_len x head_size]
        key (AnyRankedTensor): The key tensor.              [batch x num_kv_heads x kv_seq_len x head_size]
        value (AnyRankedTensor): The value tensor.          [batch x num_kv_heads x kv_seq_len x head_size]
        attention_mask (AnyRankedTensor, optional): Defaults to `None`. [batch x 1 x query_seq_len x kv_seq_len]. Head broadcasting is implied.
        output (AnyRankedTensor): The output DPS operand [batch x num_heads x query_seq_len x head_size].
        is_causal (bool): Whether The attention is causal. Defaults to `true`.
        scale (float, optional): Defaults to `None`.

    Returns:
        AnyRankedTensor: The output tensor [batch x num_heads x seq_len x head_size].
  }];

  let arguments = (ins AnyRankedTensor:$query,
                       AnyRankedTensor:$key,
                       AnyRankedTensor:$value,
                       Optional<AnyRankedTensor>:$attention_mask,
                       AnyRankedTensor:$output,
                       DefaultValuedAttr<BoolAttr, "true">:$is_causal,
                       OptionalAttr<F32Attr>:$scale);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_GlobalAvgPool2dOp: TTIR_NamedOp<"global_avg_pool2d">{
  let summary = "A global average pooling 2d operation";
  let description = [{
    The `global_avg_pool2d` operation applies global average pooling over the spatial dimensions
    (height and width) of a 4D input tensor. In essence, it should be realised as the sum-reduce style operation
    under the hood, for performance reasons (since we include all elements, there is no need for kernel allocation).
    It reduces spatial dimensions to 1.

    This operation yields the same result as the standard avg_pool_2d operation when kernel_size == [input_height, input_width].
    However, it is much more memory efficient, as the underlying implementation doesn't need to allocate
    the kernel for pooling (since the operation boils down to a simple sum-reduce).

    Example:
    ```mlir
    // Basic global average pooling reducing 32x32 spatial dimensions to 1x1
    %input = ... : tensor<1x32x32x64xbf16>  // Input tensor with values in NHWC format
                                            // N=1 (batch), H=32 (height), W=32 (width), C=64 (channels)
    %output = ttir.empty() : tensor<1x1x1x64xbf16>
    %result = "ttir.global_avg_pool2d"(%input, %output) : (tensor<1x32x32x64xbf16>, tensor<1x1x1x64xbf16>) -> tensor<1x1x1x64xbf16>
    // Result: tensor<1x1x1x64xbf16> where each channel contains the average of all 32*32=1024 spatial locations

    // Example with different input sizes
    %large_input = ... : tensor<1x128x128x32xbf16>  // Large spatial dimensions
    %large_output = ttir.empty() : tensor<1x1x1x32xbf16>
    %large_result = "ttir.global_avg_pool2d"(%large_input, %large_output) : (tensor<1x128x128x32xbf16>, tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xbf16>
    // Each output channel averages across 128*128=16384 spatial locations
    ```

    Inputs:
    - `input`: 4D tensor with shape [N, H, W, C] where N is batch size, H is height, W is width, and C is channels
    - `output`: Pre-allocated output tensor with shape [N, 1, 1, C] to store the result

    Attributes:
    - None (this operation has no configurable attributes, unlike regular pooling operations)

    Outputs:
    - `result`: 4D tensor with shape [N, 1, 1, C] containing the global average pooled values

    Note: The operation reduces spatial dimensions (H, W) to (1, 1) by computing the average across
    all spatial locations for each channel independently. This is equivalent to avg_pool2d with
    kernel_size=[H, W], stride=[1, 1], and no padding, but implemented more efficiently.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       AnyRankedTensor:$output);

  let results = (outs AnyRankedTensor:$result);

  let hasVerifier = 1;
}

def TTIR_SplitQueryKeyValueAndSplitHeadsOp: TTIR_NamedOp<"split_query_key_value_and_split_heads"> {
    let summary = "Split query, key, values and split heads op used in attention layer.";
    let description = [{
      Splits input_tensor of shape [batch_size, sequence_size, 3 * hidden_size] into 3 tensors (Query, Key, Value) of shape [batch_size, sequence_size, hidden_size]. Then, reshapes and permutes the output tensors, to make them ready for computing attention scores.
      If kv_input_tensor is passed in, then input_tensor of shape [batch_size, sequence_size, hidden_size] is only used for Query, and kv_input_tensor of shape [batch_size, sequence_size, 2 * hidden_size] is used for Key and Value.
      For the sharded implementation, the input query, key and value are expected to be concatenated such that the heads are interleaved (q1 k1 v1…qn kn vn).
    }];

    let arguments = (ins AnyRankedTensor:$input_tensor,
                         Optional<AnyRankedTensor>:$kv_input_tensor,
                         AnyRankedTensor:$query_output,
                         AnyRankedTensor:$key_output,
                         AnyRankedTensor:$value_output,
                         UI32Attr:$num_heads,
                         OptionalAttr<UI32Attr>:$num_kv_heads,
                         BoolAttr:$transpose_key);

    let results = (outs AnyRankedTensor:$query,
                        AnyRankedTensor:$key,
                        AnyRankedTensor:$value);

    let extraClassDeclaration = [{
        MutableOperandRange getDpsInitsMutable() {
            // query_output is at index 2 (after input_tensor and kv_input_tensor)
            // But kv_input_tensor is optional, so we need to account for that
            unsigned startIdx = getKvInputTensor() ? 2 : 1;
            return MutableOperandRange(*this, startIdx, 3);
        }
    }];

    let hasVerifier = 1;
}
#endif
