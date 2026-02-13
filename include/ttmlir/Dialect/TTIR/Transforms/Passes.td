// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD
#define TTMLIR_TTMLIR_DIALECT_TTIR_TTIRPASSES_TD

include "mlir/Pass/PassBase.td"

def TTIRImplicitBroadcastFold: Pass<"ttir-implicit-broadcast-fold", "::mlir::ModuleOp"> {
  let summary = "Broadcast operation is folded to all the consumers.";
  let description = [{
    This pass walks through the graph and folds broadcasts operations when it is implicitly supported by the operation.

    Example:
    %0 = ttir.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i32: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = ttir.empty() : tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>

    Since MultiplyOp supports implicit broadcasting, above broadcast is folded as:
    %0 = ttir.empty() : tensor<1x16x32xf32>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  }];
}

def CPUHoistManuallyTaggedTransform: Pass<"cpu-hoist-manually-tagged", "::mlir::ModuleOp">
{
  let summary = "Hoist ops manually tagged with ttir.should_hoist to the CPU module";
  let description = [{
    Transform pass which finds ops tagged with the `ttir.should_hoist` attribute
    and hoists them into their own function in the CPU module. A function declaration
    is created in the device module, and the original ops are replaced with a call
    to that function declaration.

    Example:
    input:
      ttcore.device_module {
        builtin.module {
          func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
            %0 = ttir.empty() : tensor<32x32xbf16>
            %1 = "ttir.add"(%arg0, %arg1, %0) {ttir.should_hoist} : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16> loc("add_op1")
            return %1 : tensor<32x32xbf16>
          }
        }
      }
    output:
      ttcore.device_module {
        builtin.module {
          func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
            %0 = ttir.empty() : tensor<32x32xbf16>
            %1 = call @cpu_hoisted_ttir_add_<hash>(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
            return %1 : tensor<32x32xbf16>
          }
          func.func private @cpu_hoisted_ttir_add_<hash>(tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
        }
      }
      ttcore.cpu_module {
        builtin.module {
          func.func @cpu_hoisted_ttir_add_<hash>(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>, %arg2: tensor<32x32xbf16>) -> tensor<32x32xbf16> attributes {arg_ranks = [2, 2, 2, 2], result_ranks = [2]} {
            %0 = "ttir.add"(%arg0, %arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
            return %0 : tensor<32x32xbf16>
          }
        }
      }
  }];

  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect"];
}

def CPUHoistNonLowerableSHLOOpsTransform: Pass<"cpu-hoist-non-lowerable-shlo-ops", "::mlir::ModuleOp">
{
  let summary = "Hoist non-lowerable StableHLO ops to the CPU module";
  let description = [{
    Transform pass which finds StableHLO ops that cannot be lowered to TTIR and
    hoists them to the CPU module. The allow-list of ops to hoist is hardcoded;
    before adding an op, make sure it is supported in StableHLO to TOSA/Linalg
    conversion.

    When StableHLO support is not compiled in, this pass is a no-op.
  }];

  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect"];
}

def CPUHoistConstEvalTransform: Pass<"cpu-hoist-const-eval", "::mlir::ModuleOp">
{
  let summary = "Hoist ops from const-eval subgraphs to the CPU module";
  let description = [{
    Transform pass which finds const-eval functions and hoists ops within them
    into the CPU module functions. This improves precision (CPU uses 32-bit types)
    and reduces peak DRAM/L1 usage by keeping intermediate tensors in host memory.
  }];

  let dependentDialects = ["::mlir::tt::ttcore::TTCoreDialect"];
}

def ElementTypeNormalization: Pass<"ttir-element-type-normalization", "::mlir::ModuleOp">
{
  let summary = "Normalize element types into list of supported types.";
  let description = [{
    This pass walks through the graph and normalizes the element types into a list of supported types. This is useful for lowering
    to a target that only supports a subset of the element types.
  }];

  list<Option> options = [
    Option<"enableBfp8Conversion", "enable-bfp8-conversion", "bool", "false", "When enabled this pass will convert all bfloat16 types into bfp8_b types.">,
  ];
}

def TTIRFlattenSlidingWindow: Pass<"ttir-flatten-sliding-window", "::mlir::ModuleOp">
{
  let summary = "Flatten sliding window ops.";
  let description = [{
    This is a compatibility pass for converting to the TTNN dialect.
    This pass walks through the graph and flattens sliding window ops (ttir.conv2d, ttir.max_pool2d, ttir.avg_pool2d).

    Example:
      Before:
         %dps = ttir.empty() : tensor<3x15x31x16xbf16>
         %1 = "ttir.conv2d"(%input, %weight, %bias, %dps)
            <{
              stride = 2: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<3x32x64x8xbf16>, tensor<16x8x3x3xbf16>, tensor<1x1x1x16xbf16>, tensor<3x15x31x16xbf16>) -> tensor<3x15x31x16xbf16>

      After:
        %reshape_dps = ttir.empty() : tensor<1x1x6144x8xbf16>
        %0 = "ttir.reshape"(%input, %reshape_dps) <{[i32: 1, i32: 1, i32: 6144, i32: 8]}> : (tensor<3x32x64x8xbf16>, tensor<1x1x6144x8xbf16>) -> tensor<1x1x6144x8xbf16>
        %new_conv_dps = ttir.empty() : tensor<1x1x1395x16xbf16>
        %1 = "ttir.conv2d"(%0, %weight, %bias, %new_conv_dps)
            <{
              stride = 2: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32,
              flattened_compat_info = #ttir<flattened_compat in_channels = 8, out_channels = 16, batch_size = 3, input_height = 32, input_width = 64,>
            }> : (tensor<1x1x6144x8xbf16>, tensor<16x8x3x3xbf16>, tensor<1x1x1x16xbf16>, tensor<1x1x1395x16xbf16>) -> tensor<1x1x1395x16xbf16>
          %output_reshape_dps = ttir.empty() : tensor<3x15x30x16xbf16>
          %2 = "ttir.reshape"(%1, %output_reshape_dps) <{[i32: 3, i32: 15, i32: 31, i32: 16]}> : (tensor<1x1x1395x16xbf16>, tensor<3x15x31x16xbf16>) -> tensor<3x15x31x16xbf16>
  }];
}

def TTIREraseInverseOps: Pass<"ttir-erase-inverse-ops", "::mlir::ModuleOp">
{
  let summary = "Erase inverse ops.";
  let description = [{
    This pass walks through the graph and erases inverse operations.

    For example:
      ttir.permute(0, 1, 3, 2) -> ttir.exp -> ttir.permute(0, 1, 3, 2)

    The above sequence can be reduced to simply: "ttir.exp" as the permutations
    on either end are inverses.
  }];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect", "mlir::tt::ttir::TTIRDialect"];

  list<Option> options = [
    Option<"enableCommuteUpwards", "enable-commute-upwards", "bool", "true", "Enable commuting upwards. This should only be false for testing purposes (i.e you want to test a commute downwards pattern)">,
    Option<"enableCommuteDownwards", "enable-commute-downwards", "bool", "true", "Enable commuting downwards. This should only be false for testing purposes (i.e you want to test a commute upwards pattern)">,
    Option<"maxIterations", "max-iterations", "uint64_t", "100", "Maximum number of iterations to perform commuting. The number of TMs is expected to converge before this limit is reached.">,
    Option<"force", "force", "bool", "true", "Force the pass to run even if we don't expect any inverse TMs (no operations have TTIR_FlattenedCompatInfoAttr). When false, the pass only runs on funcOps containing at least one operation with the flattened_compat_info attribute.">
  ];
}

def TTIRExplicateTMs: Pass<"ttir-explicate-tms", "::mlir::ModuleOp">
{
  let summary = "This pass walks through the graph and explicates implicit broadcasts and reshapes on the graph edges.";
  let description = [{
    This pass walks through the graph and explicates implicit broadcasts and reshapes on the graph edges.
  }];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect", "mlir::tt::ttir::TTIRDialect"];
}

def TTIRQuantDataTypeConversionPass : Pass<"ttir-quant-data-type-conversion", "::mlir::ModuleOp"> {
  let summary = "Convert integer data types in quantized types to a specified bit width";
  let description = [{
    This pass converts all integer data types in quantized types (e.g., i8) to a specified
    bit width (e.g., i32) during ttir-to-ttnn conversion. This is a temporary workaround as
    tt-metal currently only supports i32 quantized types.

    Example:
    Input:
      %0 = "ttir.quantize"(%arg0, %1) : (tensor<1x3x224x224xf32>, tensor<1x3x224x224x!quant.uniform<i8:f32, 1.000000e-01>>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 1.000000e-01>>

    Output (with quant_bit_width=32):
      %0 = "ttir.quantize"(%arg0, %1) : (tensor<1x3x224x224xf32>, tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<1x3x224x224x!quant.uniform<i32:f32, 1.000000e-01>>
  }];
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::quant::QuantDialect"];

  list<Option> options = [
    Option<"targetBitWidth", "target-bit-width", "uint32_t", "32", "Target integer bit width for quantized types (8, 16, 32, 64)">
  ];
}

def TTIRQuantDequantConversion : Pass<"ttir-quant-dequant-conversion", "::mlir::ModuleOp">
{
  let summary = "Convert floating-point ops surrounded by quantize/dequantize into TTIR quantized operations.";
  let description = [{
    This pass detects and fuses the (x -> dequantize → floating-point operation)
    pattern into a single TTIR operation that consumes quantized inputs and produces
    quantized outputs. If the target operation implements the QuantizableOpInterface,
    it is rewritten directly into its quantized form. Otherwise, a fallback transformation
    inserts dequantize and quantize nodes around the original op to maintain correctness.
  }];
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::quant::QuantDialect"];
}

def TTIRFusing: Pass<"ttir-fusing", "::mlir::ModuleOp">
{
  let summary = "TTIR fusing pass.";
  let description = "This pass tries to fuse operations together with goal to reduce the number of operations in the graph.";

  let options = [
      Option<"conv2dWithMultiplyEnabled",
             "ttnn-enable-conv2d-with-multiply-pattern",
             "bool", /*default=*/"false",
             "Controls if we should enable the Conv2dWithMultiply pattern">,
      Option<"permuteMatmulEnabled",
             "enable-permute-matmul-fusion",
             "bool", /*default=*/"true",
             "Fuse permute ops into matmul/linear transpose attributes">,
  ];
}

def TTIRMoveReshapeToConstant: Pass<"ttir-move-reshape-to-constant", "::mlir::ModuleOp">
{
  let summary = "Move reshapes from activation paths to constant paths.";
  let description = [{
    This pass identifies reshape operations that exist solely to match the shape
    of a constant tensor operand in elementwise operations. Instead of reshaping
    the activation tensor, it moves the inverse reshape onto the constant path,
    which can then be folded or const-eval'd away.

    Example:
      Before:
        %const = ttir.constant() {value = dense<2.0> : tensor<32x1x2560xf32>} : () -> tensor<32x1x2560xf32>
        %activation = ... : tensor<32x2560xf32>
        %reshaped = ttir.reshape(%activation) {shape = [32, 1, 2560]} : ... -> tensor<32x1x2560xf32>
        %result = ttir.pow(%reshaped, %const) : ... -> tensor<32x1x2560xf32>

      After:
        %const = ttir.constant() {value = dense<2.0> : tensor<32x1x2560xf32>} : () -> tensor<32x1x2560xf32>
        %const_reshaped = ttir.reshape(%const) {shape = [32, 2560]} : ... -> tensor<32x2560xf32>
        %activation = ... : tensor<32x2560xf32>
        %result = ttir.pow(%activation, %const_reshaped) : ... -> tensor<32x2560xf32>

    The reshape on the constant path can be folded away during const-eval, removing
    the reshape from the critical activation path.
  }];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect", "mlir::tt::ttir::TTIRDialect"];
}

def TTIRFoldConstantReshapeBroadcast: Pass<"ttir-fold-constant-reshape-broadcast", "::mlir::ModuleOp">
{
  let summary = "Fold reshape and broadcast operations on constants into new constants.";
  let description = [{
    This pass folds reshape and broadcast operations on constant tensors when the
    result is consumed by elementwise operations (add, sub, mul, div, pow). Instead
    of keeping the reshape/broadcast op, it creates a new constant with the target shape.

    Reshape Example:
      Before:
        %const = ttir.constant() {value = dense<2.0> : tensor<32x1x2560xf32>} : () -> tensor<32x1x2560xf32>
        %const_reshaped = ttir.reshape(%const) {shape = [32, 2560]} : ... -> tensor<32x2560xf32>
        %result = ttir.pow(%activation, %const_reshaped) : ... -> tensor<32x2560xf32>

      After:
        %const = ttir.constant() {value = dense<2.0> : tensor<32x2560xf32>} : () -> tensor<32x2560xf32>
        %result = ttir.pow(%activation, %const) : ... -> tensor<32x2560xf32>

    Broadcast Example:
      Before:
        %const = ttir.constant() {value = dense<2.0> : tensor<1x2560xf32>} : () -> tensor<1x2560xf32>
        %const_bcast = ttir.broadcast(%const) : ... -> tensor<32x2560xf32>
        %result = ttir.multiply(%activation, %const_bcast) : ... -> tensor<32x2560xf32>

      After:
        %const = ttir.constant() {value = dense<2.0> : tensor<32x2560xf32>} : () -> tensor<32x2560xf32>
        %result = ttir.multiply(%activation, %const) : ... -> tensor<32x2560xf32>

    This eliminates the reshape/broadcast operation entirely from the graph.
  }];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect", "mlir::tt::ttir::TTIRDialect"];
}

def TTIRInferKVCacheArgumentTypes: Pass<"ttir-infer-kv-cache-argument-types", "::mlir::ModuleOp">
{
  let summary = "Infer KV cache arguments from cache operations.";
  let description = [{
    This pass identifies function arguments used as cache inputs to FillCacheOp,
    UpdateCacheOp, PagedFillCacheOp, or PagedUpdateCacheOp and marks them with
    the ttcore.kv_cache unit attribute.

    This is used to distinguish KV cache arguments from regular input arguments
    and avoid e.g. forcing to Row Major layout.
  }];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect", "mlir::tt::ttir::TTIRDialect"];
}

def TTIRDecomposeComplexReshape: Pass<"ttir-decompose-complex-reshape", "::mlir::ModuleOp"> {
  let summary = "Decompose reshapes that flip trailing 1s into permute + reshape.";
  let description = [{
    Detects two kinds of reshape that require a permute for correct data movement:

    1. Singleton transpose: non-1 dims are the same but trailing-1 status differs.
       Decomposed into at most reshape + swap-last-two-dims permute + reshape.
         [1, 128] -> [128, 1]:  permute [1, 0]
         [32, 8, 17] -> [32, 8, 17, 1]:  reshape to [32, 8, 1, 17], permute [0, 1, 3, 2]
         [64, 1, 1] -> [1, 1, 64]:  reshape to [1, 64, 1], permute [0, 2, 1]

    2. Flatten and swap: non-1 dims differ AND trailing 1s need to become
       leading 1s. Flattened to (X, 1), swapped to (1, X), then reshaped.
         [128, 4, 1] -> [1, 512]:  reshape to [512, 1], permute [1, 0]
         [512, 1] -> [1, 32, 16]:  permute [1, 0] to [1, 512], reshape to [1, 32, 16]
  }];
}

def TTIRMultiDeviceTensorAnnotation: Pass<"ttir-multi-device-tensor-annotation", "::mlir::ModuleOp">
{
  let summary = "TTIR multi-device tensor annotation pass.";
  let description = [{
    This pass walks through the graph with mesh and add multi-device tensor annotations to tensor encoding.

    Example:
      Before:
         %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1024x2048xf32>) -> tensor<512x512xf32>
        %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1024x2048xf32>) -> tensor<512x512xf32>
        %2 = ttir.empty() : tensor<512x512xf32>
        %3 = "ttir.add"(%0, %1, %2) : (tensor<512x512xf32>, tensor<512x512xf32>, tensor<512x512xf32>) -> tensor<512x512xf32>
        %4 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512x512xf32>) -> tensor<1024x2048xf32>
        return %4 : tensor<1024x2048xf32>

      After:
        %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1024x2048xf32>) -> tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>
        %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1024x2048xf32>) -> tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>
        %2 = ttir.empty() : tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>
        %3 = "ttir.add"(%0, %1, %2) : (tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>, tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>, tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>) -> tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>
        %4 = "ttir.mesh_shard"(%3) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<512x512xf32, #ttcore.tensor_mesh<"mesh">>) -> tensor<1024x2048xf32>
        return %4 : tensor<1024x2048xf32>
  }];

  let dependentDialects = ["mlir::tt::ttcore::TTCoreDialect", "mlir::tt::ttir::TTIRDialect"];
}

def TTIRFoldFullToScalar: Pass<"ttir-fold-full-to-scalar", "::mlir::ModuleOp"> {
  let summary = "Fold creation ops into scalar when all users support implicit broadcasting.";
  let description = [{
    This pass finds ttir.full, ttir.zeros, and ttir.ones ops whose users all
    have the Broadcastable trait and replaces them with a volume-1 (scalar)
    equivalent. Broadcastable ops will implicitly broadcast the scalar to
    the required shape, saving memory by materializing only a single element
    instead of a full tensor.

    When the creation op was the shape carrier for a consumer, the consumer's
    result type is updated and an explicit broadcast is added on its output
    to preserve the original shape.

    Example:
      Before:
        %0 = "ttir.full"() <{shape = array<i32: 64, 32>, fill_value = 1.0 : f32}> : () -> tensor<64x32xf32>
        %1 = "ttir.add"(%arg0, %0) : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>

      After:
        %0 = "ttir.full"() <{shape = array<i32: 1>, fill_value = 1.0 : f32}> : () -> tensor<1xf32>
        %1 = "ttir.add"(%arg0, %0) : (tensor<64x32xf32>, tensor<1xf32>) -> tensor<64x32xf32>
  }];
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect"];
}

#endif
