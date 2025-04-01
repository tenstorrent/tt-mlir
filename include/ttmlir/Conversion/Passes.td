// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_CONVERSION_PASSES
#define TTMLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

#ifdef TTMLIR_ENABLE_STABLEHLO
def ConvertStableHLOToTTIR : Pass<"convert-stablehlo-to-ttir", "::mlir::ModuleOp"> {
let summary = "Convert StableHLO dialect to TTIR dialect.";
  let constructor = "createConvertStableHLOToTTIRPass()";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect", "mlir::sdy::SdyDialect", "mlir::tt::ttir::TTIRDialect"];
}
def ConvertArithToStableHLO : Pass<"convert-arith-to-stablehlo", "::mlir::ModuleOp"> {
let summary = "Convert Arith Dialect to StableHLO dialect.";
  let constructor = "createConvertArithToStableHLOPass()";
  let dependentDialects = ["mlir::stablehlo::StablehloDialect", "mlir::sdy::SdyDialect", "mlir::arith::ArithDialect"];
}
#endif

def ConvertTosaToTTIR : Pass<"convert-tosa-to-ttir", "::mlir::ModuleOp"> {
  let summary = "Convert TOSA dialect to TTIR dialect.";
  let constructor = "createConvertTosaToTTIRPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect"];
}

def TTIRToTTIRDecomposition: Pass<"ttir-to-ttir-decomposition", "::mlir::ModuleOp"> {
  let summary = "Decomposes TTIR operations into simpler TTIR operations.";
  let constructor = "createTTIRToTTIRDecompositionPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect"];
}

def TTIRToTTIRGeneric: Pass<"ttir-to-ttir-generic", "::mlir::ModuleOp"> {
  let summary = "Convert named TTIR operations to their ttir.generic form.";
  let description = [{
    This pass converts "named" ops to a nested ttir.generic/linalg.generic structure, with
    ttir.generic denoting the available degrees of parallelism across a grid of cores
    and linalg.generic adding another level of nesting for a single core's data movement/compute
    task. This conversion will do an appropriate decomposition of the original op
    into lower-level tiled/blocked ops.

    Example:
    ```mlir
      %0 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<128x96xf32>, tensor<128x96xf32>, tensor<128x96xf32>) -> tensor<128x96xf32>
      %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<128x96xf32>, tensor<1x96xf32>) -> tensor<1x96xf32>
      %2 = "ttir.matmul"(%arg0, %arg1, %arg2) <{transpose_a = false, transpose_b = false}> : (tensor<128x96xf32>, tensor<96x64xf32>, tensor<128x64xf32>) -> tensor<128x64xf32>
    ```
    becomes
    ```mlir
      #map = affine_map<(d0, d1) -> (d0, d1)>
      #map1 = affine_map<(d0, d1) -> (0, 0)>
      #map2 = affine_map<(d0, d1) -> (0, d1)>
      #map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
      #map4 = affine_map<(d0, d1, d2) -> (d2, d1)>
      #map5 = affine_map<(d0, d1, d2) -> (d0, d1)>
      #parallel = #tt.iterator_type<parallel>
      #reduction = #tt.iterator_type<reduction>
      #layout = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<4x3x!tt.tile<32x32, f32>, #l1_>>
      #layout1 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<1x3x!tt.tile<32x32, f32>, #l1_>>
      #layout2 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<1x1x!tt.tile<32x32, f32>, #l1_>>
      #layout3 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<3x2x!tt.tile<32x32, f32>, #l1_>>
      #layout4 = #tt.metal_layout<(d0, d1) -> (d0, d1), undef, <1x1>, memref<4x2x!tt.tile<32x32, f32>, #l1_>>

      %0 = "ttir.generic"(%arg0, %arg1, %arg2) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^compute(%cb0: memref<4x3x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x3x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x3x!tt.tile<32x32, f32>, #l1_>):
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<4x3x!tt.tile<32x32, f32>, #l1_>, memref<4x3x!tt.tile<32x32, f32>, #l1_>) outs(%cb2 : memref<4x3x!tt.tile<32x32, f32>, #l1_>) {
        ^bb0(%in: !tt.tile<32x32, f32>, %in_0: !tt.tile<32x32, f32>, %out: !tt.tile<32x32, f32>):
          %1 = "ttir.tile_add"(%in, %in_0) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
          linalg.yield %1 : !tt.tile<32x32, f32>
        }
      }) : (tensor<128x96xf32, #layout>, tensor<128x96xf32, #layout>, tensor<128x96xf32, #layout>) -> tensor<128x96xf32, #layout>
      return %0 : tensor<128x96xf32, #layout>

      %scaler = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<32x32xf32, #layout2>}> : () -> tensor<32x32xf32, #layout2>
      %1 = "ttir.generic"(%arg0, %scaler, %0) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#reduction, #parallel], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^compute(%cb0: memref<4x3x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<1x1x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<1x3x!tt.tile<32x32, f32>, #l1_>):
        linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["reduction", "parallel"]} ins(%cb0, %cb1 : memref<4x3x!tt.tile<32x32, f32>, #l1_>, memref<1x1x!tt.tile<32x32, f32>, #l1_>) outs(%cb2 : memref<1x3x!tt.tile<32x32, f32>, #l1_>) {
        ^bb0(%in: !tt.tile<32x32, f32>, %in_0: !tt.tile<32x32, f32>, %out: !tt.tile<32x32, f32>):
          %3 = "ttir.tile_reduce_sum"(%in, %in_0) <{reduce_dim = #ttir<reduce_dim R>}> : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
          linalg.yield %3 : !tt.tile<32x32, f32>
        }
      }) : (tensor<128x96xf32, #layout>, tensor<32x32xf32, #layout2>, tensor<1x96xf32, #layout1>) -> tensor<1x96xf32, #layout1>

      %2 = "ttir.generic"(%arg0, %arg1, %arg2) <{grid = #tt.grid<1x1>, indexing_maps = [#map3, #map4, #map5], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
      ^compute(%cb0: memref<4x3x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<3x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<4x2x!tt.tile<32x32, f32>, #l1_>):
        "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<4x3x!tt.tile<32x32, f32>, #l1_>, memref<3x2x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>) -> ()
      }) : (tensor<128x96xf32, #layout>, tensor<96x64xf32, #layout3>, tensor<128x64xf32, #layout4>) -> tensor<128x64xf32, #layout4>
    ```
  }];
  let constructor = "createTTIRToTTIRGenericPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::linalg::LinalgDialect"];
}

def ConvertTTIRToTTNN: Pass<"convert-ttir-to-ttnn", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to TTNN dialect.";
  let constructor = "createConvertTTIRToTTNNPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::ttnn::TTNNDialect"];
}

def ConvertTTIRToTTMetal: Pass<"convert-ttir-to-ttmetal", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to TTMetal dialect.";
  let constructor = "createConvertTTIRToTTMetalPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::tt::ttmetal::TTMetalDialect", "mlir::tt::ttkernel::TTKernelDialect"];
}

def ConvertTTNNToEmitC : Pass<"convert-ttnn-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert TTNN dialect to EmitC dialect.";
  let constructor = "createConvertTTNNToEmitCPass()";
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::tt::ttnn::TTNNDialect"];
}

def ConvertTTKernelToEmitC : Pass<"convert-ttkernel-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert TTKernel dialect to EmitC dialect.";
  let dependentDialects = ["mlir::emitc::EmitCDialect", "mlir::func::FuncDialect",
                           "mlir::tt::ttkernel::TTKernelDialect"];
}

def ConvertTTIRToLinalg: Pass<"convert-ttir-to-linalg", "::mlir::ModuleOp"> {
  let summary = "Convert TTIR dialect to Linalg dialect.";
  let description = [{
    Conversion pass to convert TTIR ops with defined conversion pattern into linalg ops, with broadcast and collapse tensor ops as needed.
    Example:
    Input:
      func.func @add_with_broadcast(
        %arg0: tensor<32x32xf32>,
        %arg1: tensor<32x1xf32>,
        %arg2: tensor<32x32xf32>
      ) -> tensor<32x32xf32> {
        %1 = "ttir.add"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
      }
    Output:
      func.func @add_with_broadcast(
        %arg0: tensor<32x32xf32>,
        %arg1: tensor<32x1xf32>,
        %arg2: tensor<32x32xf32>
      ) -> tensor<32x32xf32> {
        %collapsed = tensor.collapse_shape %arg1 [[0, 1]] : tensor<32x1xf32> into tensor<32xf32>
        %0 = ttir.empty() : tensor<32x32xf32>
        %broadcasted = linalg.broadcast ins(%collapsed : tensor<32xf32>) outs(%0 : tensor<32x32xf32>) dimensions = [1]
        %1 = linalg.add ins(%arg0, %broadcasted : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
        return %1 : tensor<32x32xf32>
    }
  }];
  let constructor = "createConvertTTIRToLinalgPass()";
  let dependentDialects = ["mlir::tt::ttir::TTIRDialect", "mlir::linalg::LinalgDialect"];
}


#endif // TTMLIR_CONVERSION_PASSES
