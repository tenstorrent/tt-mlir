// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_module_reshape attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @test_reshape(%arg0: tensor<1x64x64x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x1x4096x64xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    // CHECK-LABEL: func.func public @test_reshape
    // CHECK: ttnn.reshape
    // CHECK-SAME: {shape = [1 : i32, 1 : i32, 4096 : i32, 64 : i32]}
    // CHECK-SAME: tensor<1x64x64x64xf32
    // CHECK-SAME: -> tensor<1x1x4096x64xf32
    %0 = stablehlo.reshape %arg0 : (tensor<1x64x64x64xf32>) -> tensor<1x1x4096x64xf32>
    return %0 : tensor<1x1x4096x64xf32>
  }

  func.func public @test_reshape_i64(%arg0: tensor<1x1x1xi64>) -> tensor<1x1xi64> {
    // CHECK-LABEL: func.func public @test_reshape_i64
    // CHECK: ttnn.reshape
    // CHECK-SAME: {shape = [1 : i32, 1 : i32]}
    // CHECK-SAME: tensor<1x1x1xsi32,
    // CHECK-SAME: -> tensor<1x1xsi32,
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x1xi64>) -> tensor<1x1xi64>
    return %0 : tensor<1x1xi64>
  }

  func.func public @test_reshape_i1(%arg0: tensor<1x1x2x7xi1>) -> tensor<1x1x7x2xi1> {
    // CHECK-LABEL: func.func public @test_reshape_i1
    // CHECK: ttnn.reshape
    // CHECK-SAME: {shape = [1 : i32, 1 : i32, 7 : i32, 2 : i32]}
    // CHECK-SAME: tensor<1x1x2x7xbf16,
    // CHECK-SAME: -> tensor<1x1x7x2xbf16,
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x2x7xi1>) -> tensor<1x1x7x2xi1>
    return %0 : tensor<1x1x7x2xi1>
  }

  func.func public @test_reshape_ui8(%arg0: tensor<1x1x2x7xui8>) -> tensor<1x1x7x2xui8> {
    // CHECK-LABEL: func.func public @test_reshape_ui8
    // CHECK: ttnn.reshape
    // CHECK-SAME: {shape = [1 : i32, 1 : i32, 7 : i32, 2 : i32]}
    // CHECK-SAME: tensor<1x1x2x7xsi32,
    // CHECK-SAME: -> tensor<1x1x7x2xsi32,
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x2x7xui8>) -> tensor<1x1x7x2xui8>
    return %0 : tensor<1x1x7x2xui8>
  }
}
