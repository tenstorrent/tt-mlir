
// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @test_invoke_external(%arg0: tensor<128x4x32x256xbf16>, %arg1: tensor<i32>, %arg2: tensor<i1>) -> tensor<128x4x32x256xbf16> {
    // NOTE: the only way to pass scalars is as 0D tensors.
    // CHECK: "ttir.invoke_external"(%arg0, %arg1, %arg2)
    // CHECK-SAME: entry = "my_func"
    // CHECK-SAME: path = "/path/to/external.mlir"
    %0 = stablehlo.custom_call @tt.invoke_external(%arg0, %arg1, %arg2)
      {mhlo.frontend_attributes = {path = "/path/to/external.mlir", entry = "my_func"}}
      : (tensor<128x4x32x256xbf16>, tensor<i32>, tensor<i1>) -> tensor<128x4x32x256xbf16>
    return %0 : tensor<128x4x32x256xbf16>
  }
}
