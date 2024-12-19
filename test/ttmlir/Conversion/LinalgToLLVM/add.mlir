// RUN: ttmlir-opt --linalg-to-llvm-pipeline %s | FileCheck %s
module {
  func.func @add(
    %arg0: tensor<32x32xf32>,  // First input tensor
    %arg1: tensor<32x32xf32>,  // Second input tensor
    %arg2: tensor<32x32xf32>   // Output tensor (result stored here)
  ) -> tensor<32x32xf32> {
    // Perform linalg.add and store the result in %arg2
    %1 = linalg.add ins(%arg0, %arg1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%arg2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
  // CHECK: llvm.func @add(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !llvm.ptr, %arg8: !llvm.ptr, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64, %arg13: i64, %arg14: !llvm.ptr, %arg15: !llvm.ptr, %arg16: i64, %arg17: i64, %arg18: i64, %arg19: i64, %arg20: i64) -> !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
}
