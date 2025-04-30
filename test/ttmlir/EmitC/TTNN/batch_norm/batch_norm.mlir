// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

module @jit_batch_norm {
  func.func public @test_batch_norm(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<2x2x2x2xf32> {
    %0 = ttir.empty() : tensor<2x2x2x2xf32>
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.reshape"(%arg3)
    // CHECK: [[VAL1:%[0-9]+]] = "ttnn.reshape"(%arg4)
    // CHECK: [[VAL2:%[0-9]+]] = "ttnn.reshape"(%arg1)
    // CHECK: [[VAL3:%[0-9]+]] = "ttnn.reshape"(%arg2)
    %1 = "ttir.batch_norm"(%arg0, %arg1, %arg2, %arg3, %arg4, %0) <{dimension = 1 : i32, epsilon = 0.000000e+00 : f32, training = false}> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
    // CHECK: [[VAL4:%[0-9]+]] = "ttnn.batch_norm"(%arg0, [[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]]) <{epsilon = 0.000000e+00 : f32, training = false}> : (tensor<2x2x2x2xf32, #ttnn_layout>, tensor<1x2x1x1xf32, #ttnn_layout2>, tensor<1x2x1x1xf32, #ttnn_layout2>, tensor<1x2x1x1xf32, #ttnn_layout2>, tensor<1x2x1x1xf32, #ttnn_layout2>) -> tensor<2x2x2x2xf32, #ttnn_layout>
    return %1 : tensor<2x2x2x2xf32>
    // CHECK: return [[VAL4]]
  }
}
