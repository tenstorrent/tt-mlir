// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_batch_norm {
  func.func public @test_batch_norm(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<2x2x2x2xf32> {
    // CHECK: [[ARG3_RM:%[0-9]+]] = "ttnn.to_layout"(%arg3)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: [[VAL0_RM:%[0-9]+]] = "ttnn.reshape"([[ARG3_RM]])
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.to_layout"([[VAL0_RM]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK: [[ARG4_RM:%[0-9]+]] = "ttnn.to_layout"(%arg4)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: [[VAL1_RM:%[0-9]+]] = "ttnn.reshape"([[ARG4_RM]])
    // CHECK: [[VAL1:%[0-9]+]] = "ttnn.to_layout"([[VAL1_RM]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK: [[ARG1_RM:%[0-9]+]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: [[VAL2_RM:%[0-9]+]] = "ttnn.reshape"([[ARG1_RM]])
    // CHECK: [[VAL2:%[0-9]+]] = "ttnn.to_layout"([[VAL2_RM]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK: [[ARG2_RM:%[0-9]+]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: [[VAL3_RM:%[0-9]+]] = "ttnn.reshape"([[ARG2_RM]])
    // CHECK: [[VAL3:%[0-9]+]] = "ttnn.to_layout"([[VAL3_RM]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    %1 = "ttir.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) <{dimension = 1 : i32, epsilon = 0.000000e+00 : f32}> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<2x2x2x2xf32>
    // CHECK: [[VAL4:%[0-9]+]] = "ttnn.batch_norm_inference"(%arg0, [[VAL0]], [[VAL1]], [[VAL2]], [[VAL3]]) <{{{.*}}epsilon = 0.000000e+00 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>}>
    return %1 : tensor<2x2x2x2xf32>
    // CHECK: return [[VAL4]]
  }
}
