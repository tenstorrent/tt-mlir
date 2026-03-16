// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_batch_norm_training {
  func.func public @test_batch_norm_training(%arg0: tensor<2x2x2x2xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) {
    // CHECK: [[ARG1_RM:%[0-9]+]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: [[VAL0_RM:%[0-9]+]] = "ttnn.reshape"([[ARG1_RM]])
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.to_layout"([[VAL0_RM]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK: [[ARG2_RM:%[0-9]+]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: [[VAL1_RM:%[0-9]+]] = "ttnn.reshape"([[ARG2_RM]])
    // CHECK: [[VAL1:%[0-9]+]] = "ttnn.to_layout"([[VAL1_RM]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK: [[ARG3_RM:%[0-9]+]] = "ttnn.to_layout"(%arg3)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: [[VAL2_RM:%[0-9]+]] = "ttnn.reshape"([[ARG3_RM]])
    // CHECK: [[VAL2:%[0-9]+]] = "ttnn.to_layout"([[VAL2_RM]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK: [[ARG4_RM:%[0-9]+]] = "ttnn.to_layout"(%arg4)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: [[VAL3_RM:%[0-9]+]] = "ttnn.reshape"([[ARG4_RM]])
    // CHECK: [[VAL3:%[0-9]+]] = "ttnn.to_layout"([[VAL3_RM]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    %3:3 = "ttir.batch_norm_training"(%arg0, %arg1, %arg2, %arg3, %arg4) <{dimension = 1 : i32, epsilon = 1.000000e-05 : f32, momentum = 1.000000e-01 : f32}> : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
    // CHECK: [[RESULT:%[0-9]+]] = "ttnn.batch_norm_training"(%arg0, [[VAL2]], [[VAL3]], [[VAL0]], [[VAL1]]) <{{{.*}}epsilon = {{.*}}, momentum = {{.*}}, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>}>
    return %3#0, %3#1, %3#2 : tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>
    // CHECK: return [[RESULT]], %arg3, %arg4
  }
}
