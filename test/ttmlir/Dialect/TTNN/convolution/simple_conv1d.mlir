// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>
module {
  func.func @main(%arg0: tensor<1x256x512xf32>, %arg1: tensor<1024x256x1xf32>, %arg2: tensor<1024xf32>) -> tensor<1x1024x512xf32> {
    %0 = tensor.empty() : tensor<1x1024x512xf32>
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.reshape"(%{{.*}}) <{shape = [1 : i32, 256 : i32, 512 : i32, 1 : i32]}> : (tensor<[[TENSOR_SHAPE0:[0-9]+x[0-9]+x[0-9]+xf32]], #{{.*}}) -> tensor<[[TENSOR_SHAPE1:[0-9]+x[0-9]+x[0-9]+x1xf32]], #{{.*}}>
    // CHECK: [[VAL1:%[0-9]+]] = "ttnn.reshape"(%{{.*}}) <{shape = [1024 : i32, 256 : i32, 1 : i32, 1 : i32]}> : (tensor<[[TENSOR_SHAPE2:[0-9]+x[0-9]+x[0-9]+xf32]], #{{.*}}>) -> tensor<[[TENSOR_SHAPE3:[0-9]+x[0-9]+x[0-9]+x1xf32]], #{{.*}}>
    // CHECK: [[VAL2:%[0-9]+]] = "ttnn.transpose"([[VAL0]]) <{dim0 = 1 : si32, dim1 = 2 : si32}> : (tensor<[[TENSOR_SHAPE1]], #{{.*}}>) -> tensor<[[TENSOR_SHAPE4:[0-9]+x[0-9]+x[0-9]+x1xf32]], #{{.*}}>
    // CHECK: [[VAL3:%[0-9]+]] = "ttnn.transpose"([[VAL2]]) <{dim0 = 2 : si32, dim1 = 3 : si32}> : (tensor<[[TENSOR_SHAPE4]], #{{.*}}>) -> tensor<[[TENSOR_SHAPE5:[0-9]+x[0-9]+x[0-9]+x[0-9]+xf32]], #{{.*}}>
    // CHECK: [[VAL4:%[0-9]+]] = "ttnn.reshape"([[VAL3]]) <{shape = [1 : i32, 1 : i32, 512 : i32, 256 : i32]}> : (tensor<[[TENSOR_SHAPE5]], #{{.*}}>) -> tensor<[[TENSOR_SHAPE6:[0-9]+x[0-9]+x[0-9]+x[0-9]+xf32]], #{{.*}}>
    // CHECK: [[VAL5:%[0-9]+]] = "ttnn.conv2d"([[VAL4]], %10, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK: (tensor<[[TENSOR_SHAPE6]], #{{.*}}>, tensor<1024x256x1x1xf32,  #{{.*}}>, tensor<1x1x512x1024xf32, #{{.*}}>, !tt.device<#device>) -> tensor<1x1x512x1024xf32,  #{{.*}}>
    %1 = "ttir.convolution"(%arg0, %arg1, %0) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2>, feature_group_count = 1 : i64, input_dilation = array<i64: 1>, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile], padding = array<i64: 0, 0>, weight_dilation = array<i64: 1>, window_reversal = array<i1: false>, window_strides = array<i64: 1>}> : (tensor<1x256x512xf32>, tensor<1024x256x1xf32>, tensor<1x1024x512xf32>) -> tensor<1x1024x512xf32>
    // CHECK: return %{{.*}} : tensor<1x1024x512xf32, #ttnn_layout3>
    return %1 : tensor<1x1024x512xf32>
  }
}
