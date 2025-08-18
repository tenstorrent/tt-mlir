// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-fusing-pass=true enable-fusing-conv2d-with-multiply-pattern=true" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// This is common pattern throught Resnet. We have conv2d with constant weight, followed by multiply with constant input. This will be commuted through conv2d.
// Then we fuse add into conv2d with bias and lastly we fuse conv2d and relu into conv2d with activation.

// CHECK-LABEL: func.func @main
// CHECK-SAME: %arg0: tensor<64x64x3x3xf32, #ttnn.ttnn_layout<{{.*}}, memref<12288x3xf32, #ttnn.buffer_type<system_memory
module {
  func.func @main(%weight: tensor<64x64x3x3xf32> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg5: tensor<1x1x1x64xf32> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg6: tensor<1x1x1x64xf32>, %input: tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32> {
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: activation = "relu"
    %0 = ttir.empty() : tensor<1x56x56x64xf32>
    %1 = "ttir.conv2d"(%input, %weight, %0) <{dilation = array<i32: 1, 1>, groups = 1 : i32, padding = array<i32: 1, 1, 1, 1>, stride = array<i32: 1, 1>}> {channel_last = 1 : si32} : (tensor<1x56x56x64xf32>, tensor<64x64x3x3xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %2 = ttir.empty() : tensor<1x56x56x64xf32>
    // CHECK-NOT: "ttnn.multiply"
    // CHECK-NOT: "ttnn.add"
    // CHECK-NOT: "ttnn.relu"
    %3 = "ttir.multiply"(%1, %arg5, %2) : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %4 = ttir.empty() : tensor<1x56x56x64xf32>
    %5 = "ttir.add"(%3, %arg6, %4) : (tensor<1x56x56x64xf32>, tensor<1x1x1x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    %6 = ttir.empty() : tensor<1x56x56x64xf32>
    %7 = "ttir.relu"(%5, %6) : (tensor<1x56x56x64xf32>, tensor<1x56x56x64xf32>) -> tensor<1x56x56x64xf32>
    return %7 : tensor<1x56x56x64xf32>
  }
}
