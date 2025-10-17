// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=false enable-const-eval=true row-major-enabled=true" -o %t %s -mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// CHECK: func.func @conv2d_input_rm
// CHECK: = "ttnn.get_device"()
// CHECK: = "ttnn.prepare_conv2d_weights"(%{{.*}}, %{{.*}})

func.func @conv2d_input_rm(%arg0: tensor<16x32x32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                           %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                           %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<16x30x30x64xbf16> {
    // CHECK: = "ttnn.get_device"()
    %0 = ttir.empty() : tensor<16x30x30x64xbf16>
    // CHECK: = "ttnn.conv2d"(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}})
    // CHECK-NEXT: memref<16384x64xbf16,
    // CHECK-NEXT: memref<18x2x!ttcore.tile
    %1 = "ttir.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
                stride = 1: i32,
                padding = 0: i32,
                dilation = 1: i32,
                groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<16x30x30x64xbf16>) -> tensor<16x30x30x64xbf16>
    return %1 : tensor<16x30x30x64xbf16>
}
