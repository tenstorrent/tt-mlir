// RUN: ttmlir-opt --ttir-fusing="ttnn-enable-conv2d-with-multiply-pattern=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @commute_batch_norm
func.func @commute_batch_norm(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg3: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg4: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg5: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: "ttir.conv2d"
    // CHECK-NOT: "ttir.batch_norm_inference"
    // CHECK-NOT: "ttir.add"
    // CHECK-NOT: "ttir.multiply"
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
                    <{
                        stride = 1: i32,
                        padding = 0: i32,
                        dilation = 1: i32,
                        groups = 1: i32
                    }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %3 = "ttir.batch_norm_inference"(%1, %arg2, %arg3, %arg4, %arg5, %2) <{dimension = 3 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>

    return %3: tensor<1x30x30x64xbf16>
}
}
