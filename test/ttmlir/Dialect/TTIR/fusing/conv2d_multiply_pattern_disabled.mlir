// RUN: ttmlir-opt --ttir-fusing="ttnn-enable-conv2d-with-multiply-pattern=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @commute_multiply(%arg0: tensor<1x32x32x64xbf16>, %arg1: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %arg2: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<1x30x30x64xbf16> {
    // CHECK: %[[CONV:.*]] = "ttir.conv2d"
    // CHECK: %[[WEIGHT_SCALED:.*]] = "ttir.multiply"
    // CHECK-SAME: (%[[CONV]], %arg2
    %0 = ttir.empty() : tensor<1x30x30x64xbf16>
    %1 = "ttir.conv2d"(%arg0, %arg1, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<1x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    %2 = ttir.empty() : tensor<1x30x30x64xbf16>
    %3 = "ttir.multiply"(%1, %arg2, %2) : (tensor<1x30x30x64xbf16>, tensor<1x1x1x64xbf16>, tensor<1x30x30x64xbf16>) -> tensor<1x30x30x64xbf16>
    return %3: tensor<1x30x30x64xbf16>
  }
}
