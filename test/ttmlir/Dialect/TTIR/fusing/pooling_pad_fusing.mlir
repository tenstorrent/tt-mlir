// RUN: ttmlir-opt -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @max_pool2d_pad_fusion(%arg0: tensor<1x112x112x64xbf16>) -> tensor<1x56x56x64xbf16>{
    // CHECK-NOT: "ttir.pad"
    // CHECK: "ttir.max_pool2d"
    // CHECK-SAME: padding = array<i32: 1, 1, 1, 1>
    %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 1, 1, 0, 0, 1, 1, 0>, value = 0xFF800000 : f32}> : (tensor<1x112x112x64xbf16>) -> tensor<1x114x114x64xbf16>
    %3 = "ttir.max_pool2d"(%1) <{kernel = array<i32: 3, 3>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x114x114x64xbf16>) -> tensor<1x56x56x64xbf16>
    return %3: tensor<1x56x56x64xbf16>
  }
}

module {
  func.func @avg_pool2d_pad_fusion(%arg0: tensor<1x112x112x64xbf16>) -> tensor<1x56x56x64xbf16>{
    // CHECK-NOT: "ttir.pad"
    // CHECK: "ttir.avg_pool2d"
    // CHECK-SAME: padding = array<i32: 1, 1, 1, 1>
    %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 1, 1, 0, 0, 1, 1, 0>, value = 0.0 : f32}> : (tensor<1x112x112x64xbf16>) -> tensor<1x114x114x64xbf16>
    %3 = "ttir.avg_pool2d"(%1) <{kernel = array<i32: 3, 3>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0, 0, 0>, ceil_mode = false}> : (tensor<1x114x114x64xbf16>) -> tensor<1x56x56x64xbf16>
    return %3: tensor<1x56x56x64xbf16>
  }
}
