// RUN: ttmlir-opt -split-input-file -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module @PadMaxPool2dFusingTest {
  func.func @main(%arg0: tensor<1x112x112x64xbf16>) -> tensor<1x58x58x64xbf16>{
    // CHECK-NOT: "ttir.pad"
    // CHECK: padding = array<i32: 3, 3, 3, 3>,
    %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 1, 1, 1, 1, 0, 0>, value = 0x00000000 : f32}> : (tensor<1x112x112x64xbf16>) -> tensor<1x114x114x64xbf16>
    %2 = "ttir.max_pool2d"(%1) <{
        kernel = array<i32: 3, 3>,
        stride = array<i32: 2, 2>,
        dilation = array<i32: 1, 1>,
        padding = array<i32: 2, 2, 2, 2>,
        ceil_mode = false
    }> : (tensor<1x114x114x64xbf16>) -> tensor<1x58x58x64xbf16>
    return %2: tensor<1x58x58x64xbf16>
  }
}

// -----

module @PadAvgPool2dFusingTest {
  func.func @main(%arg0: tensor<1x112x112x64xbf16>) -> tensor<1x58x58x64xbf16>{
    // CHECK-NOT: "ttir.pad"
    // CHECK: padding = array<i32: 3, 3, 3, 3>,
    %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 1, 1, 1, 1, 0, 0>, value = 0x00000000 : f32}> : (tensor<1x112x112x64xbf16>) -> tensor<1x114x114x64xbf16>
    %2 = "ttir.avg_pool2d"(%1) <{
        kernel = array<i32: 3, 3>,
        stride = array<i32: 2, 2>,
        dilation = array<i32: 1, 1>,
        padding = array<i32: 2, 2, 2, 2>,
        ceil_mode = false
    }> : (tensor<1x114x114x64xbf16>) -> tensor<1x58x58x64xbf16>
    return %2: tensor<1x58x58x64xbf16>
  }
}
