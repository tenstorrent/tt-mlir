// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module attributes {} {
  func.func @max_pool2d_with_indices(%arg0: tensor<1x128x128x32xbf16>)
      -> (tensor<1x63x63x32xbf16>, tensor<1x63x63x32xi32>) {
    %0 = ttir.empty() : tensor<1x63x63x32xbf16>
    %1 = ttir.empty() : tensor<1x63x63x32xi32>
    %2, %3 = "ttir.max_pool2d_with_indices"(%arg0, %0, %1)
      <{kernel = array<i32: 3, 3>,
        stride = array<i32: 2, 2>,
        dilation = array<i32: 1, 1>,
        padding = array<i32: 0, 0, 0, 0>,
        ceil_mode = false}>
      : (tensor<1x128x128x32xbf16>,
         tensor<1x63x63x32xbf16>,
         tensor<1x63x63x32xi32>)
        -> (tensor<1x63x63x32xbf16>, tensor<1x63x63x32xi32>)
    return %2, %3 : tensor<1x63x63x32xbf16>, tensor<1x63x63x32xi32>
  }
}
