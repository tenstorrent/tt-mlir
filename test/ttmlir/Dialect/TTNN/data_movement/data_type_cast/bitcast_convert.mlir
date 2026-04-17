// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func public @bitcast_convert_ui32_to_f32(%arg0: tensor<32x32xui32>) -> tensor<32x32xf32> {
    // CHECK: "ttnn.bitcast_convert"
    %0 = "ttir.bitcast_convert"(%arg0) : (tensor<32x32xui32>) -> tensor<32x32xf32>
    // CHECK: return {{.*}} : tensor<32x32xf32
    return %0 : tensor<32x32xf32>
  }
  func.func public @bitcast_convert_f32_to_ui32(%arg0: tensor<32x32xf32>) -> tensor<32x32xui32> {
    // CHECK: "ttnn.bitcast_convert"
    %0 = "ttir.bitcast_convert"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xui32>
    // CHECK: return {{.*}} : tensor<32x32xui32
    return %0 : tensor<32x32xui32>
  }
  func.func public @bitcast_convert_bf16_to_ui16(%arg0: tensor<32x32xbf16>) -> tensor<32x32xui16> {
    // CHECK: "ttnn.bitcast_convert"
    %0 = "ttir.bitcast_convert"(%arg0) : (tensor<32x32xbf16>) -> tensor<32x32xui16>
    // CHECK: return {{.*}} : tensor<32x32xui16
    return %0 : tensor<32x32xui16>
  }
}
