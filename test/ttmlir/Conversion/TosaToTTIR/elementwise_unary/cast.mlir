// RUN: ttmlir-opt --convert-tosa-to-ttir %s | FileCheck %s
module attributes {} {
  func.func @test_cast(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xbf16> {
    %0 = tosa.cast %arg0 : (tensor<13x21x3xf32>) -> tensor<13x21x3xbf16>
    // CHECK: %[[CAST_OUT:[0-9]+]] = tensor.empty() : tensor<13x21x3xbf16>
    // CHECK: %{{[0-9]+}} = "ttir.typecast"(%arg{{[0-9]+}}, %[[CAST_OUT]]){{.+}} : (tensor<13x21x3xf32>, tensor<13x21x3xbf16>) -> tensor<13x21x3xbf16>
    return %0 : tensor<13x21x3xbf16>
  }
}
