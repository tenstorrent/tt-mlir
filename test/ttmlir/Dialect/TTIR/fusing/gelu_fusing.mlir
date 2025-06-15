// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

module {
  func.func @main(%arg0: tensor<2x2xbf16> {tt.argument_type = #tt.argument_type<input>}) -> tensor<2x2xbf16> {
    // CHECK: %[[RESULT:.*]] = "ttir.gelu"(%arg0, %0) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    // CHECK: return %[[RESULT]]

    %0 = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xbf16>
    %1 = "ttir.full"() <{fill_value = 2.000000e+00 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xbf16>
    %2 = "ttir.full"() <{fill_value = 5.000000e-01 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xbf16>
    %3 = ttir.empty() : tensor<2x2xbf16>
    %4 = "ttir.multiply"(%arg0, %2, %3) : (tensor<2x2xbf16>, tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    %5 = ttir.empty() : tensor<2x2xbf16>
    %6 = "ttir.rsqrt"(%1, %5) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    %7 = ttir.empty() : tensor<2x2xbf16>
    %8 = "ttir.multiply"(%arg0, %6, %7) : (tensor<2x2xbf16>, tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    %9 = ttir.empty() : tensor<2x2xf32>
    %10 = "ttir.typecast"(%8, %9) : (tensor<2x2xbf16>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %11 = ttir.empty() : tensor<2x2xf32>
    %12 = "ttir.erf"(%10, %11) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %13 = ttir.empty() : tensor<2x2xbf16>
    %14 = "ttir.typecast"(%12, %13) : (tensor<2x2xf32>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    %15 = ttir.empty() : tensor<2x2xbf16>
    %16 = "ttir.add"(%14, %0, %15) : (tensor<2x2xbf16>, tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    %17 = ttir.empty() : tensor<2x2xbf16>
    %18 = "ttir.multiply"(%16, %4, %17) : (tensor<2x2xbf16>, tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
    return %18 : tensor<2x2xbf16>
  }
}
