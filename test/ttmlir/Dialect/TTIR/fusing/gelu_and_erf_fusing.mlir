// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

module {
    func.func @main(%arg0: tensor<2x2xbf16> {tt.argument_type = #tt.argument_type<input>}) -> tensor<2x2xbf16> {
        // CHECK: %[[RESULT:.*]] = "ttir.gelu"(%arg0, %0) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
        // CHECK: return %[[RESULT]]

        %0 = "ttir.full"() <{fill_value = 1.000000e+00 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xbf16>
        %1 = "ttir.full"() <{fill_value = 2.000000e+00 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xbf16>
        %2 = "ttir.full"() <{fill_value = 5.000000e-01 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xbf16>
        %3 = "ttir.full"() <{fill_value = -2.72614237E-10 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %4 = "ttir.full"() <{fill_value = 2.77068146E-8 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %5 = "ttir.full"() <{fill_value = -2.10102394E-6 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %6 = "ttir.full"() <{fill_value = -5.69250624E-5 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %7 = "ttir.full"() <{fill_value = -7.34990637E-4 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %8 = "ttir.full"() <{fill_value = -2.954600e-03 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %9 = "ttir.full"() <{fill_value = -0.0160960332 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %10 = "ttir.full"() <{fill_value = -1.45660715E-5 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %11 = "ttir.full"() <{fill_value = -2.13374049E-4 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %12 = "ttir.full"() <{fill_value = -0.00168282702 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %13 = "ttir.full"() <{fill_value = -0.00737332925 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %14 = "ttir.full"() <{fill_value = -0.0142647391 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %15 = ttir.empty() : tensor<2x2xbf16>
        %16 = "ttir.multiply"(%arg0, %2, %15) : (tensor<2x2xbf16>, tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
        %17 = ttir.empty() : tensor<2x2xbf16>
        %18 = "ttir.rsqrt"(%1, %17) : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
        %19 = ttir.empty() : tensor<2x2xbf16>
        %20 = "ttir.multiply"(%arg0, %18, %19) : (tensor<2x2xbf16>, tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
        %21 = ttir.empty() : tensor<2x2xf32>
        %22 = "ttir.typecast"(%20, %21) : (tensor<2x2xbf16>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %23 = ttir.empty() : tensor<2x2xf32>
        %24 = "ttir.clamp_scalar"(%22, %23) <{max = 4.000000e+00 : f32, min = -4.000000e+00 : f32}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %25 = ttir.empty() : tensor<2x2xf32>
        %26 = "ttir.multiply"(%24, %24, %25) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %27 = ttir.empty() : tensor<2x2xf32>
        %28 = "ttir.multiply"(%3, %26, %27) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %29 = ttir.empty() : tensor<2x2xf32>
        %30 = "ttir.add"(%28, %4, %29) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %31 = ttir.empty() : tensor<2x2xf32>
        %32 = "ttir.multiply"(%30, %26, %31) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %33 = ttir.empty() : tensor<2x2xf32>
        %34 = "ttir.add"(%32, %5, %33) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %35 = ttir.empty() : tensor<2x2xf32>
        %36 = "ttir.multiply"(%34, %26, %35) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %37 = ttir.empty() : tensor<2x2xf32>
        %38 = "ttir.add"(%36, %6, %37) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %39 = ttir.empty() : tensor<2x2xf32>
        %40 = "ttir.multiply"(%38, %26, %39) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %41 = ttir.empty() : tensor<2x2xf32>
        %42 = "ttir.add"(%40, %7, %41) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %43 = ttir.empty() : tensor<2x2xf32>
        %44 = "ttir.multiply"(%42, %26, %43) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %45 = ttir.empty() : tensor<2x2xf32>
        %46 = "ttir.add"(%44, %8, %45) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %47 = ttir.empty() : tensor<2x2xf32>
        %48 = "ttir.multiply"(%46, %26, %47) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %49 = ttir.empty() : tensor<2x2xf32>
        %50 = "ttir.add"(%48, %9, %49) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %51 = ttir.empty() : tensor<2x2xf32>
        %52 = "ttir.multiply"(%10, %26, %51) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %53 = ttir.empty() : tensor<2x2xf32>
        %54 = "ttir.add"(%52, %11, %53) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %55 = ttir.empty() : tensor<2x2xf32>
        %56 = "ttir.multiply"(%54, %26, %55) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %57 = ttir.empty() : tensor<2x2xf32>
        %58 = "ttir.add"(%56, %12, %57) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %59 = ttir.empty() : tensor<2x2xf32>
        %60 = "ttir.multiply"(%58, %26, %59) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %61 = ttir.empty() : tensor<2x2xf32>
        %62 = "ttir.add"(%60, %13, %61) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %63 = ttir.empty() : tensor<2x2xf32>
        %64 = "ttir.multiply"(%62, %26, %63) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %65 = ttir.empty() : tensor<2x2xf32>
        %66 = "ttir.add"(%64, %14, %65) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %67 = ttir.empty() : tensor<2x2xf32>
        %68 = "ttir.multiply"(%24, %50, %67) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %69 = ttir.empty() : tensor<2x2xf32>
        %70 = "ttir.div"(%68, %66, %69) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %71 = ttir.empty() : tensor<2x2xf32>
        %72 = "ttir.clamp_scalar"(%70, %71) <{max = 1.000000e+00 : f32, min = -1.000000e+00 : f32}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %73 = ttir.empty() : tensor<2x2xbf16>
        %74 = "ttir.typecast"(%72, %73) : (tensor<2x2xf32>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
        %75 = ttir.empty() : tensor<2x2xbf16>
        %76 = "ttir.add"(%74, %0, %75) : (tensor<2x2xbf16>, tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
        %77 = ttir.empty() : tensor<2x2xbf16>
        %78 = "ttir.multiply"(%76, %16, %77) : (tensor<2x2xbf16>, tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xbf16>
        return %78 : tensor<2x2xbf16>
    }
}
