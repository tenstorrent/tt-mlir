// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

module {
    func.func @main(%arg0: tensor<2x2xf32> {tt.argument_type = #tt.argument_type<input>}) -> tensor<2x2xf32> {
        // CHECK: %[[RESULT:.*]] = "ttir.erf"(%arg0, %0) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        // CHECK: return %[[RESULT]]

        %0 = "ttir.full"() <{fill_value = -2.72614237E-10 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %1 = "ttir.full"() <{fill_value = 2.77068146E-8 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %2 = "ttir.full"() <{fill_value = -2.10102394E-6 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %3 = "ttir.full"() <{fill_value = -5.69250624E-5 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %4 = "ttir.full"() <{fill_value = -7.34990637E-4 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %5 = "ttir.full"() <{fill_value = -2.954600e-03 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %6 = "ttir.full"() <{fill_value = -0.0160960332 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %7 = "ttir.full"() <{fill_value = -1.45660715E-5 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %8 = "ttir.full"() <{fill_value = -2.13374049E-4 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %9 = "ttir.full"() <{fill_value = -0.00168282702 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %10 = "ttir.full"() <{fill_value = -0.00737332925 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %11 = "ttir.full"() <{fill_value = -0.0142647391 : f32, shape = array<i32: 2, 2>}> : () -> tensor<2x2xf32>
        %12 = ttir.empty() : tensor<2x2xf32>
        %13 = "ttir.clamp_scalar"(%arg0, %12) <{max = 4.000000e+00 : f32, min = -4.000000e+00 : f32}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %14 = ttir.empty() : tensor<2x2xf32>
        %15 = "ttir.multiply"(%13, %13, %14) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %16 = ttir.empty() : tensor<2x2xf32>
        %17 = "ttir.multiply"(%0, %15, %16) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %18 = ttir.empty() : tensor<2x2xf32>
        %19 = "ttir.add"(%17, %1, %18) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %20 = ttir.empty() : tensor<2x2xf32>
        %21 = "ttir.multiply"(%19, %15, %20) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %22 = ttir.empty() : tensor<2x2xf32>
        %23 = "ttir.add"(%21, %2, %22) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %24 = ttir.empty() : tensor<2x2xf32>
        %25 = "ttir.multiply"(%23, %15, %24) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %26 = ttir.empty() : tensor<2x2xf32>
        %27 = "ttir.add"(%25, %3, %26) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %28 = ttir.empty() : tensor<2x2xf32>
        %29 = "ttir.multiply"(%27, %15, %28) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %30 = ttir.empty() : tensor<2x2xf32>
        %31 = "ttir.add"(%29, %4, %30) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %32 = ttir.empty() : tensor<2x2xf32>
        %33 = "ttir.multiply"(%31, %15, %32) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %34 = ttir.empty() : tensor<2x2xf32>
        %35 = "ttir.add"(%33, %5, %34) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %36 = ttir.empty() : tensor<2x2xf32>
        %37 = "ttir.multiply"(%35, %15, %36) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %38 = ttir.empty() : tensor<2x2xf32>
        %39 = "ttir.add"(%37, %6, %38) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %40 = ttir.empty() : tensor<2x2xf32>
        %41 = "ttir.multiply"(%7, %15, %40) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %42 = ttir.empty() : tensor<2x2xf32>
        %43 = "ttir.add"(%41, %8, %42) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %44 = ttir.empty() : tensor<2x2xf32>
        %45 = "ttir.multiply"(%43, %15, %44) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %46 = ttir.empty() : tensor<2x2xf32>
        %47 = "ttir.add"(%45, %9, %46) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %48 = ttir.empty() : tensor<2x2xf32>
        %49 = "ttir.multiply"(%47, %15, %48) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %50 = ttir.empty() : tensor<2x2xf32>
        %51 = "ttir.add"(%49, %10, %50) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %52 = ttir.empty() : tensor<2x2xf32>
        %53 = "ttir.multiply"(%51, %15, %52) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %54 = ttir.empty() : tensor<2x2xf32>
        %55 = "ttir.add"(%53, %11, %54) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %56 = ttir.empty() : tensor<2x2xf32>
        %57 = "ttir.multiply"(%13, %39, %56) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %58 = ttir.empty() : tensor<2x2xf32>
        %59 = "ttir.div"(%57, %55, %58) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        %60 = ttir.empty() : tensor<2x2xf32>
        %61 = "ttir.clamp_scalar"(%59, %60) <{max = 1.000000e+00 : f32, min = -1.000000e+00 : f32}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        return %61 : tensor<2x2xf32>
    }
}
