// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Pipeline test for SyncTensorsGraph.779.


module @SyncTensorsGraph.779 {
  ttcore.device_module {
    builtin.module {
      func.func @main(%arg0: tensor<1x6x4x4xf32>, %arg1: tensor<1x4x4xf32>, %arg2: tensor<1x6x4x4xf32>, %arg3: tensor<2064896xf32>, %arg4: tensor<2064896x5xf32>, %arg5: tensor<3x2064896xf32>, %arg6: tensor<2xi64>, %arg7: tensor<232355x1xi64>, %arg8: tensor<232355x1xi64>, %arg9: tensor<285388x1xi64>, %arg10: tensor<285388x1xi64>, %arg11: tensor<212612x1xi64>, %arg12: tensor<212612x1xi64>, %arg13: tensor<236236x1xi64>, %arg14: tensor<236236x1xi64>, %arg15: tensor<203984x1xi64>, %arg16: tensor<203984x1xi64>, %arg17: tensor<190629x1xi64>, %arg18: tensor<190629x1xi64>) -> (tensor<1x6x1x256x704xf32>, tensor<6x4x4xf32>, tensor<4x4xf32>, tensor<6x4x4xf32>, tensor<2064896x5xf32>, tensor<6x3x2064896xf32>, tensor<6x2064896xf32>, tensor<18x2064896xf32>, tensor<6x2064896x2xf32>, tensor<6x2064896xi1>, tensor<232355x2xi64>, tensor<232355xf32>, tensor<1x6x1x256x704xf32>, tensor<1x3x4xf32>, tensor<1x3x3xf32>, tensor<1x3x4xf32>, tensor<1x3xf32>) {
        %0 = "ttir.constant"() <{value = dense<704> : tensor<232355xi64>}> : () -> tensor<232355xi64>
        %1 = "ttir.constant"() <{value = dense<256> : tensor<232355xi64>}> : () -> tensor<232355xi64>
        %2 = "ttir.constant"() <{value = dense<704> : tensor<285388xi64>}> : () -> tensor<285388xi64>
        %3 = "ttir.constant"() <{value = dense<256> : tensor<285388xi64>}> : () -> tensor<285388xi64>
        %4 = "ttir.constant"() <{value = dense<2064896> : tensor<285388xi64>}> : () -> tensor<285388xi64>
        %5 = "ttir.constant"() <{value = dense<0> : tensor<285388xi64>}> : () -> tensor<285388xi64>
        %6 = "ttir.constant"() <{value = dense<704> : tensor<212612xi64>}> : () -> tensor<212612xi64>
        %7 = "ttir.constant"() <{value = dense<256> : tensor<212612xi64>}> : () -> tensor<212612xi64>
        %8 = "ttir.constant"() <{value = dense<2064896> : tensor<212612xi64>}> : () -> tensor<212612xi64>
        %9 = "ttir.constant"() <{value = dense<0> : tensor<212612xi64>}> : () -> tensor<212612xi64>
        %10 = "ttir.constant"() <{value = dense<704> : tensor<236236xi64>}> : () -> tensor<236236xi64>
        %11 = "ttir.constant"() <{value = dense<256> : tensor<236236xi64>}> : () -> tensor<236236xi64>
        %12 = "ttir.constant"() <{value = dense<2064896> : tensor<236236xi64>}> : () -> tensor<236236xi64>
        %13 = "ttir.constant"() <{value = dense<0> : tensor<236236xi64>}> : () -> tensor<236236xi64>
        %14 = "ttir.constant"() <{value = dense<704> : tensor<203984xi64>}> : () -> tensor<203984xi64>
        %15 = "ttir.constant"() <{value = dense<256> : tensor<203984xi64>}> : () -> tensor<203984xi64>
        %16 = "ttir.constant"() <{value = dense<2064896> : tensor<203984xi64>}> : () -> tensor<203984xi64>
        %17 = "ttir.constant"() <{value = dense<0> : tensor<203984xi64>}> : () -> tensor<203984xi64>
        %18 = "ttir.constant"() <{value = dense<704> : tensor<190629xi64>}> : () -> tensor<190629xi64>
        %19 = "ttir.constant"() <{value = dense<256> : tensor<190629xi64>}> : () -> tensor<190629xi64>
        %20 = "ttir.constant"() <{value = dense<2064896> : tensor<190629xi64>}> : () -> tensor<190629xi64>
        %21 = "ttir.constant"() <{value = dense<0> : tensor<190629xi64>}> : () -> tensor<190629xi64>
        %22 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<6x1x256x704xf32>}> : () -> tensor<6x1x256x704xf32>
        %23 = "ttir.constant"() <{value = dense<2064896> : tensor<232355xi64>}> : () -> tensor<232355xi64>
        %24 = "ttir.constant"() <{value = dense<0> : tensor<232355xi64>}> : () -> tensor<232355xi64>
        %25 = "ttir.constant"() <{value = dense<7.040000e+02> : tensor<6x2064896xf32>}> : () -> tensor<6x2064896xf32>
        %26 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<6x2064896xf32>}> : () -> tensor<6x2064896xf32>
        %27 = "ttir.constant"() <{value = dense<2.560000e+02> : tensor<6x2064896xf32>}> : () -> tensor<6x2064896xf32>
        %28 = "ttir.constant"() <{value = dense<2> : tensor<2xi64>}> : () -> tensor<2xi64>
        %29 = "ttir.constant"() <{value = dense<0> : tensor<2xi64>}> : () -> tensor<2xi64>
        %30 = "ttir.constant"() <{value = dense<1.000000e+05> : tensor<6x2064896xf32>}> : () -> tensor<6x2064896xf32>
        %31 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<6x2064896xf32>}> : () -> tensor<6x2064896xf32>
        %32 = "ttir.constant"() <{value = dense<true> : tensor<6x2x2064896xi1>}> : () -> tensor<6x2x2064896xi1>
        %33 = "ttir.constant"() <{value = dense<true> : tensor<2064896x3xi1>}> : () -> tensor<2064896x3xi1>
        %34 = "ttir.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
        %35 = "ttir.constant"() <{value = dense<4> : tensor<i64>}> : () -> tensor<i64>
        %36 = "ttir.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
        %37 = "ttir.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
        %38 = "ttir.constant"() <{value = dense<2> : tensor<i64>}> : () -> tensor<i64>
        %39 = "ttir.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
        %40 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<1x6x1x256x704xf32>}> : () -> tensor<1x6x1x256x704xf32>
        %41 = "ttir.reshape"(%arg0) <{shape = [6 : i32, 4 : i32, 4 : i32]}> : (tensor<1x6x4x4xf32>) -> tensor<6x4x4xf32>
        %42 = "ttir.reshape"(%arg1) <{shape = [4 : i32, 4 : i32]}> : (tensor<1x4x4xf32>) -> tensor<4x4xf32>
        %43 = "ttir.reshape"(%arg2) <{shape = [6 : i32, 4 : i32, 4 : i32]}> : (tensor<1x6x4x4xf32>) -> tensor<6x4x4xf32>
        %44 = "ttir.pad"(%33) <{padding = array<i32: 0, 0, 0, 2>, value = 0.000000e+00 : f32}> : (tensor<2064896x3xi1>) -> tensor<2064896x5xi1>
        %45 = "ttir.reshape"(%arg3) <{shape = [2064896 : i32, 1 : i32]}> : (tensor<2064896xf32>) -> tensor<2064896x1xf32>
        %46 = ttir.empty() : tensor<i32>
        %47 = ttir.to_layout %39, %46 : tensor<i64> into tensor<i32> -> tensor<i32>
        %48 = ttir.empty() : tensor<i32>
        %49 = ttir.to_layout %38, %48 : tensor<i64> into tensor<i32> -> tensor<i32>
        %50 = call @cpu_hoisted_stablehlo_dynamic_update_slice_b1555218(%arg4, %45, %47, %49) {ttir.cpu_hoisted_call} : (tensor<2064896x5xf32>, tensor<2064896x1xf32>, tensor<i32>, tensor<i32>) -> tensor<2064896x5xf32>
        %51 = "ttir.slice_static"(%50) <{begins = [0 : i32, 0 : i32], ends = [2064896 : i32, 3 : i32], step = [1 : i32, 1 : i32]}> : (tensor<2064896x5xf32>) -> tensor<2064896x3xf32>
        %52 = "ttir.slice_static"(%42) <{begins = [0 : i32, 0 : i32], ends = [3 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<4x4xf32>) -> tensor<3x4xf32>
        %53 = "ttir.slice_static"(%52) <{begins = [0 : i32, 3 : i32], ends = [3 : i32, 4 : i32], step = [1 : i32, 1 : i32]}> : (tensor<3x4xf32>) -> tensor<3x1xf32>
        %54 = "ttir.reshape"(%53) <{shape = [3 : i32]}> : (tensor<3x1xf32>) -> tensor<3xf32>
        %55 = "ttir.reshape"(%54) <{shape = [1 : i32, 3 : i32]}> : (tensor<3xf32>) -> tensor<1x3xf32>
        %56 = "ttir.broadcast"(%55) <{broadcast_dimensions = array<i64: 2064896, 1>}> : (tensor<1x3xf32>) -> tensor<2064896x3xf32>
        %57 = "ttir.subtract"(%51, %56) : (tensor<2064896x3xf32>, tensor<2064896x3xf32>) -> tensor<2064896x3xf32>
        %58 = "ttir.pad"(%57) <{padding = array<i32: 0, 0, 0, 2>, value = 0.000000e+00 : f32}> : (tensor<2064896x3xf32>) -> tensor<2064896x5xf32>
        %59 = "ttir.where"(%44, %58, %50) : (tensor<2064896x5xi1>, tensor<2064896x5xf32>, tensor<2064896x5xf32>) -> tensor<2064896x5xf32>
        %60 = "ttir.slice_static"(%43) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 3 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x4x4xf32>) -> tensor<6x3x4xf32>
        %61 = "ttir.slice_static"(%60) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 3 : i32, 3 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x3x4xf32>) -> tensor<6x3x3xf32>
        %62 = "ttir.reshape"(%61) <{shape = [18 : i32, 3 : i32]}> : (tensor<6x3x3xf32>) -> tensor<18x3xf32>
        %63 = "ttir.dot_general"(%62, %arg5) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<18x3xf32>, tensor<3x2064896xf32>) -> tensor<18x2064896xf32>
        %64 = "ttir.reshape"(%63) <{shape = [6 : i32, 3 : i32, 2064896 : i32]}> : (tensor<18x2064896xf32>) -> tensor<6x3x2064896xf32>
        %65 = "ttir.slice_static"(%60) <{begins = [0 : i32, 0 : i32, 3 : i32], ends = [6 : i32, 3 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x3x4xf32>) -> tensor<6x3x1xf32>
        %66 = "ttir.reshape"(%65) <{shape = [6 : i32, 3 : i32]}> : (tensor<6x3x1xf32>) -> tensor<6x3xf32>
        %67 = "ttir.reshape"(%66) <{shape = [6 : i32, 3 : i32, 1 : i32]}> : (tensor<6x3xf32>) -> tensor<6x3x1xf32>
        %68 = "ttir.broadcast"(%67) <{broadcast_dimensions = array<i64: 1, 1, 2064896>}> : (tensor<6x3x1xf32>) -> tensor<6x3x2064896xf32>
        %69 = "ttir.add"(%64, %68) : (tensor<6x3x2064896xf32>, tensor<6x3x2064896xf32>) -> tensor<6x3x2064896xf32>
        %70 = "ttir.slice_static"(%69) <{begins = [0 : i32, 2 : i32, 0 : i32], ends = [6 : i32, 3 : i32, 2064896 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x3x2064896xf32>) -> tensor<6x1x2064896xf32>
        %71 = "ttir.reshape"(%70) <{shape = [6 : i32, 2064896 : i32]}> : (tensor<6x1x2064896xf32>) -> tensor<6x2064896xf32>
        %72 = "ttir.pad"(%32) <{padding = array<i32: 0, 0, 0, 1, 0, 0>, value = 0.000000e+00 : f32}> : (tensor<6x2x2064896xi1>) -> tensor<6x3x2064896xi1>
        %73 = "ttir.clamp_tensor"(%71, %31, %30) : (tensor<6x2064896xf32>, tensor<6x2064896xf32>, tensor<6x2064896xf32>) -> tensor<6x2064896xf32>
        %74 = "ttir.reshape"(%73) <{shape = [6 : i32, 1 : i32, 2064896 : i32]}> : (tensor<6x2064896xf32>) -> tensor<6x1x2064896xf32>
        %75 = ttir.empty() : tensor<i32>
        %76 = ttir.to_layout %39, %75 : tensor<i64> into tensor<i32> -> tensor<i32>
        %77 = ttir.empty() : tensor<i32>
        %78 = ttir.to_layout %38, %77 : tensor<i64> into tensor<i32> -> tensor<i32>
        %79 = ttir.empty() : tensor<i32>
        %80 = ttir.to_layout %39, %79 : tensor<i64> into tensor<i32> -> tensor<i32>
        %81 = call @cpu_hoisted_stablehlo_dynamic_update_slice_a8ec3ec4(%69, %74, %76, %78, %80) {ttir.cpu_hoisted_call} : (tensor<6x3x2064896xf32>, tensor<6x1x2064896xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3x2064896xf32>
        %82 = "ttir.slice_static"(%81) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 2 : i32, 2064896 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x3x2064896xf32>) -> tensor<6x2x2064896xf32>
        %83 = "ttir.slice_static"(%81) <{begins = [0 : i32, 2 : i32, 0 : i32], ends = [6 : i32, 3 : i32, 2064896 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x3x2064896xf32>) -> tensor<6x1x2064896xf32>
        %84 = "ttir.reshape"(%83) <{shape = [6 : i32, 2064896 : i32]}> : (tensor<6x1x2064896xf32>) -> tensor<6x2064896xf32>
        %85 = "ttir.reshape"(%84) <{shape = [6 : i32, 1 : i32, 2064896 : i32]}> : (tensor<6x2064896xf32>) -> tensor<6x1x2064896xf32>
        %86 = "ttir.broadcast"(%85) <{broadcast_dimensions = array<i64: 1, 2, 1>}> : (tensor<6x1x2064896xf32>) -> tensor<6x2x2064896xf32>
        %87 = "ttir.div"(%82, %86) : (tensor<6x2x2064896xf32>, tensor<6x2x2064896xf32>) -> tensor<6x2x2064896xf32>
        %88 = "ttir.pad"(%87) <{padding = array<i32: 0, 0, 0, 1, 0, 0>, value = 0.000000e+00 : f32}> : (tensor<6x2x2064896xf32>) -> tensor<6x3x2064896xf32>
        %89 = "ttir.where"(%72, %88, %81) : (tensor<6x3x2064896xi1>, tensor<6x3x2064896xf32>, tensor<6x3x2064896xf32>) -> tensor<6x3x2064896xf32>
        %90 = "ttir.slice_static"(%89) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 2 : i32, 2064896 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x3x2064896xf32>) -> tensor<6x2x2064896xf32>
        %91 = "ttir.pad"(%90) <{padding = array<i32: 0, 0, 0, 1, 0, 0>, value = 0.000000e+00 : f32}> : (tensor<6x2x2064896xf32>) -> tensor<6x3x2064896xf32>
        %92 = "ttir.where"(%72, %91, %89) : (tensor<6x3x2064896xi1>, tensor<6x3x2064896xf32>, tensor<6x3x2064896xf32>) -> tensor<6x3x2064896xf32>
        %93 = "ttir.reshape"(%92) <{shape = [18 : i32, 2064896 : i32]}> : (tensor<6x3x2064896xf32>) -> tensor<18x2064896xf32>
        %94 = "ttir.slice_static"(%41) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 3 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x4x4xf32>) -> tensor<6x3x4xf32>
        %95 = "ttir.slice_static"(%94) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 3 : i32, 3 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x3x4xf32>) -> tensor<6x3x3xf32>
        %96 = "ttir.dot_general"(%95, %92) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<6x3x3xf32>, tensor<6x3x2064896xf32>) -> tensor<6x3x2064896xf32>
        %97 = "ttir.slice_static"(%94) <{begins = [0 : i32, 0 : i32, 3 : i32], ends = [6 : i32, 3 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x3x4xf32>) -> tensor<6x3x1xf32>
        %98 = "ttir.reshape"(%97) <{shape = [6 : i32, 3 : i32]}> : (tensor<6x3x1xf32>) -> tensor<6x3xf32>
        %99 = "ttir.reshape"(%98) <{shape = [6 : i32, 3 : i32, 1 : i32]}> : (tensor<6x3xf32>) -> tensor<6x3x1xf32>
        %100 = "ttir.broadcast"(%99) <{broadcast_dimensions = array<i64: 1, 1, 2064896>}> : (tensor<6x3x1xf32>) -> tensor<6x3x2064896xf32>
        %101 = "ttir.add"(%96, %100) : (tensor<6x3x2064896xf32>, tensor<6x3x2064896xf32>) -> tensor<6x3x2064896xf32>
        %102 = "ttir.slice_static"(%101) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 2 : i32, 2064896 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x3x2064896xf32>) -> tensor<6x2x2064896xf32>
        %103 = "ttir.permute"(%102) <{permutation = array<i64: 0, 2, 1>}> : (tensor<6x2x2064896xf32>) -> tensor<6x2064896x2xf32>
        %104 = "ttir.lt"(%arg6, %29) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
        %105 = "ttir.add"(%arg6, %28) : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
        %106 = "ttir.where"(%104, %105, %arg6) : (tensor<2xi1>, tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
        %107 = "ttir.reshape"(%106) <{shape = [2 : i32, 1 : i32]}> : (tensor<2xi64>) -> tensor<2x1xi64>
        %108 = "ttir.permute"(%103) <{permutation = array<i64: 2, 0, 1>}> : (tensor<6x2064896x2xf32>) -> tensor<2x6x2064896xf32>
        %109 = "ttir.reshape"(%108) <{shape = [2 : i32, 12389376 : i32]}> : (tensor<2x6x2064896xf32>) -> tensor<2x12389376xf32>
        %110 = "ttir.embedding"(%107, %109) : (tensor<2x1xi64>, tensor<2x12389376xf32>) -> tensor<2x1x12389376xf32>
        %111 = "ttir.reshape"(%110) <{shape = [2 : i32, 6 : i32, 2064896 : i32]}> : (tensor<2x1x12389376xf32>) -> tensor<2x6x2064896xf32>
        %112 = "ttir.permute"(%111) <{permutation = array<i64: 1, 2, 0>}> : (tensor<2x6x2064896xf32>) -> tensor<6x2064896x2xf32>
        %113 = "ttir.slice_static"(%112) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 2064896 : i32, 1 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x2064896x2xf32>) -> tensor<6x2064896x1xf32>
        %114 = "ttir.reshape"(%113) <{shape = [6 : i32, 2064896 : i32]}> : (tensor<6x2064896x1xf32>) -> tensor<6x2064896xf32>
        %115 = "ttir.lt"(%114, %27) : (tensor<6x2064896xf32>, tensor<6x2064896xf32>) -> tensor<6x2064896xi1>
        %116 = "ttir.typecast"(%115) <{conservative_folding = false}> : (tensor<6x2064896xi1>) -> tensor<6x2064896xui8>
        %117 = "ttir.ge"(%114, %26) : (tensor<6x2064896xf32>, tensor<6x2064896xf32>) -> tensor<6x2064896xi1>
        %118 = "ttir.typecast"(%117) <{conservative_folding = false}> : (tensor<6x2064896xi1>) -> tensor<6x2064896xui8>
        %119 = "ttir.logical_and"(%115, %117) : (tensor<6x2064896xi1>, tensor<6x2064896xi1>) -> tensor<6x2064896xi1>
        %120 = "ttir.typecast"(%119) <{conservative_folding = false}> : (tensor<6x2064896xi1>) -> tensor<6x2064896xui8>
        %121 = "ttir.slice_static"(%112) <{begins = [0 : i32, 0 : i32, 1 : i32], ends = [6 : i32, 2064896 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x2064896x2xf32>) -> tensor<6x2064896x1xf32>
        %122 = "ttir.reshape"(%121) <{shape = [6 : i32, 2064896 : i32]}> : (tensor<6x2064896x1xf32>) -> tensor<6x2064896xf32>
        %123 = "ttir.lt"(%122, %25) : (tensor<6x2064896xf32>, tensor<6x2064896xf32>) -> tensor<6x2064896xi1>
        %124 = "ttir.typecast"(%123) <{conservative_folding = false}> : (tensor<6x2064896xi1>) -> tensor<6x2064896xui8>
        %125 = "ttir.logical_and"(%119, %123) : (tensor<6x2064896xi1>, tensor<6x2064896xi1>) -> tensor<6x2064896xi1>
        %126 = "ttir.typecast"(%125) <{conservative_folding = false}> : (tensor<6x2064896xi1>) -> tensor<6x2064896xui8>
        %127 = "ttir.ge"(%122, %26) : (tensor<6x2064896xf32>, tensor<6x2064896xf32>) -> tensor<6x2064896xi1>
        %128 = "ttir.typecast"(%127) <{conservative_folding = false}> : (tensor<6x2064896xi1>) -> tensor<6x2064896xui8>
        %129 = "ttir.logical_and"(%125, %127) : (tensor<6x2064896xi1>, tensor<6x2064896xi1>) -> tensor<6x2064896xi1>
        %130 = "ttir.slice_static"(%112) <{begins = [5 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 2064896 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x2064896x2xf32>) -> tensor<1x2064896x2xf32>
        %131 = "ttir.reshape"(%130) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<1x2064896x2xf32>) -> tensor<2064896x2xf32>
        %132 = "ttir.reshape"(%arg7) <{shape = [232355 : i32]}> : (tensor<232355x1xi64>) -> tensor<232355xi64>
        %133 = "ttir.lt"(%132, %24) : (tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi1>
        %134 = "ttir.add"(%132, %23) : (tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi64>
        %135 = "ttir.where"(%133, %134, %132) : (tensor<232355xi1>, tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi64>
        %136 = "ttir.reshape"(%135) <{shape = [232355 : i32, 1 : i32]}> : (tensor<232355xi64>) -> tensor<232355x1xi64>
        %137 = "ttir.permute"(%131) <{permutation = array<i64: 0, 1>}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %138 = "ttir.reshape"(%137) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %139 = "ttir.embedding"(%136, %138) : (tensor<232355x1xi64>, tensor<2064896x2xf32>) -> tensor<232355x1x2xf32>
        %140 = "ttir.reshape"(%139) <{shape = [232355 : i32, 2 : i32]}> : (tensor<232355x1x2xf32>) -> tensor<232355x2xf32>
        %141 = "ttir.permute"(%140) <{permutation = array<i64: 0, 1>}> : (tensor<232355x2xf32>) -> tensor<232355x2xf32>
        %142 = "ttir.typecast"(%141) <{conservative_folding = false}> : (tensor<232355x2xf32>) -> tensor<232355x2xi64>
        %143 = "ttir.slice_static"(%92) <{begins = [0 : i32, 2 : i32, 0 : i32], ends = [6 : i32, 3 : i32, 2064896 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x3x2064896xf32>) -> tensor<6x1x2064896xf32>
        %144 = "ttir.reshape"(%143) <{shape = [6 : i32, 2064896 : i32]}> : (tensor<6x1x2064896xf32>) -> tensor<6x2064896xf32>
        %145 = "ttir.slice_static"(%144) <{begins = [5 : i32, 0 : i32], ends = [6 : i32, 2064896 : i32], step = [1 : i32, 1 : i32]}> : (tensor<6x2064896xf32>) -> tensor<1x2064896xf32>
        %146 = "ttir.reshape"(%145) <{shape = [2064896 : i32]}> : (tensor<1x2064896xf32>) -> tensor<2064896xf32>
        %147 = "ttir.reshape"(%arg8) <{shape = [232355 : i32]}> : (tensor<232355x1xi64>) -> tensor<232355xi64>
        %148 = "ttir.lt"(%147, %24) : (tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi1>
        %149 = "ttir.add"(%147, %23) : (tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi64>
        %150 = "ttir.where"(%148, %149, %147) : (tensor<232355xi1>, tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi64>
        %151 = "ttir.reshape"(%150) <{shape = [232355 : i32, 1 : i32]}> : (tensor<232355xi64>) -> tensor<232355x1xi64>
        %152 = "ttir.permute"(%146) <{permutation = array<i64: 0>}> : (tensor<2064896xf32>) -> tensor<2064896xf32>
        %153 = "ttir.reshape"(%152) <{shape = [2064896 : i32, 1 : i32]}> : (tensor<2064896xf32>) -> tensor<2064896x1xf32>
        %154 = "ttir.embedding"(%151, %153) : (tensor<232355x1xi64>, tensor<2064896x1xf32>) -> tensor<232355x1x1xf32>
        %155 = "ttir.reshape"(%154) <{shape = [232355 : i32]}> : (tensor<232355x1x1xf32>) -> tensor<232355xf32>
        %156 = "ttir.permute"(%155) <{permutation = array<i64: 0>}> : (tensor<232355xf32>) -> tensor<232355xf32>
        %157 = "ttir.slice_static"(%22) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 256 : i32, 704 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x1x256x704xf32>) -> tensor<1x1x256x704xf32>
        %158 = "ttir.reshape"(%157) <{shape = [256 : i32, 704 : i32]}> : (tensor<1x1x256x704xf32>) -> tensor<256x704xf32>
        %159 = "ttir.slice_static"(%112) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 2064896 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x2064896x2xf32>) -> tensor<1x2064896x2xf32>
        %160 = "ttir.reshape"(%159) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<1x2064896x2xf32>) -> tensor<2064896x2xf32>
        %161 = "ttir.reshape"(%arg18) <{shape = [190629 : i32]}> : (tensor<190629x1xi64>) -> tensor<190629xi64>
        %162 = "ttir.lt"(%161, %21) : (tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi1>
        %163 = "ttir.add"(%161, %20) : (tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi64>
        %164 = "ttir.where"(%162, %163, %161) : (tensor<190629xi1>, tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi64>
        %165 = "ttir.reshape"(%164) <{shape = [190629 : i32, 1 : i32]}> : (tensor<190629xi64>) -> tensor<190629x1xi64>
        %166 = "ttir.permute"(%160) <{permutation = array<i64: 0, 1>}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %167 = "ttir.reshape"(%166) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %168 = "ttir.embedding"(%165, %167) : (tensor<190629x1xi64>, tensor<2064896x2xf32>) -> tensor<190629x1x2xf32>
        %169 = "ttir.reshape"(%168) <{shape = [190629 : i32, 2 : i32]}> : (tensor<190629x1x2xf32>) -> tensor<190629x2xf32>
        %170 = "ttir.permute"(%169) <{permutation = array<i64: 0, 1>}> : (tensor<190629x2xf32>) -> tensor<190629x2xf32>
        %171 = "ttir.typecast"(%170) <{conservative_folding = false}> : (tensor<190629x2xf32>) -> tensor<190629x2xi64>
        %172 = "ttir.slice_static"(%171) <{begins = [0 : i32, 0 : i32], ends = [190629 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<190629x2xi64>) -> tensor<190629x1xi64>
        %173 = "ttir.reshape"(%172) <{shape = [190629 : i32]}> : (tensor<190629x1xi64>) -> tensor<190629xi64>
        %174 = "ttir.lt"(%173, %21) : (tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi1>
        %175 = "ttir.add"(%173, %19) : (tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi64>
        %176 = "ttir.where"(%174, %175, %173) : (tensor<190629xi1>, tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi64>
        %177 = "ttir.reshape"(%176) <{shape = [190629 : i32, 1 : i32]}> : (tensor<190629xi64>) -> tensor<190629x1xi64>
        %178 = "ttir.slice_static"(%171) <{begins = [0 : i32, 1 : i32], ends = [190629 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<190629x2xi64>) -> tensor<190629x1xi64>
        %179 = "ttir.reshape"(%178) <{shape = [190629 : i32]}> : (tensor<190629x1xi64>) -> tensor<190629xi64>
        %180 = "ttir.lt"(%179, %21) : (tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi1>
        %181 = "ttir.add"(%179, %18) : (tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi64>
        %182 = "ttir.where"(%180, %181, %179) : (tensor<190629xi1>, tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi64>
        %183 = "ttir.reshape"(%182) <{shape = [190629 : i32, 1 : i32]}> : (tensor<190629xi64>) -> tensor<190629x1xi64>
        %184 = "ttir.concat"(%177, %183) <{dim = 1 : si32}> : (tensor<190629x1xi64>, tensor<190629x1xi64>) -> tensor<190629x2xi64>
        %185 = "ttir.slice_static"(%144) <{begins = [0 : i32, 0 : i32], ends = [1 : i32, 2064896 : i32], step = [1 : i32, 1 : i32]}> : (tensor<6x2064896xf32>) -> tensor<1x2064896xf32>
        %186 = "ttir.reshape"(%185) <{shape = [2064896 : i32]}> : (tensor<1x2064896xf32>) -> tensor<2064896xf32>
        %187 = "ttir.reshape"(%arg17) <{shape = [190629 : i32]}> : (tensor<190629x1xi64>) -> tensor<190629xi64>
        %188 = "ttir.lt"(%187, %21) : (tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi1>
        %189 = "ttir.add"(%187, %20) : (tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi64>
        %190 = "ttir.where"(%188, %189, %187) : (tensor<190629xi1>, tensor<190629xi64>, tensor<190629xi64>) -> tensor<190629xi64>
        %191 = "ttir.reshape"(%190) <{shape = [190629 : i32, 1 : i32]}> : (tensor<190629xi64>) -> tensor<190629x1xi64>
        %192 = "ttir.permute"(%186) <{permutation = array<i64: 0>}> : (tensor<2064896xf32>) -> tensor<2064896xf32>
        %193 = "ttir.reshape"(%192) <{shape = [2064896 : i32, 1 : i32]}> : (tensor<2064896xf32>) -> tensor<2064896x1xf32>
        %194 = "ttir.embedding"(%191, %193) : (tensor<190629x1xi64>, tensor<2064896x1xf32>) -> tensor<190629x1x1xf32>
        %195 = "ttir.reshape"(%194) <{shape = [190629 : i32]}> : (tensor<190629x1x1xf32>) -> tensor<190629xf32>
        %196 = "ttir.permute"(%195) <{permutation = array<i64: 0>}> : (tensor<190629xf32>) -> tensor<190629xf32>
        %197 = "ttir.reshape"(%184) <{shape = [190629 : i32, 2 : i32]}> : (tensor<190629x2xi64>) -> tensor<190629x2xi64>
        %198 = "ttir.reshape"(%197) <{shape = [190629 : i32, 1 : i32, 2 : i32]}> : (tensor<190629x2xi64>) -> tensor<190629x1x2xi64>
        %199 = "ttir.repeat"(%198) <{repeat_dimensions = array<i64: 1, 1, 1>}> : (tensor<190629x1x2xi64>) -> tensor<190629x1x2xi64>
        %200 = "ttir.reshape"(%199) <{shape = [190629 : i32, 2 : i32]}> : (tensor<190629x1x2xi64>) -> tensor<190629x2xi64>
        %201 = "ttir.slice_static"(%200) <{begins = [0 : i32, 0 : i32], ends = [190629 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<190629x2xi64>) -> tensor<190629x1xi64>
        %202 = "ttir.slice_static"(%200) <{begins = [0 : i32, 1 : i32], ends = [190629 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<190629x2xi64>) -> tensor<190629x1xi64>
        %203 = "ttir.full"() <{fill_value = 704 : i32, shape = array<i32: 190629, 1>}> : () -> tensor<190629x1xi64>
        %204 = "ttir.multiply"(%201, %203) : (tensor<190629x1xi64>, tensor<190629x1xi64>) -> tensor<190629x1xi64>
        %205 = "ttir.add"(%204, %202) : (tensor<190629x1xi64>, tensor<190629x1xi64>) -> tensor<190629x1xi64>
        %206 = "ttir.reshape"(%205) <{shape = [190629 : i32]}> : (tensor<190629x1xi64>) -> tensor<190629xi64>
        %207 = "ttir.reshape"(%158) <{shape = [180224 : i32]}> : (tensor<256x704xf32>) -> tensor<180224xf32>
        %208 = "ttir.reshape"(%196) <{shape = [190629 : i32]}> : (tensor<190629xf32>) -> tensor<190629xf32>
        %209 = "ttir.scatter"(%207, %206, %208) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<180224xf32>, tensor<190629xi64>, tensor<190629xf32>) -> tensor<180224xf32>
        %210 = "ttir.reshape"(%209) <{shape = [256 : i32, 704 : i32]}> : (tensor<180224xf32>) -> tensor<256x704xf32>
        %211 = "ttir.reshape"(%210) <{shape = [1 : i32, 1 : i32, 256 : i32, 704 : i32]}> : (tensor<256x704xf32>) -> tensor<1x1x256x704xf32>
        %212 = ttir.empty() : tensor<i32>
        %213 = ttir.to_layout %39, %212 : tensor<i64> into tensor<i32> -> tensor<i32>
        %214 = ttir.empty() : tensor<i32>
        %215 = ttir.to_layout %39, %214 : tensor<i64> into tensor<i32> -> tensor<i32>
        %216 = ttir.empty() : tensor<i32>
        %217 = ttir.to_layout %39, %216 : tensor<i64> into tensor<i32> -> tensor<i32>
        %218 = ttir.empty() : tensor<i32>
        %219 = ttir.to_layout %39, %218 : tensor<i64> into tensor<i32> -> tensor<i32>
        %220 = call @cpu_hoisted_stablehlo_dynamic_update_slice_1d952611(%22, %211, %213, %215, %217, %219) {ttir.cpu_hoisted_call} : (tensor<6x1x256x704xf32>, tensor<1x1x256x704xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x1x256x704xf32>
        %221 = "ttir.slice_static"(%220) <{begins = [1 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 1 : i32, 256 : i32, 704 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x1x256x704xf32>) -> tensor<1x1x256x704xf32>
        %222 = "ttir.reshape"(%221) <{shape = [256 : i32, 704 : i32]}> : (tensor<1x1x256x704xf32>) -> tensor<256x704xf32>
        %223 = "ttir.slice_static"(%112) <{begins = [1 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 2064896 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x2064896x2xf32>) -> tensor<1x2064896x2xf32>
        %224 = "ttir.reshape"(%223) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<1x2064896x2xf32>) -> tensor<2064896x2xf32>
        %225 = "ttir.reshape"(%arg16) <{shape = [203984 : i32]}> : (tensor<203984x1xi64>) -> tensor<203984xi64>
        %226 = "ttir.lt"(%225, %17) : (tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi1>
        %227 = "ttir.add"(%225, %16) : (tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi64>
        %228 = "ttir.where"(%226, %227, %225) : (tensor<203984xi1>, tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi64>
        %229 = "ttir.reshape"(%228) <{shape = [203984 : i32, 1 : i32]}> : (tensor<203984xi64>) -> tensor<203984x1xi64>
        %230 = "ttir.permute"(%224) <{permutation = array<i64: 0, 1>}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %231 = "ttir.reshape"(%230) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %232 = "ttir.embedding"(%229, %231) : (tensor<203984x1xi64>, tensor<2064896x2xf32>) -> tensor<203984x1x2xf32>
        %233 = "ttir.reshape"(%232) <{shape = [203984 : i32, 2 : i32]}> : (tensor<203984x1x2xf32>) -> tensor<203984x2xf32>
        %234 = "ttir.permute"(%233) <{permutation = array<i64: 0, 1>}> : (tensor<203984x2xf32>) -> tensor<203984x2xf32>
        %235 = "ttir.typecast"(%234) <{conservative_folding = false}> : (tensor<203984x2xf32>) -> tensor<203984x2xi64>
        %236 = "ttir.slice_static"(%235) <{begins = [0 : i32, 0 : i32], ends = [203984 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<203984x2xi64>) -> tensor<203984x1xi64>
        %237 = "ttir.reshape"(%236) <{shape = [203984 : i32]}> : (tensor<203984x1xi64>) -> tensor<203984xi64>
        %238 = "ttir.lt"(%237, %17) : (tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi1>
        %239 = "ttir.add"(%237, %15) : (tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi64>
        %240 = "ttir.where"(%238, %239, %237) : (tensor<203984xi1>, tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi64>
        %241 = "ttir.reshape"(%240) <{shape = [203984 : i32, 1 : i32]}> : (tensor<203984xi64>) -> tensor<203984x1xi64>
        %242 = "ttir.slice_static"(%235) <{begins = [0 : i32, 1 : i32], ends = [203984 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<203984x2xi64>) -> tensor<203984x1xi64>
        %243 = "ttir.reshape"(%242) <{shape = [203984 : i32]}> : (tensor<203984x1xi64>) -> tensor<203984xi64>
        %244 = "ttir.lt"(%243, %17) : (tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi1>
        %245 = "ttir.add"(%243, %14) : (tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi64>
        %246 = "ttir.where"(%244, %245, %243) : (tensor<203984xi1>, tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi64>
        %247 = "ttir.reshape"(%246) <{shape = [203984 : i32, 1 : i32]}> : (tensor<203984xi64>) -> tensor<203984x1xi64>
        %248 = "ttir.concat"(%241, %247) <{dim = 1 : si32}> : (tensor<203984x1xi64>, tensor<203984x1xi64>) -> tensor<203984x2xi64>
        %249 = "ttir.slice_static"(%144) <{begins = [1 : i32, 0 : i32], ends = [2 : i32, 2064896 : i32], step = [1 : i32, 1 : i32]}> : (tensor<6x2064896xf32>) -> tensor<1x2064896xf32>
        %250 = "ttir.reshape"(%249) <{shape = [2064896 : i32]}> : (tensor<1x2064896xf32>) -> tensor<2064896xf32>
        %251 = "ttir.reshape"(%arg15) <{shape = [203984 : i32]}> : (tensor<203984x1xi64>) -> tensor<203984xi64>
        %252 = "ttir.lt"(%251, %17) : (tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi1>
        %253 = "ttir.add"(%251, %16) : (tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi64>
        %254 = "ttir.where"(%252, %253, %251) : (tensor<203984xi1>, tensor<203984xi64>, tensor<203984xi64>) -> tensor<203984xi64>
        %255 = "ttir.reshape"(%254) <{shape = [203984 : i32, 1 : i32]}> : (tensor<203984xi64>) -> tensor<203984x1xi64>
        %256 = "ttir.permute"(%250) <{permutation = array<i64: 0>}> : (tensor<2064896xf32>) -> tensor<2064896xf32>
        %257 = "ttir.reshape"(%256) <{shape = [2064896 : i32, 1 : i32]}> : (tensor<2064896xf32>) -> tensor<2064896x1xf32>
        %258 = "ttir.embedding"(%255, %257) : (tensor<203984x1xi64>, tensor<2064896x1xf32>) -> tensor<203984x1x1xf32>
        %259 = "ttir.reshape"(%258) <{shape = [203984 : i32]}> : (tensor<203984x1x1xf32>) -> tensor<203984xf32>
        %260 = "ttir.permute"(%259) <{permutation = array<i64: 0>}> : (tensor<203984xf32>) -> tensor<203984xf32>
        %261 = "ttir.reshape"(%248) <{shape = [203984 : i32, 2 : i32]}> : (tensor<203984x2xi64>) -> tensor<203984x2xi64>
        %262 = "ttir.reshape"(%261) <{shape = [203984 : i32, 1 : i32, 2 : i32]}> : (tensor<203984x2xi64>) -> tensor<203984x1x2xi64>
        %263 = "ttir.repeat"(%262) <{repeat_dimensions = array<i64: 1, 1, 1>}> : (tensor<203984x1x2xi64>) -> tensor<203984x1x2xi64>
        %264 = "ttir.reshape"(%263) <{shape = [203984 : i32, 2 : i32]}> : (tensor<203984x1x2xi64>) -> tensor<203984x2xi64>
        %265 = "ttir.slice_static"(%264) <{begins = [0 : i32, 0 : i32], ends = [203984 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<203984x2xi64>) -> tensor<203984x1xi64>
        %266 = "ttir.slice_static"(%264) <{begins = [0 : i32, 1 : i32], ends = [203984 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<203984x2xi64>) -> tensor<203984x1xi64>
        %267 = "ttir.full"() <{fill_value = 704 : i32, shape = array<i32: 203984, 1>}> : () -> tensor<203984x1xi64>
        %268 = "ttir.multiply"(%265, %267) : (tensor<203984x1xi64>, tensor<203984x1xi64>) -> tensor<203984x1xi64>
        %269 = "ttir.add"(%268, %266) : (tensor<203984x1xi64>, tensor<203984x1xi64>) -> tensor<203984x1xi64>
        %270 = "ttir.reshape"(%269) <{shape = [203984 : i32]}> : (tensor<203984x1xi64>) -> tensor<203984xi64>
        %271 = "ttir.reshape"(%222) <{shape = [180224 : i32]}> : (tensor<256x704xf32>) -> tensor<180224xf32>
        %272 = "ttir.reshape"(%260) <{shape = [203984 : i32]}> : (tensor<203984xf32>) -> tensor<203984xf32>
        %273 = "ttir.scatter"(%271, %270, %272) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<180224xf32>, tensor<203984xi64>, tensor<203984xf32>) -> tensor<180224xf32>
        %274 = "ttir.reshape"(%273) <{shape = [256 : i32, 704 : i32]}> : (tensor<180224xf32>) -> tensor<256x704xf32>
        %275 = "ttir.reshape"(%274) <{shape = [1 : i32, 1 : i32, 256 : i32, 704 : i32]}> : (tensor<256x704xf32>) -> tensor<1x1x256x704xf32>
        %276 = ttir.empty() : tensor<i32>
        %277 = ttir.to_layout %37, %276 : tensor<i64> into tensor<i32> -> tensor<i32>
        %278 = ttir.empty() : tensor<i32>
        %279 = ttir.to_layout %39, %278 : tensor<i64> into tensor<i32> -> tensor<i32>
        %280 = ttir.empty() : tensor<i32>
        %281 = ttir.to_layout %39, %280 : tensor<i64> into tensor<i32> -> tensor<i32>
        %282 = ttir.empty() : tensor<i32>
        %283 = ttir.to_layout %39, %282 : tensor<i64> into tensor<i32> -> tensor<i32>
        %284 = call @cpu_hoisted_stablehlo_dynamic_update_slice_1d952611(%220, %275, %277, %279, %281, %283) {ttir.cpu_hoisted_call} : (tensor<6x1x256x704xf32>, tensor<1x1x256x704xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x1x256x704xf32>
        %285 = "ttir.slice_static"(%284) <{begins = [2 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [3 : i32, 1 : i32, 256 : i32, 704 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x1x256x704xf32>) -> tensor<1x1x256x704xf32>
        %286 = "ttir.reshape"(%285) <{shape = [256 : i32, 704 : i32]}> : (tensor<1x1x256x704xf32>) -> tensor<256x704xf32>
        %287 = "ttir.slice_static"(%112) <{begins = [2 : i32, 0 : i32, 0 : i32], ends = [3 : i32, 2064896 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x2064896x2xf32>) -> tensor<1x2064896x2xf32>
        %288 = "ttir.reshape"(%287) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<1x2064896x2xf32>) -> tensor<2064896x2xf32>
        %289 = "ttir.reshape"(%arg14) <{shape = [236236 : i32]}> : (tensor<236236x1xi64>) -> tensor<236236xi64>
        %290 = "ttir.lt"(%289, %13) : (tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi1>
        %291 = "ttir.add"(%289, %12) : (tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi64>
        %292 = "ttir.where"(%290, %291, %289) : (tensor<236236xi1>, tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi64>
        %293 = "ttir.reshape"(%292) <{shape = [236236 : i32, 1 : i32]}> : (tensor<236236xi64>) -> tensor<236236x1xi64>
        %294 = "ttir.permute"(%288) <{permutation = array<i64: 0, 1>}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %295 = "ttir.reshape"(%294) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %296 = "ttir.embedding"(%293, %295) : (tensor<236236x1xi64>, tensor<2064896x2xf32>) -> tensor<236236x1x2xf32>
        %297 = "ttir.reshape"(%296) <{shape = [236236 : i32, 2 : i32]}> : (tensor<236236x1x2xf32>) -> tensor<236236x2xf32>
        %298 = "ttir.permute"(%297) <{permutation = array<i64: 0, 1>}> : (tensor<236236x2xf32>) -> tensor<236236x2xf32>
        %299 = "ttir.typecast"(%298) <{conservative_folding = false}> : (tensor<236236x2xf32>) -> tensor<236236x2xi64>
        %300 = "ttir.slice_static"(%299) <{begins = [0 : i32, 0 : i32], ends = [236236 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<236236x2xi64>) -> tensor<236236x1xi64>
        %301 = "ttir.reshape"(%300) <{shape = [236236 : i32]}> : (tensor<236236x1xi64>) -> tensor<236236xi64>
        %302 = "ttir.lt"(%301, %13) : (tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi1>
        %303 = "ttir.add"(%301, %11) : (tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi64>
        %304 = "ttir.where"(%302, %303, %301) : (tensor<236236xi1>, tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi64>
        %305 = "ttir.reshape"(%304) <{shape = [236236 : i32, 1 : i32]}> : (tensor<236236xi64>) -> tensor<236236x1xi64>
        %306 = "ttir.slice_static"(%299) <{begins = [0 : i32, 1 : i32], ends = [236236 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<236236x2xi64>) -> tensor<236236x1xi64>
        %307 = "ttir.reshape"(%306) <{shape = [236236 : i32]}> : (tensor<236236x1xi64>) -> tensor<236236xi64>
        %308 = "ttir.lt"(%307, %13) : (tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi1>
        %309 = "ttir.add"(%307, %10) : (tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi64>
        %310 = "ttir.where"(%308, %309, %307) : (tensor<236236xi1>, tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi64>
        %311 = "ttir.reshape"(%310) <{shape = [236236 : i32, 1 : i32]}> : (tensor<236236xi64>) -> tensor<236236x1xi64>
        %312 = "ttir.concat"(%305, %311) <{dim = 1 : si32}> : (tensor<236236x1xi64>, tensor<236236x1xi64>) -> tensor<236236x2xi64>
        %313 = "ttir.slice_static"(%144) <{begins = [2 : i32, 0 : i32], ends = [3 : i32, 2064896 : i32], step = [1 : i32, 1 : i32]}> : (tensor<6x2064896xf32>) -> tensor<1x2064896xf32>
        %314 = "ttir.reshape"(%313) <{shape = [2064896 : i32]}> : (tensor<1x2064896xf32>) -> tensor<2064896xf32>
        %315 = "ttir.reshape"(%arg13) <{shape = [236236 : i32]}> : (tensor<236236x1xi64>) -> tensor<236236xi64>
        %316 = "ttir.lt"(%315, %13) : (tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi1>
        %317 = "ttir.add"(%315, %12) : (tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi64>
        %318 = "ttir.where"(%316, %317, %315) : (tensor<236236xi1>, tensor<236236xi64>, tensor<236236xi64>) -> tensor<236236xi64>
        %319 = "ttir.reshape"(%318) <{shape = [236236 : i32, 1 : i32]}> : (tensor<236236xi64>) -> tensor<236236x1xi64>
        %320 = "ttir.permute"(%314) <{permutation = array<i64: 0>}> : (tensor<2064896xf32>) -> tensor<2064896xf32>
        %321 = "ttir.reshape"(%320) <{shape = [2064896 : i32, 1 : i32]}> : (tensor<2064896xf32>) -> tensor<2064896x1xf32>
        %322 = "ttir.embedding"(%319, %321) : (tensor<236236x1xi64>, tensor<2064896x1xf32>) -> tensor<236236x1x1xf32>
        %323 = "ttir.reshape"(%322) <{shape = [236236 : i32]}> : (tensor<236236x1x1xf32>) -> tensor<236236xf32>
        %324 = "ttir.permute"(%323) <{permutation = array<i64: 0>}> : (tensor<236236xf32>) -> tensor<236236xf32>
        %325 = "ttir.reshape"(%312) <{shape = [236236 : i32, 2 : i32]}> : (tensor<236236x2xi64>) -> tensor<236236x2xi64>
        %326 = "ttir.reshape"(%325) <{shape = [236236 : i32, 1 : i32, 2 : i32]}> : (tensor<236236x2xi64>) -> tensor<236236x1x2xi64>
        %327 = "ttir.repeat"(%326) <{repeat_dimensions = array<i64: 1, 1, 1>}> : (tensor<236236x1x2xi64>) -> tensor<236236x1x2xi64>
        %328 = "ttir.reshape"(%327) <{shape = [236236 : i32, 2 : i32]}> : (tensor<236236x1x2xi64>) -> tensor<236236x2xi64>
        %329 = "ttir.slice_static"(%328) <{begins = [0 : i32, 0 : i32], ends = [236236 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<236236x2xi64>) -> tensor<236236x1xi64>
        %330 = "ttir.slice_static"(%328) <{begins = [0 : i32, 1 : i32], ends = [236236 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<236236x2xi64>) -> tensor<236236x1xi64>
        %331 = "ttir.full"() <{fill_value = 704 : i32, shape = array<i32: 236236, 1>}> : () -> tensor<236236x1xi64>
        %332 = "ttir.multiply"(%329, %331) : (tensor<236236x1xi64>, tensor<236236x1xi64>) -> tensor<236236x1xi64>
        %333 = "ttir.add"(%332, %330) : (tensor<236236x1xi64>, tensor<236236x1xi64>) -> tensor<236236x1xi64>
        %334 = "ttir.reshape"(%333) <{shape = [236236 : i32]}> : (tensor<236236x1xi64>) -> tensor<236236xi64>
        %335 = "ttir.reshape"(%286) <{shape = [180224 : i32]}> : (tensor<256x704xf32>) -> tensor<180224xf32>
        %336 = "ttir.reshape"(%324) <{shape = [236236 : i32]}> : (tensor<236236xf32>) -> tensor<236236xf32>
        %337 = "ttir.scatter"(%335, %334, %336) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<180224xf32>, tensor<236236xi64>, tensor<236236xf32>) -> tensor<180224xf32>
        %338 = "ttir.reshape"(%337) <{shape = [256 : i32, 704 : i32]}> : (tensor<180224xf32>) -> tensor<256x704xf32>
        %339 = "ttir.reshape"(%338) <{shape = [1 : i32, 1 : i32, 256 : i32, 704 : i32]}> : (tensor<256x704xf32>) -> tensor<1x1x256x704xf32>
        %340 = ttir.empty() : tensor<i32>
        %341 = ttir.to_layout %38, %340 : tensor<i64> into tensor<i32> -> tensor<i32>
        %342 = ttir.empty() : tensor<i32>
        %343 = ttir.to_layout %39, %342 : tensor<i64> into tensor<i32> -> tensor<i32>
        %344 = ttir.empty() : tensor<i32>
        %345 = ttir.to_layout %39, %344 : tensor<i64> into tensor<i32> -> tensor<i32>
        %346 = ttir.empty() : tensor<i32>
        %347 = ttir.to_layout %39, %346 : tensor<i64> into tensor<i32> -> tensor<i32>
        %348 = call @cpu_hoisted_stablehlo_dynamic_update_slice_1d952611(%284, %339, %341, %343, %345, %347) {ttir.cpu_hoisted_call} : (tensor<6x1x256x704xf32>, tensor<1x1x256x704xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x1x256x704xf32>
        %349 = "ttir.slice_static"(%348) <{begins = [3 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [4 : i32, 1 : i32, 256 : i32, 704 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x1x256x704xf32>) -> tensor<1x1x256x704xf32>
        %350 = "ttir.reshape"(%349) <{shape = [256 : i32, 704 : i32]}> : (tensor<1x1x256x704xf32>) -> tensor<256x704xf32>
        %351 = "ttir.slice_static"(%112) <{begins = [3 : i32, 0 : i32, 0 : i32], ends = [4 : i32, 2064896 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x2064896x2xf32>) -> tensor<1x2064896x2xf32>
        %352 = "ttir.reshape"(%351) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<1x2064896x2xf32>) -> tensor<2064896x2xf32>
        %353 = "ttir.reshape"(%arg12) <{shape = [212612 : i32]}> : (tensor<212612x1xi64>) -> tensor<212612xi64>
        %354 = "ttir.lt"(%353, %9) : (tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi1>
        %355 = "ttir.add"(%353, %8) : (tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi64>
        %356 = "ttir.where"(%354, %355, %353) : (tensor<212612xi1>, tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi64>
        %357 = "ttir.reshape"(%356) <{shape = [212612 : i32, 1 : i32]}> : (tensor<212612xi64>) -> tensor<212612x1xi64>
        %358 = "ttir.permute"(%352) <{permutation = array<i64: 0, 1>}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %359 = "ttir.reshape"(%358) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %360 = "ttir.embedding"(%357, %359) : (tensor<212612x1xi64>, tensor<2064896x2xf32>) -> tensor<212612x1x2xf32>
        %361 = "ttir.reshape"(%360) <{shape = [212612 : i32, 2 : i32]}> : (tensor<212612x1x2xf32>) -> tensor<212612x2xf32>
        %362 = "ttir.permute"(%361) <{permutation = array<i64: 0, 1>}> : (tensor<212612x2xf32>) -> tensor<212612x2xf32>
        %363 = "ttir.typecast"(%362) <{conservative_folding = false}> : (tensor<212612x2xf32>) -> tensor<212612x2xi64>
        %364 = "ttir.slice_static"(%363) <{begins = [0 : i32, 0 : i32], ends = [212612 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<212612x2xi64>) -> tensor<212612x1xi64>
        %365 = "ttir.reshape"(%364) <{shape = [212612 : i32]}> : (tensor<212612x1xi64>) -> tensor<212612xi64>
        %366 = "ttir.lt"(%365, %9) : (tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi1>
        %367 = "ttir.add"(%365, %7) : (tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi64>
        %368 = "ttir.where"(%366, %367, %365) : (tensor<212612xi1>, tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi64>
        %369 = "ttir.reshape"(%368) <{shape = [212612 : i32, 1 : i32]}> : (tensor<212612xi64>) -> tensor<212612x1xi64>
        %370 = "ttir.slice_static"(%363) <{begins = [0 : i32, 1 : i32], ends = [212612 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<212612x2xi64>) -> tensor<212612x1xi64>
        %371 = "ttir.reshape"(%370) <{shape = [212612 : i32]}> : (tensor<212612x1xi64>) -> tensor<212612xi64>
        %372 = "ttir.lt"(%371, %9) : (tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi1>
        %373 = "ttir.add"(%371, %6) : (tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi64>
        %374 = "ttir.where"(%372, %373, %371) : (tensor<212612xi1>, tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi64>
        %375 = "ttir.reshape"(%374) <{shape = [212612 : i32, 1 : i32]}> : (tensor<212612xi64>) -> tensor<212612x1xi64>
        %376 = "ttir.concat"(%369, %375) <{dim = 1 : si32}> : (tensor<212612x1xi64>, tensor<212612x1xi64>) -> tensor<212612x2xi64>
        %377 = "ttir.slice_static"(%144) <{begins = [3 : i32, 0 : i32], ends = [4 : i32, 2064896 : i32], step = [1 : i32, 1 : i32]}> : (tensor<6x2064896xf32>) -> tensor<1x2064896xf32>
        %378 = "ttir.reshape"(%377) <{shape = [2064896 : i32]}> : (tensor<1x2064896xf32>) -> tensor<2064896xf32>
        %379 = "ttir.reshape"(%arg11) <{shape = [212612 : i32]}> : (tensor<212612x1xi64>) -> tensor<212612xi64>
        %380 = "ttir.lt"(%379, %9) : (tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi1>
        %381 = "ttir.add"(%379, %8) : (tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi64>
        %382 = "ttir.where"(%380, %381, %379) : (tensor<212612xi1>, tensor<212612xi64>, tensor<212612xi64>) -> tensor<212612xi64>
        %383 = "ttir.reshape"(%382) <{shape = [212612 : i32, 1 : i32]}> : (tensor<212612xi64>) -> tensor<212612x1xi64>
        %384 = "ttir.permute"(%378) <{permutation = array<i64: 0>}> : (tensor<2064896xf32>) -> tensor<2064896xf32>
        %385 = "ttir.reshape"(%384) <{shape = [2064896 : i32, 1 : i32]}> : (tensor<2064896xf32>) -> tensor<2064896x1xf32>
        %386 = "ttir.embedding"(%383, %385) : (tensor<212612x1xi64>, tensor<2064896x1xf32>) -> tensor<212612x1x1xf32>
        %387 = "ttir.reshape"(%386) <{shape = [212612 : i32]}> : (tensor<212612x1x1xf32>) -> tensor<212612xf32>
        %388 = "ttir.permute"(%387) <{permutation = array<i64: 0>}> : (tensor<212612xf32>) -> tensor<212612xf32>
        %389 = "ttir.reshape"(%376) <{shape = [212612 : i32, 2 : i32]}> : (tensor<212612x2xi64>) -> tensor<212612x2xi64>
        %390 = "ttir.reshape"(%389) <{shape = [212612 : i32, 1 : i32, 2 : i32]}> : (tensor<212612x2xi64>) -> tensor<212612x1x2xi64>
        %391 = "ttir.repeat"(%390) <{repeat_dimensions = array<i64: 1, 1, 1>}> : (tensor<212612x1x2xi64>) -> tensor<212612x1x2xi64>
        %392 = "ttir.reshape"(%391) <{shape = [212612 : i32, 2 : i32]}> : (tensor<212612x1x2xi64>) -> tensor<212612x2xi64>
        %393 = "ttir.slice_static"(%392) <{begins = [0 : i32, 0 : i32], ends = [212612 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<212612x2xi64>) -> tensor<212612x1xi64>
        %394 = "ttir.slice_static"(%392) <{begins = [0 : i32, 1 : i32], ends = [212612 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<212612x2xi64>) -> tensor<212612x1xi64>
        %395 = "ttir.full"() <{fill_value = 704 : i32, shape = array<i32: 212612, 1>}> : () -> tensor<212612x1xi64>
        %396 = "ttir.multiply"(%393, %395) : (tensor<212612x1xi64>, tensor<212612x1xi64>) -> tensor<212612x1xi64>
        %397 = "ttir.add"(%396, %394) : (tensor<212612x1xi64>, tensor<212612x1xi64>) -> tensor<212612x1xi64>
        %398 = "ttir.reshape"(%397) <{shape = [212612 : i32]}> : (tensor<212612x1xi64>) -> tensor<212612xi64>
        %399 = "ttir.reshape"(%350) <{shape = [180224 : i32]}> : (tensor<256x704xf32>) -> tensor<180224xf32>
        %400 = "ttir.reshape"(%388) <{shape = [212612 : i32]}> : (tensor<212612xf32>) -> tensor<212612xf32>
        %401 = "ttir.scatter"(%399, %398, %400) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<180224xf32>, tensor<212612xi64>, tensor<212612xf32>) -> tensor<180224xf32>
        %402 = "ttir.reshape"(%401) <{shape = [256 : i32, 704 : i32]}> : (tensor<180224xf32>) -> tensor<256x704xf32>
        %403 = "ttir.reshape"(%402) <{shape = [1 : i32, 1 : i32, 256 : i32, 704 : i32]}> : (tensor<256x704xf32>) -> tensor<1x1x256x704xf32>
        %404 = ttir.empty() : tensor<i32>
        %405 = ttir.to_layout %36, %404 : tensor<i64> into tensor<i32> -> tensor<i32>
        %406 = ttir.empty() : tensor<i32>
        %407 = ttir.to_layout %39, %406 : tensor<i64> into tensor<i32> -> tensor<i32>
        %408 = ttir.empty() : tensor<i32>
        %409 = ttir.to_layout %39, %408 : tensor<i64> into tensor<i32> -> tensor<i32>
        %410 = ttir.empty() : tensor<i32>
        %411 = ttir.to_layout %39, %410 : tensor<i64> into tensor<i32> -> tensor<i32>
        %412 = call @cpu_hoisted_stablehlo_dynamic_update_slice_1d952611(%348, %403, %405, %407, %409, %411) {ttir.cpu_hoisted_call} : (tensor<6x1x256x704xf32>, tensor<1x1x256x704xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x1x256x704xf32>
        %413 = "ttir.slice_static"(%412) <{begins = [4 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [5 : i32, 1 : i32, 256 : i32, 704 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x1x256x704xf32>) -> tensor<1x1x256x704xf32>
        %414 = "ttir.reshape"(%413) <{shape = [256 : i32, 704 : i32]}> : (tensor<1x1x256x704xf32>) -> tensor<256x704xf32>
        %415 = "ttir.slice_static"(%112) <{begins = [4 : i32, 0 : i32, 0 : i32], ends = [5 : i32, 2064896 : i32, 2 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x2064896x2xf32>) -> tensor<1x2064896x2xf32>
        %416 = "ttir.reshape"(%415) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<1x2064896x2xf32>) -> tensor<2064896x2xf32>
        %417 = "ttir.reshape"(%arg10) <{shape = [285388 : i32]}> : (tensor<285388x1xi64>) -> tensor<285388xi64>
        %418 = "ttir.lt"(%417, %5) : (tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi1>
        %419 = "ttir.add"(%417, %4) : (tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi64>
        %420 = "ttir.where"(%418, %419, %417) : (tensor<285388xi1>, tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi64>
        %421 = "ttir.reshape"(%420) <{shape = [285388 : i32, 1 : i32]}> : (tensor<285388xi64>) -> tensor<285388x1xi64>
        %422 = "ttir.permute"(%416) <{permutation = array<i64: 0, 1>}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %423 = "ttir.reshape"(%422) <{shape = [2064896 : i32, 2 : i32]}> : (tensor<2064896x2xf32>) -> tensor<2064896x2xf32>
        %424 = "ttir.embedding"(%421, %423) : (tensor<285388x1xi64>, tensor<2064896x2xf32>) -> tensor<285388x1x2xf32>
        %425 = "ttir.reshape"(%424) <{shape = [285388 : i32, 2 : i32]}> : (tensor<285388x1x2xf32>) -> tensor<285388x2xf32>
        %426 = "ttir.permute"(%425) <{permutation = array<i64: 0, 1>}> : (tensor<285388x2xf32>) -> tensor<285388x2xf32>
        %427 = "ttir.typecast"(%426) <{conservative_folding = false}> : (tensor<285388x2xf32>) -> tensor<285388x2xi64>
        %428 = "ttir.slice_static"(%427) <{begins = [0 : i32, 0 : i32], ends = [285388 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<285388x2xi64>) -> tensor<285388x1xi64>
        %429 = "ttir.reshape"(%428) <{shape = [285388 : i32]}> : (tensor<285388x1xi64>) -> tensor<285388xi64>
        %430 = "ttir.lt"(%429, %5) : (tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi1>
        %431 = "ttir.add"(%429, %3) : (tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi64>
        %432 = "ttir.where"(%430, %431, %429) : (tensor<285388xi1>, tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi64>
        %433 = "ttir.reshape"(%432) <{shape = [285388 : i32, 1 : i32]}> : (tensor<285388xi64>) -> tensor<285388x1xi64>
        %434 = "ttir.slice_static"(%427) <{begins = [0 : i32, 1 : i32], ends = [285388 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<285388x2xi64>) -> tensor<285388x1xi64>
        %435 = "ttir.reshape"(%434) <{shape = [285388 : i32]}> : (tensor<285388x1xi64>) -> tensor<285388xi64>
        %436 = "ttir.lt"(%435, %5) : (tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi1>
        %437 = "ttir.add"(%435, %2) : (tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi64>
        %438 = "ttir.where"(%436, %437, %435) : (tensor<285388xi1>, tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi64>
        %439 = "ttir.reshape"(%438) <{shape = [285388 : i32, 1 : i32]}> : (tensor<285388xi64>) -> tensor<285388x1xi64>
        %440 = "ttir.concat"(%433, %439) <{dim = 1 : si32}> : (tensor<285388x1xi64>, tensor<285388x1xi64>) -> tensor<285388x2xi64>
        %441 = "ttir.slice_static"(%144) <{begins = [4 : i32, 0 : i32], ends = [5 : i32, 2064896 : i32], step = [1 : i32, 1 : i32]}> : (tensor<6x2064896xf32>) -> tensor<1x2064896xf32>
        %442 = "ttir.reshape"(%441) <{shape = [2064896 : i32]}> : (tensor<1x2064896xf32>) -> tensor<2064896xf32>
        %443 = "ttir.reshape"(%arg9) <{shape = [285388 : i32]}> : (tensor<285388x1xi64>) -> tensor<285388xi64>
        %444 = "ttir.lt"(%443, %5) : (tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi1>
        %445 = "ttir.add"(%443, %4) : (tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi64>
        %446 = "ttir.where"(%444, %445, %443) : (tensor<285388xi1>, tensor<285388xi64>, tensor<285388xi64>) -> tensor<285388xi64>
        %447 = "ttir.reshape"(%446) <{shape = [285388 : i32, 1 : i32]}> : (tensor<285388xi64>) -> tensor<285388x1xi64>
        %448 = "ttir.permute"(%442) <{permutation = array<i64: 0>}> : (tensor<2064896xf32>) -> tensor<2064896xf32>
        %449 = "ttir.reshape"(%448) <{shape = [2064896 : i32, 1 : i32]}> : (tensor<2064896xf32>) -> tensor<2064896x1xf32>
        %450 = "ttir.embedding"(%447, %449) : (tensor<285388x1xi64>, tensor<2064896x1xf32>) -> tensor<285388x1x1xf32>
        %451 = "ttir.reshape"(%450) <{shape = [285388 : i32]}> : (tensor<285388x1x1xf32>) -> tensor<285388xf32>
        %452 = "ttir.permute"(%451) <{permutation = array<i64: 0>}> : (tensor<285388xf32>) -> tensor<285388xf32>
        %453 = "ttir.reshape"(%440) <{shape = [285388 : i32, 2 : i32]}> : (tensor<285388x2xi64>) -> tensor<285388x2xi64>
        %454 = "ttir.reshape"(%453) <{shape = [285388 : i32, 1 : i32, 2 : i32]}> : (tensor<285388x2xi64>) -> tensor<285388x1x2xi64>
        %455 = "ttir.repeat"(%454) <{repeat_dimensions = array<i64: 1, 1, 1>}> : (tensor<285388x1x2xi64>) -> tensor<285388x1x2xi64>
        %456 = "ttir.reshape"(%455) <{shape = [285388 : i32, 2 : i32]}> : (tensor<285388x1x2xi64>) -> tensor<285388x2xi64>
        %457 = "ttir.slice_static"(%456) <{begins = [0 : i32, 0 : i32], ends = [285388 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<285388x2xi64>) -> tensor<285388x1xi64>
        %458 = "ttir.slice_static"(%456) <{begins = [0 : i32, 1 : i32], ends = [285388 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<285388x2xi64>) -> tensor<285388x1xi64>
        %459 = "ttir.full"() <{fill_value = 704 : i32, shape = array<i32: 285388, 1>}> : () -> tensor<285388x1xi64>
        %460 = "ttir.multiply"(%457, %459) : (tensor<285388x1xi64>, tensor<285388x1xi64>) -> tensor<285388x1xi64>
        %461 = "ttir.add"(%460, %458) : (tensor<285388x1xi64>, tensor<285388x1xi64>) -> tensor<285388x1xi64>
        %462 = "ttir.reshape"(%461) <{shape = [285388 : i32]}> : (tensor<285388x1xi64>) -> tensor<285388xi64>
        %463 = "ttir.reshape"(%414) <{shape = [180224 : i32]}> : (tensor<256x704xf32>) -> tensor<180224xf32>
        %464 = "ttir.reshape"(%452) <{shape = [285388 : i32]}> : (tensor<285388xf32>) -> tensor<285388xf32>
        %465 = "ttir.scatter"(%463, %462, %464) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<180224xf32>, tensor<285388xi64>, tensor<285388xf32>) -> tensor<180224xf32>
        %466 = "ttir.reshape"(%465) <{shape = [256 : i32, 704 : i32]}> : (tensor<180224xf32>) -> tensor<256x704xf32>
        %467 = "ttir.reshape"(%466) <{shape = [1 : i32, 1 : i32, 256 : i32, 704 : i32]}> : (tensor<256x704xf32>) -> tensor<1x1x256x704xf32>
        %468 = ttir.empty() : tensor<i32>
        %469 = ttir.to_layout %35, %468 : tensor<i64> into tensor<i32> -> tensor<i32>
        %470 = ttir.empty() : tensor<i32>
        %471 = ttir.to_layout %39, %470 : tensor<i64> into tensor<i32> -> tensor<i32>
        %472 = ttir.empty() : tensor<i32>
        %473 = ttir.to_layout %39, %472 : tensor<i64> into tensor<i32> -> tensor<i32>
        %474 = ttir.empty() : tensor<i32>
        %475 = ttir.to_layout %39, %474 : tensor<i64> into tensor<i32> -> tensor<i32>
        %476 = call @cpu_hoisted_stablehlo_dynamic_update_slice_1d952611(%412, %467, %469, %471, %473, %475) {ttir.cpu_hoisted_call} : (tensor<6x1x256x704xf32>, tensor<1x1x256x704xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x1x256x704xf32>
        %477 = "ttir.slice_static"(%476) <{begins = [5 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 1 : i32, 256 : i32, 704 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<6x1x256x704xf32>) -> tensor<1x1x256x704xf32>
        %478 = "ttir.reshape"(%477) <{shape = [256 : i32, 704 : i32]}> : (tensor<1x1x256x704xf32>) -> tensor<256x704xf32>
        %479 = "ttir.slice_static"(%142) <{begins = [0 : i32, 0 : i32], ends = [232355 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<232355x2xi64>) -> tensor<232355x1xi64>
        %480 = "ttir.reshape"(%479) <{shape = [232355 : i32]}> : (tensor<232355x1xi64>) -> tensor<232355xi64>
        %481 = "ttir.lt"(%480, %24) : (tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi1>
        %482 = "ttir.add"(%480, %1) : (tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi64>
        %483 = "ttir.where"(%481, %482, %480) : (tensor<232355xi1>, tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi64>
        %484 = "ttir.reshape"(%483) <{shape = [232355 : i32, 1 : i32]}> : (tensor<232355xi64>) -> tensor<232355x1xi64>
        %485 = "ttir.slice_static"(%142) <{begins = [0 : i32, 1 : i32], ends = [232355 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<232355x2xi64>) -> tensor<232355x1xi64>
        %486 = "ttir.reshape"(%485) <{shape = [232355 : i32]}> : (tensor<232355x1xi64>) -> tensor<232355xi64>
        %487 = "ttir.lt"(%486, %24) : (tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi1>
        %488 = "ttir.add"(%486, %0) : (tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi64>
        %489 = "ttir.where"(%487, %488, %486) : (tensor<232355xi1>, tensor<232355xi64>, tensor<232355xi64>) -> tensor<232355xi64>
        %490 = "ttir.reshape"(%489) <{shape = [232355 : i32, 1 : i32]}> : (tensor<232355xi64>) -> tensor<232355x1xi64>
        %491 = "ttir.concat"(%484, %490) <{dim = 1 : si32}> : (tensor<232355x1xi64>, tensor<232355x1xi64>) -> tensor<232355x2xi64>
        %492 = "ttir.reshape"(%491) <{shape = [232355 : i32, 2 : i32]}> : (tensor<232355x2xi64>) -> tensor<232355x2xi64>
        %493 = "ttir.reshape"(%492) <{shape = [232355 : i32, 1 : i32, 2 : i32]}> : (tensor<232355x2xi64>) -> tensor<232355x1x2xi64>
        %494 = "ttir.repeat"(%493) <{repeat_dimensions = array<i64: 1, 1, 1>}> : (tensor<232355x1x2xi64>) -> tensor<232355x1x2xi64>
        %495 = "ttir.reshape"(%494) <{shape = [232355 : i32, 2 : i32]}> : (tensor<232355x1x2xi64>) -> tensor<232355x2xi64>
        %496 = "ttir.slice_static"(%495) <{begins = [0 : i32, 0 : i32], ends = [232355 : i32, 1 : i32], step = [1 : i32, 1 : i32]}> : (tensor<232355x2xi64>) -> tensor<232355x1xi64>
        %497 = "ttir.slice_static"(%495) <{begins = [0 : i32, 1 : i32], ends = [232355 : i32, 2 : i32], step = [1 : i32, 1 : i32]}> : (tensor<232355x2xi64>) -> tensor<232355x1xi64>
        %498 = "ttir.full"() <{fill_value = 704 : i32, shape = array<i32: 232355, 1>}> : () -> tensor<232355x1xi64>
        %499 = "ttir.multiply"(%496, %498) : (tensor<232355x1xi64>, tensor<232355x1xi64>) -> tensor<232355x1xi64>
        %500 = "ttir.add"(%499, %497) : (tensor<232355x1xi64>, tensor<232355x1xi64>) -> tensor<232355x1xi64>
        %501 = "ttir.reshape"(%500) <{shape = [232355 : i32]}> : (tensor<232355x1xi64>) -> tensor<232355xi64>
        %502 = "ttir.reshape"(%478) <{shape = [180224 : i32]}> : (tensor<256x704xf32>) -> tensor<180224xf32>
        %503 = "ttir.reshape"(%156) <{shape = [232355 : i32]}> : (tensor<232355xf32>) -> tensor<232355xf32>
        %504 = "ttir.scatter"(%502, %501, %503) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<180224xf32>, tensor<232355xi64>, tensor<232355xf32>) -> tensor<180224xf32>
        %505 = "ttir.reshape"(%504) <{shape = [256 : i32, 704 : i32]}> : (tensor<180224xf32>) -> tensor<256x704xf32>
        %506 = "ttir.reshape"(%505) <{shape = [1 : i32, 1 : i32, 256 : i32, 704 : i32]}> : (tensor<256x704xf32>) -> tensor<1x1x256x704xf32>
        %507 = ttir.empty() : tensor<i32>
        %508 = ttir.to_layout %34, %507 : tensor<i64> into tensor<i32> -> tensor<i32>
        %509 = ttir.empty() : tensor<i32>
        %510 = ttir.to_layout %39, %509 : tensor<i64> into tensor<i32> -> tensor<i32>
        %511 = ttir.empty() : tensor<i32>
        %512 = ttir.to_layout %39, %511 : tensor<i64> into tensor<i32> -> tensor<i32>
        %513 = ttir.empty() : tensor<i32>
        %514 = ttir.to_layout %39, %513 : tensor<i64> into tensor<i32> -> tensor<i32>
        %515 = call @cpu_hoisted_stablehlo_dynamic_update_slice_1d952611(%476, %506, %508, %510, %512, %514) {ttir.cpu_hoisted_call} : (tensor<6x1x256x704xf32>, tensor<1x1x256x704xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x1x256x704xf32>
        %516 = "ttir.reshape"(%515) <{shape = [1 : i32, 6 : i32, 1 : i32, 256 : i32, 704 : i32]}> : (tensor<6x1x256x704xf32>) -> tensor<1x6x1x256x704xf32>
        %517 = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 3 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x4x4xf32>) -> tensor<1x3x4xf32>
        %518 = "ttir.slice_static"(%517) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 3 : i32, 3 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x3x4xf32>) -> tensor<1x3x3xf32>
        %519 = "ttir.slice_static"(%517) <{begins = [0 : i32, 0 : i32, 3 : i32], ends = [1 : i32, 3 : i32, 4 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x3x4xf32>) -> tensor<1x3x1xf32>
        %520 = "ttir.reshape"(%519) <{shape = [1 : i32, 3 : i32]}> : (tensor<1x3x1xf32>) -> tensor<1x3xf32>
        return %40, %41, %42, %43, %59, %69, %71, %93, %112, %129, %142, %156, %516, %517, %518, %517, %520 : tensor<1x6x1x256x704xf32>, tensor<6x4x4xf32>, tensor<4x4xf32>, tensor<6x4x4xf32>, tensor<2064896x5xf32>, tensor<6x3x2064896xf32>, tensor<6x2064896xf32>, tensor<18x2064896xf32>, tensor<6x2064896x2xf32>, tensor<6x2064896xi1>, tensor<232355x2xi64>, tensor<232355xf32>, tensor<1x6x1x256x704xf32>, tensor<1x3x4xf32>, tensor<1x3x3xf32>, tensor<1x3x4xf32>, tensor<1x3xf32>
      }
      func.func private @cpu_hoisted_stablehlo_dynamic_update_slice_b1555218(tensor<2064896x5xf32>, tensor<2064896x1xf32>, tensor<i32>, tensor<i32>) -> tensor<2064896x5xf32> attributes {func_hash = "b1555218eb8c9c1b09ea095ce54d409f6ccadf405feeed8f1f6889dd727e8b5c", tt.function_type = "forward_cpu_declaration"}
      func.func private @cpu_hoisted_stablehlo_dynamic_update_slice_a8ec3ec4(tensor<6x3x2064896xf32>, tensor<6x1x2064896xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3x2064896xf32> attributes {func_hash = "a8ec3ec443140cd81088485fa0c4fd28edb73c1394b0ba94c178096de2fb6a84", tt.function_type = "forward_cpu_declaration"}
      func.func private @cpu_hoisted_stablehlo_dynamic_update_slice_1d952611(tensor<6x1x256x704xf32>, tensor<1x1x256x704xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x1x256x704xf32> attributes {func_hash = "1d952611465eb629abd8d7b1e5bad510be95ab30310b0950eff02c9364b75595", tt.function_type = "forward_cpu_declaration"}
    }
  }
  ttcore.cpu_module {
    builtin.module {
      func.func @cpu_hoisted_stablehlo_dynamic_update_slice_b1555218(%arg0: tensor<2064896x5xf32> {bufferization.access = "read"}, %arg1: tensor<2064896x1xf32> {bufferization.access = "read"}, %arg2: tensor<i32> {bufferization.access = "read"}, %arg3: tensor<i32> {bufferization.access = "read"}) -> tensor<2064896x5xf32> attributes {arg_ranks = [2, 2, 0, 0], func_hash = "b1555218eb8c9c1b09ea095ce54d409f6ccadf405feeed8f1f6889dd727e8b5c", result_ranks = [2], tt.function_type = "forward_cpu"} {
        %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3 : (tensor<2064896x5xf32>, tensor<2064896x1xf32>, tensor<i32>, tensor<i32>) -> tensor<2064896x5xf32>
        return %0 : tensor<2064896x5xf32>
      }
      func.func @cpu_hoisted_stablehlo_dynamic_update_slice_a8ec3ec4(%arg0: tensor<6x3x2064896xf32> {bufferization.access = "read"}, %arg1: tensor<6x1x2064896xf32> {bufferization.access = "read"}, %arg2: tensor<i32> {bufferization.access = "read"}, %arg3: tensor<i32> {bufferization.access = "read"}, %arg4: tensor<i32> {bufferization.access = "read"}) -> tensor<6x3x2064896xf32> attributes {arg_ranks = [3, 3, 0, 0, 0], func_hash = "a8ec3ec443140cd81088485fa0c4fd28edb73c1394b0ba94c178096de2fb6a84", result_ranks = [3], tt.function_type = "forward_cpu"} {
        %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4 : (tensor<6x3x2064896xf32>, tensor<6x1x2064896xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x3x2064896xf32>
        return %0 : tensor<6x3x2064896xf32>
      }
      func.func @cpu_hoisted_stablehlo_dynamic_update_slice_1d952611(%arg0: tensor<6x1x256x704xf32> {bufferization.access = "read"}, %arg1: tensor<1x1x256x704xf32> {bufferization.access = "read"}, %arg2: tensor<i32> {bufferization.access = "read"}, %arg3: tensor<i32> {bufferization.access = "read"}, %arg4: tensor<i32> {bufferization.access = "read"}, %arg5: tensor<i32> {bufferization.access = "read"}) -> tensor<6x1x256x704xf32> attributes {arg_ranks = [4, 4, 0, 0, 0, 0], func_hash = "1d952611465eb629abd8d7b1e5bad510be95ab30310b0950eff02c9364b75595", result_ranks = [4], tt.function_type = "forward_cpu"} {
        %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 : (tensor<6x1x256x704xf32>, tensor<1x1x256x704xf32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<6x1x256x704xf32>
        return %0 : tensor<6x1x256x704xf32>
      }
    }
  }
}
