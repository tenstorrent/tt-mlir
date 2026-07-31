// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  ttcore.device_module {
    builtin.module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
      func.func @main(
          %inv_freq: tensor<64xbf16>,
          %hidden: tensor<32x18x4096xbf16>,
          %k_weight: tensor<1024x4096xbf16>,
          %gamma: tensor<4096xbf16>,
          %key_cache: tensor<32x8x128x128xbf16>,
          %v_weight: tensor<1024x4096xbf16>,
          %value_cache: tensor<32x8x128x128xbf16>,
          %q_weight: tensor<4096x4096xbf16>,
          %o_weight: tensor<4096x4096xbf16>,
          %post_gamma: tensor<4096xbf16>,
          %gate_weight: tensor<14336x4096xbf16>,
          %up_weight: tensor<14336x4096xbf16>,
          %down_weight: tensor<4096x14336xbf16>) -> (
          tensor<32x8x128x128xbf16>,
          tensor<32x8x128x128xbf16>,
          tensor<32x18x4096xbf16>) {
        %0 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 128 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<128xi64>
        %1 = "ttir.constant"() <{value = dense<0> : tensor<18xi64>}> : () -> tensor<18xi64>
        %2 = "ttir.constant"() <{value = dense<128> : tensor<18xi64>}> : () -> tensor<18xi64>
        %3 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %4 = "ttir.constant"() <{value = dense<2.44140625E-4> : tensor<f32>}> : () -> tensor<f32>
        %5 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<f32>}> : () -> tensor<f32>
        %6 = "ttir.constant"() <{value = dense<0.297301769> : tensor<f32>}> : () -> tensor<f32>
        %7 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
        %8 = "ttir.constant"() <{value = dense<0xFF80> : tensor<bf16>}> : () -> tensor<bf16>
        %9 = "ttir.constant"() <{value = dense<0xFFF0000000000000> : tensor<f64>}> : () -> tensor<f64>
        %10 = "ttir.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
        %11 = "ttir.reshape"(%10) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %12 = "ttir.broadcast"(%11) <{broadcast_dimensions = array<i64: 32, 32, 18, 128>}> : (tensor<1x1x1x1xf32>) -> tensor<32x32x18x128xf32>
        %13 = "ttir.reshape"(%9) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f64>) -> tensor<1x1x1x1xf64>
        %14 = "ttir.broadcast"(%13) <{broadcast_dimensions = array<i64: 32, 32, 18, 128>}> : (tensor<1x1x1x1xf64>) -> tensor<32x32x18x128xf64>
        %15 = "ttir.reshape"(%8) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
        %16 = "ttir.broadcast"(%15) <{broadcast_dimensions = array<i64: 32, 1, 18, 128>}> : (tensor<1x1x1x1xbf16>) -> tensor<32x1x18x128xbf16>
        %17 = "ttir.reshape"(%7) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
        %18 = "ttir.broadcast"(%17) <{broadcast_dimensions = array<i64: 32, 1, 18, 128>}> : (tensor<1x1x1x1xbf16>) -> tensor<32x1x18x128xbf16>
        %19 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
        %20 = "ttir.broadcast"(%19) <{broadcast_dimensions = array<i64: 32, 32, 128, 128>}> : (tensor<1x1x1x1xf32>) -> tensor<32x32x128x128xf32>
        %21 = "ttir.broadcast"(%19) <{broadcast_dimensions = array<i64: 32, 32, 18, 128>}> : (tensor<1x1x1x1xf32>) -> tensor<32x32x18x128xf32>
        %22 = "ttir.reshape"(%5) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %23 = "ttir.broadcast"(%22) <{broadcast_dimensions = array<i64: 32, 18, 1>}> : (tensor<1x1x1xf32>) -> tensor<32x18x1xf32>
        %24 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1xf32>
        %25 = "ttir.broadcast"(%24) <{broadcast_dimensions = array<i64: 32, 18>}> : (tensor<1x1xf32>) -> tensor<32x18xf32>
        %26 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
        %27 = "ttir.broadcast"(%26) <{broadcast_dimensions = array<i64: 32, 18, 4096>}> : (tensor<1x1x1xf32>) -> tensor<32x18x4096xf32>
        %position_values = "ttir.arange"() <{arange_dimension = 0 : i64, end = 18 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<18xi64>
        %28 = "ttir.reshape"(%position_values) <{shape = [1 : i32, 1 : i32, 18 : i32]}> : (tensor<18xi64>) -> tensor<1x1x18xi64>
        %29 = "ttir.reshape"(%28) <{shape = [18 : i32]}> : (tensor<1x1x18xi64>) -> tensor<18xi64>
        %30 = "ttir.lt"(%29, %1) : (tensor<18xi64>, tensor<18xi64>) -> tensor<18xi1>
        %31 = "ttir.add"(%29, %2) : (tensor<18xi64>, tensor<18xi64>) -> tensor<18xi64>
        %32 = "ttir.where"(%30, %31, %29) : (tensor<18xi1>, tensor<18xi64>, tensor<18xi64>) -> tensor<18xi64>
        %33 = "ttir.reshape"(%gamma) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
        %34 = "ttir.reshape"(%33) <{shape = [4096 : i32]}> : (tensor<1x1x4096xbf16>) -> tensor<4096xbf16>
        %35 = "ttir.reshape"(%34) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
        %36 = "ttir.broadcast"(%35) <{broadcast_dimensions = array<i64: 32, 18, 1>}> : (tensor<1x1x4096xbf16>) -> tensor<32x18x4096xbf16>
        %44 = "ttir.typecast"(%hidden) <{conservative_folding = false}> : (tensor<32x18x4096xbf16>) -> tensor<32x18x4096xf32>
        %45 = "ttir.pow"(%44, %27) : (tensor<32x18x4096xf32>, tensor<32x18x4096xf32>) -> tensor<32x18x4096xf32>
        %46 = "ttir.sum"(%45) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x18x4096xf32>) -> tensor<32x18xf32>
        %47 = "ttir.multiply"(%46, %25) : (tensor<32x18xf32>, tensor<32x18xf32>) -> tensor<32x18xf32>
        %48 = "ttir.reshape"(%47) <{shape = [32 : i32, 18 : i32, 1 : i32]}> : (tensor<32x18xf32>) -> tensor<32x18x1xf32>
        %49 = "ttir.add"(%48, %23) : (tensor<32x18x1xf32>, tensor<32x18x1xf32>) -> tensor<32x18x1xf32>
        %50 = "ttir.rsqrt"(%49) : (tensor<32x18x1xf32>) -> tensor<32x18x1xf32>
        %51 = "ttir.reshape"(%50) <{shape = [32 : i32, 18 : i32]}> : (tensor<32x18x1xf32>) -> tensor<32x18xf32>
        %52 = "ttir.reshape"(%51) <{shape = [32 : i32, 18 : i32, 1 : i32]}> : (tensor<32x18xf32>) -> tensor<32x18x1xf32>
        %53 = "ttir.broadcast"(%52) <{broadcast_dimensions = array<i64: 1, 1, 4096>}> : (tensor<32x18x1xf32>) -> tensor<32x18x4096xf32>
        %54 = "ttir.multiply"(%44, %53) : (tensor<32x18x4096xf32>, tensor<32x18x4096xf32>) -> tensor<32x18x4096xf32>
        %55 = "ttir.typecast"(%54) <{conservative_folding = false}> : (tensor<32x18x4096xf32>) -> tensor<32x18x4096xbf16>
        %56 = "ttir.multiply"(%36, %55) : (tensor<32x18x4096xbf16>, tensor<32x18x4096xbf16>) -> tensor<32x18x4096xbf16>
        %57 = "ttir.reshape"(%56) <{shape = [576 : i32, 4096 : i32]}> : (tensor<32x18x4096xbf16>) -> tensor<576x4096xbf16>
        %58 = "ttir.reshape"(%k_weight) <{shape = [1 : i32, 1024 : i32, 4096 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x1024x4096xbf16>
        %59 = "ttir.reshape"(%58) <{shape = [1024 : i32, 4096 : i32]}> : (tensor<1x1024x4096xbf16>) -> tensor<1024x4096xbf16>
        %60 = "ttir.permute"(%59) <{permutation = array<i64: 1, 0>}> : (tensor<1024x4096xbf16>) -> tensor<4096x1024xbf16>
        %61 = "ttir.dot_general"(%57, %60) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x4096xbf16>, tensor<4096x1024xbf16>) -> tensor<576x1024xbf16>
        %62 = "ttir.reshape"(%61) <{shape = [32 : i32, 18 : i32, 8 : i32, 128 : i32]}> : (tensor<576x1024xbf16>) -> tensor<32x18x8x128xbf16>
        %63 = "ttir.permute"(%62) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x18x8x128xbf16>) -> tensor<32x8x18x128xbf16>
        %64 = "ttir.reshape"(%inv_freq) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<64xbf16>) -> tensor<1x1x64xbf16>
        %65 = "ttir.reshape"(%64) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<1x1x64xbf16>) -> tensor<1x64x1xbf16>
        %66 = "ttir.typecast"(%65) <{conservative_folding = false}> : (tensor<1x64x1xbf16>) -> tensor<1x64x1xf32>
        %67 = "ttir.typecast"(%28) <{conservative_folding = false}> : (tensor<1x1x18xi64>) -> tensor<1x1x18xf32>
        %68 = "ttir.dot_general"(%66, %67) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<1x64x1xf32>, tensor<1x1x18xf32>) -> tensor<1x64x18xf32>
        %69 = "ttir.permute"(%68) <{permutation = array<i64: 0, 2, 1>}> : (tensor<1x64x18xf32>) -> tensor<1x18x64xf32>
        %70 = "ttir.concat"(%69, %69) <{dim = 2 : si32}> : (tensor<1x18x64xf32>, tensor<1x18x64xf32>) -> tensor<1x18x128xf32>
        %71 = "ttir.cos"(%70) : (tensor<1x18x128xf32>) -> tensor<1x18x128xf32>
        %72 = "ttir.typecast"(%71) <{conservative_folding = false}> : (tensor<1x18x128xf32>) -> tensor<1x18x128xbf16>
        %73 = "ttir.reshape"(%72) <{shape = [18 : i32, 128 : i32]}> : (tensor<1x18x128xbf16>) -> tensor<18x128xbf16>
        %74 = "ttir.reshape"(%73) <{shape = [1 : i32, 1 : i32, 18 : i32, 128 : i32]}> : (tensor<18x128xbf16>) -> tensor<1x1x18x128xbf16>
        %75 = "ttir.broadcast"(%74) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x18x128xbf16>) -> tensor<32x8x18x128xbf16>
        %76 = "ttir.multiply"(%63, %75) : (tensor<32x8x18x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x8x18x128xbf16>
        %77 = "ttir.slice_static"(%63) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [32 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<32x8x18x64xbf16>
        %78 = "ttir.neg"(%77) : (tensor<32x8x18x64xbf16>) -> tensor<32x8x18x64xbf16>
        %79 = "ttir.slice_static"(%63) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 18 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<32x8x18x64xbf16>
        %80 = "ttir.concat"(%78, %79) <{dim = 3 : si32}> : (tensor<32x8x18x64xbf16>, tensor<32x8x18x64xbf16>) -> tensor<32x8x18x128xbf16>
        %81 = "ttir.sin"(%70) : (tensor<1x18x128xf32>) -> tensor<1x18x128xf32>
        %82 = "ttir.typecast"(%81) <{conservative_folding = false}> : (tensor<1x18x128xf32>) -> tensor<1x18x128xbf16>
        %83 = "ttir.reshape"(%82) <{shape = [18 : i32, 128 : i32]}> : (tensor<1x18x128xbf16>) -> tensor<18x128xbf16>
        %84 = "ttir.reshape"(%83) <{shape = [1 : i32, 1 : i32, 18 : i32, 128 : i32]}> : (tensor<18x128xbf16>) -> tensor<1x1x18x128xbf16>
        %85 = "ttir.broadcast"(%84) <{broadcast_dimensions = array<i64: 32, 8, 1, 1>}> : (tensor<1x1x18x128xbf16>) -> tensor<32x8x18x128xbf16>
        %86 = "ttir.multiply"(%80, %85) : (tensor<32x8x18x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x8x18x128xbf16>
        %87 = "ttir.add"(%76, %86) : (tensor<32x8x18x128xbf16>, tensor<32x8x18x128xbf16>) -> tensor<32x8x18x128xbf16>
        %88 = "ttir.slice_static"(%87) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %89 = "ttir.fill_cache"(%key_cache, %88) <{batch_offset = 0 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %90 = "ttir.slice_static"(%87) <{begins = [1 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %91 = "ttir.fill_cache"(%89, %90) <{batch_offset = 1 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %92 = "ttir.slice_static"(%87) <{begins = [2 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [3 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %93 = "ttir.fill_cache"(%91, %92) <{batch_offset = 2 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %94 = "ttir.slice_static"(%87) <{begins = [3 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [4 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %95 = "ttir.fill_cache"(%93, %94) <{batch_offset = 3 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %96 = "ttir.slice_static"(%87) <{begins = [4 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [5 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %97 = "ttir.fill_cache"(%95, %96) <{batch_offset = 4 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %98 = "ttir.slice_static"(%87) <{begins = [5 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %99 = "ttir.fill_cache"(%97, %98) <{batch_offset = 5 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %100 = "ttir.slice_static"(%87) <{begins = [6 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [7 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %101 = "ttir.fill_cache"(%99, %100) <{batch_offset = 6 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %102 = "ttir.slice_static"(%87) <{begins = [7 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %103 = "ttir.fill_cache"(%101, %102) <{batch_offset = 7 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %104 = "ttir.slice_static"(%87) <{begins = [8 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [9 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %105 = "ttir.fill_cache"(%103, %104) <{batch_offset = 8 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %106 = "ttir.slice_static"(%87) <{begins = [9 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [10 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %107 = "ttir.fill_cache"(%105, %106) <{batch_offset = 9 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %108 = "ttir.slice_static"(%87) <{begins = [10 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [11 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %109 = "ttir.fill_cache"(%107, %108) <{batch_offset = 10 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %110 = "ttir.slice_static"(%87) <{begins = [11 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [12 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %111 = "ttir.fill_cache"(%109, %110) <{batch_offset = 11 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %112 = "ttir.slice_static"(%87) <{begins = [12 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [13 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %113 = "ttir.fill_cache"(%111, %112) <{batch_offset = 12 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %114 = "ttir.slice_static"(%87) <{begins = [13 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [14 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %115 = "ttir.fill_cache"(%113, %114) <{batch_offset = 13 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %116 = "ttir.slice_static"(%87) <{begins = [14 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [15 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %117 = "ttir.fill_cache"(%115, %116) <{batch_offset = 14 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %118 = "ttir.slice_static"(%87) <{begins = [15 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [16 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %119 = "ttir.fill_cache"(%117, %118) <{batch_offset = 15 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %120 = "ttir.slice_static"(%87) <{begins = [16 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [17 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %121 = "ttir.fill_cache"(%119, %120) <{batch_offset = 16 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %122 = "ttir.slice_static"(%87) <{begins = [17 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [18 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %123 = "ttir.fill_cache"(%121, %122) <{batch_offset = 17 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %124 = "ttir.slice_static"(%87) <{begins = [18 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [19 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %125 = "ttir.fill_cache"(%123, %124) <{batch_offset = 18 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %126 = "ttir.slice_static"(%87) <{begins = [19 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [20 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %127 = "ttir.fill_cache"(%125, %126) <{batch_offset = 19 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %128 = "ttir.slice_static"(%87) <{begins = [20 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [21 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %129 = "ttir.fill_cache"(%127, %128) <{batch_offset = 20 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %130 = "ttir.slice_static"(%87) <{begins = [21 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [22 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %131 = "ttir.fill_cache"(%129, %130) <{batch_offset = 21 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %132 = "ttir.slice_static"(%87) <{begins = [22 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [23 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %133 = "ttir.fill_cache"(%131, %132) <{batch_offset = 22 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %134 = "ttir.slice_static"(%87) <{begins = [23 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [24 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %135 = "ttir.fill_cache"(%133, %134) <{batch_offset = 23 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %136 = "ttir.slice_static"(%87) <{begins = [24 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [25 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %137 = "ttir.fill_cache"(%135, %136) <{batch_offset = 24 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %138 = "ttir.slice_static"(%87) <{begins = [25 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [26 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %139 = "ttir.fill_cache"(%137, %138) <{batch_offset = 25 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %140 = "ttir.slice_static"(%87) <{begins = [26 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [27 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %141 = "ttir.fill_cache"(%139, %140) <{batch_offset = 26 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %142 = "ttir.slice_static"(%87) <{begins = [27 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [28 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %143 = "ttir.fill_cache"(%141, %142) <{batch_offset = 27 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %144 = "ttir.slice_static"(%87) <{begins = [28 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [29 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %145 = "ttir.fill_cache"(%143, %144) <{batch_offset = 28 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %146 = "ttir.slice_static"(%87) <{begins = [29 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [30 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %147 = "ttir.fill_cache"(%145, %146) <{batch_offset = 29 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %148 = "ttir.slice_static"(%87) <{begins = [30 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [31 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %149 = "ttir.fill_cache"(%147, %148) <{batch_offset = 30 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %150 = "ttir.slice_static"(%87) <{begins = [31 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %151 = "ttir.fill_cache"(%149, %150) <{batch_offset = 31 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %152 = "ttir.reshape"(%v_weight) <{shape = [1 : i32, 1024 : i32, 4096 : i32]}> : (tensor<1024x4096xbf16>) -> tensor<1x1024x4096xbf16>
        %153 = "ttir.reshape"(%152) <{shape = [1024 : i32, 4096 : i32]}> : (tensor<1x1024x4096xbf16>) -> tensor<1024x4096xbf16>
        %154 = "ttir.permute"(%153) <{permutation = array<i64: 1, 0>}> : (tensor<1024x4096xbf16>) -> tensor<4096x1024xbf16>
        %155 = "ttir.dot_general"(%57, %154) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x4096xbf16>, tensor<4096x1024xbf16>) -> tensor<576x1024xbf16>
        %156 = "ttir.reshape"(%155) <{shape = [32 : i32, 18 : i32, 8 : i32, 128 : i32]}> : (tensor<576x1024xbf16>) -> tensor<32x18x8x128xbf16>
        %157 = "ttir.permute"(%156) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x18x8x128xbf16>) -> tensor<32x8x18x128xbf16>
        %158 = "ttir.slice_static"(%157) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %159 = "ttir.fill_cache"(%value_cache, %158) <{batch_offset = 0 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %160 = "ttir.slice_static"(%157) <{begins = [1 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %161 = "ttir.fill_cache"(%159, %160) <{batch_offset = 1 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %162 = "ttir.slice_static"(%157) <{begins = [2 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [3 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %163 = "ttir.fill_cache"(%161, %162) <{batch_offset = 2 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %164 = "ttir.slice_static"(%157) <{begins = [3 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [4 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %165 = "ttir.fill_cache"(%163, %164) <{batch_offset = 3 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %166 = "ttir.slice_static"(%157) <{begins = [4 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [5 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %167 = "ttir.fill_cache"(%165, %166) <{batch_offset = 4 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %168 = "ttir.slice_static"(%157) <{begins = [5 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [6 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %169 = "ttir.fill_cache"(%167, %168) <{batch_offset = 5 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %170 = "ttir.slice_static"(%157) <{begins = [6 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [7 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %171 = "ttir.fill_cache"(%169, %170) <{batch_offset = 6 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %172 = "ttir.slice_static"(%157) <{begins = [7 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [8 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %173 = "ttir.fill_cache"(%171, %172) <{batch_offset = 7 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %174 = "ttir.slice_static"(%157) <{begins = [8 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [9 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %175 = "ttir.fill_cache"(%173, %174) <{batch_offset = 8 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %176 = "ttir.slice_static"(%157) <{begins = [9 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [10 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %177 = "ttir.fill_cache"(%175, %176) <{batch_offset = 9 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %178 = "ttir.slice_static"(%157) <{begins = [10 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [11 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %179 = "ttir.fill_cache"(%177, %178) <{batch_offset = 10 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %180 = "ttir.slice_static"(%157) <{begins = [11 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [12 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %181 = "ttir.fill_cache"(%179, %180) <{batch_offset = 11 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %182 = "ttir.slice_static"(%157) <{begins = [12 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [13 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %183 = "ttir.fill_cache"(%181, %182) <{batch_offset = 12 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %184 = "ttir.slice_static"(%157) <{begins = [13 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [14 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %185 = "ttir.fill_cache"(%183, %184) <{batch_offset = 13 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %186 = "ttir.slice_static"(%157) <{begins = [14 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [15 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %187 = "ttir.fill_cache"(%185, %186) <{batch_offset = 14 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %188 = "ttir.slice_static"(%157) <{begins = [15 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [16 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %189 = "ttir.fill_cache"(%187, %188) <{batch_offset = 15 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %190 = "ttir.slice_static"(%157) <{begins = [16 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [17 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %191 = "ttir.fill_cache"(%189, %190) <{batch_offset = 16 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %192 = "ttir.slice_static"(%157) <{begins = [17 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [18 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %193 = "ttir.fill_cache"(%191, %192) <{batch_offset = 17 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %194 = "ttir.slice_static"(%157) <{begins = [18 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [19 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %195 = "ttir.fill_cache"(%193, %194) <{batch_offset = 18 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %196 = "ttir.slice_static"(%157) <{begins = [19 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [20 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %197 = "ttir.fill_cache"(%195, %196) <{batch_offset = 19 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %198 = "ttir.slice_static"(%157) <{begins = [20 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [21 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %199 = "ttir.fill_cache"(%197, %198) <{batch_offset = 20 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %200 = "ttir.slice_static"(%157) <{begins = [21 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [22 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %201 = "ttir.fill_cache"(%199, %200) <{batch_offset = 21 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %202 = "ttir.slice_static"(%157) <{begins = [22 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [23 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %203 = "ttir.fill_cache"(%201, %202) <{batch_offset = 22 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %204 = "ttir.slice_static"(%157) <{begins = [23 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [24 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %205 = "ttir.fill_cache"(%203, %204) <{batch_offset = 23 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %206 = "ttir.slice_static"(%157) <{begins = [24 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [25 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %207 = "ttir.fill_cache"(%205, %206) <{batch_offset = 24 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %208 = "ttir.slice_static"(%157) <{begins = [25 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [26 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %209 = "ttir.fill_cache"(%207, %208) <{batch_offset = 25 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %210 = "ttir.slice_static"(%157) <{begins = [26 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [27 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %211 = "ttir.fill_cache"(%209, %210) <{batch_offset = 26 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %212 = "ttir.slice_static"(%157) <{begins = [27 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [28 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %213 = "ttir.fill_cache"(%211, %212) <{batch_offset = 27 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %214 = "ttir.slice_static"(%157) <{begins = [28 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [29 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %215 = "ttir.fill_cache"(%213, %214) <{batch_offset = 28 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %216 = "ttir.slice_static"(%157) <{begins = [29 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [30 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %217 = "ttir.fill_cache"(%215, %216) <{batch_offset = 29 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %218 = "ttir.slice_static"(%157) <{begins = [30 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [31 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %219 = "ttir.fill_cache"(%217, %218) <{batch_offset = 30 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %220 = "ttir.slice_static"(%157) <{begins = [31 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 8 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x8x18x128xbf16>) -> tensor<1x8x18x128xbf16>
        %221 = "ttir.fill_cache"(%219, %220) <{batch_offset = 31 : i32}> : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>) -> tensor<32x8x128x128xbf16>
        %226 = "ttir.reshape"(%q_weight) <{shape = [1 : i32, 4096 : i32, 4096 : i32]}> : (tensor<4096x4096xbf16>) -> tensor<1x4096x4096xbf16>
        %227 = "ttir.reshape"(%226) <{shape = [4096 : i32, 4096 : i32]}> : (tensor<1x4096x4096xbf16>) -> tensor<4096x4096xbf16>
        %228 = "ttir.permute"(%227) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
        %229 = "ttir.dot_general"(%57, %228) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<576x4096xbf16>
        %230 = "ttir.reshape"(%229) <{shape = [32 : i32, 18 : i32, 32 : i32, 128 : i32]}> : (tensor<576x4096xbf16>) -> tensor<32x18x32x128xbf16>
        %231 = "ttir.permute"(%230) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x18x32x128xbf16>) -> tensor<32x32x18x128xbf16>
        %232 = "ttir.broadcast"(%74) <{broadcast_dimensions = array<i64: 32, 32, 1, 1>}> : (tensor<1x1x18x128xbf16>) -> tensor<32x32x18x128xbf16>
        %233 = "ttir.multiply"(%231, %232) : (tensor<32x32x18x128xbf16>, tensor<32x32x18x128xbf16>) -> tensor<32x32x18x128xbf16>
        %234 = "ttir.slice_static"(%231) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [32 : i32, 32 : i32, 18 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x18x128xbf16>) -> tensor<32x32x18x64xbf16>
        %235 = "ttir.neg"(%234) : (tensor<32x32x18x64xbf16>) -> tensor<32x32x18x64xbf16>
        %236 = "ttir.slice_static"(%231) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [32 : i32, 32 : i32, 18 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<32x32x18x128xbf16>) -> tensor<32x32x18x64xbf16>
        %237 = "ttir.concat"(%235, %236) <{dim = 3 : si32}> : (tensor<32x32x18x64xbf16>, tensor<32x32x18x64xbf16>) -> tensor<32x32x18x128xbf16>
        %238 = "ttir.broadcast"(%84) <{broadcast_dimensions = array<i64: 32, 32, 1, 1>}> : (tensor<1x1x18x128xbf16>) -> tensor<32x32x18x128xbf16>
        %239 = "ttir.multiply"(%237, %238) : (tensor<32x32x18x128xbf16>, tensor<32x32x18x128xbf16>) -> tensor<32x32x18x128xbf16>
        %240 = "ttir.add"(%233, %239) : (tensor<32x32x18x128xbf16>, tensor<32x32x18x128xbf16>) -> tensor<32x32x18x128xbf16>
        %241 = "ttir.typecast"(%240) <{conservative_folding = false}> : (tensor<32x32x18x128xbf16>) -> tensor<32x32x18x128xf32>
        %242 = "ttir.multiply"(%241, %21) : (tensor<32x32x18x128xf32>, tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>
        %243 = "ttir.reshape"(%151) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 128 : i32]}> : (tensor<32x8x128x128xbf16>) -> tensor<32x8x1x128x128xbf16>
        %244 = "ttir.broadcast"(%243) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x128x128xbf16>) -> tensor<32x8x4x128x128xbf16>
        %245 = "ttir.reshape"(%244) <{shape = [32 : i32, 32 : i32, 128 : i32, 128 : i32]}> : (tensor<32x8x4x128x128xbf16>) -> tensor<32x32x128x128xbf16>
        %246 = "ttir.typecast"(%245) <{conservative_folding = false}> : (tensor<32x32x128x128xbf16>) -> tensor<32x32x128x128xf32>
        %247 = "ttir.permute"(%246) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<32x32x128x128xf32>) -> tensor<32x32x128x128xf32>
        %248 = "ttir.multiply"(%247, %20) : (tensor<32x32x128x128xf32>, tensor<32x32x128x128xf32>) -> tensor<32x32x128x128xf32>
        %249 = "ttir.dot_general"(%242, %248) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<32x32x18x128xf32>, tensor<32x32x128x128xf32>) -> tensor<32x32x18x128xf32>
        %250 = "ttir.reshape"(%0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xi64>) -> tensor<1x128xi64>
        %251 = "ttir.broadcast"(%250) <{broadcast_dimensions = array<i64: 18, 1>}> : (tensor<1x128xi64>) -> tensor<18x128xi64>
        %252 = "ttir.reshape"(%29) <{shape = [18 : i32, 1 : i32]}> : (tensor<18xi64>) -> tensor<18x1xi64>
        %253 = "ttir.broadcast"(%252) <{broadcast_dimensions = array<i64: 1, 128>}> : (tensor<18x1xi64>) -> tensor<18x128xi64>
        %254 = "ttir.le"(%251, %253) : (tensor<18x128xi64>, tensor<18x128xi64>) -> tensor<18x128xi1>
        %255 = "ttir.reshape"(%254) <{shape = [1 : i32, 18 : i32, 128 : i32]}> : (tensor<18x128xi1>) -> tensor<1x18x128xi1>
        %256 = "ttir.reshape"(%255) <{shape = [1 : i32, 1 : i32, 18 : i32, 128 : i32]}> : (tensor<1x18x128xi1>) -> tensor<1x1x18x128xi1>
        %257 = "ttir.broadcast"(%256) <{broadcast_dimensions = array<i64: 32, 1, 1, 1>}> : (tensor<1x1x18x128xi1>) -> tensor<32x1x18x128xi1>
        %258 = "ttir.where"(%257, %18, %16) : (tensor<32x1x18x128xi1>, tensor<32x1x18x128xbf16>, tensor<32x1x18x128xbf16>) -> tensor<32x1x18x128xbf16>
        %259 = "ttir.typecast"(%258) <{conservative_folding = false}> : (tensor<32x1x18x128xbf16>) -> tensor<32x1x18x128xf32>
        %260 = "ttir.reshape"(%259) <{shape = [32 : i32, 18 : i32, 128 : i32]}> : (tensor<32x1x18x128xf32>) -> tensor<32x18x128xf32>
        %261 = "ttir.reshape"(%260) <{shape = [32 : i32, 1 : i32, 18 : i32, 128 : i32]}> : (tensor<32x18x128xf32>) -> tensor<32x1x18x128xf32>
        %262 = "ttir.broadcast"(%261) <{broadcast_dimensions = array<i64: 1, 32, 1, 1>}> : (tensor<32x1x18x128xf32>) -> tensor<32x32x18x128xf32>
        %263 = "ttir.add"(%249, %262) : (tensor<32x32x18x128xf32>, tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>
        %264 = "ttir.typecast"(%263) <{conservative_folding = false}> : (tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf64>
        %265 = "ttir.eq"(%264, %14) : (tensor<32x32x18x128xf64>, tensor<32x32x18x128xf64>) -> tensor<32x32x18x128xi1>
        %266 = "ttir.logical_not"(%265) : (tensor<32x32x18x128xi1>) -> tensor<32x32x18x128xi1>
        %267 = "ttir.reduce_or"(%266) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x32x18x128xi1>) -> tensor<32x32x18xi1>
        %268 = "ttir.reshape"(%267) <{shape = [32 : i32, 32 : i32, 18 : i32, 1 : i32]}> : (tensor<32x32x18xi1>) -> tensor<32x32x18x1xi1>
        %269 = "ttir.logical_not"(%268) : (tensor<32x32x18x1xi1>) -> tensor<32x32x18x1xi1>
        %270 = "ttir.reshape"(%269) <{shape = [32 : i32, 32 : i32, 18 : i32]}> : (tensor<32x32x18x1xi1>) -> tensor<32x32x18xi1>
        %271 = "ttir.reshape"(%270) <{shape = [32 : i32, 32 : i32, 18 : i32, 1 : i32]}> : (tensor<32x32x18xi1>) -> tensor<32x32x18x1xi1>
        %272 = "ttir.broadcast"(%271) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x18x1xi1>) -> tensor<32x32x18x128xi1>
        %273 = "ttir.max"(%263) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x32x18x128xf32>) -> tensor<32x32x18xf32>
        %274 = "ttir.reshape"(%273) <{shape = [32 : i32, 32 : i32, 18 : i32, 1 : i32]}> : (tensor<32x32x18xf32>) -> tensor<32x32x18x1xf32>
        %275 = "ttir.broadcast"(%274) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x18x1xf32>) -> tensor<32x32x18x128xf32>
        %276 = "ttir.subtract"(%263, %275) : (tensor<32x32x18x128xf32>, tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>
        %277 = "ttir.exp"(%276) : (tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>
        %278 = "ttir.sum"(%277) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<32x32x18x128xf32>) -> tensor<32x32x18xf32>
        %279 = "ttir.reshape"(%278) <{shape = [32 : i32, 32 : i32, 18 : i32, 1 : i32]}> : (tensor<32x32x18xf32>) -> tensor<32x32x18x1xf32>
        %280 = "ttir.broadcast"(%279) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<32x32x18x1xf32>) -> tensor<32x32x18x128xf32>
        %281 = "ttir.div"(%277, %280) : (tensor<32x32x18x128xf32>, tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>
        %282 = "ttir.where"(%272, %12, %281) : (tensor<32x32x18x128xi1>, tensor<32x32x18x128xf32>, tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xf32>
        %283 = "ttir.reshape"(%221) <{shape = [32 : i32, 8 : i32, 1 : i32, 128 : i32, 128 : i32]}> : (tensor<32x8x128x128xbf16>) -> tensor<32x8x1x128x128xbf16>
        %284 = "ttir.broadcast"(%283) <{broadcast_dimensions = array<i64: 1, 1, 4, 1, 1>}> : (tensor<32x8x1x128x128xbf16>) -> tensor<32x8x4x128x128xbf16>
        %285 = "ttir.reshape"(%284) <{shape = [32 : i32, 32 : i32, 128 : i32, 128 : i32]}> : (tensor<32x8x4x128x128xbf16>) -> tensor<32x32x128x128xbf16>
        %286 = "ttir.typecast"(%285) <{conservative_folding = false}> : (tensor<32x32x128x128xbf16>) -> tensor<32x32x128x128xf32>
        %287 = "ttir.dot_general"(%282, %286) <{batch_dims_lhs = array<i64: 0, 1>, batch_dims_rhs = array<i64: 0, 1>, contract_dims_lhs = array<i64: 3>, contract_dims_rhs = array<i64: 2>}> : (tensor<32x32x18x128xf32>, tensor<32x32x128x128xf32>) -> tensor<32x32x18x128xf32>
        %288 = "ttir.typecast"(%287) <{conservative_folding = false}> : (tensor<32x32x18x128xf32>) -> tensor<32x32x18x128xbf16>
        %289 = "ttir.permute"(%288) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<32x32x18x128xbf16>) -> tensor<32x18x32x128xbf16>
        %290 = "ttir.reshape"(%289) <{shape = [576 : i32, 4096 : i32]}> : (tensor<32x18x32x128xbf16>) -> tensor<576x4096xbf16>
        %291 = "ttir.reshape"(%o_weight) <{shape = [1 : i32, 4096 : i32, 4096 : i32]}> : (tensor<4096x4096xbf16>) -> tensor<1x4096x4096xbf16>
        %292 = "ttir.reshape"(%291) <{shape = [4096 : i32, 4096 : i32]}> : (tensor<1x4096x4096xbf16>) -> tensor<4096x4096xbf16>
        %293 = "ttir.permute"(%292) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
        %294 = "ttir.dot_general"(%290, %293) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<576x4096xbf16>
        %295 = "ttir.reshape"(%294) <{shape = [32 : i32, 18 : i32, 4096 : i32]}> : (tensor<576x4096xbf16>) -> tensor<32x18x4096xbf16>
        %296 = "ttir.add"(%hidden, %295) : (tensor<32x18x4096xbf16>, tensor<32x18x4096xbf16>) -> tensor<32x18x4096xbf16>
        %297 = "ttir.reshape"(%post_gamma) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
        %298 = "ttir.reshape"(%297) <{shape = [4096 : i32]}> : (tensor<1x1x4096xbf16>) -> tensor<4096xbf16>
        %299 = "ttir.reshape"(%298) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
        %300 = "ttir.broadcast"(%299) <{broadcast_dimensions = array<i64: 32, 18, 1>}> : (tensor<1x1x4096xbf16>) -> tensor<32x18x4096xbf16>
        %301 = "ttir.typecast"(%296) <{conservative_folding = false}> : (tensor<32x18x4096xbf16>) -> tensor<32x18x4096xf32>
        %302 = "ttir.pow"(%301, %27) : (tensor<32x18x4096xf32>, tensor<32x18x4096xf32>) -> tensor<32x18x4096xf32>
        %303 = "ttir.sum"(%302) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x18x4096xf32>) -> tensor<32x18xf32>
        %304 = "ttir.multiply"(%303, %25) : (tensor<32x18xf32>, tensor<32x18xf32>) -> tensor<32x18xf32>
        %305 = "ttir.reshape"(%304) <{shape = [32 : i32, 18 : i32, 1 : i32]}> : (tensor<32x18xf32>) -> tensor<32x18x1xf32>
        %306 = "ttir.add"(%305, %23) : (tensor<32x18x1xf32>, tensor<32x18x1xf32>) -> tensor<32x18x1xf32>
        %307 = "ttir.rsqrt"(%306) : (tensor<32x18x1xf32>) -> tensor<32x18x1xf32>
        %308 = "ttir.reshape"(%307) <{shape = [32 : i32, 18 : i32]}> : (tensor<32x18x1xf32>) -> tensor<32x18xf32>
        %309 = "ttir.reshape"(%308) <{shape = [32 : i32, 18 : i32, 1 : i32]}> : (tensor<32x18xf32>) -> tensor<32x18x1xf32>
        %310 = "ttir.broadcast"(%309) <{broadcast_dimensions = array<i64: 1, 1, 4096>}> : (tensor<32x18x1xf32>) -> tensor<32x18x4096xf32>
        %311 = "ttir.multiply"(%301, %310) : (tensor<32x18x4096xf32>, tensor<32x18x4096xf32>) -> tensor<32x18x4096xf32>
        %312 = "ttir.typecast"(%311) <{conservative_folding = false}> : (tensor<32x18x4096xf32>) -> tensor<32x18x4096xbf16>
        %313 = "ttir.multiply"(%300, %312) : (tensor<32x18x4096xbf16>, tensor<32x18x4096xbf16>) -> tensor<32x18x4096xbf16>
        %314 = "ttir.reshape"(%313) <{shape = [576 : i32, 4096 : i32]}> : (tensor<32x18x4096xbf16>) -> tensor<576x4096xbf16>
        %315 = "ttir.reshape"(%gate_weight) <{shape = [1 : i32, 14336 : i32, 4096 : i32]}> : (tensor<14336x4096xbf16>) -> tensor<1x14336x4096xbf16>
        %316 = "ttir.reshape"(%315) <{shape = [14336 : i32, 4096 : i32]}> : (tensor<1x14336x4096xbf16>) -> tensor<14336x4096xbf16>
        %317 = "ttir.permute"(%316) <{permutation = array<i64: 1, 0>}> : (tensor<14336x4096xbf16>) -> tensor<4096x14336xbf16>
        %318 = "ttir.dot_general"(%314, %317) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x4096xbf16>, tensor<4096x14336xbf16>) -> tensor<576x14336xbf16>
        %319 = "ttir.reshape"(%318) <{shape = [32 : i32, 18 : i32, 14336 : i32]}> : (tensor<576x14336xbf16>) -> tensor<32x18x14336xbf16>
        %320 = "ttir.typecast"(%319) <{conservative_folding = false}> : (tensor<32x18x14336xbf16>) -> tensor<32x18x14336xf32>
        %321 = "ttir.sigmoid"(%320) : (tensor<32x18x14336xf32>) -> tensor<32x18x14336xf32>
        %322 = "ttir.multiply"(%320, %321) : (tensor<32x18x14336xf32>, tensor<32x18x14336xf32>) -> tensor<32x18x14336xf32>
        %323 = "ttir.typecast"(%322) <{conservative_folding = false}> : (tensor<32x18x14336xf32>) -> tensor<32x18x14336xbf16>
        %324 = "ttir.reshape"(%up_weight) <{shape = [1 : i32, 14336 : i32, 4096 : i32]}> : (tensor<14336x4096xbf16>) -> tensor<1x14336x4096xbf16>
        %325 = "ttir.reshape"(%324) <{shape = [14336 : i32, 4096 : i32]}> : (tensor<1x14336x4096xbf16>) -> tensor<14336x4096xbf16>
        %326 = "ttir.permute"(%325) <{permutation = array<i64: 1, 0>}> : (tensor<14336x4096xbf16>) -> tensor<4096x14336xbf16>
        %327 = "ttir.dot_general"(%314, %326) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x4096xbf16>, tensor<4096x14336xbf16>) -> tensor<576x14336xbf16>
        %328 = "ttir.reshape"(%327) <{shape = [32 : i32, 18 : i32, 14336 : i32]}> : (tensor<576x14336xbf16>) -> tensor<32x18x14336xbf16>
        %329 = "ttir.multiply"(%323, %328) : (tensor<32x18x14336xbf16>, tensor<32x18x14336xbf16>) -> tensor<32x18x14336xbf16>
        %330 = "ttir.reshape"(%329) <{shape = [576 : i32, 14336 : i32]}> : (tensor<32x18x14336xbf16>) -> tensor<576x14336xbf16>
        %331 = "ttir.reshape"(%down_weight) <{shape = [1 : i32, 4096 : i32, 14336 : i32]}> : (tensor<4096x14336xbf16>) -> tensor<1x4096x14336xbf16>
        %332 = "ttir.reshape"(%331) <{shape = [4096 : i32, 14336 : i32]}> : (tensor<1x4096x14336xbf16>) -> tensor<4096x14336xbf16>
        %333 = "ttir.permute"(%332) <{permutation = array<i64: 1, 0>}> : (tensor<4096x14336xbf16>) -> tensor<14336x4096xbf16>
        %334 = "ttir.dot_general"(%330, %333) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<576x14336xbf16>, tensor<14336x4096xbf16>) -> tensor<576x4096xbf16>
        %335 = "ttir.reshape"(%334) <{shape = [32 : i32, 18 : i32, 4096 : i32]}> : (tensor<576x4096xbf16>) -> tensor<32x18x4096xbf16>
        %336 = "ttir.add"(%296, %335) : (tensor<32x18x4096xbf16>, tensor<32x18x4096xbf16>) -> tensor<32x18x4096xbf16>
        return %151, %221, %336 : tensor<32x8x128x128xbf16>, tensor<32x8x128x128xbf16>, tensor<32x18x4096xbf16>
      }
    }
  }
}
