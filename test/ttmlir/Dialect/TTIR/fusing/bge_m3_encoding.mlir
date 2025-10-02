// RUN: ttmlir-opt -ttir-fusing -ttir-to-ttir-decomposition -ttir-fusing %s

// module {
// func.func @transformer_attention_block(
//     %input: tensor<2x34x1024xf16>,
//     %prev_result_1: tensor<2x34x1024xf16>,
//     %prev_result_2: tensor<2x34x1024xf16>,
//     %weight_q: tensor<1024x1024xf16>,
//     %bias_q: tensor<1024xf16>,
//     %weight_k: tensor<1024x1024xf16>,
//     %bias_k: tensor<1024xf16>,
//     %weight_v: tensor<1024x1024xf16>,
//     %bias_v: tensor<1024xf16>,
//     %scale_factor: tensor<f16>,
//     %attention_mask: tensor<2x34xi64>,
//     %mask_value: tensor<f32>,
//     %constant_1: tensor<2x1x1x34xf64>,
//     %weight_out: tensor<1024x1024xf16>,
//     %bias_out: tensor<1024xf16>,
//     %norm_scale: tensor<1024xf16>,
//     %norm_reciprocal: tensor<2x34xf16>,
//     %epsilon: tensor<2x34x1xf16>
// ) -> (tensor<2x16x34x64xf16>, tensor<2x16x64x34xf16>, tensor<2x16x34x64xf16>) {

//     // Add residual connections
//     %0 = ttir.empty() : tensor<2x34x1024xf16>
//     %1 = "ttir.add"(%input, %prev_result_1, %0) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
//     %2 = ttir.empty() : tensor<2x34x1024xf16>
//     %3 = "ttir.add"(%1, %prev_result_2, %2) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Reshape for matrix multiplication
//     %4 = ttir.empty() : tensor<68x1024xf16>
//     %5 = "ttir.reshape"(%3, %4) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x1024xf16>, tensor<68x1024xf16>) -> tensor<68x1024xf16>

//     // Query projection
//     %6 = ttir.empty() : tensor<1x1024x1024xf16>
//     %7 = "ttir.reshape"(%weight_q, %6) <{shape = [1 : i32, 1024 : i32, 1024 : i32]}> : (tensor<1024x1024xf16>, tensor<1x1024x1024xf16>) -> tensor<1x1024x1024xf16>
//     %8 = ttir.empty() : tensor<1024x1024xf16>
//     %9 = "ttir.reshape"(%7, %8) <{shape = [1024 : i32, 1024 : i32]}> : (tensor<1x1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
//     %10 = ttir.empty() : tensor<1024x1024xf16>
//     %11 = "ttir.permute"(%9, %10) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
//     %12 = "ttir.dot_general"(%5, %11) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf16>, tensor<1024x1024xf16>) -> tensor<68x1024xf16>
//     %13 = ttir.empty() : tensor<2x34x1024xf16>
//     %14 = "ttir.reshape"(%12, %13) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Add query bias
//     %15 = ttir.empty() : tensor<1x1x1024xf16>
//     %16 = "ttir.reshape"(%bias_q, %15) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
//     %17 = ttir.empty() : tensor<1024xf16>
//     %18 = "ttir.reshape"(%16, %17) <{shape = [1024 : i32]}> : (tensor<1x1x1024xf16>, tensor<1024xf16>) -> tensor<1024xf16>
//     %19 = ttir.empty() : tensor<1x1x1024xf16>
//     %20 = "ttir.reshape"(%18, %19) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
//     %21 = ttir.empty() : tensor<2x34x1024xf16>
//     %22 = "ttir.broadcast"(%20, %21) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
//     %23 = ttir.empty() : tensor<2x34x1024xf16>
//     %24 = "ttir.add"(%14, %22, %23) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Reshape to multi-head format (Query)
//     %25 = ttir.empty() : tensor<2x34x16x64xf16>
//     %26 = "ttir.reshape"(%24, %25) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<2x34x1024xf16>, tensor<2x34x16x64xf16>) -> tensor<2x34x16x64xf16>
//     %27 = ttir.empty() : tensor<2x16x34x64xf16>
//     %28 = "ttir.permute"(%26, %27) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x34x16x64xf16>, tensor<2x16x34x64xf16>) -> tensor<2x16x34x64xf16>
//     %29 = ttir.empty() : tensor<32x34x64xf16>
//     %30 = "ttir.reshape"(%28, %29) <{shape = [32 : i32, 34 : i32, 64 : i32]}> : (tensor<2x16x34x64xf16>, tensor<32x34x64xf16>) -> tensor<32x34x64xf16>

//     // Key projection
//     %31 = ttir.empty() : tensor<1x1024x1024xf16>
//     %32 = "ttir.reshape"(%weight_k, %31) <{shape = [1 : i32, 1024 : i32, 1024 : i32]}> : (tensor<1024x1024xf16>, tensor<1x1024x1024xf16>) -> tensor<1x1024x1024xf16>
//     %33 = ttir.empty() : tensor<1024x1024xf16>
//     %34 = "ttir.reshape"(%32, %33) <{shape = [1024 : i32, 1024 : i32]}> : (tensor<1x1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
//     %35 = ttir.empty() : tensor<1024x1024xf16>
//     %36 = "ttir.permute"(%34, %35) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
//     %37 = "ttir.dot_general"(%5, %36) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf16>, tensor<1024x1024xf16>) -> tensor<68x1024xf16>
//     %38 = ttir.empty() : tensor<2x34x1024xf16>
//     %39 = "ttir.reshape"(%37, %38) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Add key bias
//     %40 = ttir.empty() : tensor<1x1x1024xf16>
//     %41 = "ttir.reshape"(%bias_k, %40) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
//     %42 = ttir.empty() : tensor<1024xf16>
//     %43 = "ttir.reshape"(%41, %42) <{shape = [1024 : i32]}> : (tensor<1x1x1024xf16>, tensor<1024xf16>) -> tensor<1024xf16>
//     %44 = ttir.empty() : tensor<1x1x1024xf16>
//     %45 = "ttir.reshape"(%43, %44) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
//     %46 = ttir.empty() : tensor<2x34x1024xf16>
//     %47 = "ttir.broadcast"(%45, %46) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
//     %48 = ttir.empty() : tensor<2x34x1024xf16>
//     %49 = "ttir.add"(%39, %47, %48) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Reshape to multi-head format (Key)
//     %50 = ttir.empty() : tensor<2x34x16x64xf16>
//     %51 = "ttir.reshape"(%49, %50) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<2x34x1024xf16>, tensor<2x34x16x64xf16>) -> tensor<2x34x16x64xf16>
//     %52 = ttir.empty() : tensor<2x16x64x34xf16>
//     %53 = "ttir.permute"(%51, %52) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<2x34x16x64xf16>, tensor<2x16x64x34xf16>) -> tensor<2x16x64x34xf16>
//     %54 = ttir.empty() : tensor<32x64x34xf16>
//     %55 = "ttir.reshape"(%53, %54) <{shape = [32 : i32, 64 : i32, 34 : i32]}> : (tensor<2x16x64x34xf16>, tensor<32x64x34xf16>) -> tensor<32x64x34xf16>

//     // Compute attention scores (Q @ K^T)
//     %56 = "ttir.dot_general"(%30, %55) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<32x34x64xf16>, tensor<32x64x34xf16>) -> tensor<32x34x34xf16>
//     %57 = ttir.empty() : tensor<2x16x34x34xf16>
//     %58 = "ttir.reshape"(%56, %57) <{shape = [2 : i32, 16 : i32, 34 : i32, 34 : i32]}> : (tensor<32x34x34xf16>, tensor<2x16x34x34xf16>) -> tensor<2x16x34x34xf16>

//     // Scale attention scores
//     %59 = ttir.empty() : tensor<1x1x1x1xf16>
//     %60 = "ttir.reshape"(%scale_factor, %59) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f16>, tensor<1x1x1x1xf16>) -> tensor<1x1x1x1xf16>
//     %61 = ttir.empty() : tensor<2x16x34x34xf16>
//     %62 = "ttir.broadcast"(%60, %61) <{broadcast_dimensions = array<i64: 2, 16, 34, 34>}> : (tensor<1x1x1x1xf16>, tensor<2x16x34x34xf16>) -> tensor<2x16x34x34xf16>
//     %63 = ttir.empty() : tensor<2x16x34x34xf16>
//     %64 = "ttir.div"(%58, %62, %63) : (tensor<2x16x34x34xf16>, tensor<2x16x34x34xf16>, tensor<2x16x34x34xf16>) -> tensor<2x16x34x34xf16>

//     // Apply attention mask
//     %65 = ttir.empty() : tensor<1x2x34xi64>
//     %66 = "ttir.reshape"(%attention_mask, %65) <{shape = [1 : i32, 2 : i32, 34 : i32]}> : (tensor<2x34xi64>, tensor<1x2x34xi64>) -> tensor<1x2x34xi64>
//     %67 = ttir.empty() : tensor<2x1x1x34xi64>
//     %68 = "ttir.reshape"(%66, %67) <{shape = [2 : i32, 1 : i32, 1 : i32, 34 : i32]}> : (tensor<1x2x34xi64>, tensor<2x1x1x34xi64>) -> tensor<2x1x1x34xi64>
//     %69 = ttir.empty() : tensor<2x1x1x34xf16>
//     %70 = "ttir.typecast"(%68, %69) <{conservative_folding = false}> : (tensor<2x1x1x34xi64>, tensor<2x1x1x34xf16>) -> tensor<2x1x1x34xf16>
//     %71 = ttir.empty() : tensor<2x1x1x34xf64>
//     %72 = "ttir.typecast"(%70, %71) <{conservative_folding = false}> : (tensor<2x1x1x34xf16>, tensor<2x1x1x34xf64>) -> tensor<2x1x1x34xf64>
//     %73 = ttir.empty() : tensor<2x1x1x34xf64>
//     %74 = "ttir.subtract"(%constant_1, %72, %73) : (tensor<2x1x1x34xf64>, tensor<2x1x1x34xf64>, tensor<2x1x1x34xf64>) -> tensor<2x1x1x34xf64>
//     %75 = ttir.empty() : tensor<2x1x1x34xf32>
//     %76 = "ttir.typecast"(%74, %75) <{conservative_folding = false}> : (tensor<2x1x1x34xf64>, tensor<2x1x1x34xf32>) -> tensor<2x1x1x34xf32>
//     %77 = ttir.empty() : tensor<1x1x1x1xf32>
//     %78 = "ttir.reshape"(%mask_value, %77) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
//     %79 = ttir.empty() : tensor<2x1x1x34xf32>
//     %80 = "ttir.broadcast"(%78, %79) <{broadcast_dimensions = array<i64: 2, 1, 1, 34>}> : (tensor<1x1x1x1xf32>, tensor<2x1x1x34xf32>) -> tensor<2x1x1x34xf32>
//     %81 = ttir.empty() : tensor<2x1x1x34xf32>
//     %82 = "ttir.multiply"(%76, %80, %81) : (tensor<2x1x1x34xf32>, tensor<2x1x1x34xf32>, tensor<2x1x1x34xf32>) -> tensor<2x1x1x34xf32>
//     %83 = ttir.empty() : tensor<2x1x1x34xf16>
//     %84 = "ttir.typecast"(%82, %83) <{conservative_folding = false}> : (tensor<2x1x1x34xf32>, tensor<2x1x1x34xf16>) -> tensor<2x1x1x34xf16>
//     %85 = ttir.empty() : tensor<2x34xf16>
//     %86 = "ttir.reshape"(%84, %85) <{shape = [2 : i32, 34 : i32]}> : (tensor<2x1x1x34xf16>, tensor<2x34xf16>) -> tensor<2x34xf16>
//     %87 = ttir.empty() : tensor<2x1x1x34xf16>
//     %88 = "ttir.reshape"(%86, %87) <{shape = [2 : i32, 1 : i32, 1 : i32, 34 : i32]}> : (tensor<2x34xf16>, tensor<2x1x1x34xf16>) -> tensor<2x1x1x34xf16>
//     %89 = ttir.empty() : tensor<2x16x34x34xf16>
//     %90 = "ttir.broadcast"(%88, %89) <{broadcast_dimensions = array<i64: 1, 16, 34, 1>}> : (tensor<2x1x1x34xf16>, tensor<2x16x34x34xf16>) -> tensor<2x16x34x34xf16>
//     %91 = ttir.empty() : tensor<2x16x34x34xf16>
//     %92 = "ttir.add"(%64, %90, %91) : (tensor<2x16x34x34xf16>, tensor<2x16x34x34xf16>, tensor<2x16x34x34xf16>) -> tensor<2x16x34x34xf16>

//     // Softmax computation
//     %93 = ttir.empty() : tensor<2x16x34xf16>
//     %94 = "ttir.max"(%92, %93) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<2x16x34x34xf16>, tensor<2x16x34xf16>) -> tensor<2x16x34xf16>
//     %95 = ttir.empty() : tensor<2x16x34x1xf16>
//     %96 = "ttir.reshape"(%94, %95) <{shape = [2 : i32, 16 : i32, 34 : i32, 1 : i32]}> : (tensor<2x16x34xf16>, tensor<2x16x34x1xf16>) -> tensor<2x16x34x1xf16>
//     %97 = ttir.empty() : tensor<2x16x34x34xf16>
//     %98 = "ttir.broadcast"(%96, %97) <{broadcast_dimensions = array<i64: 1, 1, 1, 34>}> : (tensor<2x16x34x1xf16>, tensor<2x16x34x34xf16>) -> tensor<2x16x34x34xf16>
//     %99 = ttir.empty() : tensor<2x16x34x34xf16>
//     %100 = "ttir.subtract"(%92, %98, %99) : (tensor<2x16x34x34xf16>, tensor<2x16x34x34xf16>, tensor<2x16x34x34xf16>) -> tensor<2x16x34x34xf16>
//     %101 = ttir.empty() : tensor<2x16x34x34xf16>
//     %102 = "ttir.exp"(%100, %101) : (tensor<2x16x34x34xf16>, tensor<2x16x34x34xf16>) -> tensor<2x16x34x34xf16>
//     %103 = ttir.empty() : tensor<2x16x34xf16>
//     %104 = "ttir.sum"(%102, %103) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<2x16x34x34xf16>, tensor<2x16x34xf16>) -> tensor<2x16x34xf16>
//     %105 = ttir.empty() : tensor<2x16x34x1xf16>
//     %106 = "ttir.reshape"(%104, %105) <{shape = [2 : i32, 16 : i32, 34 : i32, 1 : i32]}> : (tensor<2x16x34xf16>, tensor<2x16x34x1xf16>) -> tensor<2x16x34x1xf16>
//     %107 = ttir.empty() : tensor<2x16x34x34xf16>
//     %108 = "ttir.broadcast"(%106, %107) <{broadcast_dimensions = array<i64: 1, 1, 1, 34>}> : (tensor<2x16x34x1xf16>, tensor<2x16x34x34xf16>) -> tensor<2x16x34x34xf16>
//     %109 = ttir.empty() : tensor<2x16x34x34xf16>
//     %110 = "ttir.div"(%102, %108, %109) : (tensor<2x16x34x34xf16>, tensor<2x16x34x34xf16>, tensor<2x16x34x34xf16>) -> tensor<2x16x34x34xf16>
//     %111 = ttir.empty() : tensor<32x34x34xf16>
//     %112 = "ttir.reshape"(%110, %111) <{shape = [32 : i32, 34 : i32, 34 : i32]}> : (tensor<2x16x34x34xf16>, tensor<32x34x34xf16>) -> tensor<32x34x34xf16>

//     // Value projection
//     %113 = ttir.empty() : tensor<1x1024x1024xf16>
//     %114 = "ttir.reshape"(%weight_v, %113) <{shape = [1 : i32, 1024 : i32, 1024 : i32]}> : (tensor<1024x1024xf16>, tensor<1x1024x1024xf16>) -> tensor<1x1024x1024xf16>
//     %115 = ttir.empty() : tensor<1024x1024xf16>
//     %116 = "ttir.reshape"(%114, %115) <{shape = [1024 : i32, 1024 : i32]}> : (tensor<1x1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
//     %117 = ttir.empty() : tensor<1024x1024xf16>
//     %118 = "ttir.permute"(%116, %117) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
//     %119 = "ttir.dot_general"(%5, %118) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf16>, tensor<1024x1024xf16>) -> tensor<68x1024xf16>
//     %120 = ttir.empty() : tensor<2x34x1024xf16>
//     %121 = "ttir.reshape"(%119, %120) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Add value bias
//     %122 = ttir.empty() : tensor<1x1x1024xf16>
//     %123 = "ttir.reshape"(%bias_v, %122) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
//     %124 = ttir.empty() : tensor<1024xf16>
//     %125 = "ttir.reshape"(%123, %124) <{shape = [1024 : i32]}> : (tensor<1x1x1024xf16>, tensor<1024xf16>) -> tensor<1024xf16>
//     %126 = ttir.empty() : tensor<1x1x1024xf16>
//     %127 = "ttir.reshape"(%125, %126) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
//     %128 = ttir.empty() : tensor<2x34x1024xf16>
//     %129 = "ttir.broadcast"(%127, %128) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
//     %130 = ttir.empty() : tensor<2x34x1024xf16>
//     %131 = "ttir.add"(%121, %129, %130) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Reshape to multi-head format (Value)
//     %132 = ttir.empty() : tensor<2x34x16x64xf16>
//     %133 = "ttir.reshape"(%131, %132) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<2x34x1024xf16>, tensor<2x34x16x64xf16>) -> tensor<2x34x16x64xf16>
//     %134 = ttir.empty() : tensor<2x16x34x64xf16>
//     %135 = "ttir.permute"(%133, %134) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x34x16x64xf16>, tensor<2x16x34x64xf16>) -> tensor<2x16x34x64xf16>
//     %136 = ttir.empty() : tensor<32x34x64xf16>
//     %137 = "ttir.reshape"(%135, %136) <{shape = [32 : i32, 34 : i32, 64 : i32]}> : (tensor<2x16x34x64xf16>, tensor<32x34x64xf16>) -> tensor<32x34x64xf16>

//     // Apply attention weights to values (softmax @ V)
//     %138 = "ttir.dot_general"(%112, %137) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<32x34x34xf16>, tensor<32x34x64xf16>) -> tensor<32x34x64xf16>
//     %139 = ttir.empty() : tensor<2x16x34x64xf16>
//     %140 = "ttir.reshape"(%138, %139) <{shape = [2 : i32, 16 : i32, 34 : i32, 64 : i32]}> : (tensor<32x34x64xf16>, tensor<2x16x34x64xf16>) -> tensor<2x16x34x64xf16>
//     %141 = ttir.empty() : tensor<2x34x16x64xf16>
//     %142 = "ttir.permute"(%140, %141) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x16x34x64xf16>, tensor<2x34x16x64xf16>) -> tensor<2x34x16x64xf16>
//     %143 = ttir.empty() : tensor<68x1024xf16>
//     %144 = "ttir.reshape"(%142, %143) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x16x64xf16>, tensor<68x1024xf16>) -> tensor<68x1024xf16>

//     // Output projection
//     %145 = ttir.empty() : tensor<1x1024x1024xf16>
//     %146 = "ttir.reshape"(%weight_out, %145) <{shape = [1 : i32, 1024 : i32, 1024 : i32]}> : (tensor<1024x1024xf16>, tensor<1x1024x1024xf16>) -> tensor<1x1024x1024xf16>
//     %147 = ttir.empty() : tensor<1024x1024xf16>
//     %148 = "ttir.reshape"(%146, %147) <{shape = [1024 : i32, 1024 : i32]}> : (tensor<1x1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
//     %149 = ttir.empty() : tensor<1024x1024xf16>
//     %150 = "ttir.permute"(%148, %149) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
//     %151 = "ttir.dot_general"(%144, %150) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf16>, tensor<1024x1024xf16>) -> tensor<68x1024xf16>
//     %152 = ttir.empty() : tensor<2x34x1024xf16>
//     %153 = "ttir.reshape"(%151, %152) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Add output bias
//     %154 = ttir.empty() : tensor<1x1x1024xf16>
//     %155 = "ttir.reshape"(%bias_out, %154) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
//     %156 = ttir.empty() : tensor<1024xf16>
//     %157 = "ttir.reshape"(%155, %156) <{shape = [1024 : i32]}> : (tensor<1x1x1024xf16>, tensor<1024xf16>) -> tensor<1024xf16>
//     %158 = ttir.empty() : tensor<1x1x1024xf16>
//     %159 = "ttir.reshape"(%157, %158) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
//     %160 = ttir.empty() : tensor<2x34x1024xf16>
//     %161 = "ttir.broadcast"(%159, %160) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
//     %162 = ttir.empty() : tensor<2x34x1024xf16>
//     %163 = "ttir.add"(%153, %161, %162) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Add residual connection
//     %164 = ttir.empty() : tensor<2x34x1024xf16>
//     %165 = "ttir.add"(%163, %3, %164) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Layer normalization - compute mean
//     %166 = ttir.empty() : tensor<2x34xf16>
//     %167 = "ttir.sum"(%165, %166) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<2x34x1024xf16>, tensor<2x34xf16>) -> tensor<2x34xf16>
//     %168 = ttir.empty() : tensor<2x34xf16>
//     %169 = "ttir.multiply"(%167, %norm_reciprocal, %168) : (tensor<2x34xf16>, tensor<2x34xf16>, tensor<2x34xf16>) -> tensor<2x34xf16>
//     %170 = ttir.empty() : tensor<2x34x1xf16>
//     %171 = "ttir.reshape"(%169, %170) <{shape = [2 : i32, 34 : i32, 1 : i32]}> : (tensor<2x34xf16>, tensor<2x34x1xf16>) -> tensor<2x34x1xf16>
//     %172 = ttir.empty() : tensor<2x34x1024xf16>
//     %173 = "ttir.broadcast"(%171, %172) <{broadcast_dimensions = array<i64: 1, 1, 1024>}> : (tensor<2x34x1xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
//     %174 = ttir.empty() : tensor<2x34x1024xf16>
//     %175 = "ttir.subtract"(%165, %173, %174) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Cast for variance computation
//     %176 = ttir.empty() : tensor<2x34x1024xf32>
//     %177 = "ttir.typecast"(%175, %176) <{conservative_folding = false}> : (tensor<2x34x1024xf16>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

//     // Compute variance
//     %178 = ttir.empty() : tensor<2x34x1024xf16>
//     %179 = "ttir.multiply"(%175, %175, %178) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
//     %180 = ttir.empty() : tensor<2x34xf16>
//     %181 = "ttir.sum"(%179, %180) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<2x34x1024xf16>, tensor<2x34xf16>) -> tensor<2x34xf16>
//     %182 = ttir.empty() : tensor<2x34xf16>
//     %183 = "ttir.multiply"(%181, %norm_reciprocal, %182) : (tensor<2x34xf16>, tensor<2x34xf16>, tensor<2x34xf16>) -> tensor<2x34xf16>
//     %184 = ttir.empty() : tensor<2x34x1xf16>
//     %185 = "ttir.reshape"(%183, %184) <{shape = [2 : i32, 34 : i32, 1 : i32]}> : (tensor<2x34xf16>, tensor<2x34x1xf16>) -> tensor<2x34x1xf16>
//     %186 = ttir.empty() : tensor<2x34x1xf16>
//     %187 = "ttir.add"(%185, %epsilon, %186) : (tensor<2x34x1xf16>, tensor<2x34x1xf16>, tensor<2x34x1xf16>) -> tensor<2x34x1xf16>
//     %188 = ttir.empty() : tensor<2x34x1xf16>
//     %189 = "ttir.rsqrt"(%187, %188) : (tensor<2x34x1xf16>, tensor<2x34x1xf16>) -> tensor<2x34x1xf16>

//     // Cast and normalize
//     %190 = ttir.empty() : tensor<2x34x1xf32>
//     %191 = "ttir.typecast"(%189, %190) <{conservative_folding = false}> : (tensor<2x34x1xf16>, tensor<2x34x1xf32>) -> tensor<2x34x1xf32>
//     %192 = ttir.empty() : tensor<2x34xf32>
//     %193 = "ttir.reshape"(%191, %192) <{shape = [2 : i32, 34 : i32]}> : (tensor<2x34x1xf32>, tensor<2x34xf32>) -> tensor<2x34xf32>
//     %194 = ttir.empty() : tensor<2x34x1xf32>
//     %195 = "ttir.reshape"(%193, %194) <{shape = [2 : i32, 34 : i32, 1 : i32]}> : (tensor<2x34xf32>, tensor<2x34x1xf32>) -> tensor<2x34x1xf32>
//     %196 = ttir.empty() : tensor<2x34x1024xf32>
//     %197 = "ttir.broadcast"(%195, %196) <{broadcast_dimensions = array<i64: 1, 1, 1024>}> : (tensor<2x34x1xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
//     %198 = ttir.empty() : tensor<2x34x1024xf32>
//     %199 = "ttir.multiply"(%177, %197, %198) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
//     %200 = ttir.empty() : tensor<2x34x1024xf16>
//     %201 = "ttir.typecast"(%199, %200) <{conservative_folding = false}> : (tensor<2x34x1024xf32>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>

//     // Apply scale
//     %202 = ttir.empty() : tensor<2x34x1024xf32>
//     %203 = "ttir.typecast"(%201, %202) <{conservative_folding = false}> : (tensor<2x34x1024xf16>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
//     %204 = ttir.empty() : tensor<1x1x1024xf16>
//     %205 = "ttir.reshape"(%norm_scale, %204) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
//     %206 = ttir.empty() : tensor<1024xf16>
//     %207 = "ttir.reshape"(%205, %206) <{shape = [1024 : i32]}> : (tensor<1x1x1024xf16>, tensor<1024xf16>) -> tensor<1024xf16>
//     %208 = ttir.empty() : tensor<1024xf32>

//     return %28, %53, %135 : tensor<2x16x34x64xf16>, tensor<2x16x64x34xf16>, tensor<2x16x34x64xf16>
// }
// }

// after matmul + add fusion:
module {
  func.func @transformer_attention_block(%arg0: tensor<2x34x1024xf16>, %arg1: tensor<2x34x1024xf16>, %arg2: tensor<2x34x1024xf16>, %arg3: tensor<1024x1024xf16>, %arg4: tensor<1024xf16>, %arg5: tensor<1024x1024xf16>, %arg6: tensor<1024xf16>, %arg7: tensor<1024x1024xf16>, %arg8: tensor<1024xf16>, %arg9: tensor<f16>, %arg10: tensor<2x34xi64>, %arg11: tensor<f32>, %arg12: tensor<2x1x1x34xf64>, %arg13: tensor<1024x1024xf16>, %arg14: tensor<1024xf16>, %arg15: tensor<1024xf16>, %arg16: tensor<2x34xf16>, %arg17: tensor<2x34x1xf16>) -> (tensor<2x16x34x64xf16>, tensor<2x16x64x34xf16>, tensor<2x16x34x64xf16>) {
    %0 = ttir.empty() : tensor<2x34x1024xf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
    %2 = ttir.empty() : tensor<2x34x1024xf16>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<2x34x1024xf16>, tensor<2x34x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
    %4 = ttir.empty() : tensor<68x1024xf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x1024xf16>, tensor<68x1024xf16>) -> tensor<68x1024xf16>
    %6 = ttir.empty() : tensor<1024x1024xf16>
    %7 = "ttir.permute"(%arg3, %6) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    %8 = ttir.empty() : tensor<1x1x1024xf16>
    %9 = "ttir.reshape"(%arg4, %8) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
    %10 = ttir.empty() : tensor<2x34x1024xf16>
    %11 = "ttir.broadcast"(%9, %10) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
    %12 = ttir.empty() : tensor<68x1024xf16>
    %13 = "ttir.reshape"(%11, %12) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x1024xf16>, tensor<68x1024xf16>) -> tensor<68x1024xf16>
    %14 = ttir.empty() : tensor<68x1024xf16>
    %15 = "ttir.linear"(%5, %7, %13, %14) <{transpose_a = false, transpose_b = false}> : (tensor<68x1024xf16>, tensor<1024x1024xf16>, tensor<68x1024xf16>, tensor<68x1024xf16>) -> tensor<68x1024xf16>
    %16 = ttir.empty() : tensor<2x34x16x64xf16>
    %17 = "ttir.reshape"(%15, %16) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<68x1024xf16>, tensor<2x34x16x64xf16>) -> tensor<2x34x16x64xf16>
    %18 = ttir.empty() : tensor<2x16x34x64xf16>
    %19 = "ttir.permute"(%17, %18) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x34x16x64xf16>, tensor<2x16x34x64xf16>) -> tensor<2x16x34x64xf16>
    %20 = ttir.empty() : tensor<1024x1024xf16>
    %21 = "ttir.permute"(%arg5, %20) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    %22 = ttir.empty() : tensor<1x1x1024xf16>
    %23 = "ttir.reshape"(%arg6, %22) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
    %24 = ttir.empty() : tensor<2x34x1024xf16>
    %25 = "ttir.broadcast"(%23, %24) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
    %26 = ttir.empty() : tensor<68x1024xf16>
    %27 = "ttir.reshape"(%25, %26) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x1024xf16>, tensor<68x1024xf16>) -> tensor<68x1024xf16>
    %28 = ttir.empty() : tensor<68x1024xf16>
    %29 = "ttir.linear"(%5, %21, %27, %28) <{transpose_a = false, transpose_b = false}> : (tensor<68x1024xf16>, tensor<1024x1024xf16>, tensor<68x1024xf16>, tensor<68x1024xf16>) -> tensor<68x1024xf16>
    %30 = ttir.empty() : tensor<2x34x16x64xf16>
    %31 = "ttir.reshape"(%29, %30) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<68x1024xf16>, tensor<2x34x16x64xf16>) -> tensor<2x34x16x64xf16>
    %32 = ttir.empty() : tensor<2x16x64x34xf16>
    %33 = "ttir.permute"(%31, %32) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<2x34x16x64xf16>, tensor<2x16x64x34xf16>) -> tensor<2x16x64x34xf16>
    %34 = ttir.empty() : tensor<1024x1024xf16>
    %35 = "ttir.permute"(%arg7, %34) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf16>, tensor<1024x1024xf16>) -> tensor<1024x1024xf16>
    %36 = ttir.empty() : tensor<1x1x1024xf16>
    %37 = "ttir.reshape"(%arg8, %36) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf16>, tensor<1x1x1024xf16>) -> tensor<1x1x1024xf16>
    %38 = ttir.empty() : tensor<2x34x1024xf16>
    %39 = "ttir.broadcast"(%37, %38) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf16>, tensor<2x34x1024xf16>) -> tensor<2x34x1024xf16>
    %40 = ttir.empty() : tensor<68x1024xf16>
    %41 = "ttir.reshape"(%39, %40) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x1024xf16>, tensor<68x1024xf16>) -> tensor<68x1024xf16>
    %42 = ttir.empty() : tensor<68x1024xf16>
    %43 = "ttir.linear"(%5, %35, %41, %42) <{transpose_a = false, transpose_b = false}> : (tensor<68x1024xf16>, tensor<1024x1024xf16>, tensor<68x1024xf16>, tensor<68x1024xf16>) -> tensor<68x1024xf16>
    %44 = ttir.empty() : tensor<2x34x16x64xf16>
    %45 = "ttir.reshape"(%43, %44) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<68x1024xf16>, tensor<2x34x16x64xf16>) -> tensor<2x34x16x64xf16>
    %46 = ttir.empty() : tensor<2x16x34x64xf16>
    %47 = "ttir.permute"(%45, %46) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x34x16x64xf16>, tensor<2x16x34x64xf16>) -> tensor<2x16x34x64xf16>
    return %19, %33, %47 : tensor<2x16x34x64xf16>, tensor<2x16x64x34xf16>, tensor<2x16x34x64xf16>
  }
}
