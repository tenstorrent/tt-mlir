// RUN: ttmlir-opt -ttir-to-ttir-decomposition -ttir-implicit-broadcast-fold -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test for bge-m3 model attention layer.
module {
  func.func @bge_m3_attention(%arg0: tensor<2x34x1024xf32>,
                              %arg1: tensor<1024x1024xf32>, %arg2: tensor<1024xf32>, // query weight, bias
                              %arg3: tensor<1024x1024xf32>, %arg4: tensor<1024xf32>, // key weight, bias
                              %arg5: tensor<1024x1024xf32>, %arg6: tensor<1024xf32>, // value weight, bias
                              %arg7: tensor<1024x1024xf32>, %arg8: tensor<1024xf32>, // output weight, bias
                              %arg9: tensor<f32>, %arg10: tensor<2x34xi64>) -> (tensor<2x34x1024xf32>) {

    // CHECK: func.func @bge_m3_attention

    // Concatenate Q, K, V weights:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 0 : si32}>
    // CHECK-SAME: (tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<3072x1024xf32>

    // Concatenate Q, K, V biases:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 0 : si32}>
    // CHECK-SAME: (tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>) -> tensor<3072xf32>

    // Linear Q, K, V projections with concatenated bias:
    // CHECK: "ttir.linear"
    // CHECK-SAME:  <{transpose_a = false, transpose_b = true}>
    // CHECK-SAME: (tensor<68x1024xf32>, tensor<3072x1024xf32>, tensor<3072xf32>) -> tensor<68x3072xf32>

    // Check that linear op is reshaped.
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [2 : i32, 34 : i32, 3072 : i32]}>
    // CHECK-SAME: (tensor<68x3072xf32>) -> tensor<2x34x3072xf32>

    // Split Q, K, V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 16 : ui32, transpose_key = true}>
    // CHECK-SAME: (tensor<2x34x3072xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>)

    // Reshape input to 2D for linear operations
    %1 = "ttir.reshape"(%arg0) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x1024xf32>) -> tensor<68x1024xf32>

    // Query projection
    %3 = "ttir.permute"(%arg1) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %4 = "ttir.dot_general"(%1, %3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %6 = "ttir.reshape"(%4) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>) -> tensor<2x34x1024xf32>

    // Add query bias
    %8 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %10 = "ttir.broadcast"(%8) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf32>) -> tensor<2x34x1024xf32>
    %12 = "ttir.add"(%6, %10) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    // Split query heads: [2, 34, 1024] -> [2, 34, 16, 64] -> [2, 16, 34, 64] -> [32, 34, 64]
    %14 = "ttir.reshape"(%12) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<2x34x1024xf32>) -> tensor<2x34x16x64xf32>
    %16 = "ttir.permute"(%14) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x34x16x64xf32>) -> tensor<2x16x34x64xf32>
    %18 = "ttir.reshape"(%16) <{shape = [32 : i32, 34 : i32, 64 : i32]}> : (tensor<2x16x34x64xf32>) -> tensor<32x34x64xf32>

    // Key projection
    %20 = "ttir.permute"(%arg3) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %21 = "ttir.dot_general"(%1, %20) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %23 = "ttir.reshape"(%21) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>) -> tensor<2x34x1024xf32>

    // Add key bias
    %25 = "ttir.reshape"(%arg4) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %27 = "ttir.broadcast"(%25) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf32>) -> tensor<2x34x1024xf32>
    %29 = "ttir.add"(%23, %27) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    // Split key heads: [2, 34, 1024] -> [2, 34, 16, 64] -> [2, 16, 64, 34] -> [32, 64, 34] (transposed for attention)
    %31 = "ttir.reshape"(%29) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<2x34x1024xf32>) -> tensor<2x34x16x64xf32>
    %33 = "ttir.permute"(%31) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<2x34x16x64xf32>) -> tensor<2x16x64x34xf32>
    %35 = "ttir.reshape"(%33) <{shape = [32 : i32, 64 : i32, 34 : i32]}> : (tensor<2x16x64x34xf32>) -> tensor<32x64x34xf32>

    // Attention scores: Q @ K^T
    %36 = "ttir.dot_general"(%18, %35) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<32x34x64xf32>, tensor<32x64x34xf32>) -> tensor<32x34x34xf32>
    %37 = "ttir.reshape"(%36) <{shape = [2 : i32, 16 : i32, 34 : i32, 34 : i32]}> : (tensor<32x34x34xf32>) -> tensor<2x16x34x34xf32>

    // Scale attention scores
    %38 = "ttir.reshape"(%arg9) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1x1xf32>
    %39 = "ttir.broadcast"(%38) <{broadcast_dimensions = array<i64: 2, 16, 34, 34>}> : (tensor<1x1x1x1xf32>) -> tensor<2x16x34x34xf32>
    %40 = "ttir.div"(%37, %39) : (tensor<2x16x34x34xf32>, tensor<2x16x34x34xf32>) -> tensor<2x16x34x34xf32>

    // Apply attention mask (simplified - just add large negative values for masked positions)
    %41 = "ttir.typecast"(%40) <{conservative_folding = false}> : (tensor<2x16x34x34xf32>) -> tensor<2x16x34x34xf64>

    // Softmax: subtract max for numerical stability
    %42 = "ttir.max"(%41) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<2x16x34x34xf64>) -> tensor<2x16x34xf64>
    %43 = "ttir.reshape"(%42) <{shape = [2 : i32, 16 : i32, 34 : i32, 1 : i32]}> : (tensor<2x16x34xf64>) -> tensor<2x16x34x1xf64>
    %44 = "ttir.broadcast"(%43) <{broadcast_dimensions = array<i64: 1, 1, 1, 34>}> : (tensor<2x16x34x1xf64>) -> tensor<2x16x34x34xf64>
    %45 = "ttir.subtract"(%41, %44) : (tensor<2x16x34x34xf64>, tensor<2x16x34x34xf64>) -> tensor<2x16x34x34xf64>

    // Exponentiate
    %46 = "ttir.exp"(%45) : (tensor<2x16x34x34xf64>) -> tensor<2x16x34x34xf64>

    // Sum for normalization
    %47 = "ttir.sum"(%46) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<2x16x34x34xf64>) -> tensor<2x16x34xf64>
    %48 = "ttir.reshape"(%47) <{shape = [2 : i32, 16 : i32, 34 : i32, 1 : i32]}> : (tensor<2x16x34xf64>) -> tensor<2x16x34x1xf64>
    %49 = "ttir.broadcast"(%48) <{broadcast_dimensions = array<i64: 1, 1, 1, 34>}> : (tensor<2x16x34x1xf64>) -> tensor<2x16x34x34xf64>

    // Normalize to get attention weights
    %50 = "ttir.div"(%46, %49) : (tensor<2x16x34x34xf64>, tensor<2x16x34x34xf64>) -> tensor<2x16x34x34xf64>
    %51 = "ttir.typecast"(%50) <{conservative_folding = false}> : (tensor<2x16x34x34xf64>) -> tensor<2x16x34x34xf32>
    %52 = "ttir.reshape"(%51) <{shape = [32 : i32, 34 : i32, 34 : i32]}> : (tensor<2x16x34x34xf32>) -> tensor<32x34x34xf32>

    // Value projection
    %53 = "ttir.permute"(%arg5) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %54 = "ttir.dot_general"(%1, %53) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %55 = "ttir.reshape"(%54) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>) -> tensor<2x34x1024xf32>

    // Add value bias
    %56 = "ttir.reshape"(%arg6) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %57 = "ttir.broadcast"(%56) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf32>) -> tensor<2x34x1024xf32>
    %58 = "ttir.add"(%55, %57) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    // Split value heads: [2, 34, 1024] -> [2, 34, 16, 64] -> [2, 16, 34, 64] -> [32, 34, 64]
    %59 = "ttir.reshape"(%58) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<2x34x1024xf32>) -> tensor<2x34x16x64xf32>
    %60 = "ttir.permute"(%59) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x34x16x64xf32>) -> tensor<2x16x34x64xf32>
    %61 = "ttir.reshape"(%60) <{shape = [32 : i32, 34 : i32, 64 : i32]}> : (tensor<2x16x34x64xf32>) -> tensor<32x34x64xf32>

    // Apply attention: attention_weights @ V
    %62 = "ttir.dot_general"(%52, %61) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<32x34x34xf32>, tensor<32x34x64xf32>) -> tensor<32x34x64xf32>
    %63 = "ttir.reshape"(%62) <{shape = [2 : i32, 16 : i32, 34 : i32, 64 : i32]}> : (tensor<32x34x64xf32>) -> tensor<2x16x34x64xf32>

    // Merge heads: [2, 16, 34, 64] -> [2, 34, 16, 64] -> [2, 34, 1024]
    %64 = "ttir.permute"(%63) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x16x34x64xf32>) -> tensor<2x34x16x64xf32>
    %65 = "ttir.reshape"(%64) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x16x64xf32>) -> tensor<68x1024xf32>

    // Output projection
    %66 = "ttir.permute"(%arg7) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %67 = "ttir.dot_general"(%65, %66) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %68 = "ttir.reshape"(%67) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>) -> tensor<2x34x1024xf32>

    // Add output bias
    %69 = "ttir.reshape"(%arg8) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %70 = "ttir.broadcast"(%69) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf32>) -> tensor<2x34x1024xf32>
    %71 = "ttir.add"(%68, %70) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    return %71 : tensor<2x34x1024xf32>
  }
}

module {
  func.func @bert_attention(%arg0: tensor<1x128x768xbf16>,
                          %arg1: tensor<768x768xbf16>, %arg2: tensor<768xbf16>, // query weight, bias
                          %arg3: tensor<768x768xbf16>, %arg4: tensor<768xbf16>, // key weight, bias
                          %arg5: tensor<768x768xbf16>, %arg6: tensor<768xbf16>, // value weight, bias
                          %arg7: tensor<768x768xbf16>, %arg8: tensor<768xbf16>, // output weight, bias
                          %arg9: tensor<1x128xi64>) -> (tensor<1x128x768xbf16>) {

    // CHECK: func.func @bert_attention

    // Concatenate Q, K, V weights:
    // CHECK: "ttir.concat"
    // CHECK-SAME:  <{dim = 0 : si32}>
    // CHECK-SAME: (tensor<768x768xbf16>, tensor<768x768xbf16>, tensor<768x768xbf16>) -> tensor<2304x768xbf16>

    // Concatenate Q, K, V biases:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 0 : si32}>
    // CHECK-SAME: (tensor<768xbf16>, tensor<768xbf16>, tensor<768xbf16>) -> tensor<2304xbf16>

    // Linear Q, K, V projections with concatenated bias:
    // CHECK: "ttir.linear"
    // CHECK-SAME: <{transpose_a = false, transpose_b = true}>
    // CHECK-SAME: (tensor<128x768xbf16>, tensor<2304x768xbf16>, tensor<2304xbf16>) -> tensor<128x2304xbf16>

    // Check that linear op is reshaped.
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 128 : i32, 2304 : i32]}>
    // CHECK-SAME: (tensor<128x2304xbf16>) -> tensor<1x128x2304xbf16>

    // Split Q, K, V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 12 : ui32, transpose_key = true}>
    // CHECK-SAME: (tensor<1x128x2304xbf16>) -> (tensor<1x12x128x64xbf16>, tensor<1x12x64x128xbf16>, tensor<1x12x128x64xbf16>)

    // Reshape input to 2D for linear operations
    %0 = "ttir.reshape"(%arg0) <{shape = [128 : i32, 768 : i32]}> : (tensor<1x128x768xbf16>) -> tensor<128x768xbf16>

    // Query projection
    %1 = "ttir.permute"(%arg1) <{permutation = array<i64: 1, 0>}> : (tensor<768x768xbf16>) -> tensor<768x768xbf16>
    %2 = "ttir.dot_general"(%0, %1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x768xbf16>, tensor<768x768xbf16>) -> tensor<128x768xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 128 : i32, 768 : i32]}> : (tensor<128x768xbf16>) -> tensor<1x128x768xbf16>

    // Add query bias
    %4 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xbf16>) -> tensor<1x1x768xbf16>
    %5 = "ttir.broadcast"(%4) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x768xbf16>) -> tensor<1x128x768xbf16>
    %6 = "ttir.add"(%3, %5) : (tensor<1x128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    // Split query heads: [1, 128, 768] -> [1, 128, 12, 64] -> [1, 12, 128, 64] -> [12, 128, 64]
    %7 = "ttir.reshape"(%6) <{shape = [1 : i32, 128 : i32, 12 : i32, 64 : i32]}> : (tensor<1x128x768xbf16>) -> tensor<1x128x12x64xbf16>
    %8 = "ttir.permute"(%7) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x12x64xbf16>) -> tensor<1x12x128x64xbf16>
    %9 = "ttir.reshape"(%8) <{shape = [12 : i32, 128 : i32, 64 : i32]}> : (tensor<1x12x128x64xbf16>) -> tensor<12x128x64xbf16>

    // Key projection
    %10 = "ttir.permute"(%arg3) <{permutation = array<i64: 1, 0>}> : (tensor<768x768xbf16>) -> tensor<768x768xbf16>
    %11 = "ttir.dot_general"(%0, %10) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x768xbf16>, tensor<768x768xbf16>) -> tensor<128x768xbf16>
    %12 = "ttir.reshape"(%11) <{shape = [1 : i32, 128 : i32, 768 : i32]}> : (tensor<128x768xbf16>) -> tensor<1x128x768xbf16>

    // Add key bias
    %13 = "ttir.reshape"(%arg4) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xbf16>) -> tensor<1x1x768xbf16>
    %14 = "ttir.broadcast"(%13) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x768xbf16>) -> tensor<1x128x768xbf16>
    %15 = "ttir.add"(%12, %14) : (tensor<1x128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    // Split key heads: [1, 128, 768] -> [1, 128, 12, 64] -> [1, 12, 64, 128] -> [12, 64, 128] (transposed)
    %16 = "ttir.reshape"(%15) <{shape = [1 : i32, 128 : i32, 12 : i32, 64 : i32]}> : (tensor<1x128x768xbf16>) -> tensor<1x128x12x64xbf16>
    %17 = "ttir.permute"(%16) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x128x12x64xbf16>) -> tensor<1x12x64x128xbf16>
    %18 = "ttir.reshape"(%17) <{shape = [12 : i32, 64 : i32, 128 : i32]}> : (tensor<1x12x64x128xbf16>) -> tensor<12x64x128xbf16>

    // Attention scores: Q @ K^T with batched matmul
    %19 = "ttir.dot_general"(%9, %18) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<12x128x64xbf16>, tensor<12x64x128xbf16>) -> tensor<12x128x128xbf16>
    %20 = "ttir.reshape"(%19) <{shape = [1 : i32, 12 : i32, 128 : i32, 128 : i32]}> : (tensor<12x128x128xbf16>) -> tensor<1x12x128x128xbf16>

    // Scale by sqrt(d_k) = sqrt(64) = 8
    %21 = "ttir.constant"() <{value = dense<8.0> : tensor<bf16>}> : () -> tensor<bf16>
    %22 = "ttir.reshape"(%21) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
    %23 = "ttir.broadcast"(%22) <{broadcast_dimensions = array<i64: 1, 12, 128, 128>}> : (tensor<1x1x1x1xbf16>) -> tensor<1x12x128x128xbf16>
    %24 = "ttir.div"(%20, %23) : (tensor<1x12x128x128xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>

    // Apply attention mask (add mask before softmax)
    // Mask processing would go here using arg9

    // Softmax
    %25 = "ttir.max"(%24) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x128x128xbf16>) -> tensor<1x12x128xbf16>
    %26 = "ttir.reshape"(%25) <{shape = [1 : i32, 12 : i32, 128 : i32, 1 : i32]}> : (tensor<1x12x128xbf16>) -> tensor<1x12x128x1xbf16>
    %27 = "ttir.broadcast"(%26) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x12x128x1xbf16>) -> tensor<1x12x128x128xbf16>
    %28 = "ttir.subtract"(%24, %27) : (tensor<1x12x128x128xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %29 = "ttir.exp"(%28) : (tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %30 = "ttir.sum"(%29) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x128x128xbf16>) -> tensor<1x12x128xbf16>
    %31 = "ttir.reshape"(%30) <{shape = [1 : i32, 12 : i32, 128 : i32, 1 : i32]}> : (tensor<1x12x128xbf16>) -> tensor<1x12x128x1xbf16>
    %32 = "ttir.broadcast"(%31) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x12x128x1xbf16>) -> tensor<1x12x128x128xbf16>
    %33 = "ttir.div"(%29, %32) : (tensor<1x12x128x128xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %34 = "ttir.reshape"(%33) <{shape = [12 : i32, 128 : i32, 128 : i32]}> : (tensor<1x12x128x128xbf16>) -> tensor<12x128x128xbf16>

    // Value projection
    %35 = "ttir.permute"(%arg5) <{permutation = array<i64: 1, 0>}> : (tensor<768x768xbf16>) -> tensor<768x768xbf16>
    %36 = "ttir.dot_general"(%0, %35) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x768xbf16>, tensor<768x768xbf16>) -> tensor<128x768xbf16>
    %37 = "ttir.reshape"(%36) <{shape = [1 : i32, 128 : i32, 768 : i32]}> : (tensor<128x768xbf16>) -> tensor<1x128x768xbf16>

    // Add value bias
    %38 = "ttir.reshape"(%arg6) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xbf16>) -> tensor<1x1x768xbf16>
    %39 = "ttir.broadcast"(%38) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x768xbf16>) -> tensor<1x128x768xbf16>
    %40 = "ttir.add"(%37, %39) : (tensor<1x128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    // Split value heads
    %41 = "ttir.reshape"(%40) <{shape = [1 : i32, 128 : i32, 12 : i32, 64 : i32]}> : (tensor<1x128x768xbf16>) -> tensor<1x128x12x64xbf16>
    %42 = "ttir.permute"(%41) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x12x64xbf16>) -> tensor<1x12x128x64xbf16>
    %43 = "ttir.reshape"(%42) <{shape = [12 : i32, 128 : i32, 64 : i32]}> : (tensor<1x12x128x64xbf16>) -> tensor<12x128x64xbf16>

    // Apply attention: attention_weights @ V
    %44 = "ttir.dot_general"(%34, %43) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<12x128x128xbf16>, tensor<12x128x64xbf16>) -> tensor<12x128x64xbf16>
    %45 = "ttir.reshape"(%44) <{shape = [1 : i32, 12 : i32, 128 : i32, 64 : i32]}> : (tensor<12x128x64xbf16>) -> tensor<1x12x128x64xbf16>

    // Merge heads: [1, 12, 128, 64] -> [1, 128, 12, 64] -> [1, 128, 768]
    %46 = "ttir.permute"(%45) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x128x64xbf16>) -> tensor<1x128x12x64xbf16>
    %47 = "ttir.reshape"(%46) <{shape = [128 : i32, 768 : i32]}> : (tensor<1x128x12x64xbf16>) -> tensor<128x768xbf16>

    // Output projection
    %48 = "ttir.permute"(%arg7) <{permutation = array<i64: 1, 0>}> : (tensor<768x768xbf16>) -> tensor<768x768xbf16>
    %49 = "ttir.dot_general"(%47, %48) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x768xbf16>, tensor<768x768xbf16>) -> tensor<128x768xbf16>
    %50 = "ttir.reshape"(%49) <{shape = [1 : i32, 128 : i32, 768 : i32]}> : (tensor<128x768xbf16>) -> tensor<1x128x768xbf16>

    // Add output bias
    %51 = "ttir.reshape"(%arg8) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xbf16>) -> tensor<1x1x768xbf16>
    %52 = "ttir.broadcast"(%51) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x768xbf16>) -> tensor<1x128x768xbf16>
    %53 = "ttir.add"(%50, %52) : (tensor<1x128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    return %53 : tensor<1x128x768xbf16>
  }
}

module {
  func.func @llama_attention(%arg0: tensor<1x32x4096xbf16>, // input [batch, seq, hidden]
                             %arg1: tensor<4096x4096xbf16>, // query weight
                             %arg2: tensor<4096x4096xbf16>, // key weight
                             %arg3: tensor<4096x4096xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>)
  {

    // CHECK: func.func @llama_attention

    // Concatenate Q, K, V weights:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 1 : si32}>
    // CHECK-SAME: (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x12288xbf16>

    // Matmul Q, K, V projections with concatenated bias:
    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = false}>
    // CHECK-SAME: (tensor<32x4096xbf16>, tensor<4096x12288xbf16>) -> tensor<32x12288xbf16>

    // Reshape matmul output.
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 32 : i32, 12288 : i32]}>
    // CHECK-SAME: (tensor<32x12288xbf16>) -> tensor<1x32x12288xbf16>

    // Split Q, K, V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 32 : ui32, transpose_key = false}>
    // CHECK-SAME: (tensor<1x32x12288xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>)

    %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>) -> tensor<32x4096xbf16>

    // Query head

    %1 = "ttir.permute"(%arg1) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>

    %2 = "ttir.dot_general"(%0, %1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>) -> tensor<1x32x32x128xbf16>
    %4 = "ttir.permute"(%3) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>

    // Key head

    %5 = "ttir.permute"(%arg2) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>

    %6 = "ttir.dot_general"(%0, %5) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %7 = "ttir.reshape"(%6) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>) -> tensor<1x32x32x128xbf16>
    %8 = "ttir.permute"(%7) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>

    // Value head

    %9 = "ttir.permute"(%arg3) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>

    %10 = "ttir.dot_general"(%0, %9) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %11 = "ttir.reshape"(%10) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>) -> tensor<1x32x32x128xbf16>
    %12 = "ttir.permute"(%11) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>

    return %4, %8, %12 : tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>
  }
}

module {
  func.func @flan_t5_attention(%arg0: tensor<1x61x512xbf16>, %arg1: tensor<384x512xbf16>, %arg2: tensor<384x512xbf16>, %arg3: tensor<384x512xbf16>) -> (tensor<1x6x61x64xbf16>, tensor<1x6x64x61xbf16>, tensor<1x6x61x64xbf16>) {
    // Concatenate Q, K, V weights:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 0 : si32}>

    // Matmul Q, K, V projections with concatenated bias:
    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = true}>
    // CHECK-SAME: (tensor<61x512xbf16>, tensor<1152x512xbf16>) -> tensor<61x1152xbf16>

    // Reshape matmul output:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 61 : i32, 1152 : i32]}>
    // CHECK-SAME: (tensor<61x1152xbf16>) -> tensor<1x61x1152xbf16>

    // Split Q, K, V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 6 : ui32, transpose_key = true}>
    // CHECK-SAME: (tensor<1x61x1152xbf16>) -> (tensor<1x6x61x64xbf16>, tensor<1x6x64x61xbf16>, tensor<1x6x61x64xbf16>)


    %0 = "ttir.reshape"(%arg0) <{shape = [61 : i32, 512 : i32]}> : (tensor<1x61x512xbf16>) -> tensor<61x512xbf16> // [batch_size, sequence_length, hidden_dimensions]

    // Query projection
    %1 = "ttir.matmul"(%0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<61x512xbf16>, tensor<384x512xbf16>) -> tensor<61x384xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 61 : i32, 6 : i32, 64 : i32]}> : (tensor<61x384xbf16>) -> tensor<1x61x6x64xbf16>
    %3 = "ttir.permute"(%2) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x61x6x64xbf16>) -> tensor<1x6x61x64xbf16> // [batch_size, number of kv heads, sequence_length, head_size].

    // Second branch: Key projection
    %4 = "ttir.matmul"(%0, %arg2) <{transpose_a = false, transpose_b = true}> : (tensor<61x512xbf16>, tensor<384x512xbf16>) -> tensor<61x384xbf16>
    %5 = "ttir.reshape"(%4) <{shape = [1 : i32, 61 : i32, 6 : i32, 64 : i32]}> : (tensor<61x384xbf16>) -> tensor<1x61x6x64xbf16>
    %6 = "ttir.permute"(%5) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x61x6x64xbf16>) -> tensor<1x6x64x61xbf16>

    // Third branch: Value projection
    %7 = "ttir.matmul"(%0, %arg3) <{transpose_a = false, transpose_b = true}> : (tensor<61x512xbf16>, tensor<384x512xbf16>) -> tensor<61x384xbf16>
    %8 = "ttir.reshape"(%7) <{shape = [1 : i32, 61 : i32, 6 : i32, 64 : i32]}> : (tensor<61x384xbf16>) -> tensor<1x61x6x64xbf16>
    %9 = "ttir.permute"(%8) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x61x6x64xbf16>) -> tensor<1x6x61x64xbf16>

    return %3, %6, %9 : tensor<1x6x61x64xbf16>, tensor<1x6x64x61xbf16>, tensor<1x6x61x64xbf16>
  }
}


// The following test is after ttir-to-ttir-decomposition and canonicalization.
// Transpose and matmul is canonicalized to matmul with transposeB = true.
// This will change weight concatenation dimension.
module {
  func.func @llama_attention_transpose_b(%arg0: tensor<1x32x4096xbf16>, %arg1: tensor<4096x4096xbf16>, %arg2: tensor<4096x4096xbf16>, %arg3: tensor<4096x4096xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) {
    // CHECK: func.func @llama_attention_transpose_b

    // Concatenate Q, K, V weights:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 0 : si32}>
    // CHECK-SAME: (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<12288x4096xbf16>

    // Matmul Q, K, V projections with concatenated bias:
    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = true}>
    // CHECK-SAME: (tensor<32x4096xbf16>, tensor<12288x4096xbf16>) -> tensor<32x12288xbf16>

    // Reshape matmul output:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 32 : i32, 12288 : i32]}>
    // CHECK-SAME: (tensor<32x12288xbf16>) -> tensor<1x32x12288xbf16>

    // Split Q, K, V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 32 : ui32, transpose_key = false}>
    // CHECK-SAME: (tensor<1x32x12288xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>)

    %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.matmul"(%0, %arg1) <{transpose_a = false, transpose_b = true}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>) -> tensor<1x32x32x128xbf16>
    %3 = "ttir.permute"(%2) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %4 = "ttir.matmul"(%0, %arg2) <{transpose_a = false, transpose_b = true}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %5 = "ttir.reshape"(%4) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>) -> tensor<1x32x32x128xbf16>
    %6 = "ttir.permute"(%5) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %7 = "ttir.matmul"(%0, %arg3) <{transpose_a = false, transpose_b = true}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %8 = "ttir.reshape"(%7) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>) -> tensor<1x32x32x128xbf16>
    %9 = "ttir.permute"(%8) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    return %3, %6, %9 : tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>
  }
}
