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
    // CHECK-SAME: <{dim = 1 : si32}>
    // CHECK-SAME: (tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x3072xf32>) -> tensor<1024x3072xf32>

    // Concatenate Q, K, V biases:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 2 : si32}>
    // CHECK-SAME: (tensor<1x1x1024xf32>, tensor<1x1x1024xf32>, tensor<1x1x1024xf32>, tensor<1x1x3072xf32>) -> tensor<1x1x3072xf32>

    // Linear Q, K, V projections with concatenated bias:
    // CHECK: "ttir.linear"
    // CHECK-SAME:  <{transpose_a = false, transpose_b = false}>
    // CHECK-SAME: (tensor<68x1024xf32>, tensor<1024x3072xf32>, tensor<1x1x3072xf32>, tensor<1x68x3072xf32>) -> tensor<1x68x3072xf32>

    // Check that linear op is reshaped.
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [2 : i32, 34 : i32, 3072 : i32]}>
    // CHECK-SAME: (tensor<1x68x3072xf32>, tensor<2x34x3072xf32>) -> tensor<2x34x3072xf32>

    // Split Q, K, V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 16 : ui32, transpose_key = true}>
    // CHECK-SAME: (tensor<2x34x3072xf32>, tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>) -> (tensor<2x16x34x64xf32>, tensor<2x16x64x34xf32>, tensor<2x16x34x64xf32>)

    // Reshape input to 2D for linear operations
    %0 = ttir.empty() : tensor<68x1024xf32>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>

    // Query projection
    %2 = ttir.empty() : tensor<1024x1024xf32>
    %3 = "ttir.permute"(%arg1, %2) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %4 = "ttir.dot_general"(%1, %3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %5 = ttir.empty() : tensor<2x34x1024xf32>
    %6 = "ttir.reshape"(%4, %5) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    // Add query bias
    %7 = ttir.empty() : tensor<1x1x1024xf32>
    %8 = "ttir.reshape"(%arg2, %7) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %9 = ttir.empty() : tensor<2x34x1024xf32>
    %10 = "ttir.broadcast"(%8, %9) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    %11 = ttir.empty() : tensor<2x34x1024xf32>
    %12 = "ttir.add"(%6, %10, %11) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    // Split query heads: [2, 34, 1024] -> [2, 34, 16, 64] -> [2, 16, 34, 64] -> [32, 34, 64]
    %13 = ttir.empty() : tensor<2x34x16x64xf32>
    %14 = "ttir.reshape"(%12, %13) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<2x34x1024xf32>, tensor<2x34x16x64xf32>) -> tensor<2x34x16x64xf32>
    %15 = ttir.empty() : tensor<2x16x34x64xf32>
    %16 = "ttir.permute"(%14, %15) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x34x16x64xf32>, tensor<2x16x34x64xf32>) -> tensor<2x16x34x64xf32>
    %17 = ttir.empty() : tensor<32x34x64xf32>
    %18 = "ttir.reshape"(%16, %17) <{shape = [32 : i32, 34 : i32, 64 : i32]}> : (tensor<2x16x34x64xf32>, tensor<32x34x64xf32>) -> tensor<32x34x64xf32>

    // Key projection
    %19 = ttir.empty() : tensor<1024x1024xf32>
    %20 = "ttir.permute"(%arg3, %19) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %21 = "ttir.dot_general"(%1, %20) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %22 = ttir.empty() : tensor<2x34x1024xf32>
    %23 = "ttir.reshape"(%21, %22) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    // Add key bias
    %24 = ttir.empty() : tensor<1x1x1024xf32>
    %25 = "ttir.reshape"(%arg4, %24) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %26 = ttir.empty() : tensor<2x34x1024xf32>
    %27 = "ttir.broadcast"(%25, %26) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    %28 = ttir.empty() : tensor<2x34x1024xf32>
    %29 = "ttir.add"(%23, %27, %28) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    // Split key heads: [2, 34, 1024] -> [2, 34, 16, 64] -> [2, 16, 64, 34] -> [32, 64, 34] (transposed for attention)
    %30 = ttir.empty() : tensor<2x34x16x64xf32>
    %31 = "ttir.reshape"(%29, %30) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<2x34x1024xf32>, tensor<2x34x16x64xf32>) -> tensor<2x34x16x64xf32>
    %32 = ttir.empty() : tensor<2x16x64x34xf32>
    %33 = "ttir.permute"(%31, %32) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<2x34x16x64xf32>, tensor<2x16x64x34xf32>) -> tensor<2x16x64x34xf32>
    %34 = ttir.empty() : tensor<32x64x34xf32>
    %35 = "ttir.reshape"(%33, %34) <{shape = [32 : i32, 64 : i32, 34 : i32]}> : (tensor<2x16x64x34xf32>, tensor<32x64x34xf32>) -> tensor<32x64x34xf32>

    // Attention scores: Q @ K^T
    %36 = "ttir.dot_general"(%18, %35) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<32x34x64xf32>, tensor<32x64x34xf32>) -> tensor<32x34x34xf32>
    %37 = ttir.empty() : tensor<2x16x34x34xf32>
    %38 = "ttir.reshape"(%36, %37) <{shape = [2 : i32, 16 : i32, 34 : i32, 34 : i32]}> : (tensor<32x34x34xf32>, tensor<2x16x34x34xf32>) -> tensor<2x16x34x34xf32>

    // Scale attention scores
    %39 = ttir.empty() : tensor<1x1x1x1xf32>
    %40 = "ttir.reshape"(%arg9, %39) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
    %41 = ttir.empty() : tensor<2x16x34x34xf32>
    %42 = "ttir.broadcast"(%40, %41) <{broadcast_dimensions = array<i64: 2, 16, 34, 34>}> : (tensor<1x1x1x1xf32>, tensor<2x16x34x34xf32>) -> tensor<2x16x34x34xf32>
    %43 = ttir.empty() : tensor<2x16x34x34xf32>
    %44 = "ttir.div"(%38, %42, %43) : (tensor<2x16x34x34xf32>, tensor<2x16x34x34xf32>, tensor<2x16x34x34xf32>) -> tensor<2x16x34x34xf32>

    // Apply attention mask (simplified - just add large negative values for masked positions)
    %45 = ttir.empty() : tensor<2x16x34x34xf64>
    %46 = "ttir.typecast"(%44, %45) <{conservative_folding = false}> : (tensor<2x16x34x34xf32>, tensor<2x16x34x34xf64>) -> tensor<2x16x34x34xf64>

    // Softmax: subtract max for numerical stability
    %47 = ttir.empty() : tensor<2x16x34xf64>
    %48 = "ttir.max"(%46, %47) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<2x16x34x34xf64>, tensor<2x16x34xf64>) -> tensor<2x16x34xf64>
    %49 = ttir.empty() : tensor<2x16x34x1xf64>
    %50 = "ttir.reshape"(%48, %49) <{shape = [2 : i32, 16 : i32, 34 : i32, 1 : i32]}> : (tensor<2x16x34xf64>, tensor<2x16x34x1xf64>) -> tensor<2x16x34x1xf64>
    %51 = ttir.empty() : tensor<2x16x34x34xf64>
    %52 = "ttir.broadcast"(%50, %51) <{broadcast_dimensions = array<i64: 1, 1, 1, 34>}> : (tensor<2x16x34x1xf64>, tensor<2x16x34x34xf64>) -> tensor<2x16x34x34xf64>
    %53 = ttir.empty() : tensor<2x16x34x34xf64>
    %54 = "ttir.subtract"(%46, %52, %53) : (tensor<2x16x34x34xf64>, tensor<2x16x34x34xf64>, tensor<2x16x34x34xf64>) -> tensor<2x16x34x34xf64>

    // Exponentiate
    %55 = ttir.empty() : tensor<2x16x34x34xf64>
    %56 = "ttir.exp"(%54, %55) : (tensor<2x16x34x34xf64>, tensor<2x16x34x34xf64>) -> tensor<2x16x34x34xf64>

    // Sum for normalization
    %57 = ttir.empty() : tensor<2x16x34xf64>
    %58 = "ttir.sum"(%56, %57) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<2x16x34x34xf64>, tensor<2x16x34xf64>) -> tensor<2x16x34xf64>
    %59 = ttir.empty() : tensor<2x16x34x1xf64>
    %60 = "ttir.reshape"(%58, %59) <{shape = [2 : i32, 16 : i32, 34 : i32, 1 : i32]}> : (tensor<2x16x34xf64>, tensor<2x16x34x1xf64>) -> tensor<2x16x34x1xf64>
    %61 = ttir.empty() : tensor<2x16x34x34xf64>
    %62 = "ttir.broadcast"(%60, %61) <{broadcast_dimensions = array<i64: 1, 1, 1, 34>}> : (tensor<2x16x34x1xf64>, tensor<2x16x34x34xf64>) -> tensor<2x16x34x34xf64>

    // Normalize to get attention weights
    %63 = ttir.empty() : tensor<2x16x34x34xf64>
    %64 = "ttir.div"(%56, %62, %63) : (tensor<2x16x34x34xf64>, tensor<2x16x34x34xf64>, tensor<2x16x34x34xf64>) -> tensor<2x16x34x34xf64>
    %65 = ttir.empty() : tensor<2x16x34x34xf32>
    %66 = "ttir.typecast"(%64, %65) <{conservative_folding = false}> : (tensor<2x16x34x34xf64>, tensor<2x16x34x34xf32>) -> tensor<2x16x34x34xf32>
    %67 = ttir.empty() : tensor<32x34x34xf32>
    %68 = "ttir.reshape"(%66, %67) <{shape = [32 : i32, 34 : i32, 34 : i32]}> : (tensor<2x16x34x34xf32>, tensor<32x34x34xf32>) -> tensor<32x34x34xf32>

    // Value projection
    %69 = ttir.empty() : tensor<1024x1024xf32>
    %70 = "ttir.permute"(%arg5, %69) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %71 = "ttir.dot_general"(%1, %70) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %72 = ttir.empty() : tensor<2x34x1024xf32>
    %73 = "ttir.reshape"(%71, %72) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    // Add value bias
    %74 = ttir.empty() : tensor<1x1x1024xf32>
    %75 = "ttir.reshape"(%arg6, %74) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %76 = ttir.empty() : tensor<2x34x1024xf32>
    %77 = "ttir.broadcast"(%75, %76) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    %78 = ttir.empty() : tensor<2x34x1024xf32>
    %79 = "ttir.add"(%73, %77, %78) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    // Split value heads: [2, 34, 1024] -> [2, 34, 16, 64] -> [2, 16, 34, 64] -> [32, 34, 64]
    %80 = ttir.empty() : tensor<2x34x16x64xf32>
    %81 = "ttir.reshape"(%79, %80) <{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<2x34x1024xf32>, tensor<2x34x16x64xf32>) -> tensor<2x34x16x64xf32>
    %82 = ttir.empty() : tensor<2x16x34x64xf32>
    %83 = "ttir.permute"(%81, %82) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x34x16x64xf32>, tensor<2x16x34x64xf32>) -> tensor<2x16x34x64xf32>
    %84 = ttir.empty() : tensor<32x34x64xf32>
    %85 = "ttir.reshape"(%83, %84) <{shape = [32 : i32, 34 : i32, 64 : i32]}> : (tensor<2x16x34x64xf32>, tensor<32x34x64xf32>) -> tensor<32x34x64xf32>

    // Apply attention: attention_weights @ V
    %86 = "ttir.dot_general"(%68, %85) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<32x34x34xf32>, tensor<32x34x64xf32>) -> tensor<32x34x64xf32>
    %87 = ttir.empty() : tensor<2x16x34x64xf32>
    %88 = "ttir.reshape"(%86, %87) <{shape = [2 : i32, 16 : i32, 34 : i32, 64 : i32]}> : (tensor<32x34x64xf32>, tensor<2x16x34x64xf32>) -> tensor<2x16x34x64xf32>

    // Merge heads: [2, 16, 34, 64] -> [2, 34, 16, 64] -> [2, 34, 1024]
    %89 = ttir.empty() : tensor<2x34x16x64xf32>
    %90 = "ttir.permute"(%88, %89) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<2x16x34x64xf32>, tensor<2x34x16x64xf32>) -> tensor<2x34x16x64xf32>
    %91 = ttir.empty() : tensor<68x1024xf32>
    %92 = "ttir.reshape"(%90, %91) <{shape = [68 : i32, 1024 : i32]}> : (tensor<2x34x16x64xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>

    // Output projection
    %93 = ttir.empty() : tensor<1024x1024xf32>
    %94 = "ttir.permute"(%arg7, %93) <{permutation = array<i64: 1, 0>}> : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %95 = "ttir.dot_general"(%92, %94) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %96 = ttir.empty() : tensor<2x34x1024xf32>
    %97 = "ttir.reshape"(%95, %96) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    // Add output bias
    %98 = ttir.empty() : tensor<1x1x1024xf32>
    %99 = "ttir.reshape"(%arg8, %98) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf32>, tensor<1x1x1024xf32>) -> tensor<1x1x1024xf32>
    %100 = ttir.empty() : tensor<2x34x1024xf32>
    %101 = "ttir.broadcast"(%99, %100) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    %102 = ttir.empty() : tensor<2x34x1024xf32>
    %103 = "ttir.add"(%97, %101, %102) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>

    return %103 : tensor<2x34x1024xf32>
  }

  func.func @bert_attention(%arg0: tensor<1x128x768xbf16>,
                          %arg1: tensor<768x768xbf16>, %arg2: tensor<768xbf16>, // query weight, bias
                          %arg3: tensor<768x768xbf16>, %arg4: tensor<768xbf16>, // key weight, bias
                          %arg5: tensor<768x768xbf16>, %arg6: tensor<768xbf16>, // value weight, bias
                          %arg7: tensor<768x768xbf16>, %arg8: tensor<768xbf16>, // output weight, bias
                          %arg9: tensor<1x128xi64>) -> (tensor<1x128x768xbf16>) {

    // CHECK: func.func @bert_attention

    // Concatenate Q, K, V weights:
    // CHECK: "ttir.concat"
    // CHECK-SAME:  <{dim = 1 : si32}>
    // CHECK-SAME: (tensor<768x768xbf16>, tensor<768x768xbf16>, tensor<768x768xbf16>, tensor<768x2304xbf16>) -> tensor<768x2304xbf16>

    // Concatenate Q, K, V biases:
    // CHECK: "ttir.concat"
    // CHECK-SAME: <{dim = 2 : si32}>
    // CHECK-SAME: (tensor<1x1x768xbf16>, tensor<1x1x768xbf16>, tensor<1x1x768xbf16>, tensor<1x1x2304xbf16>) -> tensor<1x1x2304xbf16>

    // Linear Q, K, V projections with concatenated bias:
    // CHECK: "ttir.linear"
    // CHECK-SAME: <{transpose_a = false, transpose_b = false}>
    // CHECK-SAME: (tensor<128x768xbf16>, tensor<768x2304xbf16>, tensor<1x1x2304xbf16>, tensor<1x128x2304xbf16>) -> tensor<1x128x2304xbf16>

    // Split Q, K, V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 12 : ui32, transpose_key = true}>
    // CHECK-SAME: (tensor<1x128x2304xbf16>, tensor<1x12x128x64xbf16>, tensor<1x12x64x128xbf16>, tensor<1x12x128x64xbf16>) -> (tensor<1x12x128x64xbf16>, tensor<1x12x64x128xbf16>, tensor<1x12x128x64xbf16>)

    // Reshape input to 2D for linear operations
    %0 = ttir.empty() : tensor<128x768xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [128 : i32, 768 : i32]}> : (tensor<1x128x768xbf16>, tensor<128x768xbf16>) -> tensor<128x768xbf16>

    // Query projection
    %2 = ttir.empty() : tensor<768x768xbf16>
    %3 = "ttir.permute"(%arg1, %2) <{permutation = array<i64: 1, 0>}> : (tensor<768x768xbf16>, tensor<768x768xbf16>) -> tensor<768x768xbf16>
    %4 = "ttir.dot_general"(%1, %3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x768xbf16>, tensor<768x768xbf16>) -> tensor<128x768xbf16>
    %5 = ttir.empty() : tensor<1x128x768xbf16>
    %6 = "ttir.reshape"(%4, %5) <{shape = [1 : i32, 128 : i32, 768 : i32]}> : (tensor<128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    // Add query bias
    %7 = ttir.empty() : tensor<1x1x768xbf16>
    %8 = "ttir.reshape"(%arg2, %7) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xbf16>, tensor<1x1x768xbf16>) -> tensor<1x1x768xbf16>
    %9 = ttir.empty() : tensor<1x128x768xbf16>
    %10 = "ttir.broadcast"(%8, %9) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>
    %11 = ttir.empty() : tensor<1x128x768xbf16>
    %12 = "ttir.add"(%6, %10, %11) : (tensor<1x128x768xbf16>, tensor<1x128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    // Split query heads: [1, 128, 768] -> [1, 128, 12, 64] -> [1, 12, 128, 64] -> [12, 128, 64]
    %13 = ttir.empty() : tensor<1x128x12x64xbf16>
    %14 = "ttir.reshape"(%12, %13) <{shape = [1 : i32, 128 : i32, 12 : i32, 64 : i32]}> : (tensor<1x128x768xbf16>, tensor<1x128x12x64xbf16>) -> tensor<1x128x12x64xbf16>
    %15 = ttir.empty() : tensor<1x12x128x64xbf16>
    %16 = "ttir.permute"(%14, %15) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x12x64xbf16>, tensor<1x12x128x64xbf16>) -> tensor<1x12x128x64xbf16>
    %17 = ttir.empty() : tensor<12x128x64xbf16>
    %18 = "ttir.reshape"(%16, %17) <{shape = [12 : i32, 128 : i32, 64 : i32]}> : (tensor<1x12x128x64xbf16>, tensor<12x128x64xbf16>) -> tensor<12x128x64xbf16>

    // Key projection
    %19 = ttir.empty() : tensor<768x768xbf16>
    %20 = "ttir.permute"(%arg3, %19) <{permutation = array<i64: 1, 0>}> : (tensor<768x768xbf16>, tensor<768x768xbf16>) -> tensor<768x768xbf16>
    %21 = "ttir.dot_general"(%1, %20) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x768xbf16>, tensor<768x768xbf16>) -> tensor<128x768xbf16>
    %22 = ttir.empty() : tensor<1x128x768xbf16>
    %23 = "ttir.reshape"(%21, %22) <{shape = [1 : i32, 128 : i32, 768 : i32]}> : (tensor<128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    // Add key bias
    %24 = ttir.empty() : tensor<1x1x768xbf16>
    %25 = "ttir.reshape"(%arg4, %24) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xbf16>, tensor<1x1x768xbf16>) -> tensor<1x1x768xbf16>
    %26 = ttir.empty() : tensor<1x128x768xbf16>
    %27 = "ttir.broadcast"(%25, %26) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>
    %28 = ttir.empty() : tensor<1x128x768xbf16>
    %29 = "ttir.add"(%23, %27, %28) : (tensor<1x128x768xbf16>, tensor<1x128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    // Split key heads: [1, 128, 768] -> [1, 128, 12, 64] -> [1, 12, 64, 128] -> [12, 64, 128] (transposed)
    %30 = ttir.empty() : tensor<1x128x12x64xbf16>
    %31 = "ttir.reshape"(%29, %30) <{shape = [1 : i32, 128 : i32, 12 : i32, 64 : i32]}> : (tensor<1x128x768xbf16>, tensor<1x128x12x64xbf16>) -> tensor<1x128x12x64xbf16>
    %32 = ttir.empty() : tensor<1x12x64x128xbf16>
    %33 = "ttir.permute"(%31, %32) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x128x12x64xbf16>, tensor<1x12x64x128xbf16>) -> tensor<1x12x64x128xbf16>
    %34 = ttir.empty() : tensor<12x64x128xbf16>
    %35 = "ttir.reshape"(%33, %34) <{shape = [12 : i32, 64 : i32, 128 : i32]}> : (tensor<1x12x64x128xbf16>, tensor<12x64x128xbf16>) -> tensor<12x64x128xbf16>

    // Attention scores: Q @ K^T with batched matmul
    %36 = "ttir.dot_general"(%18, %35) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<12x128x64xbf16>, tensor<12x64x128xbf16>) -> tensor<12x128x128xbf16>
    %37 = ttir.empty() : tensor<1x12x128x128xbf16>
    %38 = "ttir.reshape"(%36, %37) <{shape = [1 : i32, 12 : i32, 128 : i32, 128 : i32]}> : (tensor<12x128x128xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>

    // Scale by sqrt(d_k) = sqrt(64) = 8
    %39 = "ttir.constant"() <{value = dense<8.0> : tensor<bf16>}> : () -> tensor<bf16>
    %40 = ttir.empty() : tensor<1x1x1x1xbf16>
    %41 = "ttir.reshape"(%39, %40) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>, tensor<1x1x1x1xbf16>) -> tensor<1x1x1x1xbf16>
    %42 = ttir.empty() : tensor<1x12x128x128xbf16>
    %43 = "ttir.broadcast"(%41, %42) <{broadcast_dimensions = array<i64: 1, 12, 128, 128>}> : (tensor<1x1x1x1xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %44 = ttir.empty() : tensor<1x12x128x128xbf16>
    %45 = "ttir.div"(%38, %43, %44) : (tensor<1x12x128x128xbf16>, tensor<1x12x128x128xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>

    // Apply attention mask (add mask before softmax)
    // Mask processing would go here using arg9

    // Softmax
    %46 = ttir.empty() : tensor<1x12x128xbf16>
    %47 = "ttir.max"(%45, %46) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x128x128xbf16>, tensor<1x12x128xbf16>) -> tensor<1x12x128xbf16>
    %48 = ttir.empty() : tensor<1x12x128x1xbf16>
    %49 = "ttir.reshape"(%47, %48) <{shape = [1 : i32, 12 : i32, 128 : i32, 1 : i32]}> : (tensor<1x12x128xbf16>, tensor<1x12x128x1xbf16>) -> tensor<1x12x128x1xbf16>
    %50 = ttir.empty() : tensor<1x12x128x128xbf16>
    %51 = "ttir.broadcast"(%49, %50) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x12x128x1xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %52 = ttir.empty() : tensor<1x12x128x128xbf16>
    %53 = "ttir.subtract"(%45, %51, %52) : (tensor<1x12x128x128xbf16>, tensor<1x12x128x128xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %54 = ttir.empty() : tensor<1x12x128x128xbf16>
    %55 = "ttir.exp"(%53, %54) : (tensor<1x12x128x128xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %56 = ttir.empty() : tensor<1x12x128xbf16>
    %57 = "ttir.sum"(%55, %56) <{dim_arg = [3 : i32], keep_dim = false}> : (tensor<1x12x128x128xbf16>, tensor<1x12x128xbf16>) -> tensor<1x12x128xbf16>
    %58 = ttir.empty() : tensor<1x12x128x1xbf16>
    %59 = "ttir.reshape"(%57, %58) <{shape = [1 : i32, 12 : i32, 128 : i32, 1 : i32]}> : (tensor<1x12x128xbf16>, tensor<1x12x128x1xbf16>) -> tensor<1x12x128x1xbf16>
    %60 = ttir.empty() : tensor<1x12x128x128xbf16>
    %61 = "ttir.broadcast"(%59, %60) <{broadcast_dimensions = array<i64: 1, 1, 1, 128>}> : (tensor<1x12x128x1xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %62 = ttir.empty() : tensor<1x12x128x128xbf16>
    %63 = "ttir.div"(%55, %61, %62) : (tensor<1x12x128x128xbf16>, tensor<1x12x128x128xbf16>, tensor<1x12x128x128xbf16>) -> tensor<1x12x128x128xbf16>
    %64 = ttir.empty() : tensor<12x128x128xbf16>
    %65 = "ttir.reshape"(%63, %64) <{shape = [12 : i32, 128 : i32, 128 : i32]}> : (tensor<1x12x128x128xbf16>, tensor<12x128x128xbf16>) -> tensor<12x128x128xbf16>

    // Value projection
    %66 = ttir.empty() : tensor<768x768xbf16>
    %67 = "ttir.permute"(%arg5, %66) <{permutation = array<i64: 1, 0>}> : (tensor<768x768xbf16>, tensor<768x768xbf16>) -> tensor<768x768xbf16>
    %68 = "ttir.dot_general"(%1, %67) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x768xbf16>, tensor<768x768xbf16>) -> tensor<128x768xbf16>
    %69 = ttir.empty() : tensor<1x128x768xbf16>
    %70 = "ttir.reshape"(%68, %69) <{shape = [1 : i32, 128 : i32, 768 : i32]}> : (tensor<128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    // Add value bias
    %71 = ttir.empty() : tensor<1x1x768xbf16>
    %72 = "ttir.reshape"(%arg6, %71) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xbf16>, tensor<1x1x768xbf16>) -> tensor<1x1x768xbf16>
    %73 = ttir.empty() : tensor<1x128x768xbf16>
    %74 = "ttir.broadcast"(%72, %73) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>
    %75 = ttir.empty() : tensor<1x128x768xbf16>
    %76 = "ttir.add"(%70, %74, %75) : (tensor<1x128x768xbf16>, tensor<1x128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    // Split value heads
    %77 = ttir.empty() : tensor<1x128x12x64xbf16>
    %78 = "ttir.reshape"(%76, %77) <{shape = [1 : i32, 128 : i32, 12 : i32, 64 : i32]}> : (tensor<1x128x768xbf16>, tensor<1x128x12x64xbf16>) -> tensor<1x128x12x64xbf16>
    %79 = ttir.empty() : tensor<1x12x128x64xbf16>
    %80 = "ttir.permute"(%78, %79) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x128x12x64xbf16>, tensor<1x12x128x64xbf16>) -> tensor<1x12x128x64xbf16>
    %81 = ttir.empty() : tensor<12x128x64xbf16>
    %82 = "ttir.reshape"(%80, %81) <{shape = [12 : i32, 128 : i32, 64 : i32]}> : (tensor<1x12x128x64xbf16>, tensor<12x128x64xbf16>) -> tensor<12x128x64xbf16>

    // Apply attention: attention_weights @ V
    %83 = "ttir.dot_general"(%65, %82) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<12x128x128xbf16>, tensor<12x128x64xbf16>) -> tensor<12x128x64xbf16>
    %84 = ttir.empty() : tensor<1x12x128x64xbf16>
    %85 = "ttir.reshape"(%83, %84) <{shape = [1 : i32, 12 : i32, 128 : i32, 64 : i32]}> : (tensor<12x128x64xbf16>, tensor<1x12x128x64xbf16>) -> tensor<1x12x128x64xbf16>

    // Merge heads: [1, 12, 128, 64] -> [1, 128, 12, 64] -> [1, 128, 768]
    %86 = ttir.empty() : tensor<1x128x12x64xbf16>
    %87 = "ttir.permute"(%85, %86) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x12x128x64xbf16>, tensor<1x128x12x64xbf16>) -> tensor<1x128x12x64xbf16>
    %88 = ttir.empty() : tensor<128x768xbf16>
    %89 = "ttir.reshape"(%87, %88) <{shape = [128 : i32, 768 : i32]}> : (tensor<1x128x12x64xbf16>, tensor<128x768xbf16>) -> tensor<128x768xbf16>

    // Output projection
    %90 = ttir.empty() : tensor<768x768xbf16>
    %91 = "ttir.permute"(%arg7, %90) <{permutation = array<i64: 1, 0>}> : (tensor<768x768xbf16>, tensor<768x768xbf16>) -> tensor<768x768xbf16>
    %92 = "ttir.dot_general"(%89, %91) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<128x768xbf16>, tensor<768x768xbf16>) -> tensor<128x768xbf16>
    %93 = ttir.empty() : tensor<1x128x768xbf16>
    %94 = "ttir.reshape"(%92, %93) <{shape = [1 : i32, 128 : i32, 768 : i32]}> : (tensor<128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    // Add output bias
    %95 = ttir.empty() : tensor<1x1x768xbf16>
    %96 = "ttir.reshape"(%arg8, %95) <{shape = [1 : i32, 1 : i32, 768 : i32]}> : (tensor<768xbf16>, tensor<1x1x768xbf16>) -> tensor<1x1x768xbf16>
    %97 = ttir.empty() : tensor<1x128x768xbf16>
    %98 = "ttir.broadcast"(%96, %97) <{broadcast_dimensions = array<i64: 1, 128, 1>}> : (tensor<1x1x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>
    %99 = ttir.empty() : tensor<1x128x768xbf16>
    %100 = "ttir.add"(%94, %98, %99) : (tensor<1x128x768xbf16>, tensor<1x128x768xbf16>, tensor<1x128x768xbf16>) -> tensor<1x128x768xbf16>

    return %100 : tensor<1x128x768xbf16>
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
    // CHECK-SAME: (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>, tensor<4096x4096xbf16>, tensor<4096x12288xbf16>) -> tensor<4096x12288xbf16>

    // Matmul Q, K, V projections with concatenated bias:
    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = false}>
    // CHECK-SAME: (tensor<32x4096xbf16>, tensor<4096x12288xbf16>, tensor<32x12288xbf16>) -> tensor<32x12288xbf16>

    // Reshape matmul output.
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 32 : i32, 12288 : i32]}>
    // CHECK-SAME: (tensor<32x12288xbf16>, tensor<1x32x12288xbf16>) -> tensor<1x32x12288xbf16>

    // Split Q, K, V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 32 : ui32, transpose_key = false}>
    // CHECK-SAME: (tensor<1x32x12288xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>)

    %0 = ttir.empty() : tensor<32x4096xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [32 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>

    // Query head

    %2 = ttir.empty() : tensor<4096x4096xbf16>
    %3 = "ttir.permute"(%arg1, %2) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>

    %4 = "ttir.dot_general"(%1, %3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %5 = ttir.empty() : tensor<1x32x32x128xbf16>
    %6 = "ttir.reshape"(%4, %5) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %7 = ttir.empty() : tensor<1x32x32x128xbf16>
    %8 = "ttir.permute"(%6, %7) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>

    // Key head

    %9 = ttir.empty() : tensor<4096x4096xbf16>
    %10 = "ttir.permute"(%arg2, %9) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>

    %11 = "ttir.dot_general"(%1, %10) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %12 = ttir.empty() : tensor<1x32x32x128xbf16>
    %13 = "ttir.reshape"(%11, %12) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %14 = ttir.empty() : tensor<1x32x32x128xbf16>
    %15 = "ttir.permute"(%13, %14) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>

    // Value head

    %16 = ttir.empty() : tensor<4096x4096xbf16>
    %17 = "ttir.permute"(%arg3, %16) <{permutation = array<i64: 1, 0>}> : (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>

    %18 = "ttir.dot_general"(%1, %17) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %19 = ttir.empty() : tensor<1x32x32x128xbf16>
    %20 = "ttir.reshape"(%18, %19) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %21 = ttir.empty() : tensor<1x32x32x128xbf16>
    %22 = "ttir.permute"(%20, %21) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>

    return %8, %15, %22 : tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>
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
    // CHECK-SAME: (tensor<61x512xbf16>, tensor<1152x512xbf16>, tensor<61x1152xbf16>) -> tensor<61x1152xbf16>

    // Reshape matmul output:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 61 : i32, 1152 : i32]}>
    // CHECK-SAME: (tensor<61x1152xbf16>, tensor<1x61x1152xbf16>) -> tensor<1x61x1152xbf16>

    // Split Q, K, V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 6 : ui32, transpose_key = true}>
    // CHECK-SAME: (tensor<1x61x1152xbf16>, tensor<1x6x61x64xbf16>, tensor<1x6x64x61xbf16>, tensor<1x6x61x64xbf16>) -> (tensor<1x6x61x64xbf16>, tensor<1x6x64x61xbf16>, tensor<1x6x61x64xbf16>)


    %0 = ttir.empty() : tensor<61x512xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [61 : i32, 512 : i32]}> : (tensor<1x61x512xbf16>, tensor<61x512xbf16>) -> tensor<61x512xbf16> // [batch_size, sequence_length, hidden_dimensions]

    // Query projection
    %2 = ttir.empty() : tensor<61x384xbf16>
    %3 = "ttir.matmul"(%1, %arg1, %2) <{transpose_a = false, transpose_b = true}> : (tensor<61x512xbf16>, tensor<384x512xbf16>, tensor<61x384xbf16>) -> tensor<61x384xbf16>
    %4 = ttir.empty() : tensor<1x61x6x64xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 61 : i32, 6 : i32, 64 : i32]}> : (tensor<61x384xbf16>, tensor<1x61x6x64xbf16>) -> tensor<1x61x6x64xbf16>
    %6 = ttir.empty() : tensor<1x6x61x64xbf16>
    %7 = "ttir.permute"(%5, %6) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x61x6x64xbf16>, tensor<1x6x61x64xbf16>) -> tensor<1x6x61x64xbf16> // [batch_size, number of kv heads, sequence_length, head_size].

    // Second branch: Key projection
    %8 = ttir.empty() : tensor<61x384xbf16>
    %9 = "ttir.matmul"(%1, %arg2, %8) <{transpose_a = false, transpose_b = true}> : (tensor<61x512xbf16>, tensor<384x512xbf16>, tensor<61x384xbf16>) -> tensor<61x384xbf16>
    %10 = ttir.empty() : tensor<1x61x6x64xbf16>
    %11 = "ttir.reshape"(%9, %10) <{shape = [1 : i32, 61 : i32, 6 : i32, 64 : i32]}> : (tensor<61x384xbf16>, tensor<1x61x6x64xbf16>) -> tensor<1x61x6x64xbf16>
    %12 = ttir.empty() : tensor<1x6x64x61xbf16>
    %13 = "ttir.permute"(%11, %12) <{permutation = array<i64: 0, 2, 3, 1>}> : (tensor<1x61x6x64xbf16>, tensor<1x6x64x61xbf16>) -> tensor<1x6x64x61xbf16>

    // Third branch: Value projection
    %14 = ttir.empty() : tensor<61x384xbf16>
    %15 = "ttir.matmul"(%1, %arg3, %14) <{transpose_a = false, transpose_b = true}> : (tensor<61x512xbf16>, tensor<384x512xbf16>, tensor<61x384xbf16>) -> tensor<61x384xbf16>
    %16 = ttir.empty() : tensor<1x61x6x64xbf16>
    %17 = "ttir.reshape"(%15, %16) <{shape = [1 : i32, 61 : i32, 6 : i32, 64 : i32]}> : (tensor<61x384xbf16>, tensor<1x61x6x64xbf16>) -> tensor<1x61x6x64xbf16>
    %18 = ttir.empty() : tensor<1x6x61x64xbf16>
    %19 = "ttir.permute"(%17, %18) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x61x6x64xbf16>, tensor<1x6x61x64xbf16>) -> tensor<1x6x61x64xbf16>

    return %7, %13, %19 : tensor<1x6x61x64xbf16>, tensor<1x6x64x61xbf16>, tensor<1x6x61x64xbf16>
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
    // CHECK-SAME: (tensor<4096x4096xbf16>, tensor<4096x4096xbf16>, tensor<4096x4096xbf16>, tensor<12288x4096xbf16>) -> tensor<12288x4096xbf16>

    // Matmul Q, K, V projections with concatenated bias:
    // CHECK: "ttir.matmul"
    // CHECK-SAME: <{transpose_a = false, transpose_b = true}>
    // CHECK-SAME: (tensor<32x4096xbf16>, tensor<12288x4096xbf16>, tensor<32x12288xbf16>) -> tensor<32x12288xbf16>

    // Reshape matmul output:
    // CHECK: "ttir.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 32 : i32, 12288 : i32]}>
    // CHECK-SAME: (tensor<32x12288xbf16>, tensor<1x32x12288xbf16>) -> tensor<1x32x12288xbf16>

    // Split Q, K, V heads:
    // CHECK: "ttir.split_query_key_value_and_split_heads"
    // CHECK-SAME: <{num_heads = 32 : ui32, transpose_key = false}>
    // CHECK-SAME: (tensor<1x32x12288xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>)

    %0 = ttir.empty() : tensor<32x4096xbf16>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [32 : i32, 4096 : i32]}> : (tensor<1x32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %2 = ttir.empty() : tensor<32x4096xbf16>
    %3 = "ttir.matmul"(%1, %arg1, %2) <{transpose_a = false, transpose_b = true}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %4 = ttir.empty() : tensor<1x32x32x128xbf16>
    %5 = "ttir.reshape"(%3, %4) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %6 = ttir.empty() : tensor<1x32x32x128xbf16>
    %7 = "ttir.permute"(%5, %6) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %8 = ttir.empty() : tensor<32x4096xbf16>
    %9 = "ttir.matmul"(%1, %arg2, %8) <{transpose_a = false, transpose_b = true}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %10 = ttir.empty() : tensor<1x32x32x128xbf16>
    %11 = "ttir.reshape"(%9, %10) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %12 = ttir.empty() : tensor<1x32x32x128xbf16>
    %13 = "ttir.permute"(%11, %12) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %14 = ttir.empty() : tensor<32x4096xbf16>
    %15 = "ttir.matmul"(%1, %arg3, %14) <{transpose_a = false, transpose_b = true}> : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %16 = ttir.empty() : tensor<1x32x32x128xbf16>
    %17 = "ttir.reshape"(%15, %16) <{shape = [1 : i32, 32 : i32, 32 : i32, 128 : i32]}> : (tensor<32x4096xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    %18 = ttir.empty() : tensor<1x32x32x128xbf16>
    %19 = "ttir.permute"(%17, %18) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>) -> tensor<1x32x32x128xbf16>
    return %7, %13, %19 : tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>, tensor<1x32x32x128xbf16>
  }
}
