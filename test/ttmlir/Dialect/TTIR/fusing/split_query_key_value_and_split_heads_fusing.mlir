// RUN: ttmlir-opt -ttir-fusing -ttir-to-ttir-decomposition -ttir-fusing -o %t %s
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
    // CHECK-SAME: <{dim = 1 : si32}>
    // CHECK-SAME: (tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x3072xf32>) -> tensor<68x3072xf32>

    // Linear Q, K, V projections with concatenated bias:
    // CHECK: "ttir.linear"
    // CHECK-SAME: (tensor<68x1024xf32>, tensor<1024x3072xf32>, tensor<68x3072xf32>, tensor<68x3072xf32>) -> tensor<68x3072xf32>

    // Check reshape [batch*seq, hidden*3] to [batch, seq, hidden*3]
    // CHECK: "ttir.reshape"
    // CHECK-SAME: (tensor<68x3072xf32>, tensor<2x34x3072xf32>) -> tensor<2x34x3072xf32>

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
}
