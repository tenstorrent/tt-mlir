// RUN: ttmlir-opt --canonicalize --ttir-fusing --ttir-erase-inverse-ops="force=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

module {
    // First RMS norm from llama - input comes from reshape (32x2048 -> 32x1x2048)
    // Reshape pairs around rms_norm should be folded away.
    // CHECK-LABEL: func.func @rms_norm_fusion_1
    func.func @rms_norm_fusion_1(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<32x2048xbf16> {
        // CHECK: %[[RESULT:.*]] = "ttir.rms_norm"(%arg0, %arg1)
        // CHECK-SAME: (tensor<32x2048xbf16>, tensor<2048xbf16>) -> tensor<32x2048xbf16>
        // CHECK: return %[[RESULT]]
        %3 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<32x1x1xf32>}> : () -> tensor<32x1x1xf32>
        %4 = "ttir.constant"() <{value = dense<4.8828125E-4> : tensor<32x1xf32>}> : () -> tensor<32x1xf32>
        %5 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x1x2048xf32>}> : () -> tensor<32x1x2048xf32>
        %16 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
        %17 = "ttir.reshape"(%16) <{shape = [2048 : i32]}> : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
        %18 = "ttir.reshape"(%17) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
        %19 = "ttir.broadcast"(%18) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        %26 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2048 : i32]}> : (tensor<32x2048xbf16>) -> tensor<32x1x2048xbf16>
        %27 = "ttir.typecast"(%26) <{conservative_folding = false}> : (tensor<32x1x2048xbf16>) -> tensor<32x1x2048xf32>
        %28 = "ttir.pow"(%27, %5) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
        %29 = "ttir.sum"(%28) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1xf32>
        %30 = "ttir.multiply"(%29, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
        %31 = "ttir.reshape"(%30) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %32 = "ttir.add"(%31, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %33 = "ttir.rsqrt"(%32) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %34 = "ttir.reshape"(%33) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
        %35 = "ttir.reshape"(%34) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %36 = "ttir.broadcast"(%35) <{broadcast_dimensions = array<i64: 1, 1, 2048>}> : (tensor<32x1x1xf32>) -> tensor<32x1x2048xf32>
        %37 = "ttir.multiply"(%27, %36) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
        %38 = "ttir.typecast"(%37) <{conservative_folding = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1x2048xbf16>
        %39 = "ttir.multiply"(%19, %38) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        %40 = "ttir.reshape"(%39) <{shape = [32 : i32, 2048 : i32]}> : (tensor<32x1x2048xbf16>) -> tensor<32x2048xbf16>
        return %40 : tensor<32x2048xbf16>
    }

    // Second RMS norm from llama - input is directly 32x1x2048xbf16 (no reshape before typecast)
    // CHECK-LABEL: func.func @rms_norm_fusion_2
    func.func @rms_norm_fusion_2(%arg0: tensor<32x1x2048xbf16>, %arg1: tensor<2048xbf16>) -> tensor<32x1x2048xbf16> {
        // CHECK: %[[RESULT:.*]] = "ttir.rms_norm"(%arg0, %arg1)
        // CHECK-SAME: (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        // CHECK: return %[[RESULT]]
        %3 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<32x1x1xf32>}> : () -> tensor<32x1x1xf32>
        %4 = "ttir.constant"() <{value = dense<4.8828125E-4> : tensor<32x1xf32>}> : () -> tensor<32x1xf32>
        %5 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x1x2048xf32>}> : () -> tensor<32x1x2048xf32>
        %140 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
        %141 = "ttir.reshape"(%140) <{shape = [2048 : i32]}> : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
        %142 = "ttir.reshape"(%141) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
        %143 = "ttir.broadcast"(%142) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        %144 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<32x1x2048xbf16>) -> tensor<32x1x2048xf32>
        %145 = "ttir.pow"(%144, %5) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
        %146 = "ttir.sum"(%145) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1xf32>
        %147 = "ttir.multiply"(%146, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
        %148 = "ttir.reshape"(%147) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %149 = "ttir.add"(%148, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %150 = "ttir.rsqrt"(%149) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %151 = "ttir.reshape"(%150) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
        %152 = "ttir.reshape"(%151) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %153 = "ttir.broadcast"(%152) <{broadcast_dimensions = array<i64: 1, 1, 2048>}> : (tensor<32x1x1xf32>) -> tensor<32x1x2048xf32>
        %154 = "ttir.multiply"(%144, %153) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
        %155 = "ttir.typecast"(%154) <{conservative_folding = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1x2048xbf16>
        %156 = "ttir.multiply"(%143, %155) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        return %156 : tensor<32x1x2048xbf16>
    }

    // Third RMS norm from llama - input comes from add (residual connection)
    // CHECK-LABEL: func.func @rms_norm_fusion_3
    func.func @rms_norm_fusion_3(%arg0: tensor<32x1x2048xbf16>, %arg1: tensor<32x1x2048xbf16>, %arg2: tensor<2048xbf16>) -> tensor<32x1x2048xbf16> {
        // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %arg1)
        // CHECK: %[[RESULT:.*]] = "ttir.rms_norm"(%[[ADD]], %arg2)
        // CHECK-SAME: (tensor<32x1x2048xbf16>, tensor<2048xbf16>) -> tensor<32x1x2048xbf16>
        // CHECK: return %[[RESULT]]
        %3 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<32x1x1xf32>}> : () -> tensor<32x1x1xf32>
        %4 = "ttir.constant"() <{value = dense<4.8828125E-4> : tensor<32x1xf32>}> : () -> tensor<32x1xf32>
        %5 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x1x2048xf32>}> : () -> tensor<32x1x2048xf32>
        %80 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
        %81 = "ttir.reshape"(%80) <{shape = [2048 : i32]}> : (tensor<1x1x2048xbf16>) -> tensor<2048xbf16>
        %82 = "ttir.reshape"(%81) <{shape = [1 : i32, 1 : i32, 2048 : i32]}> : (tensor<2048xbf16>) -> tensor<1x1x2048xbf16>
        %83 = "ttir.broadcast"(%82) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        %176 = "ttir.add"(%arg0, %arg1) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        %177 = "ttir.typecast"(%176) <{conservative_folding = false}> : (tensor<32x1x2048xbf16>) -> tensor<32x1x2048xf32>
        %178 = "ttir.pow"(%177, %5) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
        %179 = "ttir.sum"(%178) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1xf32>
        %180 = "ttir.multiply"(%179, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
        %181 = "ttir.reshape"(%180) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %182 = "ttir.add"(%181, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %183 = "ttir.rsqrt"(%182) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %184 = "ttir.reshape"(%183) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
        %185 = "ttir.reshape"(%184) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %186 = "ttir.broadcast"(%185) <{broadcast_dimensions = array<i64: 1, 1, 2048>}> : (tensor<32x1x1xf32>) -> tensor<32x1x2048xf32>
        %187 = "ttir.multiply"(%177, %186) : (tensor<32x1x2048xf32>, tensor<32x1x2048xf32>) -> tensor<32x1x2048xf32>
        %188 = "ttir.typecast"(%187) <{conservative_folding = false}> : (tensor<32x1x2048xf32>) -> tensor<32x1x2048xbf16>
        %189 = "ttir.multiply"(%83, %188) : (tensor<32x1x2048xbf16>, tensor<32x1x2048xbf16>) -> tensor<32x1x2048xbf16>
        return %189 : tensor<32x1x2048xbf16>
    }

    // Negative test: reshape changes the last dimension (64 -> 2048 -> 64).
    // The reshape fold pattern should not apply because normalization dim changes.
    // CHECK-LABEL: func.func @rms_norm_no_reshape_fold
    func.func @rms_norm_no_reshape_fold(%arg0: tensor<1x32x64xbf16>, %arg1: tensor<2048xbf16>) -> tensor<1x32x64xbf16> {
        // CHECK: "ttir.reshape"
        // CHECK: "ttir.rms_norm"
        // CHECK-SAME: tensor<1x2048xbf16>
        // CHECK: "ttir.reshape"
        %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 2048 : i32]}> : (tensor<1x32x64xbf16>) -> tensor<1x2048xbf16>
        %1 = "ttir.rms_norm"(%0, %arg1) <{epsilon = 9.99999974E-6 : f32, normalized_shape = array<i64: 2048>, operandSegmentSizes = array<i32: 1, 1, 0>}> : (tensor<1x2048xbf16>, tensor<2048xbf16>) -> tensor<1x2048xbf16>
        %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 64 : i32]}> : (tensor<1x2048xbf16>) -> tensor<1x32x64xbf16>
        return %2 : tensor<1x32x64xbf16>
    }
}
