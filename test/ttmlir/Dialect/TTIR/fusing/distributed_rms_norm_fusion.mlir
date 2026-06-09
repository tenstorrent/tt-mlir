// RUN: ttmlir-opt --canonicalize --ttir-fusing --ttir-erase-inverse-ops="force=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

module {
    // Tensor-parallel RMSNorm as decomposed by torch-xla: the mean-of-squares
    // is sum(x^2, dim=-1) -> all_reduce<sum> -> *(1/N_global), with no single
    // ttir.mean op. Here the local shard is 1024 wide and N_global = 2048
    // (2 devices x 1024), so the scale 1/2048 encodes the device factor.
    // Same layout noise (reshape/typecast/broadcast) as rms_norm_fusion_1.
    // CHECK-LABEL: func.func @distributed_rms_norm_fusion
    func.func @distributed_rms_norm_fusion(%arg0: tensor<32x1024xbf16>, %arg1: tensor<1024xbf16>) -> tensor<32x1024xbf16> {
        // CHECK: %[[RESULT:.*]] = "ttir.distributed_rms_norm"(%arg0, %arg1)
        // CHECK-SAME: cluster_axis = 1 : ui32
        // CHECK-SAME: epsilon = 9.99999974E-6 : f32
        // CHECK-SAME: (tensor<32x1024xbf16>, tensor<1024xbf16>) -> tensor<32x1024xbf16>
        // CHECK: return %[[RESULT]]
        // CHECK-NOT: "ttir.all_reduce"
        // CHECK-NOT: "ttir.sum"
        // CHECK-NOT: "ttir.pow"
        %3 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<32x1x1xf32>}> : () -> tensor<32x1x1xf32>
        %4 = "ttir.constant"() <{value = dense<4.8828125E-4> : tensor<32x1xf32>}> : () -> tensor<32x1xf32>
        %5 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x1x1024xf32>}> : () -> tensor<32x1x1024xf32>
        %16 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
        %17 = "ttir.reshape"(%16) <{shape = [1024 : i32]}> : (tensor<1x1x1024xbf16>) -> tensor<1024xbf16>
        %18 = "ttir.reshape"(%17) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
        %19 = "ttir.broadcast"(%18) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x1024xbf16>) -> tensor<32x1x1024xbf16>
        %26 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 1024 : i32]}> : (tensor<32x1024xbf16>) -> tensor<32x1x1024xbf16>
        %27 = "ttir.typecast"(%26) <{conservative_folding = false}> : (tensor<32x1x1024xbf16>) -> tensor<32x1x1024xf32>
        %28 = "ttir.pow"(%27, %5) : (tensor<32x1x1024xf32>, tensor<32x1x1024xf32>) -> tensor<32x1x1024xf32>
        %29 = "ttir.sum"(%28) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x1024xf32>) -> tensor<32x1xf32>
        %ar = "ttir.all_reduce"(%29) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x1xf32>) -> tensor<32x1xf32>
        %30 = "ttir.multiply"(%ar, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
        %31 = "ttir.reshape"(%30) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %32 = "ttir.add"(%31, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %33 = "ttir.rsqrt"(%32) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %34 = "ttir.reshape"(%33) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
        %35 = "ttir.reshape"(%34) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %36 = "ttir.broadcast"(%35) <{broadcast_dimensions = array<i64: 1, 1, 1024>}> : (tensor<32x1x1xf32>) -> tensor<32x1x1024xf32>
        %37 = "ttir.multiply"(%27, %36) : (tensor<32x1x1024xf32>, tensor<32x1x1024xf32>) -> tensor<32x1x1024xf32>
        %38 = "ttir.typecast"(%37) <{conservative_folding = false}> : (tensor<32x1x1024xf32>) -> tensor<32x1x1024xbf16>
        %39 = "ttir.multiply"(%19, %38) : (tensor<32x1x1024xbf16>, tensor<32x1x1024xbf16>) -> tensor<32x1x1024xbf16>
        %40 = "ttir.reshape"(%39) <{shape = [32 : i32, 1024 : i32]}> : (tensor<32x1x1024xbf16>) -> tensor<32x1024xbf16>
        return %40 : tensor<32x1024xbf16>
    }

    // Distributed RMSNorm fed by a residual add. The residual add is not a
    // fused operand of the xla form, so it must stay separate and the
    // distributed op fuses on its output.
    // CHECK-LABEL: func.func @distributed_rms_norm_fusion_residual
    func.func @distributed_rms_norm_fusion_residual(%arg0: tensor<32x1x1024xbf16>, %arg1: tensor<32x1x1024xbf16>, %arg2: tensor<1024xbf16>) -> tensor<32x1x1024xbf16> {
        // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %arg1)
        // CHECK: %[[RESULT:.*]] = "ttir.distributed_rms_norm"(%[[ADD]], %arg2)
        // CHECK-SAME: cluster_axis = 1 : ui32
        // CHECK-SAME: (tensor<32x1x1024xbf16>, tensor<1024xbf16>) -> tensor<32x1x1024xbf16>
        // CHECK: return %[[RESULT]]
        // CHECK-NOT: "ttir.all_reduce"
        %3 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<32x1x1xf32>}> : () -> tensor<32x1x1xf32>
        %4 = "ttir.constant"() <{value = dense<4.8828125E-4> : tensor<32x1xf32>}> : () -> tensor<32x1xf32>
        %5 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x1x1024xf32>}> : () -> tensor<32x1x1024xf32>
        %80 = "ttir.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
        %81 = "ttir.reshape"(%80) <{shape = [1024 : i32]}> : (tensor<1x1x1024xbf16>) -> tensor<1024xbf16>
        %82 = "ttir.reshape"(%81) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
        %83 = "ttir.broadcast"(%82) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x1024xbf16>) -> tensor<32x1x1024xbf16>
        %176 = "ttir.add"(%arg0, %arg1) : (tensor<32x1x1024xbf16>, tensor<32x1x1024xbf16>) -> tensor<32x1x1024xbf16>
        %177 = "ttir.typecast"(%176) <{conservative_folding = false}> : (tensor<32x1x1024xbf16>) -> tensor<32x1x1024xf32>
        %178 = "ttir.pow"(%177, %5) : (tensor<32x1x1024xf32>, tensor<32x1x1024xf32>) -> tensor<32x1x1024xf32>
        %179 = "ttir.sum"(%178) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x1024xf32>) -> tensor<32x1xf32>
        %ar = "ttir.all_reduce"(%179) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x1xf32>) -> tensor<32x1xf32>
        %180 = "ttir.multiply"(%ar, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
        %181 = "ttir.reshape"(%180) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %182 = "ttir.add"(%181, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %183 = "ttir.rsqrt"(%182) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %184 = "ttir.reshape"(%183) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
        %185 = "ttir.reshape"(%184) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %186 = "ttir.broadcast"(%185) <{broadcast_dimensions = array<i64: 1, 1, 1024>}> : (tensor<32x1x1xf32>) -> tensor<32x1x1024xf32>
        %187 = "ttir.multiply"(%177, %186) : (tensor<32x1x1024xf32>, tensor<32x1x1024xf32>) -> tensor<32x1x1024xf32>
        %188 = "ttir.typecast"(%187) <{conservative_folding = false}> : (tensor<32x1x1024xf32>) -> tensor<32x1x1024xbf16>
        %189 = "ttir.multiply"(%83, %188) : (tensor<32x1x1024xbf16>, tensor<32x1x1024xbf16>) -> tensor<32x1x1024xbf16>
        return %189 : tensor<32x1x1024xbf16>
    }

    // Negative case: the scale is 1/1024 == 1/local_hidden with no device factor
    // and no all_reduce. --canonicalize collapses sum*(1/1024) into ttir.mean, so
    // this fuses to a plain (local) ttir.rms_norm, NOT distributed_rms_norm.
    // CHECK-LABEL: func.func @local_rms_norm_not_distributed
    func.func @local_rms_norm_not_distributed(%arg0: tensor<32x1024xbf16>, %arg1: tensor<1024xbf16>) -> tensor<32x1024xbf16> {
        // CHECK: %[[RESULT:.*]] = "ttir.rms_norm"(%arg0, %arg1)
        // CHECK-SAME: (tensor<32x1024xbf16>, tensor<1024xbf16>) -> tensor<32x1024xbf16>
        // CHECK: return %[[RESULT]]
        // CHECK-NOT: "ttir.distributed_rms_norm"
        %3 = "ttir.constant"() <{value = dense<9.99999974E-6> : tensor<32x1x1xf32>}> : () -> tensor<32x1x1xf32>
        %4 = "ttir.constant"() <{value = dense<9.765625E-4> : tensor<32x1xf32>}> : () -> tensor<32x1xf32>
        %5 = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x1x1024xf32>}> : () -> tensor<32x1x1024xf32>
        %16 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
        %17 = "ttir.reshape"(%16) <{shape = [1024 : i32]}> : (tensor<1x1x1024xbf16>) -> tensor<1024xbf16>
        %18 = "ttir.reshape"(%17) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
        %19 = "ttir.broadcast"(%18) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x1024xbf16>) -> tensor<32x1x1024xbf16>
        %26 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 1024 : i32]}> : (tensor<32x1024xbf16>) -> tensor<32x1x1024xbf16>
        %27 = "ttir.typecast"(%26) <{conservative_folding = false}> : (tensor<32x1x1024xbf16>) -> tensor<32x1x1024xf32>
        %28 = "ttir.pow"(%27, %5) : (tensor<32x1x1024xf32>, tensor<32x1x1024xf32>) -> tensor<32x1x1024xf32>
        %29 = "ttir.sum"(%28) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x1024xf32>) -> tensor<32x1xf32>
        %30 = "ttir.multiply"(%29, %4) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
        %31 = "ttir.reshape"(%30) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %32 = "ttir.add"(%31, %3) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %33 = "ttir.rsqrt"(%32) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
        %34 = "ttir.reshape"(%33) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
        %35 = "ttir.reshape"(%34) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
        %36 = "ttir.broadcast"(%35) <{broadcast_dimensions = array<i64: 1, 1, 1024>}> : (tensor<32x1x1xf32>) -> tensor<32x1x1024xf32>
        %37 = "ttir.multiply"(%27, %36) : (tensor<32x1x1024xf32>, tensor<32x1x1024xf32>) -> tensor<32x1x1024xf32>
        %38 = "ttir.typecast"(%37) <{conservative_folding = false}> : (tensor<32x1x1024xf32>) -> tensor<32x1x1024xbf16>
        %39 = "ttir.multiply"(%19, %38) : (tensor<32x1x1024xbf16>, tensor<32x1x1024xbf16>) -> tensor<32x1x1024xbf16>
        %40 = "ttir.reshape"(%39) <{shape = [32 : i32, 1024 : i32]}> : (tensor<32x1x1024xbf16>) -> tensor<32x1024xbf16>
        return %40 : tensor<32x1024xbf16>
    }
}
