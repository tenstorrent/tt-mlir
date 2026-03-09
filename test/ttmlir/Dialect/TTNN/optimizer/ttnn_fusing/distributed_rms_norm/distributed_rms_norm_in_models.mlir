// Distributed RMSNorm fusing test extracted from a real model run.
//
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false system-desc-path=%system_desc_path% enable-optimizer=true" %s | FileCheck %s

module {
  // CHECK-LABEL: @llama70b_rmsnorm_distributed
  // CHECK: "ttnn.distributed_rms_norm"
  // CHECK-SAME: cluster_axis = 0 : ui32
  // CHECK-NOT: "ttnn.pow_scalar"
  // CHECK-NOT: "ttnn.reduce_scatter"
  // CHECK-NOT: "ttnn.all_gather"
  // CHECK-NOT: "ttnn.rsqrt"
  func.func @llama70b_rmsnorm_distributed(%arg0: tensor<32x4096xbf16>, %arg1: tensor<4096xbf16>) -> tensor<32x4096xbf16> {
    %84 = "ttir.full"() <{fill_value = 1.000000e-06 : f32, shape = array<i32: 32, 1, 1>}> : () -> tensor<32x1x1xf32>
    %86 = "ttir.full"() <{fill_value = 1.220703e-04 : f32, shape = array<i32: 32, 1>}> : () -> tensor<32x1xf32>
    %88 = "ttir.full"() <{fill_value = 2.000000e+00 : f32, shape = array<i32: 32, 1, 4096>}> : () -> tensor<32x1x4096xf32>
    %95 = "ttir.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %96 = "ttir.reshape"(%95) <{shape = [4096 : i32]}> : (tensor<1x1x4096xbf16>) -> tensor<4096xbf16>
    %97 = "ttir.reshape"(%96) <{shape = [1 : i32, 1 : i32, 4096 : i32]}> : (tensor<4096xbf16>) -> tensor<1x1x4096xbf16>
    %98 = "ttir.broadcast"(%97) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x4096xbf16>) -> tensor<32x1x4096xbf16>

    %104 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<32x4096xbf16>) -> tensor<32x1x4096xbf16>
    %105 = "ttir.typecast"(%104) <{conservative_folding = false}> : (tensor<32x1x4096xbf16>) -> tensor<32x1x4096xf32>
    %106 = "ttir.pow"(%105, %88) : (tensor<32x1x4096xf32>, tensor<32x1x4096xf32>) -> tensor<32x1x4096xf32>
    %107 = "ttir.sum"(%106) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x4096xf32>) -> tensor<32x1xf32>
    %108 = "ttir.all_reduce"(%107) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x1xf32>) -> tensor<32x1xf32>
    %109 = "ttir.multiply"(%108, %86) : (tensor<32x1xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
    %110 = "ttir.reshape"(%109) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %111 = "ttir.add"(%110, %84) : (tensor<32x1x1xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %112 = "ttir.rsqrt"(%111) : (tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    %113 = "ttir.reshape"(%112) <{shape = [32 : i32, 1 : i32]}> : (tensor<32x1x1xf32>) -> tensor<32x1xf32>
    %114 = "ttir.reshape"(%113) <{shape = [32 : i32, 1 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<32x1x1xf32>
    %115 = "ttir.broadcast"(%114) <{broadcast_dimensions = array<i64: 1, 1, 4096>}> : (tensor<32x1x1xf32>) -> tensor<32x1x4096xf32>
    %116 = "ttir.multiply"(%105, %115) : (tensor<32x1x4096xf32>, tensor<32x1x4096xf32>) -> tensor<32x1x4096xf32>
    %117 = "ttir.typecast"(%116) <{conservative_folding = false}> : (tensor<32x1x4096xf32>) -> tensor<32x1x4096xbf16>
    %118 = "ttir.multiply"(%98, %117) : (tensor<32x1x4096xbf16>, tensor<32x1x4096xbf16>) -> tensor<32x1x4096xbf16>
    %119 = "ttir.reshape"(%118) <{shape = [32 : i32, 4096 : i32]}> : (tensor<32x1x4096xbf16>) -> tensor<32x4096xbf16>
    return %119 : tensor<32x4096xbf16>
  }
}
