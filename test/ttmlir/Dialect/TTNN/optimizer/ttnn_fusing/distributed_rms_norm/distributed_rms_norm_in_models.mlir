// Distributed RMSNorm fusing test extracted from a real model run.
//
// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 experimental-bfp8-weights=true enable-permute-matmul-fusion=false system-desc-path=%system_desc_path% enable-optimizer=true" %s | FileCheck %s

module {
  // CHECK-LABEL: @llama70b_rmsnorm_distributed
  // CHECK: "ttnn.distributed_rms_norm"
  // CHECK-SAME: cluster_axis = 0 : ui32
  // CHECK-SAME: epsilon = 1.000000e-06 : f32
  // CHECK-NOT: "ttnn.pow_scalar"
  // CHECK-NOT: "ttnn.reduce_scatter"
  // CHECK-NOT: "ttnn.all_gather"
  // CHECK-NOT: "ttnn.rsqrt"
  func.func @llama70b_rmsnorm_distributed(%arg0: tensor<1x32x4096xbf16>, %arg1: tensor<1x4096xbf16>) -> tensor<32x4096xbf16> {
    // local_hidden=4096, cluster_axis_size=2 => scale=1/(4096*2)=1/8192
    %scale = "ttir.full"() <{fill_value = 1.220703e-04 : f32, shape = array<i32: 1, 1>}> : () -> tensor<1x1xf32>
    %eps = "ttir.full"() <{fill_value = 1.000000e-06 : f32, shape = array<i32: 1, 1>}> : () -> tensor<1x1xf32>
    %pow2 = "ttir.full"() <{fill_value = 2.000000e+00 : f32, shape = array<i32: 32, 1, 4096>}> : () -> tensor<32x1x4096xf32>

    %0 = "ttir.typecast"(%arg0) <{conservative_folding = false}> : (tensor<1x32x4096xbf16>) -> tensor<1x32x4096xf32>
    %1 = "ttir.reshape"(%0) <{shape = [32 : i32, 1 : i32, 4096 : i32]}> : (tensor<1x32x4096xf32>) -> tensor<32x1x4096xf32>
    %2 = "ttir.pow"(%1, %pow2) : (tensor<32x1x4096xf32>, tensor<32x1x4096xf32>) -> tensor<32x1x4096xf32>
    %3 = "ttir.sum"(%2) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<32x1x4096xf32>) -> tensor<32x1xf32>
    %4 = "ttir.reshape"(%3) <{shape = [1 : i32, 1 : i32, 32 : i32, 1 : i32]}> : (tensor<32x1xf32>) -> tensor<1x1x32x1xf32>
    %5 = "ttir.reduce_scatter"(%4) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 2 : si32}> : (tensor<1x1x32x1xf32>) -> tensor<1x1x16x1xf32>
    %6 = "ttir.all_gather"(%5) <{all_gather_dim = 2 : si32, cluster_axis = 0 : ui32}> : (tensor<1x1x16x1xf32>) -> tensor<1x1x32x1xf32>
    %7 = "ttir.reshape"(%6) <{shape = [32 : i32, 1 : i32]}> : (tensor<1x1x32x1xf32>) -> tensor<32x1xf32>
    %8 = "ttir.multiply"(%7, %scale) : (tensor<32x1xf32>, tensor<1x1xf32>) -> tensor<32x1xf32>
    %9 = "ttir.add"(%8, %eps) : (tensor<32x1xf32>, tensor<1x1xf32>) -> tensor<32x1xf32>
    %10 = "ttir.rsqrt"(%9) : (tensor<32x1xf32>) -> tensor<32x1xf32>
    %11 = "ttir.reshape"(%0) <{shape = [32 : i32, 4096 : i32]}> : (tensor<1x32x4096xf32>) -> tensor<32x4096xf32>
    %12 = "ttir.multiply"(%11, %10) : (tensor<32x4096xf32>, tensor<32x1xf32>) -> tensor<32x4096xf32>
    %13 = "ttir.typecast"(%12) <{conservative_folding = false}> : (tensor<32x4096xf32>) -> tensor<32x4096xbf16>
    %14 = "ttir.multiply"(%arg1, %13) : (tensor<1x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %14 : tensor<32x4096xbf16>
  }
}
