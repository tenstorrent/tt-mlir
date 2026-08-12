// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

// Layer norm forward composite with a single result.
module {
  func.func @layernorm_fw(%input: tensor<1x1x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>,
                          %bias: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK: "ttir.layernorm_fw"
    // CHECK-SAME: epsilon = 9.99999974E-6 : f32
    // CHECK-SAME: return_mean_rstd = false
    // CHECK-NOT: stablehlo.composite
    %0 = stablehlo.composite "tenstorrent.layernorm_fw" %input, %weight, %bias {
      composite_attributes = {
        epsilon = 1.000000e-05 : f32
      },
      decomposition = @tenstorrent.layernorm_fw.impl
    } : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }
  func.func private @tenstorrent.layernorm_fw.impl(%input: tensor<1x1x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>, %bias: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    return %input : tensor<1x1x128x256xbf16>
  }
}

// -----

// Layer norm forward composite returning the mean and rstd (3 results).
module {
  func.func @layernorm_fw_mean_rstd(%input: tensor<1x1x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>,
                                    %bias: tensor<1x1x1x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>) {
    // CHECK: "ttir.layernorm_fw"
    // CHECK-SAME: return_mean_rstd = true
    // CHECK-NOT: stablehlo.composite
    %0:3 = stablehlo.composite "tenstorrent.layernorm_fw" %input, %weight, %bias {
      composite_attributes = {
        epsilon = 1.000000e-05 : f32
      },
      decomposition = @tenstorrent.layernorm_fw.mean_rstd.impl
    } : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>
  }
  func.func private @tenstorrent.layernorm_fw.mean_rstd.impl(%input: tensor<1x1x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>, %bias: tensor<1x1x1x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>) {
    %0 = stablehlo.constant dense<0.0> : tensor<1x1x128x1xbf16>
    %1 = stablehlo.constant dense<1.0> : tensor<1x1x128x1xbf16>
    return %input, %0, %1 : tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>
  }
}
