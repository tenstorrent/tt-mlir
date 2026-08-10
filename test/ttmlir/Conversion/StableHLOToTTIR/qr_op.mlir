// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline -o %t2 %t
// RUN: FileCheck %s --check-prefix=TTNN --input-file=%t2

// JAX can emit either a LAPACK FFI pair or `Qr` plus
// `ProductOfElementaryHouseholderReflectors`. Both forms include an R
// extraction (slice plus strict-upper-triangle select). The graph must collapse
// to one ttir.qr created from the QR input.
module {
  // m >= n: The generic QR producer feeds ORGQR directly.
  // CHECK-LABEL: func.func @qr_m_ge_n
  func.func @qr_m_ge_n(%arg0: tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<3x3xf32>) {
    // CHECK: "ttir.qr"(%arg0)
    // CHECK-SAME: -> (tensor<4x3xf32>, tensor<3x3xf32>)
    // CHECK-NOT: custom_call
    // CHECK-NOT: stablehlo
    // CHECK: return
    // TTNN-LABEL: func.func @qr_m_ge_n
    // TTNN-NOT: ttir.qr
    // TTNN-NOT: xi1
    // TTNN: "ttnn.matmul"
    %0:2 = stablehlo.custom_call @Qr(%arg0) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m]) {i=4, j=3, k=4, l=3, m=3}, custom>} : (tensor<4x3xf32>) -> (tensor<4x3xf32>, tensor<3xf32>)
    %1 = stablehlo.custom_call @ProductOfElementaryHouseholderReflectors(%0#0, %0#1) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k])->([l, m]) {i=4, j=3, k=3, l=4, m=3}, custom>} : (tensor<4x3xf32>, tensor<3xf32>) -> tensor<4x3xf32>
    %2 = stablehlo.slice %0#0 [0:3, 0:3] : (tensor<4x3xf32>) -> tensor<3x3xf32>
    %3 = stablehlo.iota dim = 0 : tensor<3x3xi32>
    %c = stablehlo.constant dense<-1> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x3xi32>
    %5 = stablehlo.add %3, %4 : tensor<3x3xi32>
    %6 = stablehlo.iota dim = 1 : tensor<3x3xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x3xf32>
    %9 = stablehlo.select %7, %8, %2 : tensor<3x3xi1>, tensor<3x3xf32>
    return %1, %9 : tensor<4x3xf32>, tensor<3x3xf32>
  }

  // m < n: ORGQR consumes the leading k x k block slice of the packed
  // reflectors; the R extraction masks the full packed tensor.
  // CHECK-LABEL: func.func @qr_m_lt_n
  func.func @qr_m_lt_n(%arg0: tensor<3x4xf32>) -> (tensor<3x3xf32>, tensor<3x4xf32>) {
    // CHECK: "ttir.qr"(%arg0)
    // CHECK-SAME: -> (tensor<3x3xf32>, tensor<3x4xf32>)
    // CHECK-NOT: custom_call
    // CHECK-NOT: stablehlo
    // CHECK: return
    // TTNN-LABEL: func.func @qr_m_lt_n
    // TTNN-NOT: ttir.qr
    // TTNN-NOT: xi1
    // TTNN: "ttnn.matmul"
    %0:2 = stablehlo.custom_call @lapack_sgeqrf_ffi(%arg0) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j])->([k, l], [m]) {i=3, j=4, k=3, l=4, m=3}, custom>} : (tensor<3x4xf32>) -> (tensor<3x4xf32>, tensor<3xf32>)
    %1 = stablehlo.slice %0#0 [0:3, 0:3] : (tensor<3x4xf32>) -> tensor<3x3xf32>
    %2 = stablehlo.custom_call @lapack_sorgqr_ffi(%1, %0#1) {backend_config = "", mhlo.backend_config = {}, mhlo.frontend_attributes = {num_batch_dims = "0"}, operand_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>], output_operand_aliases = [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index = 0, operand_tuple_indices = []>], result_layouts = [dense<[0, 1]> : tensor<2xindex>], sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [k])->([l, m]) {i=3, j=3, k=3, l=3, m=3}, custom>} : (tensor<3x3xf32>, tensor<3xf32>) -> tensor<3x3xf32>
    %3 = stablehlo.iota dim = 0 : tensor<3x4xi32>
    %c = stablehlo.constant dense<-1> : tensor<i32>
    %4 = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<i32>) -> tensor<3x4xi32>
    %5 = stablehlo.add %3, %4 : tensor<3x4xi32>
    %6 = stablehlo.iota dim = 1 : tensor<3x4xi32>
    %7 = stablehlo.compare  GE, %5, %6,  SIGNED : (tensor<3x4xi32>, tensor<3x4xi32>) -> tensor<3x4xi1>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %8 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<3x4xf32>
    %9 = stablehlo.select %7, %8, %0#0 : tensor<3x4xi1>, tensor<3x4xf32>
    return %2, %9 : tensor<3x3xf32>, tensor<3x4xf32>
  }

  // No lapack custom call or stablehlo op may survive the rewrite; each
  // function above ends with CHECK-NOT covering its tail up to the next label.
}
