// RUN: ttmlir-opt --split-input-file --ttir-to-ttmetal-fe-pipeline --ttir-to-ttmetal-me-pipeline --convert-d2m-to-ttkernel %s | FileCheck %s

// Float reductions lower via the FPU reduce_tile kernel (3-operand form
// with a scaler tile materialized by tile_fill).

!ttype_f32 = tensor<128x96xf32>
// CHECK-LABEL: func.func @test_sum_f32
func.func @test_sum_f32(%in: !ttype_f32) -> (tensor<128x1xf32>) {
  // CHECK: ttkernel.fill_tile
  // CHECK: ttkernel.reduce_init
  // CHECK: ttkernel.reduce_tile
  %0 = "ttir.sum"(%in) <{dim_arg = [-1 : i32], keep_dim = true}> : (!ttype_f32) -> tensor<128x1xf32>
  return %0 : tensor<128x1xf32>
}

// -----

!ttype_f32 = tensor<128x96xf32>
// CHECK-LABEL: func.func @test_max_f32
func.func @test_max_f32(%in: !ttype_f32) -> (tensor<128x1xf32>) {
  // CHECK: ttkernel.fill_tile
  // CHECK: ttkernel.reduce_init
  // CHECK: ttkernel.reduce_tile
  %0 = "ttir.max"(%in) <{dim_arg = [-1 : i32], keep_dim = true}> : (!ttype_f32) -> tensor<128x1xf32>
  return %0 : tensor<128x1xf32>
}

// -----

!ttype_f32 = tensor<128x96xf32>
// CHECK-LABEL: func.func @test_mean_f32
func.func @test_mean_f32(%in: !ttype_f32) -> (tensor<128x1xf32>) {
  // CHECK: ttkernel.fill_tile
  // CHECK: ttkernel.reduce_init
  // CHECK: ttkernel.reduce_tile
  %0 = "ttir.mean"(%in) <{dim_arg = [-1 : i32], keep_dim = true}> : (!ttype_f32) -> tensor<128x1xf32>
  return %0 : tensor<128x1xf32>
}

// -----

// Integer reductions lower via the SFPU sfpu_reduce kernel (no scaler tile,
// identity seed + accumulator for inter-tile accumulation).

!ttype_i32 = tensor<128x96xsi32>
// CHECK-LABEL: func.func @test_sum_i32
func.func @test_sum_i32(%in: !ttype_i32) -> (tensor<128x1xsi32>) {
  // CHECK-NOT: ttkernel.reduce_tile
  // Identity (0) is seeded into DST via fill_tile_int.
  // CHECK: ttkernel.fill_tile_int
  // CHECK: ttkernel.sfpu_reduce_init
  // CHECK: ttkernel.sfpu_reduce
  // Accumulator across tiles uses add_int_tile for sum.
  // CHECK: ttkernel.add_int_tile_init
  // CHECK: ttkernel.add_int_tile(
  %0 = "ttir.sum"(%in) <{dim_arg = [-1 : i32], keep_dim = true}> : (!ttype_i32) -> tensor<128x1xsi32>
  return %0 : tensor<128x1xsi32>
}

// -----

!ttype_i32 = tensor<128x96xsi32>
// CHECK-LABEL: func.func @test_max_i32
func.func @test_max_i32(%in: !ttype_i32) -> (tensor<128x1xsi32>) {
  // CHECK-NOT: ttkernel.reduce_tile
  // Identity (INT_MIN) is seeded into DST via fill_tile_int.
  // CHECK: ttkernel.fill_tile_int
  // CHECK: ttkernel.sfpu_reduce_init
  // CHECK: ttkernel.sfpu_reduce
  // Accumulator across tiles uses binary_max_int32_tile for max.
  // CHECK: ttkernel.binary_max_int32_tile_init
  // CHECK: ttkernel.binary_max_int32_tile(
  %0 = "ttir.max"(%in) <{dim_arg = [-1 : i32], keep_dim = true}> : (!ttype_i32) -> tensor<128x1xsi32>
  return %0 : tensor<128x1xsi32>
}

// -----

// i32 sum reducing the second-to-last dim exercises reduce_dim C.

!ttype_i32 = tensor<128x96xsi32>
// CHECK-LABEL: func.func @test_sum_i32_C
func.func @test_sum_i32_C(%in: !ttype_i32) -> (tensor<1x96xsi32>) {
  // CHECK: ttkernel.fill_tile_int
  // CHECK: ttkernel.sfpu_reduce_init
  // CHECK: ttkernel.sfpu_reduce({{.*}}, <reduce_sum>, <{{.*}}>, <reduce_dim_col>)
  // CHECK: ttkernel.add_int_tile(
  %0 = "ttir.sum"(%in) <{dim_arg = [-2 : i32], keep_dim = true}> : (!ttype_i32) -> tensor<1x96xsi32>
  return %0 : tensor<1x96xsi32>
}

// -----

// i32 max reducing the last dim exercises reduce_dim R.

!ttype_i32 = tensor<128x96xsi32>
// CHECK-LABEL: func.func @test_max_i32_R
func.func @test_max_i32_R(%in: !ttype_i32) -> (tensor<128x1xsi32>) {
  // CHECK: ttkernel.fill_tile_int
  // CHECK: ttkernel.sfpu_reduce_init
  // CHECK: ttkernel.sfpu_reduce({{.*}}, <reduce_max>, <{{.*}}>, <reduce_dim_row>)
  // CHECK: ttkernel.binary_max_int32_tile(
  %0 = "ttir.max"(%in) <{dim_arg = [-1 : i32], keep_dim = true}> : (!ttype_i32) -> tensor<128x1xsi32>
  return %0 : tensor<128x1xsi32>
}

// -----

// i32 reduce over both inner dims (RC) → the SFPU rewriter decomposes
// ReduceDim::Scalar into two sfpu_reduce kernels (Col then Row).

!ttype_i32 = tensor<128x96xsi32>
// CHECK-LABEL: func.func @test_sum_i32_RC
func.func @test_sum_i32_RC(%in: !ttype_i32) -> (tensor<1x1xsi32>) {
  // CHECK: ttkernel.fill_tile_int
  // CHECK: ttkernel.sfpu_reduce_init
  // CHECK: ttkernel.sfpu_reduce({{.*}}, <reduce_sum>, <{{.*}}>, <reduce_dim_col>)
  // CHECK: ttkernel.sfpu_reduce({{.*}}, <reduce_sum>, <{{.*}}>, <reduce_dim_row>)
  // CHECK: ttkernel.add_int_tile(
  %0 = "ttir.sum"(%in) <{dim_arg = [-2 : i32, -1 : i32], keep_dim = true}> : (!ttype_i32) -> tensor<1x1xsi32>
  return %0 : tensor<1x1xsi32>
}

// -----

!ttype_i32 = tensor<128x96xsi32>
// CHECK-LABEL: func.func @test_max_i32_RC
func.func @test_max_i32_RC(%in: !ttype_i32) -> (tensor<1x1xsi32>) {
  // CHECK: ttkernel.fill_tile_int
  // CHECK: ttkernel.sfpu_reduce_init
  // CHECK: ttkernel.sfpu_reduce({{.*}}, <reduce_max>, <{{.*}}>, <reduce_dim_col>)
  // CHECK: ttkernel.sfpu_reduce({{.*}}, <reduce_max>, <{{.*}}>, <reduce_dim_row>)
  // CHECK: ttkernel.binary_max_int32_tile(
  %0 = "ttir.max"(%in) <{dim_arg = [-2 : i32, -1 : i32], keep_dim = true}> : (!ttype_i32) -> tensor<1x1xsi32>
  return %0 : tensor<1x1xsi32>
}

// -----

// i32 min decomposes to neg → max → neg, so we still see the SFPU reduce_max
// sequence, surrounded by negative int tile ops.

!ttype_i32 = tensor<128x96xsi32>
// CHECK-LABEL: func.func @test_min_i32
func.func @test_min_i32(%in: !ttype_i32) -> (tensor<128x1xsi32>) {
  // CHECK: ttkernel.negative_tile_int32
  // CHECK: ttkernel.sfpu_reduce_init
  // CHECK: ttkernel.sfpu_reduce
  // CHECK: ttkernel.binary_max_int32_tile(
  // CHECK: ttkernel.negative_tile_int32
  %0 = "ttir.min"(%in) <{dim_arg = [-1 : i32], keep_dim = true}> : (!ttype_i32) -> tensor<128x1xsi32>
  return %0 : tensor<128x1xsi32>
}
