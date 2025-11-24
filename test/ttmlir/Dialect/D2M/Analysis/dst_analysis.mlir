// RUN: ttmlir-opt --ttcore-register-device --d2m-dst-analysis="strategy=basic emit-diagnostics=true" %s 2>&1 | FileCheck %s --check-prefix=BASIC
// RUN: ttmlir-opt --ttcore-register-device --d2m-dst-analysis="strategy=graph-coloring emit-diagnostics=true" %s 2>&1 | FileCheck %s --check-prefix=GC
// RUN: ttmlir-opt --ttcore-register-device --d2m-dst-analysis="strategy=greedy emit-diagnostics=true" %s 2>&1 | FileCheck %s --check-prefix=GREEDY

// Test DST analysis with different strategies on multiple operation types.

#l1_ = #ttcore.memory_space<l1>

module {

  // CHECK-LABEL: func.func @simple_matmul
  // BASIC: remark: DST analysis (basic): 1 slices required
  // GC: remark: DST analysis (graph-coloring): 1 slices required
  // GREEDY: remark: DST analysis (greedy): 1 slices required
  func.func @simple_matmul(%in0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                            %in1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                            %out0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                       memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>)
      outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

      // Load %a - operand 0 of tile_matmul, not in getOperandsLoadFromDstRegister() - NOT COUNTED
      // Load %b - operand 1 of tile_matmul, not in getOperandsLoadFromDstRegister() - NOT COUNTED
      // Load %c - operand 2 of tile_matmul, IS in getOperandsLoadFromDstRegister() - COUNTED (1 slice)
      // Store %result - getDstRegInPlace()=true, reuses operand 2's slice - NOT COUNTED
      // Total: 1 slice
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          affine.for %k = 0 to 2 {
            %a = affine.load %mem0[%i, %k] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %b = affine.load %mem1[%k, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %c = affine.load %mem2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %result = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            affine.store %result, %mem2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          }
        }
      }
    }
    return
  }

  // Test DST analysis with a unary operation (tile_recip).
  // CHECK-LABEL: func.func @unary_op
  // BASIC: remark: DST analysis (basic): 1 slices required
  // GC: remark: DST analysis (graph-coloring): 1 slices required
  // GREEDY: remark: DST analysis (greedy): 1 slices required
  func.func @unary_op(%in0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                       %out0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>)
      outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

      // tile_recip has getOperandsLoadFromDstRegister() = {0} (unary in-place)
      // Load %a - operand 0 of tile_recip, IS in getOperandsLoadFromDstRegister() - COUNTED (1 slice)
      // Store %result - getDstRegInPlace()=true, reuses operand 0's slice - NOT COUNTED
      // Total: 1 slice
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          %a = affine.load %mem0[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %result = "d2m.tile_recip"(%a) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %result, %mem1[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    }
    return
  }

  // Test DST analysis with a binary operation (tile_add).
  // CHECK-LABEL: func.func @binary_op
  // BASIC: remark: DST analysis (basic): 3 slices required
  // GC: remark: DST analysis (graph-coloring): 3 slices required
  // GREEDY: remark: DST analysis (greedy): 3 slices required
  func.func @binary_op(%in0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                        %in1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                        %out0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                       memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>)
      outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

      // tile_add has getOperandsLoadFromDstRegister() = {0, 1} (binary, both operands in DST)
      // Load %a - operand 0 of tile_add, IS in getOperandsLoadFromDstRegister() - COUNTED (1 slice)
      // Load %b - operand 1 of tile_add, IS in getOperandsLoadFromDstRegister() - COUNTED (1 slice)
      // Store %result - getDstRegInPlace()=false (default), allocates new slice - COUNTED (1 slice)
      // Total: 3 slices
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          %a = affine.load %mem0[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %b = affine.load %mem1[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %result = "d2m.tile_add"(%a, %b) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %result, %mem2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    }
    return
  }

  // Test DST analysis with a reduction operation (tile_reduce_sum).
  // CHECK-LABEL: func.func @reduction_op
  // BASIC: remark: DST analysis (basic): 1 slices required
  // GC: remark: DST analysis (graph-coloring): 1 slices required
  // GREEDY: remark: DST analysis (greedy): 1 slices required
  func.func @reduction_op(%in0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                        %in1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                        %out0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                       memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>)
      outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

      // tile_reduce_sum has getOperandsLoadFromDstRegister() = {2} (accumulator only)
      // Load %a - operand 0 of tile_reduce_sum, not in getOperandsLoadFromDstRegister() - NOT COUNTED
      // Load %b - operand 1 of tile_reduce_sum, not in getOperandsLoadFromDstRegister() - NOT COUNTED
      // Load %c - operand 2 of tile_reduce_sum, IS in getOperandsLoadFromDstRegister() - COUNTED (1 slice)
      // Store %result - getDstRegInPlace()=true, reuses operand 2's slice - NOT COUNTED
      // Total: 1 slice
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          affine.for %k = 0 to 2 {
            %a = affine.load %mem0[%i, %k] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %b = affine.load %mem1[%k, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %c = affine.load %mem2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %result = "d2m.tile_reduce_sum"(%a, %b, %c) <{reduce_dim = #d2m<reduce_dim R>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            affine.store %result, %mem2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          }
        }
      }
    }
    return
  }

  // Test DST analysis with multiple operations in sequence (matmul + exp + multiply).
  // This tests interference between consecutive operations.
  // CHECK-LABEL: func.func @multi_op_sequence
  // BASIC: remark: DST analysis (basic): 5 slices required
  // GC: remark: DST analysis (graph-coloring): 3 slices required
  // GREEDY: remark: DST analysis (greedy): 3 slices required
  func.func @multi_op_sequence(%in0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                                %in1: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                                %in2: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                                %out0: memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>) {
    d2m.generic {
      block_factors = [1, 1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<reduction>],
      threads = [#d2m.thread<compute>]
    } ins(%in0, %in1, %in2 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                             memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>,
                             memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>)
      outs(%out0 : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x8192, 1>, #l1_>) {
    ^compute0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>,
              %cb3: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
      %mem0 = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem1 = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem2 = d2m.wait %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
      %mem3 = d2m.reserve %cb3 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

      // Operation sequence: matmul -> exp -> multiply
      // Basic strategy counts each DST access separately:
      //   - matmul: 1 load (%accum from DST) = 1 slice
      //   - exp: 1 load (%matmul_result from DST) = 1 slice
      //   - multiply: 2 loads (%exp_result, %scale from DST) + 1 store (not in-place) = 3 slices
      //   - Total: 5 slices
      //
      // Graph coloring can reuse slices by recognizing non-overlapping lifetimes:
      //   - After matmul stores, the %accum slice can be reused for subsequent operations, so
      //     3 colors are sufficient for this fused sequence.
      //   - Total: 3 slices
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 2 {
          affine.for %k = 0 to 2 {
            // Matmul: in-place operation (getDstRegInPlace=true)
            %a = affine.load %mem0[%i, %k] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %b = affine.load %mem1[%k, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %accum = affine.load %mem3[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
            %matmul_result = "d2m.tile_matmul"(%a, %b, %accum) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            affine.store %matmul_result, %mem3[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          }
          // Exp: in-place operation (getDstRegInPlace=true)
          %matmul_out = affine.load %mem3[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %exp_result = "d2m.tile_exp"(%matmul_out) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %exp_result, %mem3[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>

          // Multiply: NOT in-place (getDstRegInPlace=false)
          %exp_out = affine.load %mem3[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %scale = affine.load %mem2[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
          %mul_result = "d2m.tile_mul"(%exp_out, %scale) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          affine.store %mul_result, %mem3[%i, %j] : memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
        }
      }
    }
    return
  }
}
