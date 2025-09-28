// RUN: ttmlir-opt --ttcore-register-device --ttir-bufferization-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Below is kind of an obscure error message check, but the out of place
// bufferization will result in IR that looks like:
//
//   %t0_tmp = memref.alloc
//   %tx = d2m.dma %view [%core0, %arg3], %t0_tmp
//
// This results in the verifier thinking that both src and dst are remote
// since neither refer to a block argument of the parent generic.

// CHECK-NOT: error: 'd2m.dma' op cannot have both src and dst remote

#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#layout1 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = (d0, d1, d2, d3) -> (d0 + d2 floordiv 2, d1 + d3 floordiv 2, d2 mod 2, d3 mod 2)>

func.func @matmul(%arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, %arg1: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, %arg2: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1> {
  %view = d2m.view_layout %arg0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>
  %view_0 = d2m.view_layout %arg1 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>
  %view_1 = d2m.view_layout %arg2 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>
  %0 = d2m.generic {block_factors = [1, 1, 1, 1, 1, 1], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>]}
      ins(%view, %view_0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>, tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>)
      outs(%view_1 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>)  {
  ^datamovement0(%t0: tensor<2x2x!ttcore.tile<32x32, f32>>, %t1: tensor<2x2x!ttcore.tile<32x32, f32>>, %t2: tensor<2x2x!ttcore.tile<32x32, f32>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %core0 = d2m.core_index(0) : index
    %core1 = d2m.core_index(1) : index
    scf.for %arg3 = %c0 to %c2 step %c1 {
      %1 = arith.cmpi eq, %core1, %c0 : index
      scf.if %1 {
        %tx = d2m.dma %view [%core0, %arg3], %t0 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
        d2m.dma_wait %tx
        d2m.semaphore_wait %sem0, %c1 reset %c0
        %tx_2 = d2m.dma %t0, %t0 core[%core0, %c1] mcast[%c1, %c1] : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
        d2m.dma_wait %tx_2
        d2m.semaphore_set %sem1, %c1, core[%core0, %c1] mcast[%c1, %c1]
      } else {
        d2m.semaphore_inc %sem0, %c1, core[%core0, %c0]
        d2m.semaphore_wait %sem1, %c1 reset %c0
      }
      d2m.yield %t0 : (tensor<2x2x!ttcore.tile<32x32, f32>>)
    }
    d2m.yield %t2 : (tensor<2x2x!ttcore.tile<32x32, f32>>)
  } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>
  return %0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout1>
}
