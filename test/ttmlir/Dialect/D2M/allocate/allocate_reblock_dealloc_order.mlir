// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=test-assume-l1-capacity=8388608" -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test verifies that when reblockGenerics creates a view on a generic's output,
// the underlying alloc is not deallocated before subsequent view_layout ops that
// depend on the reblocked view.
//
// The bug scenario:
//   %out_reblocked = d2m.view_layout %out     // view of %out
//   d2m.generic outs(%out_reblocked)          // uses reblocked view
//   memref.dealloc %out                       // WRONG: deallocated too early!
//   %out_restored = d2m.view_layout %out_reblocked  // use-after-free!
//
// This can occur when %out has no direct use after the generic in the original IR.
// The bug doesn't trigger when %out has a later direct use (e.g., func.return %out)
// because liveness extends past the view_layout.

#l1 = #ttcore.memory_space<l1>
#eltwise = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

module {
  // CHECK-LABEL: func.func @reblock_dealloc_order
  //
  // The generic output gets reblocked, creating a view before the generic.
  // After the generic, a view_layout restores the original shape.
  // The dealloc of the underlying alloc must come AFTER the restoring view_layout,
  // NOT between the generic and the restoring view_layout.
  //
  // CHECK: d2m.view_layout
  // CHECK: d2m.generic {block_factors = [2, 1]
  //
  // Between the generic closing brace and the restoring view_layout,
  // there must NOT be a memref.dealloc of the output's underlying alloc.
  // CHECK: }
  // CHECK-NOT: memref.dealloc
  // CHECK: d2m.view_layout
  func.func @reblock_dealloc_order() {
    %lhs = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %add_out = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#eltwise, #eltwise, #eltwise], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%add_out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
    ^bb0:
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %i = 0 to %bf0 {
        affine.for %j = 0 to %bf1 {
          %off0 = d2m.block_offset(0) : index
          %idx0 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%i)[%off0]
          %off1 = d2m.block_offset(1) : index
          %idx1 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)>(%j)[%off1]

          %in0 = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>
          %ld0 = d2m.remote_load %in0 %lhs[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %in1 = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>
          %ld1 = d2m.remote_load %in1 %rhs[%idx0, %idx1] : memref<2x4x!ttcore.tile<32x32, f32>>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          %out = memref.alloc() {alignment = 64 : i64} : memref<2x4x!ttcore.tile<32x32, f32>>

          linalg.generic {indexing_maps = [#eltwise, #eltwise, #eltwise], iterator_types = ["parallel", "parallel"]}
              ins(%in0, %in1 : memref<2x4x!ttcore.tile<32x32, f32>>, memref<2x4x!ttcore.tile<32x32, f32>>)
              outs(%out : memref<2x4x!ttcore.tile<32x32, f32>>) {
          ^bb0(%t0: !ttcore.tile<32x32, f32>, %t1: !ttcore.tile<32x32, f32>, %t2: !ttcore.tile<32x32, f32>):
            %sum = "d2m.tile_add"(%t0, %t1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
            linalg.yield %sum : !ttcore.tile<32x32, f32>
          }

          d2m.remote_store %add_out[%idx0, %idx1] %out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
        } {d2m.blocking_loop = 1 : i64}
      } {d2m.blocking_loop = 0 : i64}
    }
    return
  }
}
