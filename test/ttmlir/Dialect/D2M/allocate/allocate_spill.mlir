// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=test-assume-l1-capacity=6291456 test-buffer-size-policy=max" -o %t %s
// RUN: FileCheck %s --input-file=%t

// This test uses max-sized stream buffers but succeeds by spilling some tensors to DRAM.

// CHECK: %{{.+}} = memref.alloc() {address = {{[0-9]+}} : i64, alignment = {{[0-9]+}} : i64}{{.+}} #dram>

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  func.func @main() -> memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096>, #l1> {
    %lhs = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096>, #l1>
    %rhs = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096>, #l1>
    %r = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%lhs, %rhs : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096>, #l1>, memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096>, #l1>)
        outs(%r : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096>, #l1>)  {
    ^compute0(%cb0: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>):
      %0 = d2m.wait %cb0 : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.wait %cb1 : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
      %2 = d2m.reserve %cb2 : <memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
      "d2m.tile_matmul_block"(%0, %1, %2) : (memref<16x16x!ttcore.tile<32x32, f32>, #l1>, memref<16x16x!ttcore.tile<32x32, f32>, #l1>, memref<16x16x!ttcore.tile<32x32, f32>, #l1>) -> ()
    }
    return %r : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096>, #l1>
  }
}
