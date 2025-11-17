// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-tile-compute-loops --d2m-insert-dst-register-access --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

func.func @cosh(%alloc_1 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048>, #ttcore.memory_space<l1>>,
                %alloc   : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048>, #ttcore.memory_space<l1>>,
                %alloc_3 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048>, #ttcore.memory_space<l1>>,
                %alloc_5 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048>, #ttcore.memory_space<l1>>) {
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
      ins(%alloc_1, %alloc, %alloc_3 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048>, #ttcore.memory_space<l1>>, memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048>, #ttcore.memory_space<l1>>, memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048>, #ttcore.memory_space<l1>>)
      outs(%alloc_5 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048>, #ttcore.memory_space<l1>>)  {
  ^compute0(%cb0_arg: !d2m.cb<memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>>, %cb1_arg: !d2m.cb<memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>>, %cb2_arg: !d2m.cb<memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>>, %cb3_arg: !d2m.cb<memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>>):
    %cb0 = d2m.wait %cb0_arg : <memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>> -> memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>
    %cb1 = d2m.wait %cb1_arg : <memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>> -> memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>
    %cb2 = d2m.wait %cb2_arg : <memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>> -> memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>
    %cb3 = d2m.reserve %cb3_arg : <memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>> -> memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>
    // CHECK: d2m.acquire_dst
    // CHECK: d2m.tile_negative
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cb1 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) outs(%cb3 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %0 = "d2m.tile_negative"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %0 : !ttcore.tile<32x32, bf16>
    }
    // CHECK: d2m.acquire_dst
    // CHECK: d2m.tile_exp
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cb3 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) outs(%alloc_8 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %0 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %0 : !ttcore.tile<32x32, bf16>
    }
    // CHECK: d2m.acquire_dst
    // CHECK: d2m.tile_exp
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) outs(%cb3 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %0 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %0 : !ttcore.tile<32x32, bf16>
    }
    // CHECK: d2m.acquire_dst
    // CHECK: d2m.tile_add
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cb3, %alloc_8 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) outs(%cb3 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %in_9: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %0 = "d2m.tile_add"(%in, %in_9) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %0 : !ttcore.tile<32x32, bf16>
    }
    // CHECK: d2m.acquire_dst
    // CHECK: d2m.tile_mul
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cb3, %cb2 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) outs(%cb3 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %in_9: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %0 = "d2m.tile_mul"(%in, %in_9) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %0 : !ttcore.tile<32x32, bf16>
    }
  }
  return
}
