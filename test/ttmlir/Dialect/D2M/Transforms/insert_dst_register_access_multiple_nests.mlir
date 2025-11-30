// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-tile-compute-loops --d2m-linalg-to-affine --d2m-insert-dst-register-access='allocation-strategy=basic' --canonicalize %s | FileCheck %s --check-prefixes=CHECK,BASIC
// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-tile-compute-loops --d2m-linalg-to-affine --d2m-insert-dst-register-access='allocation-strategy=greedy' --canonicalize %s | FileCheck %s --check-prefixes=CHECK,GREEDY
// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-tile-compute-loops --d2m-linalg-to-affine --d2m-insert-dst-register-access='allocation-strategy=chaitin-briggs' --canonicalize %s | FileCheck %s --check-prefixes=CHECK,CHAITIN

func.func @cosh(%alloc_1 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>,
                %alloc   : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>,
                %alloc_3 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>,
                %alloc_5 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>) {
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<4x4>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
      ins(%alloc_1, %alloc, %alloc_3 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>, memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>, memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>)
      outs(%alloc_5 : memref<4x4x8x12x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>)  {
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
    // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHECK: affine.for %[[I1:.*]] =
    // CHECK: affine.for %[[J1:.*]] =
    // BASIC-DAG: affine.store %{{.*}}, %[[DST]][0, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // BASIC-DAG: affine.store %{{.*}}, %[[DST]][1, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY-DAG: affine.store %{{.*}}, %[[DST]][0, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY-DAG: affine.store %{{.*}}, %[[DST]][1, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN-DAG: affine.store %{{.*}}, %[[DST]][1, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN-DAG: affine.store %{{.*}}, %[[DST]][2, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHECK: affine.for
    // CHECK: affine.for
    // BASIC-DAG: %{{.*}} = affine.load %[[DST]][0, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // BASIC-DAG: %{{.*}} = affine.load %[[DST]][1, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY-DAG: %{{.*}} = affine.load %[[DST]][0, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY-DAG: %{{.*}} = affine.load %[[DST]][1, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN-DAG: %{{.*}} = affine.load %[[DST]][1, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN-DAG: %{{.*}} = affine.load %[[DST]][2, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHECK: %{{.*}} = "d2m.tile_add"
    // BASIC: affine.store %{{.*}}, %[[DST]][2, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY: affine.store %{{.*}}, %[[DST]][2, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN: affine.store %{{.*}}, %[[DST]][0, %[[I1]], %[[J1]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHECK: affine.for %[[I3:.*]] =
    // CHECK: affine.for %[[J3:.*]] =
    // BASIC: %{{.*}} = affine.load %[[DST]][2, %[[I3]], %[[J3]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY: %{{.*}} = affine.load %[[DST]][2, %[[I3]], %[[J3]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN: %{{.*}} = affine.load %[[DST]][0, %[[I3]], %[[J3]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cb3, %alloc_8 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) outs(%cb3 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %in_9: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %0 = "d2m.tile_add"(%in, %in_9) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %0 : !ttcore.tile<32x32, bf16>
    }
    // CHECK: %[[DST2:.*]] = d2m.acquire_dst() : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHECK: affine.for %[[I2:.*]] =
    // CHECK: affine.for %[[J2:.*]] =
    // BASIC-DAG: affine.store %{{.*}}, %[[DST2]][0, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // BASIC-DAG: affine.store %{{.*}}, %[[DST2]][1, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY-DAG: affine.store %{{.*}}, %[[DST2]][0, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY-DAG: affine.store %{{.*}}, %[[DST2]][1, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN-DAG: affine.store %{{.*}}, %[[DST2]][1, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN-DAG: affine.store %{{.*}}, %[[DST2]][2, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHECK: affine.for
    // CHECK: affine.for
    // BASIC-DAG: %{{.*}} = affine.load %[[DST2]][0, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // BASIC-DAG: %{{.*}} = affine.load %[[DST2]][1, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY-DAG: %{{.*}} = affine.load %[[DST2]][0, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY-DAG: %{{.*}} = affine.load %[[DST2]][1, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN-DAG: %{{.*}} = affine.load %[[DST2]][1, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN-DAG: %{{.*}} = affine.load %[[DST2]][2, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHECK: %{{.*}} = "d2m.tile_mul"
    // BASIC: affine.store %{{.*}}, %[[DST2]][2, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY: affine.store %{{.*}}, %[[DST2]][2, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN: affine.store %{{.*}}, %[[DST2]][0, %[[I2]], %[[J2]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHECK: affine.for %[[I4:.*]] =
    // CHECK: affine.for %[[J4:.*]] =
    // BASIC: %{{.*}} = affine.load %[[DST2]][2, %[[I4]], %[[J4]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // GREEDY: %{{.*}} = affine.load %[[DST2]][2, %[[I4]], %[[J4]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    // CHAITIN: %{{.*}} = affine.load %[[DST2]][0, %[[I4]], %[[J4]]] : memref<8x1x1x!ttcore.tile<32x32, bf16>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%cb3, %cb2 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) outs(%cb3 : memref<8x12x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%in: !ttcore.tile<32x32, bf16>, %in_9: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
      %0 = "d2m.tile_mul"(%in, %in_9) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %0 : !ttcore.tile<32x32, bf16>
    }
  }
  return
}
