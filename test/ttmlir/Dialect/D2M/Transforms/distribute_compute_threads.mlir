// RUN: ttmlir-opt --split-input-file \
// RUN:   --d2m-distribute-compute-threads %s | FileCheck %s
// RUN: ttmlir-opt --split-input-file \
// RUN:   --d2m-distribute-compute-threads="num-compute-threads=4 split-dim=n" \
// RUN:   %s | FileCheck %s --check-prefix=SPLITN

// -----

// Default options (num=4, split-dim=m): matmul-shaped linalg.generic inside
// a d2m.generic is tiled along M into scf.forall with #d2m.compute_thread
// mapping. Three memref.subview ops cover A, B, C; B's subview spans full
// N since N is untiled.

func.func @distribute_matmul_m(
    %A: memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %B: memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %C: memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
  // CHECK-LABEL: func.func @distribute_matmul_m
  // CHECK: d2m.generic
  // CHECK: scf.forall (%{{.*}}) in (4)
  // CHECK: memref.subview
  // CHECK: memref.subview
  // CHECK: memref.subview
  // CHECK: linalg.generic
  // CHECK: } {mapping = [#d2m.compute_thread<num = 4>]}
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%A, %B : memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%C : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%C : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>, %c: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
  }
  return
}

// -----

// split-dim=n distributes along N instead. The B operand now has the
// per-thread slice; A's subview spans full M.

func.func @distribute_matmul_n(
    %A: memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %B: memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %C: memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
  // SPLITN-LABEL: func.func @distribute_matmul_n
  // SPLITN: scf.forall (%{{.*}}) in (4)
  // SPLITN: } {mapping = [#d2m.compute_thread<num = 4>]}
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%A, %B : memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%C : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%A, %B : memref<8x1x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%C : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>, %c: !ttcore.tile<32x32, bf16>):
      %r = "d2m.tile_matmul"(%a, %b, %c) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
      linalg.yield %r : !ttcore.tile<32x32, bf16>
    }
  }
  return
}

// -----

// A non-matmul linalg.generic (elementwise: only parallel iterators, single
// input) is left untouched in v1.

func.func @leave_elementwise(
    %A: memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>,
    %B: memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
  // CHECK-LABEL: func.func @leave_elementwise
  // CHECK: d2m.generic
  // CHECK-NOT: scf.forall
  // CHECK: linalg.generic
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
      ins(%A : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%B : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
    } ins(%A : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>)
      outs(%B : memref<8x8x!ttcore.tile<32x32, bf16>, #ttcore.memory_space<l1>>) {
    ^bb0(%a: !ttcore.tile<32x32, bf16>, %b: !ttcore.tile<32x32, bf16>):
      linalg.yield %a : !ttcore.tile<32x32, bf16>
    }
  }
  return
}
