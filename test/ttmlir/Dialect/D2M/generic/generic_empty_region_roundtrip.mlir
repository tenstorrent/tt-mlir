// RUN: ttmlir-opt %s | ttmlir-opt | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-regions-to-funcs %s | FileCheck %s --check-prefix=OUTLINE

#l1 = #ttcore.memory_space<l1>

ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>

func.func @empty_datamovement_region(%arg0: memref<1xf32, #l1>) {
  // ROUNDTRIP-LABEL: func.func @empty_datamovement_region
  // ROUNDTRIP: d2m.generic
  // ROUNDTRIP-SAME: threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
  // ROUNDTRIP: {
  // ROUNDTRIP-NEXT: ^datamovement0:
  // ROUNDTRIP: }, {
  d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
      ins()
      outs(%arg0 : memref<1xf32, #l1>) {
  ^datamovement0:
  }, {
  ^compute0:
  }
  return
}

// OUTLINE: func.func private @datamovement_kernel0() attributes {d2m.thread = #d2m.thread<datamovement, dm_core = {{[0-9]+}}>, tt.function_type = "kernel"} {
// OUTLINE-NEXT: return
// OUTLINE: func.func private @compute_kernel1() attributes {d2m.thread = #d2m.thread<compute>, tt.function_type = "kernel"} {
// OUTLINE-NEXT: return
