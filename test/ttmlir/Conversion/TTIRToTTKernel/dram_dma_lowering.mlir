// RUN: ttmlir-opt --tt-register-device --ttir-generic-lower-dmas %s | FileCheck %s
// ttmlir-opt --tt-register-device --ttir-generic-lower-dmas --convert-ttir-to-ttkernel %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#dram = #tt.memory_space<dram>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>
#reduction = #tt.iterator_type<reduction>

#input_memspace = #dram
!tensor_t     = memref<1x1x2x2x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #input_memspace>

!stream_t     = memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.view<map(4)>,     #input_memspace>
!stream_buf_t = memref<1x1x2x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>
!cb0_t        = memref<2x4x!tt.tile<32x32, f32>, #l1_>

!streamB_t     = memref<1x1x4x2x!tt.tile<32x32, f32>, #tt.view<map(4)>,     #input_memspace>
!streamB_buf_t = memref<1x1x4x2x!tt.tile<32x32, f32>, #tt.shard<32768x4096>, #l1_>
!cb1_t         = memref<4x2x!tt.tile<32x32, f32>, #l1_>

!output_t  = memref<1x1x2x2x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>
!cb2_t      = memref<2x2x!tt.tile<32x32, f32>, #l1_>

module {

func.func @matmul_single_core_stream() {

  %arg0 = ttir.get_global_operand(0) : !tensor_t
  %arg1 = ttir.get_global_operand(1) : !tensor_t
  %arg2 = ttir.get_global_operand(2) : !tensor_t

  %streamBufA = memref.alloc() {address = 102400 : i64, alignment = 64 : i64} : !stream_buf_t
  %streamA = "ttir.stream_layout"(%arg0, %streamBufA) : ( !tensor_t, !stream_buf_t) -> !stream_t 

  %streamBufB = memref.alloc() {address = 204800 : i64, alignment = 64 : i64} : !streamB_buf_t
  %streamB = "ttir.stream_layout"(%arg1, %streamBufB) : ( !tensor_t, !streamB_buf_t) -> !streamB_t

  %out = memref.alloc() {address = 307200 : i64, alignment = 64 : i64} : !output_t

  "ttir.generic"(%streamA, %streamB, %out) <{
    block_factors = [1, 1, 1], 
    grid = #tt.grid<1x1>, 
    indexing_maps = [#map1, #map2, #map3], 
    iterator_types = [#parallel, #parallel, #reduction], 
    threads = [#ttir.thread<datamovement>, #ttir.thread<datamovement>, #ttir.thread<compute>], 
    operandSegmentSizes = array<i32: 2, 1>}> 
  ({
  ^datamovement0(%cb0: !cb0_t, %cb1: !cb1_t, %cb2: !cb2_t):
    %tx = ttir.dma %streamA<#map1>, %cb0 : (!stream_t, !cb0_t) -> !ttir.mem_tx
    ttir.dma_wait %tx
    ttir.yield %cb0 : (!cb0_t)
  }, {
  ^datamovement1(%cb0: !cb0_t, %cb1: !cb1_t, %cb2: !cb2_t):
    %tx = ttir.dma %streamB<#map2>, %cb1 : (!streamB_t, !cb1_t) -> !ttir.mem_tx
    ttir.dma_wait %tx
    ttir.yield %cb1 : (!cb1_t)
  }, {
  ^compute(%cb0: !cb0_t, %cb1: !cb1_t, %cb2: !cb2_t):

  }) : ( !stream_t, !streamB_t, !output_t) -> ()

  return 
}

}