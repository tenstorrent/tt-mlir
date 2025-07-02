// RUN: ttmlir-opt --tt-register-device --ttir-generic-lower-dmas %s | FileCheck %s
// ttmlir-opt --tt-register-device --ttir-generic-lower-dmas --convert-ttir-to-ttkernel %s | FileCheck %s

#l1_ = #tt.memory_space<l1>
#dram = #tt.memory_space<dram>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>
#reduction = #tt.iterator_type<reduction>

#input_memspace = #dram

!tensor_t     = memref<64x128xf16>
!dram_buffer_t   = memref<1x1x2x4x!tt.tile<32x32, f16>, #tt.shard<8192x2048>, #dram>

!stream_t     = memref<1x1x2x4x!tt.tile<32x32, f16>, #tt.view<map(4)>,     #input_memspace>
!stream_buf_t = memref<1x1x2x4x!tt.tile<32x32, f16>, #tt.shard<8192x2048>, #l1_>

!cb_in_t  = memref<2x4x!tt.tile<32x32, f16>, #l1_>
!cb_out_t = memref<2x4x!tt.tile<32x32, f16>, #l1_>

module {

func.func @dram_dma(%arg0 : !tensor_t) -> !tensor_t {

  // allocate input/output dram buffers 
  %dram_buffer_in = memref.alloc()  {address = 32768 : i64, alignment = 64 : i64} : !dram_buffer_t
  %dram_buffer_out = memref.alloc() {address = 102400 : i64, alignment = 64 : i64} : !dram_buffer_t

  //---- copy system buffers to device input buffer in tilized layout ----//
  ttir.to_layout %arg0, %dram_buffer_in : 
    memref<64x128xf16> into !dram_buffer_t 
    hostInfo = <(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf16, #tt.memory_space<system>>>

  //---- setup stream buffers for dma ----//
  %streamBufIn = memref.alloc() {address = 196608 : i64, alignment = 64 : i64} : !stream_buf_t
  %streamIn = "ttir.stream_layout"(%dram_buffer_in, %streamBufIn) : (!dram_buffer_t, !stream_buf_t) -> !stream_t 
  
  %streamBufOut = memref.alloc() {address = 307200 : i64, alignment = 64 : i64} : !stream_buf_t
  %streamOut = "ttir.stream_layout"(%dram_buffer_out, %streamBufOut) : (!dram_buffer_t, !stream_buf_t) -> !stream_t 

  ttir.generic {
      block_factors = [1, 1], 
      grid = #tt.grid<1x1>, 
      indexing_maps = [#map1, #map1], 
      iterator_types = [#parallel, #parallel], 
      threads = [#ttir.thread<datamovement>, #ttir.thread<datamovement>, #ttir.thread<compute>], 
      operandSegmentSizes = array<i32: 1, 1>
    }
    ins(%streamIn : !stream_t) 
    outs(%streamOut : !stream_t)
  {
  ^datamovement0(%cb_in: !cb_in_t, %cb_out: !cb_out_t):
    %tx = ttir.dma %streamIn<#map1>, %cb_in : (!stream_t, !cb_in_t) -> !ttir.mem_tx
    ttir.dma_wait %tx
    ttir.yield %cb_in : (!cb_in_t)
  }, {
  ^datamovement1(%cb_in: !cb_in_t, %cb_out: !cb_out_t):
    //  Local to remote dram
    //  %tx = ttir.dma %src, %dst : (memref<6x8x!tt.tile<32x32, f32>, #l1_>, memref<1x1x6x8x!tt.tile<32x32, f32>, $tt.shard<...>, #dram>) -> !ttir.mem_tx
    //ttir.await %cb_in : (!cb_in_t)
    //%tx = ttir.dma %cb_in, %streamOut<#map1> : (!cb_in_t, !stream_t) -> !ttir.mem_tx
    //ttir.dma_wait %tx
  }, {
  ^compute(%cb_in: !cb_in_t, %cb_out: !cb_out_t):

  }

  // copy outputs back to host 
  %host_output_ref = memref.alloc() : !tensor_t
  //ttir.to_layout %dram_buffer_out, %host_output_ref : !dram_buffer_t into !tensor_t hostInfo = <(d0, d1) -> (d0, d1), undef, <1x1>, memref<64x128xf16, #tt.memory_space<system>>>
  return %host_output_ref : !tensor_t
}

}