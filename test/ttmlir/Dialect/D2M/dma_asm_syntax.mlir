// RUN: ttmlir-opt %s

// This test just checks that various dma assembly syntaxes are parsed correctly.

#dram = #ttcore.memory_space<dram>
#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

func.func @parse(%remote_src: memref<2x4x2x2xf32, #dram>,
                 %remote_dst: memref<2x4x2x2xf32, #dram>,
                 %local_src: memref<2x2xf32, #l1_>,
                 %local_dst: memref<2x2xf32, #l1_>) -> () {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // Local to local
  %tx0 = d2m.dma %local_src, %local_dst : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx1 = d2m.dma %local_src[%c0], %local_dst[%c0] : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx2 = d2m.dma %local_src[%c0, %c0], %local_dst[%c0, %c0] : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx3 = d2m.dma %local_src[%c0, %c0], %local_dst[%c0, %c0], <4> : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx4 = d2m.dma %local_src, %local_dst core[%c0, %c0] mcast[%c1, %c1] : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx5 = d2m.dma %local_src, %local_dst core[%c0, %c0] mcast[%c1, %c1], <4> : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx6 = d2m.dma %local_src, %local_src core[%c0, %c0] mcast[%c1, %c1], <4> : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx

  // Remote to local
  %tx7 = d2m.dma %remote_src[%c0, %c0], %local_dst : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx8 = d2m.dma %remote_src[%c0, %c0, %c0], %local_dst[%c0] : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx9 = d2m.dma %remote_src[%c0, %c0, %c0, %c0], %local_dst[%c0, %c0] : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx10 = d2m.dma %remote_src[%c0, %c0, %c0, %c0], %local_dst[%c0, %c0], <4> : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx11 = d2m.dma %remote_src[%c0, %c0], %local_dst core[%c0, %c0] mcast[%c1, %c1] : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx
  %tx12 = d2m.dma %remote_src[%c0, %c0], %local_dst core[%c0, %c0] mcast[%c1, %c1], <4> : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !d2m.mem_tx

  // Local to remote
  %tx13 = d2m.dma %local_src, %remote_dst[%c0, %c0] : (memref<2x2xf32, #l1_>, memref<2x4x2x2xf32, #dram>) -> !d2m.mem_tx
  %tx14 = d2m.dma %local_src[%c0], %remote_dst[%c0, %c0, %c0] : (memref<2x2xf32, #l1_>, memref<2x4x2x2xf32, #dram>) -> !d2m.mem_tx
  %tx15 = d2m.dma %local_src[%c0, %c0], %remote_dst[%c0, %c0, %c0, %c0] : (memref<2x2xf32, #l1_>, memref<2x4x2x2xf32, #dram>) -> !d2m.mem_tx
  %tx16 = d2m.dma %local_src[%c0, %c0], %remote_dst[%c0, %c0, %c0, %c0], <4> : (memref<2x2xf32, #l1_>, memref<2x4x2x2xf32, #dram>) -> !d2m.mem_tx

  func.return
}
