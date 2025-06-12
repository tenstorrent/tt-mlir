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
  %tx0 = ttir.dma %local_src, %local_dst : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx1 = ttir.dma %local_src[%c0], %local_dst[%c0] : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx2 = ttir.dma %local_src[%c0, %c0], %local_dst[%c0, %c0] : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx3 = ttir.dma %local_src[%c0, %c0], %local_dst[%c0, %c0], <4> : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx4 = ttir.dma %local_src, %local_dst core[%c0, %c0] mcast[%c1, %c1] : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx5 = ttir.dma %local_src, %local_dst core[%c0, %c0] mcast[%c1, %c1], <4> : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx6 = ttir.dma %local_src, %local_src core[%c0, %c0] mcast[%c1, %c1], <4> : (memref<2x2xf32, #l1_>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx

  // Remote to local
  %tx7 = ttir.dma %remote_src[%c0, %c0], %local_dst : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx8 = ttir.dma %remote_src[%c0, %c0, %c0], %local_dst[%c0] : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx9 = ttir.dma %remote_src[%c0, %c0, %c0, %c0], %local_dst[%c0, %c0] : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx10 = ttir.dma %remote_src[%c0, %c0, %c0, %c0], %local_dst[%c0, %c0], <4> : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx11 = ttir.dma %remote_src[%c0, %c0], %local_dst core[%c0, %c0] mcast[%c1, %c1] : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx
  %tx12 = ttir.dma %remote_src[%c0, %c0], %local_dst core[%c0, %c0] mcast[%c1, %c1], <4> : (memref<2x4x2x2xf32, #dram>, memref<2x2xf32, #l1_>) -> !ttir.mem_tx

  // Local to remote
  %tx13 = ttir.dma %local_src, %remote_dst[%c0, %c0] : (memref<2x2xf32, #l1_>, memref<2x4x2x2xf32, #dram>) -> !ttir.mem_tx
  %tx14 = ttir.dma %local_src[%c0], %remote_dst[%c0, %c0, %c0] : (memref<2x2xf32, #l1_>, memref<2x4x2x2xf32, #dram>) -> !ttir.mem_tx
  %tx15 = ttir.dma %local_src[%c0, %c0], %remote_dst[%c0, %c0, %c0, %c0] : (memref<2x2xf32, #l1_>, memref<2x4x2x2xf32, #dram>) -> !ttir.mem_tx
  %tx16 = ttir.dma %local_src[%c0, %c0], %remote_dst[%c0, %c0, %c0, %c0], <4> : (memref<2x2xf32, #l1_>, memref<2x4x2x2xf32, #dram>) -> !ttir.mem_tx

  func.return
}
