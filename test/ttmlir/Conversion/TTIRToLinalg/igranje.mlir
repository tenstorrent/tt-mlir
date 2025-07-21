// RUN: ttmlir-opt --convert-linalg-to-affine-loops %s | FileCheck %s
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu-decompose-memrefs,expand-strided-metadata,normalize-memrefs,gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv}),nvvm-attach-target{chip="sm75" features=+ptx80 O=3},convert-nvvm-to-llvm,reconcile-unrealized-casts,gpu-to-llvm{use-bare-pointers-for-host use-bare-pointers-for-kernels})" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu-decompose-memrefs,expand-strided-metadata,normalize-memrefs,gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv}))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu-decompose-memrefs,expand-strided-metadata,normalize-memrefs)" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops,func.func(convert-affine-for-to-gpu),gpu-kernel-outlining)" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops,func.func(convert-affine-for-to-gpu))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops)" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops,func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv}))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops,func.func(convert-affine-for-to-gpu),gpu-kernel-outlining)" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir | mlir-opt --pass-pipeline="builtin.module(gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv}))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir

module {
func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x128xf32>, %C: memref<128x128xf32>) {
  linalg.matmul ins(%A, %B : memref<128x128xf32>, memref<128x128xf32>)
                outs(%C : memref<128x128xf32>)
  return
}
}
