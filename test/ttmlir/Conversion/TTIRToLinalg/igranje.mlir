// RUN: ttmlir-opt --convert-ttir-to-nvvm %s
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu-decompose-memrefs,expand-strided-metadata,normalize-memrefs,gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv}),nvvm-attach-target{chip="sm75" features=+ptx80 O=3},convert-nvvm-to-llvm,reconcile-unrealized-casts,gpu-to-llvm{use-bare-pointers-for-host use-bare-pointers-for-kernels})" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu-decompose-memrefs,expand-strided-metadata,normalize-memrefs,gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv}))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu-decompose-memrefs,expand-strided-metadata,normalize-memrefs)" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops,func.func(convert-affine-for-to-gpu),gpu-kernel-outlining)" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops,func.func(convert-affine-for-to-gpu))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops)" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops,func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv}))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir

module {
  func.func @relu_test(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.relu"(%arg0, %0) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
