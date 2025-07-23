// RUN: ttmlir-opt --convert-ttir-to-nvvm %s
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu-decompose-memrefs,expand-strided-metadata,normalize-memrefs,gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv}),nvvm-attach-target{chip="sm75" features=+ptx80 O=3},convert-nvvm-to-llvm,reconcile-unrealized-casts,gpu-to-llvm{use-bare-pointers-for-host use-bare-pointers-for-kernels})" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu-decompose-memrefs,expand-strided-metadata,normalize-memrefs,gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv}))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu-decompose-memrefs,expand-strided-metadata,normalize-memrefs)" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(canonicalize,one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},canonicalize,convert-linalg-to-affine-loops,func.func(affine-loop-invariant-code-motion),func.func(convert-affine-for-to-gpu))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops,func.func(convert-affine-for-to-gpu),gpu-kernel-outlining)" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops,func.func(convert-affine-for-to-gpu))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops)" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir
//ttmlir-opt --pass-pipeline="builtin.module(convert-linalg-to-affine-loops,func.func(convert-affine-for-to-gpu),gpu-kernel-outlining,lower-affine,gpu.module(convert-gpu-to-nvvm{index-bitwidth=0 use-bare-ptr-memref-call-conv}))" test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir

//mlir-translate --mlir-to-llvmir test/ttmlir/Conversion/TTIRToLinalg/izlaz.mlir -o test/ttmlir/Conversion/TTIRToLinalg/llvmir.mlir

//ttmlir-opt --convert-ttir-to-nvvm test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir -o test/ttmlir/Conversion/TTIRToLinalg/izlaz.mlir
//ttmlir-opt --convert-ttir-to-linalg test/ttmlir/Conversion/TTIRToLinalg/igranje.mlir -o test/ttmlir/Conversion/TTIRToLinalg/izlaz.mlir

module @MNISTLinear attributes {} {
  func.func @forward(%arg0: tensor<1x784xf32> {ttir.name = "input_1"}, %arg1: tensor<784x512xf32> {ttir.name = "linear_relu_stack.0.weight"}, %arg2: tensor<1x512xf32> {ttir.name = "linear_relu_stack.0.bias"}, %arg3: tensor<512x512xf32> {ttir.name = "linear_relu_stack.2.weight"}, %arg4: tensor<1x512xf32> {ttir.name = "linear_relu_stack.2.bias"}, %arg5: tensor<512x10xf32> {ttir.name = "linear_relu_stack.4.weight"}, %arg6: tensor<1x10xf32> {ttir.name = "linear_relu_stack.4.bias"}) -> (tensor<1x10xf32> {ttir.name = "MNISTLinear_350.output_add_981"}) {
    %0 = ttir.empty() : tensor<1x512xf32>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x784xf32>, tensor<784x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %2 = ttir.empty() : tensor<1x512xf32>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<1x512xf32>, tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %4 = ttir.empty() : tensor<1x512xf32>
    %5 = "ttir.relu"(%3, %4) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %6 = ttir.empty() : tensor<1x512xf32>
    %7 = "ttir.matmul"(%5, %arg3, %6) : (tensor<1x512xf32>, tensor<512x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %8 = ttir.empty() : tensor<1x512xf32>
    %9 = "ttir.add"(%7, %arg4, %8) : (tensor<1x512xf32>, tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %10 = ttir.empty() : tensor<1x512xf32>
    %11 = "ttir.relu"(%9, %10) : (tensor<1x512xf32>, tensor<1x512xf32>) -> tensor<1x512xf32>
    %12 = ttir.empty() : tensor<1x10xf32>
    %13 = "ttir.matmul"(%11, %arg5, %12) : (tensor<1x512xf32>, tensor<512x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    %14 = ttir.empty() : tensor<1x10xf32>
    %15 = "ttir.add"(%13, %arg6, %14) : (tensor<1x10xf32>, tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %15 : tensor<1x10xf32>
  }
}
