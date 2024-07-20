load("//include:macros.bzl", "all_inc_gen")

cc_library(
  name = "Dialect",
  deps = [
    "//lib/Dialect/TT/IR:MLIRTTDialect",
    "//lib/Dialect/TTIR/Analysis:MLIRTTIRAnalysis",
    "//lib/Dialect/TTIR/IR:MLIRTTIRDialect",
    "//lib/Dialect/TTIR/Transforms:MLIRTTIRTransforms",
    "//lib/Dialect/TTKernel/IR:MLIRTTKernelDialect",
    "//lib/Dialect/TTMetal/IR:MLIRTTMetalDialect",
    "//lib/Dialect/TTMetal/Transforms:MLIRTTMetalTransforms",
    "//lib/Dialect/TTNN/IR:MLIRTTNNDialect",
    "//lib/Dialect/TTNN/Transforms:MLIRTTNNTransforms",
    "@llvm-project//mlir:AllPassesAndDialects",
    "@llvm-project//llvm:Support",
    "@llvm-project//mlir:Dialect",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:Support",
  ] + all_inc_gen(),
  copts=["-I$(GENDIR)/include"],
  visibility = ["//visibility:public"],
)