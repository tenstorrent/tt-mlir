load("//include:macros.bzl", "all_inc_gen")

cc_library(
  name = "MLIRTTNNPipelines",
  srcs = [
    "TTNNPipelines.cpp",
  ],
  deps = [
    "//include:headers",
    "//include/ttmlir/Target/TTNN:fbs_lib",
    "@llvm-project//mlir:Analysis",
    "@llvm-project//mlir:BufferizationTransforms",
    "@llvm-project//mlir:Dialect",
    "@llvm-project//mlir:EmitCDialect",
    "@llvm-project//mlir:TargetCpp",
    "@llvm-project//mlir:FuncDialect",
    "@llvm-project//mlir:LinalgDialect",
    "@llvm-project//mlir:LinalgTransforms",
    "@llvm-project//mlir:MLProgramDialect",
    "@llvm-project//mlir:TensorDialect",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:TransformUtils",
    "@llvm-project//mlir:Support",
    "@llvm-project//mlir:Rewrite",
    "@llvm-project//mlir:Pass",
    "@llvm-project//llvm:Support",
  ] + all_inc_gen(),
  copts = ["-I$(GENDIR)/include"],
  visibility = ["//visibility:public"],
)
