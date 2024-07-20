load("//include:macros.bzl", "all_inc_gen")

cc_library(
  name = "MLIRTTIRTransforms",
  srcs = [
    "Passes.cpp",
  ],
  deps = [
    "//include:headers",
    "@llvm-project//mlir:Analysis",
    "@llvm-project//mlir:BufferizationTransforms",
    "@llvm-project//mlir:Dialect",
    "@llvm-project//mlir:FuncDialect",
    "@llvm-project//mlir:LinalgDialect",
    "@llvm-project//mlir:LinalgTransforms",
    "@llvm-project//mlir:MLProgramDialect",
    "@llvm-project//mlir:TensorDialect",
    "@llvm-project//mlir:TosaDialect",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:TransformUtils",
    "@llvm-project//mlir:Support",
    "@llvm-project//mlir:Rewrite",
    "@llvm-project//llvm:Support",
  ] + all_inc_gen(),
  copts=["-I$(GENDIR)/include"],
  visibility = ["//visibility:public"],
)
