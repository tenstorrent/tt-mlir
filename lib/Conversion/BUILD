load("//include:macros.bzl", "all_inc_gen")

cc_library(
  name = "TTNNToEmitC",
  srcs = [
    "TTNNToEmitC/TTNNToEmitC.cpp",
  ],
  hdrs = [
    "TTNNToEmitC/PopulatePatterns.h",
    "TTNNToEmitC/TypeConverter.h",
    "PassDetail.h",
  ],
  deps = [
    "//include:headers",
    "@llvm-project//mlir:EmitCDialect",
    "@llvm-project//mlir:FuncDialect",
    "@llvm-project//mlir:FuncTransforms",
    "@llvm-project//mlir:Pass",
    "@llvm-project//mlir:TransformUtils",
    "@llvm-project//mlir:AllPassesAndDialects",
    "@llvm-project//llvm:Support",
    "@llvm-project//mlir:Dialect",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:Support",
  ] + all_inc_gen(),
  copts=["-I$(GENDIR)/include"],
  visibility = ["//visibility:public"],
)
