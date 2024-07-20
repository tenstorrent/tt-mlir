load("//include:macros.bzl", "all_inc_gen")

cc_library(
  name = "MLIRTTIRAnalysis",
  srcs = [
    "GridAnalysis.cpp",
  ],
  deps = [
    "//include:headers",
    "@llvm-project//mlir:AllPassesAndDialects",
    "@llvm-project//llvm:Support",
    "@llvm-project//mlir:Dialect",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:Support",
  ] + all_inc_gen(),
  copts=["-I$(GENDIR)/include"],
  visibility = ["//visibility:public"],
)
