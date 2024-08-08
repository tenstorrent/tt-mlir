load("//include:macros.bzl", "all_inc_gen")

cc_library(
  name = "TTMLIR",
  srcs = ["RegisterAll.cpp"],
  deps = [
    "//include:headers",
    "//lib/CAPI:TTMLIRCAPI",
    "//lib/Dialect:Dialect",
    "//lib/Conversion:TTNNToEmitC",
    "@llvm-project//mlir:AllPassesAndDialects",
    "@llvm-project//llvm:Support",
    "@llvm-project//mlir:Dialect",
    "@llvm-project//mlir:IR",
    "@llvm-project//mlir:Support",
  ] + all_inc_gen(),
  copts=["-I$(GENDIR)/include"],
  visibility = ["//visibility:public"],
)