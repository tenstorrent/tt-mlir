load("//include:macros.bzl", "all_inc_gen")

cc_library(
  name = "TTMLIRCAPI",
  srcs = [
    "Dialects.cpp",
    "TTKernelTypes.cpp",
  ],
  hdrs = [
    "@llvm-project//mlir:c_headers"
  ],
  deps = [
    "//include:headers",
    "//lib/Dialect/TTKernel/IR:MLIRTTKernelDialect",
    "@llvm-project//mlir:Support",
    "@llvm-project//mlir:CAPIIR",
  ] + all_inc_gen(),
  copts=["-I$(GENDIR)/include"],
  visibility = ["//visibility:public"],
)