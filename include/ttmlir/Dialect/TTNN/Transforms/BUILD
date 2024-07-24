load(
    "@llvm-project//mlir:tblgen.bzl",
    "gentbl_cc_library",
    "td_library",
)

filegroup(
    name="headers",
    srcs=glob(["**/*.h"]),
    visibility=["//visibility:public"],
)

gentbl_cc_library(
    name="ttnn_pass_inc_gen",
    strip_include_prefix=".",
    tbl_outs=[
        (
            ["-gen-pass-decls"],
            "Passes.h.inc",
        ),
    ],
    tblgen="@llvm-project//mlir:mlir-tblgen",
    td_file="Passes.td",
    deps=[
        ":ttnn_td_files",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
    visibility=["//visibility:public"],
)


td_library(
    name="ttnn_td_files",
    srcs=[
        "Passes.td",
    ],
    includes=["."],
    deps=[
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ShapeOpsTdFiles",
    ],
)