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
    name="ttconversion_pass_inc_gen",
    strip_include_prefix=".",
    tbl_outs=[
        (
            [
                "-gen-pass-decls",
                "-name=TTMLIRConversion",
            ],
            "Passes.h.inc",
        ),
    ],
    tblgen="@llvm-project//mlir:mlir-tblgen",
    td_file="Passes.td",
    deps=[
        ":ttconversion_td_files",
        "@llvm-project//mlir:PassBaseTdFiles",
    ],
    visibility=["//visibility:public"],
)

td_library(
    name="ttconversion_td_files",
    srcs=[
        "Passes.td",
    ],
    deps=[
        "@llvm-project//mlir:BuiltinDialectTdFiles",
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:ShapeOpsTdFiles",
    ],
    includes=["../../"],
    visibility=["//visibility:public"],
)

