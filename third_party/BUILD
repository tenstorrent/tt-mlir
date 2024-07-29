load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

cmake(
    name = "tt-metal",
    env = {
        "ARCH_NAME": "wormhole_b0",
        "ARCH_EXTRA_DIR": "wormhole/wormhole_b0_defines",
    },
    generate_args = ["-G Ninja"],
    cache_entries = {
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_C_COMPILER": "/usr/bin/clang-17",
        "CMAKE_CXX_COMPILER": "/usr/bin/clang++-17",
    },
    lib_source = "@tt-metal//:all_srcs",
    visibility = ["//visibility:public"],
    out_shared_libs = ["libtt_metal.so", "libtt_eager.so", "_ttnn.so"],
    out_lib_dir = "/../tt-metal.build_tmpdir/lib",
    copts = ["-Wno-ignored-qualifiers",
            "-Wno-unused-parameter",
            "-Wno-unused-variable",
            "-Wno-reorder-ctor",
            "-Wno-unused-but-set-variable",
            "-Wno-unused-private-field",
            "-Wno-mismatched-tags",
            "-Wno-unused-lambda-capture",
            "-Wno-unused-local-typedef",
            "-Wno-pessimizing-move",
            "-Wno-range-loop-construct",
            "-Wno-unused-function",
            "-Wno-self-assign-overloaded",
            "-Wno-missing-braces",

            "-Wsometimes-uninitialized",
            "-Wno-c++11-narrowing",
            "-Wno-c++23-extensions",
            "-Wno-error=local-type-template-args",
            "-Wno-delete-non-abstract-non-virtual-dtor",
            "-Wno-c99-designator",
            "-Wno-shift-op-parentheses",
            "-Wno-non-c-typedef-for-linkage",
            "-Wno-deprecated-this-capture",
            "-Wno-deprecated-volatile",
            "-Wno-deprecated-builtins",
            "-Wno-deprecated-declarations",
            "-Wno-logical-op-parentheses",
    ],
)

# cc_library(
#     name = "headers",
#     srcs = glob(["**"]),
#     visibility = ["//visibility:public"],
# )