
cc_library(
    name = "headers",
    srcs = glob(["**/*.h"]),
    hdrs = [":other_headers"],
    includes = ["."],
    visibility=["//visibility:public"],
)
filegroup(
    name = "other_headers",
    srcs = glob(["**/*.h"]) + 
        ["//include/ttmlir/Dialect/TT/IR:headers",
        "//include/ttmlir/Dialect/TT:headers",
        "//include/ttmlir/Dialect/TTIR/IR:headers",
        "//include/ttmlir/Dialect/TTIR:headers",
        "//include/ttmlir/Dialect/TTKernel/IR:headers",
        "//include/ttmlir/Dialect/TTKernel:headers",
        "//include/ttmlir/Dialect/TTMetal/IR:headers",
        "//include/ttmlir/Dialect/TTMetal:headers",
        "//include/ttmlir/Dialect/TTNN/IR:headers",
        "//include/ttmlir/Dialect/TTNN:headers",
        "//include/ttmlir/Dialect:headers",
        "//include/ttmlir/Target/Common:headers",
        "//include/ttmlir/Target/TTMetal:headers",
        "//include/ttmlir/Target/TTNN:headers",
        "//include/ttmlir/Target:headers",
        "//include/ttmlir/Conversion:headers",
        "//include/ttmlir:headers",
    ],
    visibility=["//visibility:public"],
)

