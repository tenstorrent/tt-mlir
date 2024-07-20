filegroup(
    name="headers",
    srcs=glob(["**/*.h"]),
    visibility=["//visibility:public"],
)

filegroup(
    name="fbs_files",
    srcs=glob(["*.fbs"]),
    visibility=["//visibility:public"],
)

# TODO: Shouldn't have to call each genrule twice
genrule(
    name = "debug_info",
    srcs = ["debug_info.fbs"],
    outs = ["debug_info_generated.h"],
    cmd = "($(location @flatbuffers//:flatc) --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location debug_info.fbs))",
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "debug_info_bfbs",
    srcs = ["debug_info.fbs"],
    outs = ["debug_info_bfbs_generated.h"],
    cmd = "($(location @flatbuffers//:flatc) --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location debug_info.fbs))",
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "types",
    srcs = ["types.fbs"],
    outs = ["types_generated.h"],
    cmd = "($(location @flatbuffers//:flatc) --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location types.fbs))",
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "types_bfbs",
    srcs = ["types.fbs"],
    outs = ["types_bfbs_generated.h"],
    cmd = "($(location @flatbuffers//:flatc) --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location types.fbs))",
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "version",
    srcs = ["version.fbs"],
    outs = ["version_generated.h"],
    cmd = "($(location @flatbuffers//:flatc) --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location version.fbs))",
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "version_bfbs",
    srcs = ["version.fbs"],
    outs = ["version_bfbs_generated.h"],
    cmd = "($(location @flatbuffers//:flatc) --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location version.fbs))",
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "system_desc",
    srcs = glob(["*.fbs"]),
    outs = ["system_desc_generated.h",],
    cmd = "($(location @flatbuffers//:flatc) --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location system_desc.fbs))",
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "system_desc_bfbs",
    srcs = glob(["*.fbs"]),
    outs = ["system_desc_bfbs_generated.h",],
    cmd = "($(location @flatbuffers//:flatc) --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location system_desc.fbs))",
    tools = ["@flatbuffers//:flatc"],
)


cc_library(
    name = "fbs_lib",
    strip_include_prefix=".",
    srcs = [
        "debug_info",
        "debug_info_bfbs",
        "types",
        "types_bfbs",
        "version",
        "version_bfbs",
        "system_desc",
        "system_desc_bfbs",
    ],
    hdrs = [
        "debug_info_generated.h",
        "debug_info_bfbs_generated.h",
        "types_generated.h",
        "types_bfbs_generated.h",
        "version_generated.h",
        "version_bfbs_generated.h",
        "system_desc_generated.h",
        "system_desc_bfbs_generated.h",
    ],
    deps = [
        "@flatbuffers//:flatbuffers",
    ],
    visibility = ["//visibility:public"],
)