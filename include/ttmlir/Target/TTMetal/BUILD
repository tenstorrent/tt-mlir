filegroup(
    name="headers",
    srcs=glob(["**/*.h"]),
    visibility=["//visibility:public"],
)

# TODO: Shouldn't have to call each genrule twice
genrule(
    name = "command",
    srcs = glob(["*.fbs"]) + ["//include/ttmlir/Target/Common:fbs_files"],
    outs = ["command_generated.h"],
    cmd = """
    NAME=$(location program.fbs)
    PATH=$${NAME%/*/*}
    ($(location @flatbuffers//:flatc) -I $$PATH --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location command.fbs))
    """,
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "command_bfbs",
    srcs = glob(["*.fbs"]) + ["//include/ttmlir/Target/Common:fbs_files"],
    outs = ["command_bfbs_generated.h"],
    cmd = """
    NAME=$(location program.fbs)
    PATH=$${NAME%/*/*}
    ($(location @flatbuffers//:flatc) -I $$PATH --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location command.fbs))
    """,
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "program",
    srcs = glob(["*.fbs"]) + ["//include/ttmlir/Target/Common:fbs_files"],
    outs = ["program_generated.h"],
    cmd = """
    NAME=$(location program.fbs)
    PATH=$${NAME%/*/*}
    ($(location @flatbuffers//:flatc) -I $$PATH --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location program.fbs))
    """,
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "program_bfbs",
    srcs = glob(["*.fbs"]) + ["//include/ttmlir/Target/Common:fbs_files"],
    outs = ["program_bfbs_generated.h"],
    cmd = """
    NAME=$(location program.fbs)
    PATH=$${NAME%/*/*}
    ($(location @flatbuffers//:flatc) -I $$PATH --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location program.fbs))
    """,
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "binary",
    srcs = glob(["*.fbs"]) + ["//include/ttmlir/Target/Common:fbs_files"],
    outs = ["binary_generated.h"],
    cmd = """
    NAME=$(location program.fbs)
    PATH=$${NAME%/*/*}
    ($(location @flatbuffers//:flatc) -I $$PATH --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location binary.fbs))
    """,
    tools = ["@flatbuffers//:flatc"],
)
genrule(
    name = "binary_bfbs",
    srcs = glob(["*.fbs"]) + ["//include/ttmlir/Target/Common:fbs_files"],
    outs = ["binary_bfbs_generated.h"],
    cmd = """
    NAME=$(location program.fbs)
    PATH=$${NAME%/*/*}
    ($(location @flatbuffers//:flatc) -I $$PATH --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location binary.fbs))
    """,
    tools = ["@flatbuffers//:flatc"],
)


cc_library(
    name = "fbs_lib",
    strip_include_prefix=".",
    srcs = [
        "command",
        "command_bfbs",
        "program",
        "program_bfbs",
        "binary",
        "binary_bfbs",
    ],
    hdrs = [
        "command_generated.h",
        "command_bfbs_generated.h",
        "program_generated.h",
        "program_bfbs_generated.h",
        "binary_generated.h",
        "binary_bfbs_generated.h",
    ],
    deps = [
        "//include/ttmlir/Target/Common:fbs_lib",
        "@flatbuffers//:flatbuffers",
    ],
    copts = ["-Ittmlir/Target/Common"],
    visibility = ["//visibility:public"],
)
