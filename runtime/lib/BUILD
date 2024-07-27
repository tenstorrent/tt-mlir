load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
bool_flag(
    name = "enable_runtime",
    build_setting_default = True,
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_ttnn",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

bool_flag(
    name = "enable_metal",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

config_setting(
    name = "_enable_runtime",
    flag_values = {
        ":enable_runtime": "true",
    },
)

config_setting(
    name = "_enable_ttnn",
    flag_values = {
        ":enable_ttnn": "true",
    },
)

cc_library(
  name = "TTRuntime",
  srcs = [
    "binary.cpp",
    "runtime.cpp",
  ],
  linkstatic = 1,
  deps = [
    "//include/ttmlir/Target/TTNN:fbs_lib",
    "//include/ttmlir/Target/TTMetal:fbs_lib",
    "//include/ttmlir/Target/Common:fbs_lib",
    "//include:headers",
    "//runtime/include:headers",
    "@flatbuffers//:flatbuffers",
  ] + select({
        ":_enable_ttnn": [":TTNNRuntime"],
        "//conditions:default": [],
  }),
  copts = ["-I$(GENDIR)/include"],
)


cc_library(
  name = "TTNNRuntime",
  srcs = [
    "ttnn/runtime.cpp",
    "ttnn/program.cpp",
  ],
  target_compatible_with = select({
      ":_enable_ttnn": [],
      "//conditions:default": ["@platforms//:incompatible"],
  }),
  deps = [
    "//include:headers",
    "//runtime/include:headers",
    "//third_party:tt-metal",
  ]
)