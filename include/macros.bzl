# Collect all inc_gen targets in one place
def all_inc_gen():
    include_deps = [
        "//include/ttmlir/Dialect/TT/IR:tt_ops_inc_gen",
        "//include/ttmlir/Dialect/TT/IR:tt_enums_inc_gen",
        "//include/ttmlir/Dialect/TT/IR:tt_attrs_inc_gen",
        "//include/ttmlir/Dialect/TT/IR:tt_types_inc_gen",
        "//include/ttmlir/Dialect/TT/IR:tt_dialect_inc_gen",
        "//include/ttmlir/Dialect/TTIR/IR:ttir_ops_inc_gen",
        "//include/ttmlir/Dialect/TTIR/IR:ttir_interfaces_inc_gen",
        "//include/ttmlir/Dialect/TTIR/IR:ttir_dialect_inc_gen",
        "//include/ttmlir/Dialect/TTIR:ttir_pass_inc_gen",
        "//include/ttmlir/Dialect/TTKernel/IR:ttkernel_ops_inc_gen",
        "//include/ttmlir/Dialect/TTKernel/IR:ttkernel_enums_inc_gen",
        "//include/ttmlir/Dialect/TTKernel/IR:ttkernel_attrs_inc_gen",
        "//include/ttmlir/Dialect/TTKernel/IR:ttkernel_types_inc_gen",
        "//include/ttmlir/Dialect/TTKernel/IR:ttkernel_dialect_inc_gen",
        "//include/ttmlir/Dialect/TTMetal/IR:ttmetal_ops_inc_gen",
        "//include/ttmlir/Dialect/TTMetal/IR:ttmetal_enums_inc_gen",
        "//include/ttmlir/Dialect/TTMetal/IR:ttmetal_attrs_inc_gen",
        "//include/ttmlir/Dialect/TTMetal/IR:ttmetal_types_inc_gen",
        "//include/ttmlir/Dialect/TTMetal/IR:ttmetal_dialect_inc_gen",
        "//include/ttmlir/Dialect/TTMetal:ttmetal_pass_inc_gen",
        "//include/ttmlir/Dialect/TTNN/IR:ttnn_ops_inc_gen",
        "//include/ttmlir/Dialect/TTNN/IR:ttnn_enums_inc_gen",
        "//include/ttmlir/Dialect/TTNN/IR:ttnn_attrs_inc_gen",
        "//include/ttmlir/Dialect/TTNN/IR:ttnn_types_inc_gen",
        "//include/ttmlir/Dialect/TTNN/IR:ttnn_dialect_inc_gen",
        "//include/ttmlir/Dialect/TTNN:ttnn_pass_inc_gen",
        "//include/ttmlir/Conversion:ttconversion_pass_inc_gen",
    ]
    return include_deps

def create_flatbuffer(name):
    native.genrule(
        name = "name_generation",
        srcs = [name],
        outs = [name.split('.')[0] + "_generated.h"],
        cmd = "($(location @flatbuffers//:flatc) --bfbs-gen-embed --cpp --cpp-std c++17 --scoped-enums --warnings-as-errors -o $(@D) $(location " + name + "))",
        tools = ["@flatbuffers//:flatc"],
    )
