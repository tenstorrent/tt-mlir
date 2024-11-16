// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD

include "mlir/IR/OpBase.td"

def TTNN_OpModelInterface : OpInterface<"OpModel"> {
    let description = [{
        Interface to access TTNN op model methods.
    }];
    let cppNamespace = "::mlir::tt::ttnn";
    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Returns the op kernel estimate in clock cycles.
            }],
            /*retTy=*/"size_t",
            /*methodName=*/"getOpPerfCycles",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$input_layouts, "const TTNNLayoutAttr&":$output_layout),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return std::numeric_limits<size_t>::max();"
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns the op memory L1 usage estimate in bytes. The return value is a tuple of 3 values:
                - The first value is CB L1 peak allocation in bytes.
                - The second value is Tensor L1 peak allocation in bytes.
                - The third value is Output L1 buffer allocation in bytes.
            }],
            /*retTy=*/"std::tuple<size_t, size_t, size_t>",
            /*methodName=*/"getOpL1Usage",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$input_layouts, "const TTNNLayoutAttr&":$output_layout),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return std::make_tuple(0,0,0);"
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns if input/output layouts are legal for the op.
            }],
            /*retTy=*/"bool",
            /*methodName=*/"isOpLegal",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$input_layouts, "const TTNNLayoutAttr&":$output_layout),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return true;"
        >,
        ];
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD
