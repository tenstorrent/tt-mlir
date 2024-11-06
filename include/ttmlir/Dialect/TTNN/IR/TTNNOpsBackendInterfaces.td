// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD

include "mlir/IR/OpBase.td"

def TTNNOpBackendInterface : OpInterface<"TTNNOpBackend"> {
    let description = [{
        Interface to access a registered method to infer the return types for an
        operation that can be used during type inference.
    }];
    let cppNamespace = "::mlir::tt::ttnn";
    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Return the op kernel estimate in clock cycles.
            }],
            /*retTy=*/"size_t",
            /*methodName=*/"getOpPerfCycles",
            /*args=*/(ins "const std::vector<tt::LayoutAttr>&":$input_layouts, "const tt::LayoutAttr&":$output_layout),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return std::numeric_limits<size_t>::max();"
        >,
        InterfaceMethod<
            /*desc=*/[{
                Return the op kernel estimate in clock cycles.
            }],
            /*retTy=*/"size_t",
            /*methodName=*/"getOpL1Usage",
            /*args=*/(ins "const std::vector<tt::LayoutAttr>&":$input_layouts, "const tt::LayoutAttr&":$output_layout),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return 0;"
        >,
        InterfaceMethod<
            /*desc=*/[{
                Return the op kernel estimate in clock cycles.
            }],
            /*retTy=*/"bool",
            /*methodName=*/"isOpLegal",
            /*args=*/(ins "const std::vector<tt::LayoutAttr>&":$input_layouts, "const tt::LayoutAttr&":$output_layout),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return true;"
        >,
        ];
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD
