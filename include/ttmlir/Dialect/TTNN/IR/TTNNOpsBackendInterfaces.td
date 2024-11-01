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
            /*args=*/(ins), // TBD
            /*methodBody=*/"",
            /*defaultImplementation=*/"return std::numeric_limits<size_t>::max();"
        >,
        InterfaceMethod<
            /*desc=*/[{
                Return the op kernel estimate in clock cycles.
            }],
            /*retTy=*/"size_t",
            /*methodName=*/"getOpL1Usage",
            /*args=*/(ins), // TBD
            /*methodBody=*/"",
            /*defaultImplementation=*/"return 0;"
        >,
        InterfaceMethod<
            /*desc=*/[{
                Return the op kernel estimate in clock cycles.
            }],
            /*retTy=*/"bool",
            /*methodName=*/"isOpLegal",
            /*args=*/(ins), // TBD
            /*methodBody=*/"",
            /*defaultImplementation=*/"return true;"
        >,
        ];
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD
