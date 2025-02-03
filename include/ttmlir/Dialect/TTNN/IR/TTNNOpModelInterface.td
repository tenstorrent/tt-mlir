// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD

include "mlir/IR/OpBase.td"

// TODO(odjuricic): support ops with multiple outputs
def TTNN_OpModelInterface : OpInterface<"OpModel"> {
    let description = [{
        Interface to access TTNN op model methods.
    }];
    let cppNamespace = "::mlir::tt::ttnn";
    let methods = [
        InterfaceMethod<
            /*desc=*/[{
                Measure and return the runtime of the op in nanoseconds by running it on the device.
                If the op is illegal or execution fails, return an Error with a string describing the failure.
            }],
            /*retTy=*/"llvm::Expected<size_t>",
            /*methodName=*/"getOpRuntime",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$inputs, "const TTNNLayoutAttr&":$output),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return llvm::createStringError(\"Not Implemented\");"
        >,
        InterfaceMethod<
            /*desc=*/[{
                Check if the op is legal for the given input/output layout. If it is, returns a tuple of three values:**
                   - The first value is the CB L1 peak allocation in bytes.
                   - The second value is the Tensor L1 peak allocation in bytes.
                   - The third value is the Output L1 buffer allocation in bytes.
                If the op is illegal, return an Error with a string describing the failure.
            }],
            /*retTy=*/"llvm::Expected<std::tuple<size_t, size_t, size_t>>",
            /*methodName=*/"getOpConstraints",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$inputs, "const TTNNLayoutAttr&":$output),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return std::make_tuple(0,0,0);"
        >,
        ];
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD
