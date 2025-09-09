// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNOPMODELINTERFACE_TD
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNOPMODELINTERFACE_TD

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
                Measures and returns the runtime of the op in nanoseconds by running it on the device.
                If the op is illegal or execution fails, returns an Error with a string describing the failure.
            }],
            /*retTy=*/"llvm::Expected<size_t>",
            /*methodName=*/"getOpRuntime",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$inputs, "const OpConfig&":$config),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return llvm::createStringError(\"Not Implemented\");"
        >,
        InterfaceMethod<
            /*desc=*/[{
                Checks if the op is legal for the given input/output layout. If it is, returns a tuple of three values:**
                   - The first value is the CB L1 peak allocation in bytes.
                   - The second value is the Tensor L1 peak allocation in bytes.
                   - The third value is the Output L1 buffer allocation in bytes.
                   - The fourth value is the actual TTNNLayoutAttr of the output tensor, as returned by the operation. For
                     some operations this may be different from the requested output layout
                If the op is illegal, returns an Error with a string describing the failure.
            }],
            /*retTy=*/"llvm::Expected<mlir::tt::ttnn::op_model::OpConstraints>",
            /*methodName=*/"getOpConstraints",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$inputs, "const OpConfig&":$config),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return llvm::createStringError(\"Not Implemented\");"
        >,
        ];
}

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNOPMODELINTERFACE_TD
