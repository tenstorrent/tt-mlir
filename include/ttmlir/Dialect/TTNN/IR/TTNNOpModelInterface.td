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
                Returns the op kernel estimate in clock cycles.
            }],
            /*retTy=*/"size_t",
            /*methodName=*/"getOpPerfCycles",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$inputs, "const TTNNLayoutAttr&":$output),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return std::numeric_limits<size_t>::max();"
        >,
        InterfaceMethod<
            /*desc=*/[{
                Returns a tuple of three values:**
                1. A boolean indicating if the op is legal for the given input/output layouts.
                2. If the op is legal, a tuple of three values representing the op memory L1 usage estimate in bytes.
                   - The first value is the CB L1 peak allocation in bytes.
                   - The second value is the Tensor L1 peak allocation in bytes.
                   - The third value is the Output L1 buffer allocation in bytes.
                3. If the op is illegal, a string describing the failure.
            }],
            /*retTy=*/"std::tuple<bool, std::optional<std::tuple<size_t, size_t, size_t>>, std::optional<std::string>>",
            /*methodName=*/"getOpConstraints",
            /*args=*/(ins "const std::vector<TTNNLayoutAttr>&":$inputs, "const TTNNLayoutAttr&":$output),
            /*methodBody=*/"",
            /*defaultImplementation=*/"return std::make_tuple(true,std::make_tuple(0,0,0), std::nullopt);"
        >,
        ];
}

#endif // TTMLIR_TTMLIR_DIALECT_TTNN_TTNNOPSPERFINTERFACES_TD
