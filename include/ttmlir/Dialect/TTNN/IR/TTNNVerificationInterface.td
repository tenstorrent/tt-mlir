// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_TTMLIR_DIALECT_TTNN_IR_TTNN_VERIFICATION_INTERFACE_TD
#define TTMLIR_TTMLIR_DIALECT_TTNN_IR_TTNN_VERIFICATION_INTERFACE_TD

include "mlir/IR/OpBase.td"

def TTNN_VerificationInterface : OpInterface<"TTNNVerificationInterface"> {
    let cppNamespace = "::mlir::tt::ttnn";

    let verify = [{
        return verifyTTNNVerificationInterface($_op);
    }];
}

def ChangesDataType : NativeOpTrait<"ChangesDataType">;
def ChangesBufferType : NativeOpTrait<"ChangesBufferType">;
def ChangesMemoryLayout: NativeOpTrait<"ChangesMemoryLayout">;
def ChangesTensorLayout : NativeOpTrait<"ChangesTensorLayout">;
def DoesNotChangeTensorEncoding : NativeOpTrait<"DoesNotChangeTensorEncoding">;

#endif
