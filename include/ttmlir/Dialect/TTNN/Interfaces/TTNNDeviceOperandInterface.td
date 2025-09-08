// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_DIALECT_TTNN_INTERFACES_TTNNDEVICEOPERANDINTERFACE_TD
#define TTMLIR_DIALECT_TTNN_INTERFACES_TTNNDEVICEOPERANDINTERFACE_TD

include "mlir/IR/OpBase.td"

// This interface is used to specify device operands for TTNN operations.
def TTNN_DeviceOperandInterface : OpInterface<"TTNNDeviceOperandInterface"> {
  let cppNamespace = "::mlir::tt::ttnn";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Get the device operand value.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"getDevice",
      /*args=*/(ins),
      /*methodBody=*/[{
        return $_op.getDevice();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Set the device operand value. For operations with required device operands,
        this sets the device value. For operations with optional device operands,
        this assigns the device value if provided, or clears the operand if null.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setDevice",
      /*args=*/(ins "Value": $device),
      /*methodBody=*/[{
        // Use SFINAE to detect if this is an optional operand
        if constexpr (std::is_same_v<decltype($_op.getDeviceMutable()), ::mlir::MutableOperandRange>) {
          // Optional operand case
          if (device) {
            $_op.getDeviceMutable().assign(device);
          } else {
            $_op.getDeviceMutable().clear();
          }
        } else {
          // Required operand case - set the value directly
          if (device) {
            $_op.getDeviceMutable().assign(device);
          }
          else {
            llvm::report_fatal_error("Device operand cannot be null for operations with required device operands.");
          }
        }
      }]
    >,
  ];
}

#endif // TTMLIR_DIALECT_TTNN_INTERFACES_TTNNDEVICEOPERANDINTERFACE_TD
