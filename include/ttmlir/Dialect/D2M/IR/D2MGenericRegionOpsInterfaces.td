// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTMLIR_D2M_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_TD
#define TTMLIR_D2M_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_TD

include "mlir/IR/OpBase.td"
include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.td"

def D2M_SynchronizableOpInterface : OpInterface<"SynchronizableOpInterface"> {
  let description = [{
     Define a base interface for synchronizable ops in the generic region.
  }];

  let cppNamespace = "::mlir::tt::d2m";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{Returns true if the given operand is a producer (i.e. written
                 to) for synchronization purposes.}],
      /*retTy=*/"bool",
      /*methodName=*/"isProducer",
      /*args=*/(ins "mlir::OpOperand &":$operand),
      /*methodBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/[{Returns true if the given operand is a consumer (i.e. read
                 from) for synchronization purposes.}],
      /*retTy=*/"bool",
      /*methodName=*/"isConsumer",
      /*args=*/(ins "mlir::OpOperand &":$operand),
      /*methodBody=*/""
    >,
  ];

}

#endif // TTMLIR_D2M_DIALECT_D2M_IR_D2MGENERICREGIONOPSINTERFACES_TD
