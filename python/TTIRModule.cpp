// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

#include "mlir/CAPI/IR.h"

namespace mlir::ttmlir::python {
void populateTTIRModule(py::module &m) {
  m.def("is_dps", [](MlirOperation op) {
    return mlir::isa<DestinationStyleOpInterface>(unwrap(op));
  });
}
} // namespace mlir::ttmlir::python
