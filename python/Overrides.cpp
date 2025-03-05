// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/TTMLIRModule.h"

namespace mlir::ttmlir::python {

void populateOverridesModule(nb::module_ &m) {

  m.def(
      "get_ptr", [](void *op) { return reinterpret_cast<uintptr_t>(op); },
      nb::arg("op").noconvert());
}

} // namespace mlir::ttmlir::python
