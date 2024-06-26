// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Bindings/Python/Overrides.h"

namespace mlir::ttmlir::python {

void populateOverridesModule(py::module &m) {

  m.def(
      "get_ptr", [](void *op) { return reinterpret_cast<uintptr_t>(op); },
      py::arg("op").noconvert());
}

} // namespace mlir::ttmlir::python
