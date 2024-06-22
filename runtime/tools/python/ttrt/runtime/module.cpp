// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/binary.h"
#include "tt/runtime/runtime.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() = "ttrt.runtime python extension for interacting with the "
            "Tenstorrent devies";

  m.def("get_current_system_desc", &tt::runtime::getCurrentSystemDesc,
        "Get the current system descriptor");
  m.def("open_device", &tt::runtime::openDevice,
        "Open a device for execution");
  m.def("close_device", &tt::runtime::closeDevice, "Close a device");
  m.def("submit", &tt::runtime::submit,
        "Submit a binary for execution");
}
