// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/binary.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_C, m) {
  m.doc() =
      "ttrt.binary python extension for loading / inspecting tt binary files";

  py::class_<tt::runtime::Binary>(m, "Binary")
      .def_property_readonly("version", &tt::runtime::binary::getVersion)
      .def_property_readonly("ttmlir_git_hash",
                             &tt::runtime::binary::getTTMLIRGitHash)
      .def("as_json", &tt::runtime::binary::asJson);
  m.def("load_from_path", &tt::runtime::binary::loadFromPath);
}
