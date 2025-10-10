// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if defined(TTMLIR_ENABLE_RUNTIME_TESTS) && TTMLIR_ENABLE_RUNTIME_TESTS == 1

#include "tt/runtime/detail/test/ttnn/utils.h"
#include "tt/runtime/test/ttnn/dylib.h"

#include "tt/runtime/detail/python/nanobind_headers.h"

namespace nb = nanobind;

namespace tt::runtime::python {

void registerRuntimeTestBindings(nb::module_ &m) {
  m.def("get_dram_interleaved_tile_layout",
        &tt::runtime::test::ttnn::getDramInterleavedTileLayout,
        nb::arg("dtype"), "Get dram interleaved tile layout");
  m.def("get_dram_interleaved_row_major_layout",
        &tt::runtime::test::ttnn::getDramInterleavedRowMajorLayout,
        nb::arg("dtype"), "Get dram interleaved row major layout");
  m.def("get_host_row_major_layout",
        &tt::runtime::test::ttnn::getHostRowMajorLayout, nb::arg("dtype"),
        "Get host row major layout");
  m.def("open_so", &tt::runtime::test::ttnn::openSo, nb::arg("path"),
        "Open a shared object");
  m.def("close_so", &tt::runtime::test::ttnn::closeSo, nb::arg("handle"),
        "Close a shared object");
  m.def("get_so_programs", &tt::runtime::test::ttnn::getSoPrograms,
        nb::arg("so"), nb::arg("path"),
        "Get the program names from a shared object file");
  m.def("create_inputs", &tt::runtime::test::ttnn::createInputs, nb::arg("so"),
        nb::arg("func_name"), nb::arg("device"), nb::arg("path"),
        "Create inputs for a program from a shared object file");
  m.def("run_so_program", &tt::runtime::test::ttnn::runSoProgram, nb::arg("so"),
        nb::arg("func_name"), nb::arg("inputs"), nb::arg("device"),
        "Run a program from a shared object file");
  m.def("compare_outs", &tt::runtime::test::ttnn::compareOuts, nb::arg("lhs"),
        nb::arg("rhs"));
}
} // namespace tt::runtime::python

#endif
