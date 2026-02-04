// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/debug.h"
#include "tt/runtime/perf.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"

#include "profiler.h"

#include "tt/runtime/detail/python/nanobind_headers.h"
#include "profiler_impl.h"

namespace nb = nanobind;

namespace tt::runtime::python {
void registerProfilerBindings(nb::module_ &m) {
  m.def("start_profiler", &tt::runtime::profiler::start_profiler,
        nb::arg("outputDirectory"),
        nb::arg("address") = "localhost",
        nb::arg("port") = 8086,
        "Start the profiler with given parameters");

  m.def("stop_profiler", &tt::runtime::profiler::stop_profiler,
        "Stop the profiler");
}
} // namespace tt::runtime::python