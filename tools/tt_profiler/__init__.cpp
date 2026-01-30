// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_profiler.hpp"

#include "nanobind_headers.h"

namespace nb = nanobind;

NB_MODULE(_ttmlir_profiler, m) {
  m.doc() = "Profiler bindings for TTMLIR runtime";
  tt::profiler::python::registerProfilerBindings(m);
}
