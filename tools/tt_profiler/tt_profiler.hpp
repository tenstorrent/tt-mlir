// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_PROFILER_H
#define TT_PROFILER_H

#include "nanobind_headers.h"

namespace nb = nanobind;

namespace tt::profiler::python {

void registerProfilerBindings(nb::module_ &m);

} // namespace tt::profiler::python

#endif // TT_PROFILER_H
