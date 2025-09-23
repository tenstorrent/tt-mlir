// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if defined(TTMLIR_ENABLE_RUNTIME_TESTS) && TTMLIR_ENABLE_RUNTIME_TESTS == 1
#define RUNTIME_TEST_ENABLED
#endif

#if defined(TT_RUNTIME_ENABLE_TTNN) && TT_RUNTIME_ENABLE_TTNN == 1
#define RUNTIME_UTILS_ENABLED
#endif

#include "binary/binary.h"
#include "runtime/runtime.h"
#include "runtime/utils.h"

#if defined(RUNTIME_TEST_ENABLED)
#include "runtime/test.h"
#endif

#include "tt/runtime/detail/python/nanobind_headers.h"

namespace nb = nanobind;

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
NB_MODULE(_ttmlir_runtime, m) {
  m.doc() = "Python bindings for TTMLIR runtime";
  auto m_runtime = m.def_submodule("runtime", "Runtime API bindings");
  auto m_binary = m.def_submodule("binary", "Binary API bindings");
#if defined(RUNTIME_UTILS_ENABLED)
  auto m_utils = m.def_submodule("utils", "Runtime utility helpers");
#endif
#if defined(RUNTIME_TEST_ENABLED)
  auto m_runtime_test =
      m_runtime.def_submodule("test", "Runtime test API bindings");
#endif

  tt::runtime::python::registerRuntimeBindings(m_runtime);
  tt::runtime::python::registerBinaryBindings(m_binary);
#if defined(RUNTIME_UTILS_ENABLED)
  tt::runtime::python::registerRuntimeUtilsBindings(m_utils);
#endif
#if defined(RUNTIME_TEST_ENABLED)
  tt::runtime::python::registerRuntimeTestBindings(m_runtime_test);
#endif

#undef RUNTIME_TEST_ENABLED
}
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)
