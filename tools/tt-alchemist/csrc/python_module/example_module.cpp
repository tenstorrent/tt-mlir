// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/embed.h>

namespace py = pybind11;

// Intended to be opened via dlopen and called as a function from C++ land
// Use extern "C" to prevent name mangling and make it findable via dlsym
extern "C" {
int entrypoint() {
    py::scoped_interpreter guard{}; // start the interpreter and keep it alive

    py::exec(R"(
        import a
        print(a.message)
    )");
    return 0;
}
}
