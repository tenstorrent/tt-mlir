#include <c10/core/Device.h>
#include <nanobind/nanobind.h>

#include "torch/backend.hpp"
#include "torch/ops/fallback.hpp"

namespace nb = nanobind;

NB_MODULE(_native, m) {
    // Establish the PrivateUse1 backend's canonical name from C++. This is the
    // c10 primitive that backs ScalarType / Device / dispatch-key lookups for
    // "tt". Calling it here keeps registration co-located with the
    // TORCH_LIBRARY_IMPL kernels that ship in this .so — if anyone ever loads
    // _native.so without going through the Python wrapper (unusual but
    // possible), the kernels are still wired correctly.
    //
    // The Python-side counterpart, torch.utils.rename_privateuse1_backend("tt")
    // in tt_kurbla.torch._device.register(), adds the framework-level
    // ergonomics on top (Tensor.tt() method, the autograd-aware dispatch-key
    // alias, etc.). It calls into the same c10 primitive internally and is a
    // no-op when the name already matches — the supported order is the one
    // enforced by python/tt_kurbla/torch/__init__.py: this C++ call first,
    // then register() afterwards.
    c10::register_privateuse1_backend("tt");

    // Op kernels (ops/tensor.cpp, ops/elementwise.cpp) and the PrivateUse1
    // DeviceGuardImpl self-register via global ctors when this .so loads —
    // TORCH_LIBRARY_IMPL / C10_REGISTER_GUARD_IMPL handle the dispatcher and
    // guard sides; we just need the allocator hook.
    tt::kurbla::torch_backend::register_allocator();

    m.doc() = "tt-kurbla torch backend native module";
    m.def("loaded", []() { return true; });

    // Strict-fallback toggle. Tests flip this on to assert that a code path
    // never falls back to CPU; the catch-all fallback raises instead of
    // running while strict mode is set.
    m.def("set_fallback_strict", &tt::kurbla::torch_backend::set_fallback_strict);
    m.def("fallback_strict", &tt::kurbla::torch_backend::fallback_strict);
}
