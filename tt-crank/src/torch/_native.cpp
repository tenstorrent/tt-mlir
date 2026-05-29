#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Value.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <torch/csrc/autograd/python_variable.h>
#include <ttmlir/Target/Common/types_generated.h>

#include "torch/backend.hpp"
#include "torch/ops/builders.hpp"
#include "torch/ops/fallback.hpp"
#include "torch/ttir_module_builder.hpp"

namespace nb = nanobind;
using namespace nb::literals; // for `"name"_a` argument literals

namespace tk = ::tt::kurbla::torch_backend;

namespace {

// Holds a ModuleBuilder until `compile()` consumes it. Subsequent method calls
// after compile() raise — the underlying MLIR module has been moved out.
class PyModuleBuilder {
public:
    explicit PyModuleBuilder(std::vector<tk::TensorTypeSpec> specs) : mb_(tk::ModuleBuilder::init(specs)) {}

    PyModuleBuilder(PyModuleBuilder &&) = default;
    PyModuleBuilder &operator=(PyModuleBuilder &&) = default;
    PyModuleBuilder(const PyModuleBuilder &) = delete;
    PyModuleBuilder &operator=(const PyModuleBuilder &) = delete;

    mlir::Value arg(std::size_t index) {
        assert_builder();
        const auto args = mb_->args();
        TORCH_CHECK(index < args.size(), "tt-kurbla ModuleBuilder.arg: index ", index, " out of range (", args.size(),
                    " args)");
        return args[index];
    }

    mlir::Value add(mlir::Value lhs, mlir::Value rhs, double alpha) {
        assert_builder();
        return tk::build_add(*mb_, lhs, rhs, alpha);
    }

    mlir::Value mm(mlir::Value lhs, mlir::Value rhs) {
        assert_builder();
        return tk::build_mm(*mb_, lhs, rhs);
    }

    mlir::Value addmm(mlir::Value bias, mlir::Value mat1, mlir::Value mat2, double beta, double alpha) {
        assert_builder();
        return tk::build_addmm(*mb_, bias, mat1, mat2, beta, alpha);
    }

    // Lift a Python scalar to a broadcastable `ttir.constant` at `dtype`.
    mlir::Value scalar(::tt::target::DataType dtype, double value) {
        assert_builder();
        return tk::build_scalar(*mb_, tk::mlir_element_type_for(tk::to_torch_dtype(dtype)), value);
    }

    // Cast `value` to a tensor with `dtype`'s element type, preserving shape.
    // No-op if `value` is already at that element type.
    mlir::Value typecast(mlir::Value value, ::tt::target::DataType dtype) {
        assert_builder();
        return mb_->insert_typecast(value, tk::mlir_element_type_for(tk::to_torch_dtype(dtype)));
    }

    std::shared_ptr<::tt::kurbla::CompiledProgram> compile(std::vector<mlir::Value> outputs) {
        assert_builder();
        auto module_op = std::move(*mb_).finalize(outputs);
        mb_.reset();
        return tk::compile_module(std::move(module_op));
    }

private:
    void assert_builder() const {
        TORCH_CHECK(mb_.has_value(), "tt-kurbla ModuleBuilder: builder already consumed by compile()");
    }

    std::optional<tk::ModuleBuilder> mb_;
};

// Python `torch.Tensor` -> `at::Tensor` at the binding boundary, since
// nanobind has no built-in caster for `at::Tensor`. Throws if `h` isn't a
// `torch.Tensor`. Returns an independent strong ref on the underlying
// TensorImpl; the Python wrapper's ref is unchanged.
at::Tensor unpack_torch_tensor(nb::handle h) {
    PyObject *p = h.ptr();
    TORCH_CHECK(THPVariable_Check(p) != 0, "tt-kurbla: expected a torch.Tensor");
    return THPVariable_Unpack(p);
}

// `at::Tensor` -> Python `torch.Tensor`. `THPVariable_Wrap` returns a new
// PyObject* with refcount=1; `nb::steal` adopts that ref without bumping it
// again (which `nb::borrow` would do, leaking).
nb::object wrap_torch_tensor(at::Tensor t) {
    PyObject *p = THPVariable_Wrap(std::move(t));
    if (p == nullptr) {
        throw nb::python_error();
    }
    return nb::steal<nb::object>(p);
}

nb::list run_program(std::shared_ptr<::tt::kurbla::CompiledProgram> program, nb::list inputs,
                     std::vector<::tt::target::DataType> output_dtypes) {
    std::vector<at::Tensor> at_inputs;
    at_inputs.reserve(nb::len(inputs));
    for (auto item : inputs) {
        at_inputs.push_back(unpack_torch_tensor(item));
    }

    auto outputs = tk::run_compiled_program(program, at_inputs, output_dtypes);

    nb::list result;
    for (auto &t : outputs) {
        result.append(wrap_torch_tensor(std::move(t)));
    }
    return result;
}

} // namespace

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
    tk::register_allocator();

    m.doc() = "tt-kurbla torch backend native module";
    m.def("loaded", []() { return true; });

    // Strict-fallback toggle. Tests flip this on to assert that a code path
    // never falls back to CPU; the catch-all fallback raises instead of
    // running while strict mode is set.
    m.def("set_fallback_strict", &tk::set_fallback_strict);
    m.def("fallback_strict", &tk::fallback_strict);

    // ====== torch.compile backend bindings ======
    //
    // The Python `tt_kurbla.torch._compile` module walks an FX graph and calls
    // these to emit a single TTIR module, compile it, and run it on each
    // invocation. Phase-0 surface: add only — enough for a stacked-add model.

    nb::enum_<::tt::target::DataType>(m, "DataType")
        .value("Float32", ::tt::target::DataType::Float32)
        .value("Float64", ::tt::target::DataType::Float64)
        .value("Float16", ::tt::target::DataType::Float16)
        .value("BFloat16", ::tt::target::DataType::BFloat16)
        .value("Int32", ::tt::target::DataType::Int32)
        .value("Int64", ::tt::target::DataType::Int64)
        .value("Bool", ::tt::target::DataType::Bool)
        .value("UInt8", ::tt::target::DataType::UInt8);

    nb::class_<tk::TensorTypeSpec>(m, "TensorTypeSpec")
        .def(
            "__init__",
            [](tk::TensorTypeSpec *self, std::vector<std::int64_t> shape, ::tt::target::DataType dtype) {
                new (self) tk::TensorTypeSpec{std::move(shape), dtype};
            },
            "shape"_a, "dtype"_a);

    // Opaque handle to an `mlir::Value` living inside an in-flight
    // ModuleBuilder. Python should not construct these directly — they come
    // back from ModuleBuilder.arg() / per-op methods.
    nb::class_<mlir::Value>(m, "Value");

    nb::class_<PyModuleBuilder>(m, "ModuleBuilder")
        .def(nb::init<std::vector<tk::TensorTypeSpec>>(), "input_specs"_a)
        .def("arg", &PyModuleBuilder::arg, "index"_a)
        .def("add", &PyModuleBuilder::add, "lhs"_a, "rhs"_a, "alpha"_a = 1.0)
        .def("mm", &PyModuleBuilder::mm, "lhs"_a, "rhs"_a)
        .def("addmm", &PyModuleBuilder::addmm, "bias"_a, "mat1"_a, "mat2"_a, "beta"_a = 1.0, "alpha"_a = 1.0)
        .def("scalar", &PyModuleBuilder::scalar, "dtype"_a, "value"_a)
        .def("typecast", &PyModuleBuilder::typecast, "value"_a, "dtype"_a)
        // Consumes the builder. Subsequent calls on `self` raise.
        .def("compile", &PyModuleBuilder::compile, "outputs"_a);

    nb::class_<::tt::kurbla::CompiledProgram>(m, "CompiledProgram");

    m.def("run_program", &run_program, "program"_a, "inputs"_a, "output_dtypes"_a,
          "Bind tt-backend torch.Tensors to the compiled program's inputs and run it. "
          "`output_dtypes` is the per-output user-facing dtype the result tensors "
          "should be wrapped as (one per program output). "
          "Returns a list of tt-backend torch.Tensors.");
}
