#include "engine/execution_payload.hpp"

#include "assert.hpp"
#include "cast.hpp"
#include "engine/device.hpp"
#include <exception>
#include <optional>
#include <sstream>
#include <tracy/Tracy.hpp>
#include <tt/runtime/runtime.h>
#include <utility>

namespace tt::kurbla {

namespace {

template <class T> std::string to_string(const std::vector<T> &vec) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace

struct ExecutionPayload::Impl {
    CompiledProgram *program{};
    std::vector<std::optional<tt::runtime::Tensor>> input_slots;
};

ExecutionPayload::ExecutionPayload(CompiledProgram &program) : impl_(std::make_unique<Impl>()) {
    impl_->program = &program;
    impl_->input_slots.resize(program.num_inputs);
}

ExecutionPayload::~ExecutionPayload() = default;
ExecutionPayload::ExecutionPayload(ExecutionPayload &&) noexcept = default;
ExecutionPayload &ExecutionPayload::operator=(ExecutionPayload &&) noexcept = default;

CompiledProgram &ExecutionPayload::compiled_program() const {
    return *impl_->program;
}

void ExecutionPayload::bind_tensor(tt::runtime::Tensor &tensor, std::uint32_t index) {
    TT_FATAL(index < impl_->input_slots.size(), "bind_tensor: index {} out of range (program has {} input(s))", index,
             impl_->input_slots.size());

    // Stride/physicalVolume legitimately differ between the user's host tensor
    // and the binary's padded device layout — that's what toLayout reconciles.
    const tt::runtime::TensorDesc actual = tt::runtime::getTensorDesc(tensor);
    const tt::runtime::TensorDesc &expected = impl_->program->input_descs[index];
    TT_FATAL(actual.shape == expected.shape && actual.dataType == expected.dataType,
             "bind_tensor: tensor for input {} does not match the program's expected desc. "
             "expected shape={} dtype={}; got shape={} dtype={}",
             index, to_string(expected.shape), as<int>(expected.dataType), to_string(actual.shape),
             as<int>(actual.dataType));

    try {
        const auto &layout = impl_->program->input_layout_at(index);
        if (!tt::runtime::hasLayout(tensor, layout)) {
            tensor = tt::runtime::toLayout(tensor, runtime_device(), layout, /*retain=*/true);
        }

        impl_->input_slots[index] = tensor;
    } catch (const std::exception &e) {
        TT_THROW("bind_tensor: toLayout failed: {}", e.what());
    }
}

std::vector<tt::runtime::Tensor> ExecutionPayload::run() {
    ZoneScopedN("tt_kurbla::ExecutionPayload::run");
    std::vector<std::uint32_t> missing;
    for (std::uint32_t i = 0; i < impl_->input_slots.size(); ++i) {
        if (!impl_->input_slots[i].has_value()) {
            missing.push_back(i);
        }
    }
    TT_FATAL(missing.empty(), "run: missing input bindings at indices {}", to_string(missing));

    std::vector<tt::runtime::Tensor> inputs;
    inputs.reserve(impl_->input_slots.size());
    for (const auto &slot : impl_->input_slots) {
        // Guaranteed populated by the missing-indices check above.
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        inputs.push_back(slot.value());
    }

    try {
        auto out = tt::runtime::submit(runtime_device(), impl_->program->binary, /*program_index=*/0, inputs);
        for (auto &tensor : out) {
            tt::runtime::setTensorRetain(tensor, true);
        }
        return out;
    } catch (const std::exception &e) {
        TT_THROW("run: submit failed: {}", e.what());
    }
}

} // namespace tt::kurbla
