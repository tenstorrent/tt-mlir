#include "engine/execution_payload.hpp"

#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <tt/runtime/runtime.h>

#include "engine/device.hpp"

namespace tt::kurbla {

namespace {

std::string format_shape(const std::vector<std::uint32_t> &shape) {
    std::ostringstream oss;
    oss << "[";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

} // namespace

struct ExecutionPayload::Impl {
    std::shared_ptr<CompiledProgram> program;
    std::uint32_t program_index{};
    std::vector<tt::runtime::TensorDesc> input_descs;
    std::vector<tt::runtime::Layout> input_layouts;
    std::vector<std::optional<tt::runtime::Tensor>> input_slots;
};

ExecutionPayload::ExecutionPayload(std::shared_ptr<CompiledProgram> program, std::uint32_t program_index)
    : impl_(std::make_unique<Impl>()) {
    if (!program) {
        throw ExecuteError("ExecutionPayload: program must not be null");
    }
    if (program_index >= program->num_programs()) {
        std::ostringstream oss;
        oss << "ExecutionPayload: program_index " << program_index << " out of range (binary has "
            << program->num_programs() << " program(s))";
        throw ExecuteError(oss.str());
    }

    impl_->program = std::move(program);
    impl_->program_index = program_index;
    impl_->input_descs = impl_->program->input_descs(program_index);
    impl_->input_slots.resize(impl_->input_descs.size());

    impl_->input_layouts.reserve(impl_->input_descs.size());
    for (std::uint32_t i = 0; i < impl_->input_descs.size(); ++i) {
        impl_->input_layouts.push_back(tt::runtime::getLayout(impl_->program->binary, program_index, i));
    }
}

ExecutionPayload::~ExecutionPayload() = default;
ExecutionPayload::ExecutionPayload(ExecutionPayload &&) noexcept = default;
ExecutionPayload &ExecutionPayload::operator=(ExecutionPayload &&) noexcept = default;

const std::shared_ptr<CompiledProgram> &ExecutionPayload::compiled_program() const {
    return impl_->program;
}

std::uint32_t ExecutionPayload::program_index() const {
    return impl_->program_index;
}

void ExecutionPayload::bind_tensor(const tt::runtime::Tensor &tensor, std::uint32_t index) {
    if (index >= impl_->input_slots.size()) {
        std::ostringstream oss;
        oss << "bind_tensor: index " << index << " out of range (program has " << impl_->input_slots.size()
            << " input(s))";
        throw InputBindingError(oss.str());
    }

    // Stride/physicalVolume legitimately differ between the user's host tensor
    // and the binary's padded device layout — that's what toLayout reconciles.
    const tt::runtime::TensorDesc actual = tt::runtime::getTensorDesc(tensor);
    const tt::runtime::TensorDesc &expected = impl_->input_descs[index];
    if (actual.shape != expected.shape || actual.dataType != expected.dataType) {
        std::ostringstream oss;
        oss << "bind_tensor: tensor for input " << index << " does not match the program's expected desc. "
            << "expected shape=" << format_shape(expected.shape) << " dtype=" << static_cast<int>(expected.dataType)
            << "; got shape=" << format_shape(actual.shape) << " dtype=" << static_cast<int>(actual.dataType);
        throw InputBindingError(oss.str());
    }

    try {
        impl_->input_slots[index] =
            tt::runtime::toLayout(tensor, runtime_device(), impl_->input_layouts[index], /*retain=*/true);
    } catch (const std::exception &e) {
        throw DeviceError(std::string("bind_tensor: toLayout failed: ") + e.what());
    }
}

std::vector<tt::runtime::Tensor> ExecutionPayload::run() {
    std::vector<std::uint32_t> missing;
    for (std::uint32_t i = 0; i < impl_->input_slots.size(); ++i) {
        if (!impl_->input_slots[i].has_value()) {
            missing.push_back(i);
        }
    }
    if (!missing.empty()) {
        std::ostringstream oss;
        oss << "run: missing input bindings at indices [";
        for (std::size_t i = 0; i < missing.size(); ++i) {
            if (i != 0) {
                oss << ", ";
            }
            oss << missing[i];
        }
        oss << "]";
        throw InputBindingError(oss.str());
    }

    std::vector<tt::runtime::Tensor> inputs;
    inputs.reserve(impl_->input_slots.size());
    for (const auto &slot : impl_->input_slots) {
        // Guaranteed populated by the missing-indices check above.
        // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
        inputs.push_back(slot.value());
    }

    try {
        std::vector<tt::runtime::Tensor> outputs =
            tt::runtime::submit(runtime_device(), impl_->program->binary, impl_->program_index, inputs);
        tt::runtime::wait(outputs);
        return outputs;
    } catch (const std::exception &e) {
        throw DeviceError(std::string("run: submit failed: ") + e.what());
    }
}

} // namespace tt::kurbla
