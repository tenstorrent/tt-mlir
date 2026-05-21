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
    std::shared_ptr<CompiledProgram> program;
    std::uint32_t program_index{};
    std::vector<tt::runtime::TensorDesc> input_descs;
    std::vector<tt::runtime::Layout> input_layouts;
    std::vector<std::optional<tt::runtime::Tensor>> input_slots;
};

ExecutionPayload::ExecutionPayload(std::shared_ptr<CompiledProgram> program, std::uint32_t program_index)
    : impl_(std::make_unique<Impl>()) {
    TT_FATAL(program != nullptr, "ExecutionPayload: program must not be null");
    TT_FATAL(program_index < program->num_programs(),
             "ExecutionPayload: program_index {} out of range (binary has {} program(s))", program_index,
             program->num_programs());

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
    TT_FATAL(index < impl_->input_slots.size(), "bind_tensor: index {} out of range (program has {} input(s))", index,
             impl_->input_slots.size());

    // Stride/physicalVolume legitimately differ between the user's host tensor
    // and the binary's padded device layout — that's what toLayout reconciles.
    const tt::runtime::TensorDesc actual = tt::runtime::getTensorDesc(tensor);
    const tt::runtime::TensorDesc &expected = impl_->input_descs[index];
    TT_FATAL(actual.shape == expected.shape && actual.dataType == expected.dataType,
             "bind_tensor: tensor for input {} does not match the program's expected desc. "
             "expected shape={} dtype={}; got shape={} dtype={}",
             index, to_string(expected.shape), as<int>(expected.dataType), to_string(actual.shape),
             as<int>(actual.dataType));

    try {
        impl_->input_slots[index] =
            tt::runtime::toLayout(tensor, runtime_device(), impl_->input_layouts[index], /*retain=*/true);
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
        std::vector<tt::runtime::Tensor> outputs;
        outputs = tt::runtime::submit(runtime_device(), impl_->program->binary, impl_->program_index, inputs);
        {
            ZoneScopedN("tt_kurbla::wait");
            tt::runtime::wait(outputs);
        }
        return outputs;
    } catch (const std::exception &e) {
        TT_THROW("run: submit failed: {}", e.what());
    }
}

} // namespace tt::kurbla
