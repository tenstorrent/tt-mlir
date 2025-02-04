// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "flatbuffers/idl.h"

#include "tt/runtime/detail/logger.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/Common/system_desc_bfbs_generated.h"
#include "ttmlir/Target/Common/system_desc_generated.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/TTMetal/binary_bfbs_generated.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/TTNN/binary_bfbs_generated.h"

namespace tt::runtime {

static std::string asJson(void const *fbb, uint8_t const *binarySchema,
                          size_t schemaSize) {
  flatbuffers::IDLOptions opts;
  opts.size_prefixed = true;
  opts.strict_json = true;
  opts.output_default_scalars_in_json = true;
  flatbuffers::Parser parser(opts);

  if (not parser.Deserialize(binarySchema, schemaSize)) {
    LOG_FATAL("Failed to deserialize schema");
  }

  std::string text;
  const char *err = ::flatbuffers::GenerateText(parser, fbb, &text);
  LOG_ASSERT(not err, "Failed to generate JSON: ", err);
  return text;
}

static std::vector<uint32_t>
calculateStride(std::vector<uint32_t> const &shape) {
  LOG_ASSERT(!shape.empty());
  std::vector<uint32_t> stride(shape.size(), 1);
  for (size_t i = shape.size() - 1; i > 0; i--) {
    stride[i - 1] = stride[i] * shape[i];
  }
  return stride;
}

namespace ttnn {

::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  LOG_ASSERT(isTTNN, "Unsupported binary format");
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

std::string getVersion(Flatbuffer binary) {
  auto const *version = getBinary(binary)->version();
  return std::to_string(version->major()) + "." +
         std::to_string(version->minor()) + "." +
         std::to_string(version->patch());
}

std::string_view getTTMLIRGitHash(Flatbuffer binary) {
  return getBinary(binary)->ttmlir_git_hash()->c_str();
}

std::string asJson(Flatbuffer binary) {
  return ::tt::runtime::asJson(
      binary.handle.get(), ::tt::target::ttnn::TTNNBinaryBinarySchema::data(),
      ::tt::target::ttnn::TTNNBinaryBinarySchema::size());
}

std::vector<TensorDesc> getProgramInputs(Flatbuffer binary,
                                         std::uint32_t programIndex) {
  std::vector<TensorDesc> inputs;
  auto const *program = getBinary(binary)->programs()->Get(programIndex);
  for (auto const *input : *program->inputs()) {
    TensorDesc desc;
    desc.shape = {input->desc()->shape()->begin(),
                  input->desc()->shape()->end()};
    desc.stride = calculateStride(desc.shape);
    desc.itemsize = ::tt::runtime::utils::dataTypeElementSize(
        input->desc()->layout()->memory_desc()->data_type());
    desc.dataType = input->desc()->layout()->memory_desc()->data_type();
    inputs.push_back(desc);
  }
  return inputs;
}

std::vector<TensorDesc> getProgramOutputs(Flatbuffer binary,
                                          std::uint32_t programIndex) {
  std::vector<TensorDesc> outputs;
  auto const *program = getBinary(binary)->programs()->Get(programIndex);
  for (auto const *output : *program->outputs()) {
    TensorDesc desc;
    desc.shape = {output->desc()->shape()->begin(),
                  output->desc()->shape()->end()};
    desc.stride = calculateStride(desc.shape);
    desc.itemsize = ::tt::runtime::utils::dataTypeElementSize(
        output->desc()->layout()->memory_desc()->data_type());
    desc.dataType = output->desc()->layout()->memory_desc()->data_type();
    outputs.push_back(desc);
  }
  return outputs;
}

const ::tt::target::GoldenTensor *getDebugInfoGolden(Flatbuffer binary,
                                                     std::string &loc) {
  auto const *programs = getBinary(binary)->programs();
  for (auto const *program : *programs) {
    for (const ::tt::target::GoldenKV *goldenKV :
         *program->debug_info()->golden_info()->golden_map()) {
      if (std::string(goldenKV->key()->c_str()) == loc) {
        return goldenKV->value();
        ;
      }
    }
  }

  LOG_WARNING("Golden information not found");
  return nullptr;
}

} // namespace ttnn

namespace metal {

::tt::target::metal::TTMetalBinary const *getBinary(Flatbuffer binary) {
  bool isTTMetal =
      ::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          binary.handle.get());
  LOG_ASSERT(isTTMetal, "Unsupported binary format");
  return ::tt::target::metal::GetSizePrefixedTTMetalBinary(binary.handle.get());
}

std::string getVersion(Flatbuffer binary) {
  auto const *version = getBinary(binary)->version();
  return std::to_string(version->major()) + "." +
         std::to_string(version->minor()) + "." +
         std::to_string(version->patch());
}

std::string_view getTTMLIRGitHash(Flatbuffer binary) {
  return getBinary(binary)->ttmlir_git_hash()->c_str();
}

std::string asJson(Flatbuffer binary) {
  return ::tt::runtime::asJson(
      binary.handle.get(),
      ::tt::target::metal::TTMetalBinaryBinarySchema::data(),
      ::tt::target::metal::TTMetalBinaryBinarySchema::size());
}

std::vector<TensorDesc> getProgramInputs(Flatbuffer binary,
                                         std::uint32_t programIndex) {
  std::vector<TensorDesc> inputs;
  auto const *program = getBinary(binary)->programs()->Get(programIndex);
  LOG_ASSERT(program->device_programs()->size() == 1,
             "Currently only one device program is supported, got: ",
             program->device_programs()->size());
  for (auto const *input : *program->device_programs()->Get(0)->inputs()) {
    TensorDesc desc;
    desc.shape = {input->desc()->shape()->begin(),
                  input->desc()->shape()->end()};
    desc.stride = calculateStride(desc.shape);
    desc.itemsize = utils::dataTypeElementSize(
        input->desc()->layout()->memory_desc()->data_type());
    desc.dataType = input->desc()->layout()->memory_desc()->data_type();
    inputs.push_back(desc);
  }
  return inputs;
}

std::vector<TensorDesc> getProgramOutputs(Flatbuffer binary,
                                          std::uint32_t programIndex) {
  std::vector<TensorDesc> outputs;
  auto const *program = getBinary(binary)->programs()->Get(programIndex);
  LOG_ASSERT(program->device_programs()->size() == 1,
             "Currently only one device program is supported, got: ",
             program->device_programs()->size());
  for (auto const *output : *program->device_programs()->Get(0)->outputs()) {
    TensorDesc desc;
    desc.shape = {output->desc()->shape()->begin(),
                  output->desc()->shape()->end()};
    desc.stride = calculateStride(desc.shape);
    desc.itemsize = utils::dataTypeElementSize(
        output->desc()->layout()->memory_desc()->data_type());
    desc.dataType = output->desc()->layout()->memory_desc()->data_type();
    outputs.push_back(desc);
  }
  return outputs;
}

const ::tt::target::GoldenTensor *getDebugInfoGolden(Flatbuffer binary,
                                                     std::string &loc) {
  LOG_WARNING("Debug golden information not enabled for metal yet!");
  return nullptr;
}

} // namespace metal

namespace system_desc {

::tt::target::SystemDescRoot const *getBinary(Flatbuffer binary) {
  if (!::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          binary.handle.get())) {
    LOG_FATAL("Unsupported binary format");
  }
  return ::tt::target::GetSizePrefixedSystemDescRoot(binary.handle.get());
}

std::string getVersion(Flatbuffer binary) {
  auto const *version = getBinary(binary)->version();
  return std::to_string(version->major()) + "." +
         std::to_string(version->minor()) + "." +
         std::to_string(version->patch());
}

std::string_view getTTMLIRGitHash(Flatbuffer binary) {
  return getBinary(binary)->ttmlir_git_hash()->c_str();
}

std::string asJson(Flatbuffer binary) {
  return ::tt::runtime::asJson(
      binary.handle.get(), ::tt::target::SystemDescRootBinarySchema::data(),
      ::tt::target::SystemDescRootBinarySchema::size());
}

} // namespace system_desc

Flatbuffer Flatbuffer::loadFromPath(char const *path) {
  // load a flatbuffer from path
  std::ifstream fbb(path, std::ios::binary | std::ios::ate);
  LOG_ASSERT(fbb.is_open(), "Failed to open file: ", path);
  std::streampos size = fbb.tellg();
  fbb.seekg(0, std::ios::beg);
  auto buffer = ::tt::runtime::utils::malloc_shared(size);
  fbb.read(static_cast<char *>(buffer.get()), size);
  return Flatbuffer(buffer);
}

void Flatbuffer::store(char const *path) const {
  // store a flatbuffer to path
  std::ofstream fbb(path, std::ios::binary);
  auto size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(handle.get()));
  fbb.write(reinterpret_cast<char const *>(handle.get()), size);
}

std::string_view Flatbuffer::getFileIdentifier() const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ::tt::target::ttnn::TTNNBinaryIdentifier();
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return ::tt::target::metal::TTMetalBinaryIdentifier();
  }

  if (::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          handle.get())) {
    return ::tt::target::SystemDescRootIdentifier();
  }

  LOG_FATAL("Unsupported binary format");
}

std::string Flatbuffer::getVersion() const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getVersion(*this);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getVersion(*this);
  }

  if (::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          handle.get())) {
    return system_desc::getVersion(*this);
  }

  LOG_FATAL("Unsupported binary format");
}

std::string_view Flatbuffer::getTTMLIRGitHash() const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getTTMLIRGitHash(*this);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getTTMLIRGitHash(*this);
  }

  if (::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          handle.get())) {
    return system_desc::getTTMLIRGitHash(*this);
  }

  LOG_FATAL("Unsupported binary format");
}

std::string Flatbuffer::asJson() const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::asJson(*this);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::asJson(*this);
  }

  if (::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          handle.get())) {
    return system_desc::asJson(*this);
  }

  LOG_FATAL("Unsupported binary format");
}

SystemDesc SystemDesc::loadFromPath(char const *path) {
  return SystemDesc(Flatbuffer::loadFromPath(path).handle);
}

Binary Binary::loadFromPath(char const *path) {
  return Binary(Flatbuffer::loadFromPath(path).handle);
}

std::vector<TensorDesc>
Binary::getProgramInputs(std::uint32_t programIndex) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getProgramInputs(*this, programIndex);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getProgramInputs(*this, programIndex);
  }

  LOG_FATAL("Unsupported binary format");
}

std::vector<TensorDesc>
Binary::getProgramOutputs(std::uint32_t programIndex) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getProgramOutputs(*this, programIndex);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getProgramOutputs(*this, programIndex);
  }

  LOG_FATAL("Unsupported binary format");
}

const ::tt::target::GoldenTensor *
Binary::getDebugInfoGolden(std::string &loc) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getDebugInfoGolden(*this, loc);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getDebugInfoGolden(*this, loc);
  }

  LOG_FATAL("Unsupported binary format for obtaining golden information");
}

} // namespace tt::runtime
