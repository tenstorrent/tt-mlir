// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "flatbuffers/idl.h"

#include "tt/runtime/binary.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/Common/system_desc_bfbs_generated.h"
#include "ttmlir/Target/Common/system_desc_generated.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/TTNN/binary_bfbs_generated.h"

namespace tt::runtime::binary {

static std::string asJson(void const *fbb, uint8_t const *binarySchema,
                          size_t schemaSize) {
  flatbuffers::IDLOptions opts;
  opts.size_prefixed = true;
  opts.strict_json = true;
  flatbuffers::Parser parser(opts);

  if (not parser.Deserialize(binarySchema, schemaSize)) {
    throw std::runtime_error("Failed to deserialize schema");
  }

  std::string text;
  if (::flatbuffers::GenerateText(parser, fbb, &text)) {
    throw std::runtime_error("Failed to generate JSON");
  }

  return text;
}

namespace ttnn {

::tt::target::ttnn::TTNNBinary const *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  if (not isTTNN) {
    throw std::runtime_error("Unsupported binary format");
  }
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

std::string getVersion(Flatbuffer binary) {
  auto const *version = getBinary(binary)->version();
  return std::to_string(version->major()) + "." +
         std::to_string(version->minor()) + "." +
         std::to_string(version->release());
}

std::string_view getTTMLIRGitHash(Flatbuffer binary) {
  return getBinary(binary)->ttmlir_git_hash()->c_str();
}

std::string asJson(Flatbuffer binary) {
  return ::tt::runtime::binary::asJson(
      binary.handle.get(), ::tt::target::ttnn::TTNNBinaryBinarySchema::data(),
      ::tt::target::ttnn::TTNNBinaryBinarySchema::size());
}

std::vector<TensorDesc> getProgramInputs(Flatbuffer binary,
                                         std::uint32_t programIndex) {
  std::vector<TensorDesc> inputs;
  auto const *program = getBinary(binary)->programs()->Get(programIndex);
  for (auto const *input : *program->inputs()) {
    TensorDesc desc;
    desc.shape = {input->desc()->shape()->begin(), input->desc()->shape()->end()};
    desc.stride = {input->desc()->layout()->strides()->begin(),
                   input->desc()->layout()->strides()->end()};
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
  for (auto const *output : *program->outputs()) {
    TensorDesc desc;
    desc.shape = {output->desc()->shape()->begin(), output->desc()->shape()->end()};
    desc.stride = {output->desc()->layout()->strides()->begin(),
                   output->desc()->layout()->strides()->end()};
    desc.itemsize = utils::dataTypeElementSize(
        output->desc()->layout()->memory_desc()->data_type());
    desc.dataType = output->desc()->layout()->memory_desc()->data_type();
    outputs.push_back(desc);
  }
  return outputs;
}

} // namespace ttnn

namespace system_desc {

::tt::target::SystemDescRoot const *getBinary(Flatbuffer binary) {
  if (not ::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          binary.handle.get())) {
    throw std::runtime_error("Unsupported binary format");
  }
  return ::tt::target::GetSizePrefixedSystemDescRoot(binary.handle.get());
}

std::string getVersion(Flatbuffer binary) {
  auto const *version = getBinary(binary)->version();
  return std::to_string(version->major()) + "." +
         std::to_string(version->minor()) + "." +
         std::to_string(version->release());
}

std::string_view getTTMLIRGitHash(Flatbuffer binary) {
  return getBinary(binary)->ttmlir_git_hash()->c_str();
}

std::string asJson(Flatbuffer binary) {
  return ::tt::runtime::binary::asJson(
      binary.handle.get(), ::tt::target::SystemDescRootBinarySchema::data(),
      ::tt::target::SystemDescRootBinarySchema::size());
}

} // namespace system_desc

Flatbuffer loadFromData(void *data) {
  return Binary{utils::unsafe_borrow_shared(data)};
}

Flatbuffer loadFromPath(char const *path) {
  // load a flatbuffer from path
  std::ifstream fbb(path, std::ios::binary | std::ios::ate);
  if (!fbb.is_open()) {
    throw std::runtime_error("Failed to open file: " + std::string(path));
  }

  std::streampos size = fbb.tellg();
  fbb.seekg(0, std::ios::beg);
  auto buffer = utils::malloc_shared(size);
  fbb.read(static_cast<char *>(buffer.get()), size);
  return Binary{buffer};
}

void store(Flatbuffer binary, char const *path) {
  // store a flatbuffer to path
  std::ofstream fbb(path, std::ios::binary);
  auto size = ::flatbuffers::GetPrefixedSize(
      static_cast<const uint8_t *>(binary.handle.get()));
  fbb.write(reinterpret_cast<char const *>(binary.handle.get()), size);
}

std::string_view getFileIdentifier(Flatbuffer binary) {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          binary.handle.get())) {
    return ::tt::target::ttnn::TTNNBinaryIdentifier();
  }

  if (::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          binary.handle.get())) {
    return ::tt::target::SystemDescRootIdentifier();
  }

  throw std::runtime_error("Unsupported binary format");
}

std::string getVersion(Flatbuffer binary) {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          binary.handle.get())) {
    return ttnn::getVersion(binary);
  }

  if (::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          binary.handle.get())) {
    return system_desc::getVersion(binary);
  }

  throw std::runtime_error("Unsupported binary format");
}

std::string_view getTTMLIRGitHash(Flatbuffer binary) {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          binary.handle.get())) {
    return ttnn::getTTMLIRGitHash(binary);
  }

  if (::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          binary.handle.get())) {
    return system_desc::getTTMLIRGitHash(binary);
  }

  throw std::runtime_error("Unsupported binary format");
}

std::string asJson(Flatbuffer binary) {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          binary.handle.get())) {
    return ttnn::asJson(binary);
  }

  if (::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          binary.handle.get())) {
    return system_desc::asJson(binary);
  }

  throw std::runtime_error("Unsupported binary format");
}

std::vector<TensorDesc> getProgramInputs(Flatbuffer binary,
                                         std::uint32_t programIndex) {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          binary.handle.get())) {
    return ttnn::getProgramInputs(binary, programIndex);
  }

  throw std::runtime_error("Unsupported binary format");
}

std::vector<TensorDesc> getProgramOutputs(Flatbuffer binary,
                                          std::uint32_t programIndex) {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          binary.handle.get())) {
    return ttnn::getProgramOutputs(binary, programIndex);
  }

  throw std::runtime_error("Unsupported binary format");
}

} // namespace tt::runtime::binary
