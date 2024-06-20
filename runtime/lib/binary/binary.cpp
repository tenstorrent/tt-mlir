// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "flatbuffers/idl.h"

#include "tt/runtime/binary.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/TTNN/binary_bfbs_generated.h"

namespace tt::runtime::binary {

::tt::target::ttnn::TTNNBinary const *getTTNNBinary(Binary const &binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  if (not isTTNN) {
    throw std::runtime_error("Unsupported binary format");
  }
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

Binary loadFromData(void *data) {
  return Binary{utils::unsafe_wrap_shared(data)};
}

Binary loadFromPath(char const *path) {
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

std::string getVersion(Binary const &binary) {
  auto const *version = getTTNNBinary(binary)->version();
  return std::to_string(version->major()) + "." +
         std::to_string(version->minor()) + "." +
         std::to_string(version->release());
}

std::string getTTMLIRGitHash(Binary const &binary) {
  return getTTNNBinary(binary)->ttmlir_git_hash()->str();
}

std::string asJson(Binary const &binary) {
  flatbuffers::IDLOptions opts;
  opts.size_prefixed = true;
  opts.strict_json = true;
  flatbuffers::Parser parser(opts);

  if (not parser.Deserialize(
          ::tt::target::ttnn::TTNNBinaryBinarySchema::data(),
          ::tt::target::ttnn::TTNNBinaryBinarySchema::size())) {
    throw std::runtime_error("Failed to deserialize schema");
  }

  std::string text;
  if (::flatbuffers::GenerateText(parser, binary.handle.get(), &text)) {
    throw std::runtime_error("Failed to generate JSON");
  }

  return text;
}

} // namespace tt::runtime::binary
