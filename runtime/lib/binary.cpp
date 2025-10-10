// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <fstream>

#include "flatbuffers/idl.h"

#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/tensor_cache.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/Common/system_desc_bfbs_generated.h"
#include "ttmlir/Target/Common/system_desc_generated.h"
#include "ttmlir/Target/TTMetal/Target.h"
#include "ttmlir/Target/TTMetal/binary_bfbs_generated.h"
#include "ttmlir/Target/TTNN/Target.h"
#include "ttmlir/Target/TTNN/binary_bfbs_generated.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcovered-switch-default"
#include "ttmlir/Target/CUDA/program_generated.h"
#pragma clang diagnostic pop

namespace tt::runtime {

Binary::Binary(Flatbuffer fb)
    : Flatbuffer(fb), binaryId(nextBinaryId()),
      tensorCache(std::make_shared<TensorCache>()) {}

Binary::Binary(std::shared_ptr<void> handle)
    : Flatbuffer(handle), binaryId(nextBinaryId()),
      tensorCache(std::make_shared<TensorCache>()) {}

Binary &Binary::operator=(Flatbuffer fb) {
  this->handle = fb.handle;

  binaryId = nextBinaryId();

  // Reinitialize tensor cache since binary handle contents
  // are now different
  tensorCache = std::make_shared<TensorCache>();

  return *this;
}

Binary &Binary::operator=(std::shared_ptr<void> handle) {
  this->handle = handle;

  binaryId = nextBinaryId();

  // Reinitialize tensor cache since binary handle contents
  // are now different
  tensorCache = std::make_shared<TensorCache>();

  return *this;
}

std::uint64_t Binary::nextBinaryId() {
  static std::atomic<uint64_t> id{0};
  return id.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t Binary::id() const { return binaryId; }

static flatbuffers::Parser getParser(const uint8_t *binarySchema,
                                     size_t schemaSize) {
  flatbuffers::IDLOptions opts;
  opts.size_prefixed = true;
  opts.strict_json = true;
  opts.output_default_scalars_in_json = true;
  flatbuffers::Parser parser(opts);

  if (!parser.Deserialize(binarySchema, schemaSize)) {
    LOG_FATAL("Failed to deserialize schema");
  }

  return parser;
}

// Binary asJson functions are broken down to get individual flatbuffer
// components, allowing for bypassing the golden_map in debug_info, the loading
// and processing of which can use significant memory and time.
static std::string asJson(const void *fbb, const uint8_t *binarySchema,
                          size_t schemaSize) {
  flatbuffers::Parser parser = getParser(binarySchema, schemaSize);
  std::string text;
  const char *err = ::flatbuffers::GenerateText(parser, fbb, &text);
  LOG_ASSERT(!err, "Failed to generate JSON: ", err);
  return text;
}

template <typename T>
static std::string asJsonFromTable(const T *table, const uint8_t *binarySchema,
                                   size_t schemaSize) {
  flatbuffers::Parser parser = getParser(binarySchema, schemaSize);
  std::string text;
  const char *err = ::flatbuffers::GenTextFromTable(
      parser, table, table->GetFullyQualifiedName(), &text);
  LOG_ASSERT(!err, "Failed to generate JSON: ", err);

  return text;
}

template <typename T>
static std::string asJsonFromParentTable(const T *parent_table,
                                         const uint8_t *binarySchema,
                                         size_t schemaSize) {
  flatbuffers::Parser parser = getParser(binarySchema, schemaSize);
  std::string result_text;
  for (const auto *table : *parent_table) {
    std::string text;
    const char *err = ::flatbuffers::GenTextFromTable(
        parser, table, table->GetFullyQualifiedName(), &text);
    LOG_ASSERT(!err, "Failed to generate JSON: ", err);

    if (!result_text.empty()) {
      result_text += ",";
    }
    result_text += text;
  }
  // Wrap the tables in a JSON array
  result_text = "[" + result_text + "]";
  return result_text;
}

namespace ttnn {

const ::tt::target::ttnn::TTNNBinary *getBinary(Flatbuffer binary) {
  bool isTTNN = ::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
      binary.handle.get());
  LOG_ASSERT(isTTNN, "Unsupported binary format");
  return ::tt::target::ttnn::GetSizePrefixedTTNNBinary(binary.handle.get());
}

std::string getVersion(Flatbuffer binary) {
  const auto *version = getBinary(binary)->version();
  return std::to_string(version->major()) + "." +
         std::to_string(version->minor()) + "." +
         std::to_string(version->patch());
}

std::string getSchemaHash(Flatbuffer binary) {
  return getBinary(binary)->schema_hash()->c_str();
}

std::string getTTMLIRGitHash(Flatbuffer binary) {
  return getBinary(binary)->ttmlir_git_hash()->c_str();
}

std::string asJson(Flatbuffer binary) {
  return ::tt::runtime::asJson(
      binary.handle.get(), ::tt::target::ttnn::TTNNBinaryBinarySchema::data(),
      ::tt::target::ttnn::TTNNBinaryBinarySchema::size());
}

std::uint32_t getNumProgramInputs(Flatbuffer binary,
                                  std::uint32_t programIndex) {
  return getBinary(binary)->programs()->Get(programIndex)->inputs()->size();
}

std::uint32_t getNumProgramOutputs(Flatbuffer binary,
                                   std::uint32_t programIndex) {
  return getBinary(binary)->programs()->Get(programIndex)->outputs()->size();
}

std::vector<TensorDesc> getProgramInputs(Flatbuffer binary,
                                         std::uint32_t programIndex) {
  std::vector<TensorDesc> inputs;
  const auto *program = getBinary(binary)->programs()->Get(programIndex);
  for (const auto *input : *program->inputs()) {
    TensorDesc desc(
        {input->desc()->shape()->begin(), input->desc()->shape()->end()},
        input->desc()->layout()->memory_desc()->data_type());
    inputs.push_back(desc);
  }
  return inputs;
}

std::vector<TensorDesc> getProgramOutputs(Flatbuffer binary,
                                          std::uint32_t programIndex) {
  std::vector<TensorDesc> outputs;
  const auto *program = getBinary(binary)->programs()->Get(programIndex);
  for (const auto *output : *program->outputs()) {
    TensorDesc desc(
        {output->desc()->shape()->begin(), output->desc()->shape()->end()},
        output->desc()->layout()->memory_desc()->data_type());
    outputs.push_back(desc);
  }
  return outputs;
}

std::string getSystemDescAsJson(Flatbuffer binary) {
  const auto *system_desc = getBinary(binary)->system_desc();
  return asJsonFromTable(system_desc,
                         ::tt::target::ttnn::TTNNBinaryBinarySchema::data(),
                         ::tt::target::ttnn::TTNNBinaryBinarySchema::size());
}

std::string getProgramOpsAsJson(Flatbuffer binary, std::uint32_t programIndex) {
  const auto *programs = getBinary(binary)->programs();
  LOG_ASSERT(programIndex < programs->size(), "Program index out of bounds");
  const auto *operations = programs->Get(programIndex)->operations();
  return asJsonFromParentTable(
      operations, ::tt::target::ttnn::TTNNBinaryBinarySchema::data(),
      ::tt::target::ttnn::TTNNBinaryBinarySchema::size());
}

std::string getProgramInputsAsJson(Flatbuffer binary,
                                   std::uint32_t programIndex) {
  const auto *programs = getBinary(binary)->programs();
  LOG_ASSERT(programIndex < programs->size(), "Program index out of bounds");
  const auto *inputs = programs->Get(programIndex)->inputs();
  return asJsonFromParentTable(
      inputs, ::tt::target::ttnn::TTNNBinaryBinarySchema::data(),
      ::tt::target::ttnn::TTNNBinaryBinarySchema::size());
}

std::string getProgramOutputsAsJson(Flatbuffer binary,
                                    std::uint32_t programIndex) {
  const auto *programs = getBinary(binary)->programs();
  LOG_ASSERT(programIndex < programs->size(), "Program index out of bounds");
  const auto *outputs = programs->Get(programIndex)->outputs();
  return asJsonFromParentTable(
      outputs, ::tt::target::ttnn::TTNNBinaryBinarySchema::data(),
      ::tt::target::ttnn::TTNNBinaryBinarySchema::size());
}

std::string getMlirAsJson(Flatbuffer binary) {
  const auto *mlir = getBinary(binary)->mlir();
  return asJsonFromTable(mlir,
                         ::tt::target::ttnn::TTNNBinaryBinarySchema::data(),
                         ::tt::target::ttnn::TTNNBinaryBinarySchema::size());
}

std::unordered_map<std::uint32_t, const ::tt::target::GoldenTensor *>
getDebugInfoGolden(Flatbuffer binary, std::string &loc) {
  std::unordered_map<std::uint32_t, const ::tt::target::GoldenTensor *>
      goldenTensorDeviceMap;

  const auto *programs = getBinary(binary)->programs();
  for (const auto *program : *programs) {
    for (const ::tt::target::GoldenKV *goldenKV :
         *program->debug_info()->golden_info()->golden_map()) {
      if (std::string(goldenKV->key()->c_str()) == loc) {
        for (const ::tt::target::GoldenDeviceTensor *goldenDeviceTensor :
             *goldenKV->value()) {
          goldenTensorDeviceMap[goldenDeviceTensor->device()] =
              goldenDeviceTensor->value();
        }
      }
    }
  }

  return goldenTensorDeviceMap;
}

} // namespace ttnn

namespace metal {

const ::tt::target::metal::TTMetalBinary *getBinary(Flatbuffer binary) {
  bool isTTMetal =
      ::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          binary.handle.get());
  LOG_ASSERT(isTTMetal, "Unsupported binary format");
  return ::tt::target::metal::GetSizePrefixedTTMetalBinary(binary.handle.get());
}

std::string getVersion(Flatbuffer binary) {
  const auto *version = getBinary(binary)->version();
  return std::to_string(version->major()) + "." +
         std::to_string(version->minor()) + "." +
         std::to_string(version->patch());
}

std::string getSchemaHash(Flatbuffer binary) {
  return getBinary(binary)->schema_hash()->c_str();
}

std::string getTTMLIRGitHash(Flatbuffer binary) {
  return getBinary(binary)->ttmlir_git_hash()->c_str();
}

std::string asJson(Flatbuffer binary) {
  return ::tt::runtime::asJson(
      binary.handle.get(),
      ::tt::target::metal::TTMetalBinaryBinarySchema::data(),
      ::tt::target::metal::TTMetalBinaryBinarySchema::size());
}

std::string getSystemDescAsJson(Flatbuffer binary) {
  const auto *system_desc = getBinary(binary)->system_desc();
  return asJsonFromTable(
      system_desc, ::tt::target::metal::TTMetalBinaryBinarySchema::data(),
      ::tt::target::metal::TTMetalBinaryBinarySchema::size());
}

std::string getProgramInputsAsJson(Flatbuffer binary,
                                   std::uint32_t programIndex) {
  const auto *programs = getBinary(binary)->programs();
  LOG_ASSERT(programIndex < programs->size(), "Program index out of bounds");
  const auto *inputs = programs->Get(programIndex)->inputs();
  return asJsonFromParentTable(
      inputs, ::tt::target::metal::TTMetalBinaryBinarySchema::data(),
      ::tt::target::metal::TTMetalBinaryBinarySchema::size());
}

std::string getProgramOutputsAsJson(Flatbuffer binary,
                                    std::uint32_t programIndex) {
  const auto *programs = getBinary(binary)->programs();
  LOG_ASSERT(programIndex < programs->size(), "Program index out of bounds");
  const auto *outputs = programs->Get(programIndex)->outputs();
  return asJsonFromParentTable(
      outputs, ::tt::target::metal::TTMetalBinaryBinarySchema::data(),
      ::tt::target::metal::TTMetalBinaryBinarySchema::size());
}

std::string getMlirAsJson(Flatbuffer binary) {
  const auto *mlir = getBinary(binary)->mlir();
  return asJsonFromTable(
      mlir, ::tt::target::metal::TTMetalBinaryBinarySchema::data(),
      ::tt::target::metal::TTMetalBinaryBinarySchema::size());
}

static std::vector<TensorDesc>
getTensorDescs(const ::flatbuffers::Vector<
               ::flatbuffers::Offset<tt::target::metal::TensorRef>> *tensors) {
  std::vector<TensorDesc> tensorDescs;
  tensorDescs.reserve(tensors->size());
  for (const auto *tensor : *tensors) {
    TensorDesc desc(
        {tensor->desc()->shape()->begin(), tensor->desc()->shape()->end()},
        tensor->desc()->layout()->memory_desc()->data_type());
    tensorDescs.push_back(desc);
  }
  return tensorDescs;
}

std::uint32_t getNumProgramInputs(Flatbuffer binary,
                                  std::uint32_t programIndex) {
  return getBinary(binary)->programs()->Get(programIndex)->inputs()->size();
}

std::uint32_t getNumProgramOutputs(Flatbuffer binary,
                                   std::uint32_t programIndex) {
  return getBinary(binary)->programs()->Get(programIndex)->outputs()->size();
}

std::vector<TensorDesc> getProgramInputs(Flatbuffer binary,
                                         std::uint32_t programIndex) {
  const auto *program = getBinary(binary)->programs()->Get(programIndex);
  LOG_ASSERT(program->device_programs()->size() == 1,
             "Currently only one device program is supported, got: ",
             program->device_programs()->size());
  return getTensorDescs(program->inputs());
}

std::vector<TensorDesc> getProgramOutputs(Flatbuffer binary,
                                          std::uint32_t programIndex) {
  const auto *program = getBinary(binary)->programs()->Get(programIndex);
  LOG_ASSERT(program->device_programs()->size() == 1,
             "Currently only one device program is supported, got: ",
             program->device_programs()->size());
  return getTensorDescs(program->outputs());
}

std::unordered_map<std::uint32_t, const ::tt::target::GoldenTensor *>
getDebugInfoGolden(Flatbuffer binary, std::string &loc) {
  std::unordered_map<std::uint32_t, const ::tt::target::GoldenTensor *>
      goldenTensorDeviceMap;

  const auto *programs = getBinary(binary)->programs();
  for (const auto *program : *programs) {
    for (const ::tt::target::GoldenKV *goldenKV :
         *program->debug_info()->golden_info()->golden_map()) {
      if (std::string(goldenKV->key()->c_str()) == loc) {
        for (const ::tt::target::GoldenDeviceTensor *goldenDeviceTensor :
             *goldenKV->value()) {
          goldenTensorDeviceMap[goldenDeviceTensor->device()] =
              goldenDeviceTensor->value();
        }
      }
    }
  }

  return goldenTensorDeviceMap;
}

} // namespace metal

namespace cuda {

// Helper function to convert CUDA DataType enum to common DataType enum
static ::tt::target::DataType
cudaDataTypeToCommon(::tt::target::cuda::DataType cudaType) {
  switch (cudaType) {
  case ::tt::target::cuda::DataType::Float64:
    return ::tt::target::DataType::Float64;
  case ::tt::target::cuda::DataType::UInt64:
    return ::tt::target::DataType::UInt64;
  case ::tt::target::cuda::DataType::Int64:
    return ::tt::target::DataType::Int64;
  case ::tt::target::cuda::DataType::Float32:
    return ::tt::target::DataType::Float32;
  case ::tt::target::cuda::DataType::UInt32:
    return ::tt::target::DataType::UInt32;
  case ::tt::target::cuda::DataType::Int32:
    return ::tt::target::DataType::Int32;
  case ::tt::target::cuda::DataType::Float16:
    return ::tt::target::DataType::Float16;
  case ::tt::target::cuda::DataType::BFloat16:
    return ::tt::target::DataType::BFloat16;
  case ::tt::target::cuda::DataType::UInt16:
    return ::tt::target::DataType::UInt16;
  case ::tt::target::cuda::DataType::Int16:
    return ::tt::target::DataType::Int16;
  }
}

// Helper function to convert CUDA DataType enum to string
std::string cudaDataTypeToString(::tt::target::cuda::DataType dataType) {
  switch (dataType) {
  case ::tt::target::cuda::DataType::Float64:
    return "Float64";
  case ::tt::target::cuda::DataType::UInt64:
    return "UInt64";
  case ::tt::target::cuda::DataType::Int64:
    return "Int64";
  case ::tt::target::cuda::DataType::Float32:
    return "Float32";
  case ::tt::target::cuda::DataType::UInt32:
    return "UInt32";
  case ::tt::target::cuda::DataType::Int32:
    return "Int32";
  case ::tt::target::cuda::DataType::Float16:
    return "Float16";
  case ::tt::target::cuda::DataType::BFloat16:
    return "BFloat16";
  case ::tt::target::cuda::DataType::UInt16:
    return "UInt16";
  case ::tt::target::cuda::DataType::Int16:
    return "Int16";
  }
}

const ::tt::target::cuda::Program *getBinary(Flatbuffer binary) {
  bool isCUDA = ::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
      binary.handle.get());
  LOG_ASSERT(isCUDA, "Unsupported binary format");
  return ::tt::target::cuda::GetSizePrefixedProgram(binary.handle.get());
}

std::string getVersion(Flatbuffer binary) { return "0.0.0"; }

std::string getSchemaHash(Flatbuffer binary) { return ""; }

std::string getTTMLIRGitHash(Flatbuffer binary) { return ""; }

std::string asJson(Flatbuffer binary) {
  return ::tt::runtime::asJson(binary.handle.get(),
                               ::tt::target::cuda::ProgramBinarySchema::data(),
                               ::tt::target::cuda::ProgramBinarySchema::size());
}

std::string getSystemDescAsJson(Flatbuffer binary) { return ""; }

std::string getMlirAsJson(Flatbuffer binary) { return ""; }

std::vector<TensorDesc> getProgramInputs(Flatbuffer binary,
                                         std::uint32_t programIndex) {
  std::vector<TensorDesc> inputs;
  const auto *program = getBinary(binary);

  // CUDA schema has memrefs instead of direct inputs
  for (const auto *memref : *program->memrefs()) {
    TensorDesc desc(
        {memref->type()->shape()->begin(), memref->type()->shape()->end()},
        cudaDataTypeToCommon(memref->type()->data_type()));
    inputs.push_back(desc);
  }
  return inputs;
}

std::vector<TensorDesc> getProgramOutputs(Flatbuffer binary,
                                          std::uint32_t programIndex) {
  std::vector<TensorDesc> outputs;
  const auto *program = getBinary(binary);

  // Find the return variable in memrefs
  std::string returnVar = program->return_variable()->c_str();
  for (const auto *memref : *program->memrefs()) {
    if (memref->id()->c_str() == returnVar) {
      TensorDesc desc(
          {memref->type()->shape()->begin(), memref->type()->shape()->end()},
          cudaDataTypeToCommon(memref->type()->data_type()));
      outputs.push_back(desc);
      break;
    }
  }
  return outputs;
}

std::string getProgramName(Flatbuffer binary, std::uint32_t programIndex) {
  return "cuda_program";
}

bool isProgramPrivate(Flatbuffer binary, std::uint32_t programIndex) {
  return false;
}

std::string getProgramOpsAsJson(Flatbuffer binary, std::uint32_t programIndex) {
  const auto *program = getBinary(binary);

  // Custom JSON generation for CUDA actions since they are unions
  std::string result = "[";
  bool first = true;

  const auto *actions = program->actions();
  const auto *actionTypes = program->actions_type();

  for (unsigned int i = 0; i < actions->size(); i++) {
    if (!first) {
      result += ",";
    }
    first = false;

    auto actionType = actionTypes->Get(i);
    switch (actionType) {
    case ::tt::target::cuda::Action::Kernel: {
      const auto *kernel =
          static_cast<const ::tt::target::cuda::Kernel *>(actions->Get(i));
      result += "{\"type\":\"Kernel\",\"name\":\"" +
                std::string(kernel->name()->c_str()) + "\"}";
      break;
    }
    case ::tt::target::cuda::Action::CopyFunction: {
      const auto *copy = static_cast<const ::tt::target::cuda::CopyFunction *>(
          actions->Get(i));
      result += "{\"type\":\"CopyFunction\",\"src\":\"" +
                std::string(copy->src()->c_str()) + "\",\"dst\":\"" +
                std::string(copy->dst()->c_str()) + "\"}";
      break;
    }
    default:
      result += "{\"type\":\"Unknown\"}";
      break;
    }
  }

  result += "]";
  return result;
}

std::string getProgramInputsAsJson(Flatbuffer binary,
                                   std::uint32_t programIndex) {
  const auto *program = getBinary(binary);

  // Generate JSON in the format expected by the Python code
  std::string result = "[";
  bool first = true;

  for (const auto *memref : *program->memrefs()) {
    if (!first) {
      result += ",";
    }
    first = false;

    // Create the expected structure with "desc" key
    result += "{\"desc\":{";
    result += "\"shape\":[";
    bool firstDim = true;
    for (const auto dim : *memref->type()->shape()) {
      if (!firstDim) {
        result += ",";
      }
      firstDim = false;
      result += std::to_string(dim);
    }
    result += "],\"layout\":{\"memory_desc\":{\"data_type\":\"" +
              cudaDataTypeToString(memref->type()->data_type()) + "\"}}}}";
  }

  result += "]";
  return result;
}

std::string getProgramOutputsAsJson(Flatbuffer binary,
                                    std::uint32_t programIndex) {
  const auto *program = getBinary(binary);
  std::string returnVar = program->return_variable()->c_str();

  // Find the return variable in memrefs and format it properly
  std::string result = "[";
  bool found = false;

  for (const auto *memref : *program->memrefs()) {
    if (memref->id()->c_str() == returnVar) {
      // Create the expected structure with "desc" key
      result += "{\"desc\":{";
      result += "\"shape\":[";
      bool firstDim = true;
      for (const auto dim : *memref->type()->shape()) {
        if (!firstDim) {
          result += ",";
        }
        firstDim = false;
        result += std::to_string(dim);
      }
      result += "],\"layout\":{\"memory_desc\":{\"data_type\":\"" +
                cudaDataTypeToString(memref->type()->data_type()) + "\"}}}}";
      found = true;
      break;
    }
  }

  if (!found) {
    // Fallback if return variable not found in memrefs
    result += "{\"desc\":{\"shape\":[],\"layout\":{\"memory_desc\":{\"data_"
              "type\":\"Float32\"}}}}";
  }

  result += "]";
  return result;
}

} // namespace cuda

namespace system_desc {

const ::tt::target::SystemDescRoot *getBinary(Flatbuffer binary) {
  if (!::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          binary.handle.get())) {
    LOG_FATAL("Unsupported binary format");
  }
  return ::tt::target::GetSizePrefixedSystemDescRoot(binary.handle.get());
}

std::string getVersion(Flatbuffer binary) {
  const auto *version = getBinary(binary)->version();
  return std::to_string(version->major()) + "." +
         std::to_string(version->minor()) + "." +
         std::to_string(version->patch());
}

std::string getSchemaHash(Flatbuffer binary) {
  return getBinary(binary)->schema_hash()->c_str();
}

std::string getTTMLIRGitHash(Flatbuffer binary) {
  return getBinary(binary)->ttmlir_git_hash()->c_str();
}

std::string asJson(Flatbuffer binary) {
  return ::tt::runtime::asJson(
      binary.handle.get(), ::tt::target::SystemDescRootBinarySchema::data(),
      ::tt::target::SystemDescRootBinarySchema::size());
}

} // namespace system_desc

Flatbuffer Flatbuffer::loadFromPath(const char *path) {
  // load a flatbuffer from path
  std::ifstream fbb(path, std::ios::binary | std::ios::ate);
  LOG_ASSERT(fbb.is_open(), "Failed to open file: ", path);
  std::streampos size = fbb.tellg();
  fbb.seekg(0, std::ios::beg);
  auto buffer = ::tt::runtime::utils::mallocShared(size);
  fbb.read(static_cast<char *>(buffer.get()), size);
  return Flatbuffer(buffer);
}

Flatbuffer Flatbuffer::loadFromMemory(const void *memory, size_t size) {
  // load a flatbuffer from memory
  LOG_ASSERT(memory != nullptr, "Memory pointer is null");
  LOG_ASSERT(size > 0, "Size must be greater than zero");
  auto buffer = ::tt::runtime::utils::mallocShared(size);
  std::memcpy(buffer.get(), memory, size);
  return Flatbuffer(buffer);
}

void Flatbuffer::store(const char *path) const {
  // store a flatbuffer to path
  std::ofstream fbb(path, std::ios::binary);
  auto size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(handle.get()));
  fbb.write(reinterpret_cast<const char *>(handle.get()), size);
}

template <typename T>
void Flatbuffer::storeToMemory(std::vector<T> &serializedFlatbuffer) const {
  static_assert(sizeof(T) == 1, "Element type must be 1 byte");
  static_assert(std::is_trivially_copyable_v<T>,
                "Element type must be trivially copyable");

  size_t size = ::flatbuffers::GetSizePrefixedBufferLength(
      static_cast<const uint8_t *>(handle.get()));
  serializedFlatbuffer.resize(size);
  std::memcpy(serializedFlatbuffer.data(), handle.get(), size);
}

template void Flatbuffer::storeToMemory<std::byte>(
    std::vector<std::byte> &serializedFlatbuffer) const;

template void Flatbuffer::storeToMemory<std::uint8_t>(
    std::vector<std::uint8_t> &serializedFlatbuffer) const;

std::string Flatbuffer::getFileIdentifier() const {
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

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return ::tt::target::cuda::ProgramIdentifier();
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

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getVersion(*this);
  }

  LOG_FATAL("Unsupported binary format");
}

std::string Flatbuffer::getSchemaHash() const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getSchemaHash(*this);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getSchemaHash(*this);
  }

  if (::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          handle.get())) {
    return system_desc::getSchemaHash(*this);
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getSchemaHash(*this);
  }

  LOG_FATAL("Unsupported binary format");
}

bool Flatbuffer::checkSchemaHash() const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getSchemaHash(*this) ==
           ::tt::target::ttnn::binary_bfbs_schema_hash;
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getSchemaHash(*this) ==
           ::tt::target::ttmetal::binary_bfbs_schema_hash;
  }

  if (::tt::target::SizePrefixedSystemDescRootBufferHasIdentifier(
          handle.get())) {
    return system_desc::getSchemaHash(*this) ==
           ::tt::target::common::system_desc_bfbs_schema_hash;
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return true;
  }

  LOG_FATAL("Unsupported binary format");
}

std::string Flatbuffer::getTTMLIRGitHash() const {
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

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getTTMLIRGitHash(*this);
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

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::asJson(*this);
  }

  LOG_FATAL("Unsupported binary format");
}

std::string Binary::getSystemDescAsJson() const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getSystemDescAsJson(*this);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getSystemDescAsJson(*this);
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getSystemDescAsJson(*this);
  }

  LOG_FATAL("Unsupported binary format");
}

std::uint32_t Binary::getNumPrograms() const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getBinary(*this)->programs()->size();
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getBinary(*this)->programs()->size();
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return 1; // Currently only one program for CUDA is supported.
  }

  LOG_FATAL("Unsupported binary format");
}

const std::pair<std::uint32_t, std::uint32_t>
Binary::getProgramMeshShape(std::uint32_t programIndex) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    const tt::target::Dim2d *const mesh_shape =
        ttnn::getBinary(*this)->programs()->Get(programIndex)->mesh_shape();
    assert(mesh_shape != nullptr);
    return std::make_pair(mesh_shape->y(), mesh_shape->x());
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    const tt::target::Dim2d *const mesh_shape =
        metal::getBinary(*this)->programs()->Get(programIndex)->mesh_shape();
    assert(mesh_shape != nullptr);
    return std::make_pair(mesh_shape->y(), mesh_shape->x());
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return std::make_pair(1, 1);
  }

  LOG_FATAL("Unsupported binary format");
}

std::string Binary::getProgramName(std::uint32_t programIndex) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getBinary(*this)
        ->programs()
        ->Get(programIndex)
        ->name()
        ->c_str();
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getBinary(*this)
        ->programs()
        ->Get(programIndex)
        ->name()
        ->c_str();
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getProgramName(*this, programIndex);
  }

  LOG_FATAL("Unsupported binary format");
}

bool Binary::isProgramPrivate(std::uint32_t programIndex) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getBinary(*this)->programs()->Get(programIndex)->private_();
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getBinary(*this)->programs()->Get(programIndex)->private_();
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return false;
  }

  LOG_FATAL("Unsupported binary format");
  return false;
}

std::string Binary::getProgramOpsAsJson(std::uint32_t programIndex) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getProgramOpsAsJson(*this, programIndex);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    LOG_WARNING("getProgramOpsAsJson not supported for TTMetal");
    return "";
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getProgramOpsAsJson(*this, programIndex);
  }

  LOG_FATAL("Unsupported binary format");
}

std::string Binary::getProgramInputsAsJson(std::uint32_t programIndex) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getProgramInputsAsJson(*this, programIndex);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getProgramInputsAsJson(*this, programIndex);
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getProgramInputsAsJson(*this, programIndex);
  }

  LOG_FATAL("Unsupported binary format");
}

std::string Binary::getProgramOutputsAsJson(std::uint32_t programIndex) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getProgramOutputsAsJson(*this, programIndex);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getProgramOutputsAsJson(*this, programIndex);
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getProgramOutputsAsJson(*this, programIndex);
  }

  LOG_FATAL("Unsupported binary format");
}

std::string Binary::getMlirAsJson() const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getMlirAsJson(*this);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getMlirAsJson(*this);
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getMlirAsJson(*this);
  }

  LOG_FATAL("Unsupported binary format");
}

SystemDesc SystemDesc::loadFromPath(const char *path) {
  return SystemDesc(Flatbuffer::loadFromPath(path).handle);
}

Binary Binary::loadFromPath(const char *path) {
  return Binary(Flatbuffer::loadFromPath(path).handle);
}

std::uint32_t Binary::getNumProgramInputs(std::uint32_t programIndex) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getNumProgramInputs(*this, programIndex);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getNumProgramInputs(*this, programIndex);
  }

  LOG_FATAL("Unsupported binary format");
}

std::uint32_t Binary::getNumProgramOutputs(std::uint32_t programIndex) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getNumProgramOutputs(*this, programIndex);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getNumProgramOutputs(*this, programIndex);
  }

  LOG_FATAL("Unsupported binary format");
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

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getProgramInputs(*this, programIndex);
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

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    return cuda::getProgramOutputs(*this, programIndex);
  }

  LOG_FATAL("Unsupported binary format");
}

std::unordered_map<std::uint32_t, const ::tt::target::GoldenTensor *>
Binary::getDebugInfoGolden(std::string &loc) const {
  if (::tt::target::ttnn::SizePrefixedTTNNBinaryBufferHasIdentifier(
          handle.get())) {
    return ttnn::getDebugInfoGolden(*this, loc);
  }

  if (::tt::target::metal::SizePrefixedTTMetalBinaryBufferHasIdentifier(
          handle.get())) {
    return metal::getDebugInfoGolden(*this, loc);
  }

  if (::tt::target::cuda::SizePrefixedProgramBufferHasIdentifier(
          handle.get())) {
    // CUDA binaries don't have golden information - return empty map
    return std::unordered_map<std::uint32_t,
                              const ::tt::target::GoldenTensor *>();
  }

  LOG_FATAL("Unsupported binary format for obtaining golden information");
}

} // namespace tt::runtime
