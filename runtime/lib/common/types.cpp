// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tt/runtime/types.h"
#include "tt/runtime/detail/common/logger.h"
#include "tt/runtime/detail/common/runtime_context.h"
#include "tt/runtime/utils.h"

#include <atomic>
#include <iomanip>
#include <sstream>

namespace tt::runtime {

TensorDesc::TensorDesc(const std::vector<uint32_t> &shape,
                       const ::tt::target::DataType dataType,
                       const std::optional<uint32_t> itemsize,
                       const std::optional<std::vector<uint32_t>> &stride,
                       const std::optional<uint64_t> physicalVolume)
    : shape(shape), dataType(dataType) {
  this->itemsize = itemsize.value_or(utils::dataTypeElementSize(dataType));
  this->stride = stride.value_or(utils::calculateStride(shape));
  this->physicalVolume = physicalVolume.value_or(volume());
}

size_t TensorDesc::volume() const {
  return utils::product(shape.cbegin(), shape.cend());
}

std::string MemoryView::toString() const {
  constexpr size_t MB = 1024 * 1024;
  std::ostringstream oss;
  // KB precision for MB values
  oss << std::fixed << std::setprecision(3);

  oss << "MemoryView{";
  oss << "numBanks: " << numBanks << ", ";
  oss << "totalBytesPerBank: " << (static_cast<double>(totalBytesPerBank) / MB)
      << " MB, ";
  oss << "totalBytesAllocatedPerBank: "
      << (static_cast<double>(totalBytesAllocatedPerBank) / MB) << " MB, ";
  oss << "totalBytesFreePerBank: "
      << (static_cast<double>(totalBytesFreePerBank) / MB) << " MB, ";
  oss << "largestContiguousBytesFreePerBank: "
      << (static_cast<double>(largestContiguousBytesFreePerBank) / MB) << " MB";
  oss << "}";

  return oss.str();
}

MultiProcessArgs &
MultiProcessArgs::withHosts(const std::vector<std::string> &hosts) {
  hosts_ = hosts;
  return *this;
}

MultiProcessArgs::MultiProcessArgs(std::string_view rankBindingPath)
    : rankBindingPath_(rankBindingPath) {}

MultiProcessArgs &MultiProcessArgs::withHostsFilePath(std::string_view path) {
  hostsFilePath_ = path;
  return *this;
}

std::string MultiProcessArgs::getRankBindingPath() const {
  return rankBindingPath_.string();
}

MultiProcessArgs &MultiProcessArgs::withRankFilePath(std::string_view path) {
  rankFilePath_ = path;
  return *this;
}

MultiProcessArgs &MultiProcessArgs::withMcaOptions(
    const std::map<std::string, std::string> &mcaOptions) {
  mcaOptions_ = mcaOptions;
  return *this;
}

MultiProcessArgs &MultiProcessArgs::withTagOutput(bool tagOutput) {
  tagOutput_ = tagOutput;
  return *this;
}

MultiProcessArgs &MultiProcessArgs::withAllowRunAsRoot(bool allowRunAsRoot) {
  allowRunAsRoot_ = allowRunAsRoot;
  return *this;
}

MultiProcessArgs &MultiProcessArgs::withExtraMpiArgs(
    const std::vector<std::string> &extraMpiArgs) {
  extraMpiArgs_ = extraMpiArgs;
  return *this;
}

MultiProcessArgs &
MultiProcessArgs::withControllerHostname(std::string_view hostname) {
  controllerHostname_ = std::string(hostname);
  return *this;
}

MultiProcessArgs &
MultiProcessArgs::withTcpInterface(std::string_view interfaceName) {
  tcpInterface_ = std::string(interfaceName);
  return *this;
}

std::optional<std::string> MultiProcessArgs::getControllerHostname() const {
  return controllerHostname_;
}

std::string MultiProcessArgs::toArgString() const {
  std::ostringstream oss;

  // Rank binding path, tt-run specific
  LOG_ASSERT(!rankBindingPath_.empty(), "Rank binding path is required");
  oss << "--rank-binding " << rankBindingPath_;

  oss << " ";

  // TCP interface path, tt-run specific (replacement for mca btl_tcp_if_include, no longer supported by ttrun)
  if(tcpInterface_.has_value()) {
    oss << " ";
    oss << "--tcp-interface " << tcpInterface_.value();
  }

  oss << " ";

  oss << "--mpi-args \"";

  // Hosts
  if (!hosts_.empty()) {
    oss << " ";
    oss << "--host ";
    for (size_t i = 0; i < hosts_.size(); i++) {
      oss << hosts_[i];
      if (i != hosts_.size() - 1) {
        oss << ",";
      }
    }
  }

  // Hosts file
  if (!hostsFilePath_.empty()) {
    oss << " ";
    oss << "--hostfile " << hostsFilePath_;
  }

  // Rank file
  if (!rankFilePath_.empty()) {
    oss << " ";
    oss << "--rankfile " << rankFilePath_;
  }

  // MCA options
  for (const auto &[key, value] : mcaOptions_) {
    oss << " ";
    oss << "--mca " << key << " " << value;
  }

  // Tag output
  if (tagOutput_) {
    oss << " ";
    oss << "--tag-output";
  }

  // Allow run as root
  if (allowRunAsRoot_) {
    oss << " ";
    oss << "--allow-run-as-root";
  }

  // Extra MPI args
  for (size_t i = 0; i < extraMpiArgs_.size(); i++) {
    oss << " ";
    oss << extraMpiArgs_[i];
  }

  oss << " ";
  oss << "\"";

  return oss.str();
}

std::uint32_t Device::nextDeviceGlobalId() {
  static std::atomic<std::uint32_t> globalId = 0;
  return globalId.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t Tensor::nextTensorGlobalId() {
  static std::atomic<std::uint64_t> globalId = 0;
  return globalId.fetch_add(1, std::memory_order_relaxed);
}

std::uint64_t Layout::nextLayoutGlobalId() {
  static std::atomic<std::uint64_t> globalId = 0;
  return globalId.fetch_add(1, std::memory_order_relaxed);
}

} // namespace tt::runtime
