// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Stub implementations for macOS to allow pykernel to build without runtime
// These should never be called - they trap if invoked

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"

#include "tt/runtime/debug.h"
#include "tt/runtime/perf.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/workarounds.h"
#include <cstdlib>

#pragma clang diagnostic pop

namespace tt::runtime {

// Stub for Flatbuffer::loadFromPath
Flatbuffer Flatbuffer::loadFromPath(const char *path) { __builtin_trap(); }

// Stub for SystemDesc::loadFromPath
SystemDesc SystemDesc::loadFromPath(const char *path) { __builtin_trap(); }

// Stub for Binary::loadFromPath
Binary Binary::loadFromPath(const char *path) { __builtin_trap(); }

namespace workaround {

// Stub for workaround::Env::get
const Env &Env::get(bool swapBinaryOperands,
                    bool readUpdateIndexFromDeviceForKVCache,
                    bool traceImplicitFromDevice, bool blackholeWorkarounds) {
  __builtin_trap();
}

} // namespace workaround

// Stubs for MultiProcessArgs constructor and methods
MultiProcessArgs::MultiProcessArgs(std::string_view rankBindingPath)
    : rankBindingPath_(rankBindingPath), tagOutput_(false),
      allowRunAsRoot_(false) {}

MultiProcessArgs &
MultiProcessArgs::withHosts(const std::vector<std::string> &hosts) {
  __builtin_trap();
}

MultiProcessArgs &MultiProcessArgs::withHostsFilePath(std::string_view path) {
  __builtin_trap();
}

std::string MultiProcessArgs::getRankBindingPath() const { __builtin_trap(); }

MultiProcessArgs &MultiProcessArgs::withRankFilePath(std::string_view path) {
  __builtin_trap();
}

MultiProcessArgs &MultiProcessArgs::withMcaOptions(
    const std::map<std::string, std::string> &mcaOptions) {
  __builtin_trap();
}

MultiProcessArgs &MultiProcessArgs::withTagOutput(bool tagOutput) {
  __builtin_trap();
}

MultiProcessArgs &MultiProcessArgs::withAllowRunAsRoot(bool allowRunAsRoot) {
  __builtin_trap();
}

MultiProcessArgs &MultiProcessArgs::withExtraMpiArgs(
    const std::vector<std::string> &extraMpiArgs) {
  __builtin_trap();
}

std::string MultiProcessArgs::toArgString() const { __builtin_trap(); }

namespace perf {

// Stubs for perf::Env methods
void Env::tracyLogOpLocation(const std::string &locInfo) const {
  __builtin_trap();
}

void Env::tracyLogInputLayoutConversion(bool conversionNeeded) const {
  __builtin_trap();
}

void Env::tracyLogConstEvalProgram(bool constEvalOp) const { __builtin_trap(); }

void Env::tracyLogProgramMetadata(const std::string &metaData) const {
  __builtin_trap();
}

void Env::setProgramMetadata(const std::string &programMetadata) {
  __builtin_trap();
}

} // namespace perf

namespace debug {

// Stub for debug::Env::get
const Env &Env::get(bool dumpKernelsToDisk, bool loadKernelsFromDisk,
                    bool useLocForKernelName, std::string kernelSourceDir,
                    bool deviceAddressValidation, bool blockingCQ) {
  __builtin_trap();
}

// Stub for debug::Hooks::get
const Hooks &Hooks::get(std::optional<Hooks::CallbackFn> preOperatorCallback,
                        std::optional<Hooks::CallbackFn> postOperatorCallback) {
  __builtin_trap();
}

// Stubs for debug::Stats methods
Stats &Stats::get() { __builtin_trap(); }

void Stats::incrementStat(const std::string &stat, std::int64_t value) {
  __builtin_trap();
}

std::int64_t Stats::getStat(const std::string &stat) const { __builtin_trap(); }

void Stats::removeStat(const std::string &stat) { __builtin_trap(); }

void Stats::clear() { __builtin_trap(); }

std::string Stats::toString() const { __builtin_trap(); }

} // namespace debug

// Stubs for Flatbuffer methods
std::string Flatbuffer::asJson() const { __builtin_trap(); }
bool Flatbuffer::checkSchemaHash() const { __builtin_trap(); }
std::string Flatbuffer::getFileIdentifier() const { __builtin_trap(); }
std::string Flatbuffer::getSchemaHash() const { __builtin_trap(); }
std::string Flatbuffer::getTTMLIRGitHash() const { __builtin_trap(); }
std::string Flatbuffer::getVersion() const { __builtin_trap(); }
void Flatbuffer::store(const char *path) const { __builtin_trap(); }

// Stubs for Binary methods
Binary::Binary(std::shared_ptr<void> handle)
    : Flatbuffer(handle), binaryId(0), tensorCache(nullptr) {}
std::string Binary::getMlirAsJson() const { __builtin_trap(); }
std::uint32_t Binary::getNumPrograms() const { __builtin_trap(); }
std::string Binary::getProgramInputsAsJson(std::uint32_t programIndex) const {
  __builtin_trap();
}
const std::pair<std::uint32_t, std::uint32_t>
Binary::getProgramMeshShape(std::uint32_t programIndex) const {
  __builtin_trap();
}
std::string Binary::getProgramName(std::uint32_t programIndex) const {
  __builtin_trap();
}
std::string Binary::getProgramOpsAsJson(std::uint32_t programIndex) const {
  __builtin_trap();
}
std::string Binary::getProgramOutputsAsJson(std::uint32_t programIndex) const {
  __builtin_trap();
}
std::string Binary::getSystemDescAsJson() const { __builtin_trap(); }
bool Binary::isProgramPrivate(std::uint32_t programIndex) const {
  __builtin_trap();
}
std::unordered_map<std::uint32_t, const ::tt::target::GoldenTensor *>
Binary::getDebugInfoGolden(std::string &loc) const {
  __builtin_trap();
}

// Stubs for runtime functions
void closeMeshDevice(Device parentMesh) { __builtin_trap(); }
void deallocateBuffers(Device device) { __builtin_trap(); }
void deallocateTensor(Tensor &tensor, bool force) { __builtin_trap(); }
void disablePersistentKernelCache() { __builtin_trap(); }
void dumpMemoryReport(Device device) { __builtin_trap(); }
void enablePersistentKernelCache() { __builtin_trap(); }
tt::target::Arch getArch() { __builtin_trap(); }
std::vector<DeviceRuntime> getAvailableDeviceRuntimes() { __builtin_trap(); }
std::vector<HostRuntime> getAvailableHostRuntimes() { __builtin_trap(); }
DeviceRuntime getCurrentDeviceRuntime() { __builtin_trap(); }
HostRuntime getCurrentHostRuntime() { __builtin_trap(); }
std::vector<int> getDeviceIds(Device meshDevice) { __builtin_trap(); }
size_t getDramSizePerChannel(Device meshDevice) { __builtin_trap(); }
size_t getL1SizePerCore(Device meshDevice) { __builtin_trap(); }
size_t getL1SmallSize(Device meshDevice) { __builtin_trap(); }
Layout getLayout(Binary executableHandle, std::uint32_t programIndex,
                 std::uint32_t inputIndex) {
  __builtin_trap();
}
std::unordered_map<MemoryBufferType, MemoryView> getMemoryView(Device device) {
  __builtin_trap();
}
std::vector<std::uint32_t> getMeshShape(Device meshDevice) { __builtin_trap(); }
size_t getNumAvailableDevices() { __builtin_trap(); }
size_t getNumDramChannels(Device meshDevice) { __builtin_trap(); }
size_t getNumHwCqs(Device meshDevice) { __builtin_trap(); }
std::string getOpDebugString(OpContext opContextHandle) { __builtin_trap(); }
std::vector<TensorRef> getOpInputRefs(OpContext opContextHandle,
                                      CallbackContext programContextHandle) {
  __builtin_trap();
}
std::string getOpLocInfo(OpContext opContextHandle) { __builtin_trap(); }
std::optional<TensorRef> getOpOutputRef(OpContext opContextHandle,
                                        CallbackContext programContextHandle) {
  __builtin_trap();
}
std::unordered_map<std::uint32_t, Tensor>
getOpOutputTensor(OpContext opContextHandle,
                  CallbackContext programContextHandle) {
  __builtin_trap();
}
std::vector<std::byte> getTensorDataBuffer(Tensor tensor) { __builtin_trap(); }
tt::target::DataType getTensorDataType(Tensor tensor) { __builtin_trap(); }
TensorDesc getTensorDesc(Tensor tensor) { __builtin_trap(); }
std::uint32_t getTensorElementSize(Tensor tensor) { __builtin_trap(); }
bool getTensorRetain(Tensor tensor) { __builtin_trap(); }
std::vector<std::uint32_t> getTensorShape(Tensor tensor) { __builtin_trap(); }
std::vector<std::uint32_t> getTensorStride(Tensor tensor) { __builtin_trap(); }
std::uint32_t getTensorVolume(Tensor tensor) { __builtin_trap(); }
size_t getTraceRegionSize(Device meshDevice) { __builtin_trap(); }
bool isProgramCacheEnabled(Device meshDevice) { __builtin_trap(); }
bool isTensorAllocated(Tensor tensor) { __builtin_trap(); }
void launchDistributedRuntime(const DistributedOptions &options) {
  __builtin_trap();
}
void memcpy(Tensor dst, Tensor src) { __builtin_trap(); }
Device openMeshDevice(const MeshDeviceOptions &options) { __builtin_trap(); }
void readDeviceProfilerResults(Device device) { __builtin_trap(); }
void releaseSubMeshDevice(Device subMesh) { __builtin_trap(); }
void releaseTrace(Device meshDevice, std::uint64_t binaryId,
                  size_t mainProgramId) {
  __builtin_trap();
}
std::optional<Tensor>
retrieveTensorFromPool(CallbackContext programContextHandle,
                       TensorRef tensorRef, bool untilize) {
  __builtin_trap();
}
void setCompatibleDeviceRuntime(const Binary &binary) { __builtin_trap(); }
void setCurrentDeviceRuntime(const DeviceRuntime &runtime) { __builtin_trap(); }
void setCurrentHostRuntime(const HostRuntime &runtime) { __builtin_trap(); }
void setFabricConfig(FabricConfig config) { __builtin_trap(); }
void setTensorRetain(Tensor tensor, bool retain) { __builtin_trap(); }
void shutdownDistributedRuntime() { __builtin_trap(); }
std::vector<Tensor> toHost(Tensor tensor, bool untilize, bool blocking) {
  __builtin_trap();
}
void updateTensorInPool(CallbackContext programContextHandle,
                        TensorRef tensorRef, Tensor srcTensor) {
  __builtin_trap();
}
void wait(Event event) { __builtin_trap(); }

} // namespace tt::runtime
