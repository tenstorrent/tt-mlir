// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ort.hpp"
#include "config.hpp"
#include "engine/compile_options.hpp"
#include "tensor.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iterator>
#include <new>
#include <tt-logger/tt-logger.hpp>
#include <vector>

#include <tt/runtime/runtime.h>

#include "assert.hpp"
#include "cast.hpp"
#include "compute.hpp"
#include "engine/device.hpp"
#include "graph.hpp"
#include "onnx/options.hpp"
#include "onnx/partition.hpp"
#include "onnxruntime_c_api.h"

namespace tt::crank::onnx {

const OrtApi *ort_api_ptr = nullptr;  // runtime API
const OrtEpApi *ep_api_ptr = nullptr; // execution provider API
const OrtModelEditorApi *model_editor_ptr = nullptr;

const OrtApi &ort_api() {
    TT_FATAL(ort_api_ptr != nullptr, "runtime api null");
    return *ort_api_ptr;
}

const OrtEpApi &ep_api() {
    TT_FATAL(ep_api_ptr != nullptr, "execution provider api null");
    return *ep_api_ptr;
}

const OrtModelEditorApi &model_editor_api() {
    TT_FATAL(model_editor_ptr != nullptr, "model editor api null (minimal ORT build?)");
    return *model_editor_ptr;
}

OrtMemoryInfo *tt_meminfo = nullptr;

OrtMemoryInfo *meminfo() {
    TT_FATAL(tt_meminfo != nullptr, "tt meminfo null");
    return tt_meminfo;
}

// Session options.
struct Opts {
    Opts(const OrtSessionOptions *opts)
        : compile_options{parse_compile_options(opts)}, ep_ctx_enabled{parse_ep_ctx_enabled(opts)} {}

    CompileOptions compile_options;
    bool ep_ctx_enabled;
};

// Extended OrtEp so that it has a compile options.
// This should be rethought and implemented in a better way.
// Hacking it for now.
struct TtEp : OrtEp {
    TtEp(const OrtSessionOptions *ort_opts) : OrtEp{}, opts{ort_opts} {}

    [[nodiscard]] const CompileOptions &compile_options() const { return opts.compile_options; }
    [[nodiscard]] bool ep_ctx_enabled() const { return opts.ep_ctx_enabled; }

    Opts opts;
};

namespace {

// Makes context node which is used by ORT for creating .onnx files.
// We are using embed mode, which means that weights are stored in .onnx file (check out partition.serialize()).
OrtNode *make_ep_context_node(const OrtNode *fused_node, const Partition &partition) {
    const std::string blob = partition.serialize();

    std::vector<OrtOpAttr *> attrs;
    auto add_attr = [&attrs](const char *name, const void *data, int len, OrtOpAttrType type) {
        OrtOpAttr *attr = nullptr;
        check_call(ort_api().CreateOpAttr(name, data, len, type, &attr));
        attrs.push_back(attr);
    };

    const std::int64_t embed_mode = 1;
    const std::int64_t main_context = 1;
    const std::string source = "TTKurblaExecutionProvider";
    const std::string partition_name = node_name(fused_node);

    add_attr("ep_cache_context", blob.data(), as<int>(blob.size()), ORT_OP_ATTR_STRING);
    add_attr("embed_mode", &embed_mode, 1, ORT_OP_ATTR_INT);
    add_attr("main_context", &main_context, 1, ORT_OP_ATTR_INT);
    add_attr("source", source.data(), as<int>(source.size()), ORT_OP_ATTR_STRING);
    add_attr("partition_name", partition_name.data(), as<int>(partition_name.size()), ORT_OP_ATTR_STRING);

    std::vector<const char *> input_ptrs;
    std::ranges::transform(node_inputs(fused_node), std::back_inserter(input_ptrs), value_name_ref);

    std::vector<const char *> output_ptrs;
    std::ranges::transform(node_outputs(fused_node), std::back_inserter(output_ptrs), value_name_ref);

    OrtNode *node = nullptr;
    check_call(model_editor_api().CreateNode("EPContext", "com.microsoft", partition_name.c_str(), input_ptrs.data(),
                                             input_ptrs.size(), output_ptrs.data(), output_ptrs.size(), attrs.data(),
                                             attrs.size(), &node));
    for (OrtOpAttr *attr : attrs) {
        ort_api().ReleaseOpAttr(attr);
    }

    return node;
}

} // namespace

const char *ORT_API_CALL GetNameEpImpl(const OrtEp * /*this_ptr*/) noexcept {
    return "TTKurblaExecutionProvider";
}

OrtStatus *ORT_API_CALL IsConcurrentRunSupportedImpl(OrtEp * /*this_ptr*/, bool *is_supported) noexcept {
    *is_supported = false;
    return nullptr;
}

OrtStatus *ORT_API_CALL Partition::CreateStateImpl(OrtNodeComputeInfo *part,
                                                   OrtNodeComputeContext * /*compute_context*/,
                                                   void **compute_state) noexcept {
    try {
        *compute_state = new ComputeState(as<Partition *>(part));
        return nullptr;
    } catch (const std::exception &e) {
        return fail_status(e);
    }
}

OrtStatus *ORT_API_CALL Partition::ComputeImpl(OrtNodeComputeInfo *part, void *compute_state,
                                               OrtKernelContext *ctx) noexcept {
    try {
        return as<ComputeState *>(compute_state)->compute(ctx, as<Partition *>(part));
    } catch (const std::exception &e) {
        return fail_status(e);
    }
}

void ORT_API_CALL Partition::ReleaseStateImpl(OrtNodeComputeInfo * /*this_ptr*/, void *compute_state) noexcept {
    delete as<ComputeState *>(compute_state);
}

OrtStatus *ORT_API_CALL CompileImpl(OrtEp *this_ptr, const OrtGraph **graphs, const OrtNode **fused_nodes, size_t count,
                                    OrtNodeComputeInfo **partitions, OrtNode **ep_context_nodes) noexcept {
    try {
        auto *tt_ep = as<TtEp *>(this_ptr);
        for (std::size_t i = 0; i < count; ++i) {
            Partition *part = new Partition{graphs[i], fused_nodes[i]};
            part->compile(tt_ep->compile_options());
            partitions[i] = part;

            if (tt_ep->ep_ctx_enabled()) {
                ep_context_nodes[i] = make_ep_context_node(fused_nodes[i], *part);
            }
        }
        return nullptr;
    } catch (const std::exception &e) {
        return fail_status(e);
    }
}

// Returns whether all graph nodes are supported.
// For "context mode", all nodes must be context nodes, and true is returned, because we have already compiled this
// graph. Otherwise, all nodes must be supported.
// Note that are using count_if for node traversal, because all_of will fail after first unsupported node and remaining
// unsupported nodes will not get it's diagnostics printed.
bool all_nodes_supported(std::vector<const OrtNode *> nodes) {
    TT_FATAL(std::ranges::none_of(nodes, is_ep_context_node), "Invalid graph - there are context nodes.");
    bool all_supp = std::ranges::count_if(nodes, [](const OrtNode *node) { return !is_node_supported(node); }) == 0;
    if (!all_supp && !log_fallback_enabled()) {
        log_error(tt::LogAlways,
                  "Graph contains unsupported nodes. To get details, set TT_KURBLA_LOG_FALLBACK_ENABLED=1");
    }

    return all_supp;
}

// This function expects that we call EpGraphSupportInfo_AddNodesToFuse for all nodes that we can compile.
// It looks to me that is not trivial to implement partial graph compilation, hence, for start, we will try to
// compile whole graph or fail. For nodes that are not added with EpGraphSupportInfo_AddNodesToFuse, ORT will try to
// fallback to CPU, which we have prevented with session.disable_cpu_ep_fallback in
// tests/python/onnx/ort_ep_utils.py;
OrtStatus *ORT_API_CALL GetCapabilityImpl(OrtEp * /*this_ptr*/, const OrtGraph *graph,
                                          OrtEpGraphSupportInfo *graph_support_info) noexcept {
    try {
        auto nodes = graph_nodes(graph);
        if (!graph_ctx_serialized(nodes) && !all_nodes_supported(nodes)) {
            return nullptr;
        }

        // Weights (constant initializers) are dropped after compile, so we must copy them before compile finishes.
        OrtNodeFusionOptions fusion_options = {.ort_version_supported = ORT_API_VERSION,
                                               .drop_constant_initializers = true};
        check_call(ep_api().EpGraphSupportInfo_AddNodesToFuse(graph_support_info, nodes.data(), nodes.size(),
                                                              &fusion_options));
        return nullptr;
    } catch (const std::exception &e) {
        return fail_status(e);
    }
}

void ORT_API_CALL ReleaseNodeComputeInfosImpl(OrtEp * /*this_ptr*/, OrtNodeComputeInfo **node_compute_infos,
                                              size_t num_node_compute_infos) noexcept {
    for (std::size_t i = 0; i < num_node_compute_infos; ++i) {
        delete as<Partition *>(node_compute_infos[i]);
    }
}

static const char *ORT_API_CALL GetNameImpl(const OrtEpFactory * /*this_ptr*/) noexcept {
    return "TTKurblaExecutionProvider";
}
static const char *ORT_API_CALL GetVendorImpl(const OrtEpFactory * /*this_ptr*/) noexcept {
    return "Tenstorrent";
}
static uint32_t ORT_API_CALL GetVendorIdImpl(const OrtEpFactory * /*this_ptr*/) noexcept {
    return 0x1E52;
}
static const char *ORT_API_CALL GetVersionImpl(const OrtEpFactory * /*this_ptr*/) noexcept {
    return "0.1.0";
}

// Just register default CPU device.
// For tensor transfers, we will use custom created memory info.
static OrtStatus *ORT_API_CALL GetSupportedDevicesImpl(OrtEpFactory *factory, const OrtHardwareDevice *const *devices,
                                                       size_t num_devices, OrtEpDevice **ep_devices,
                                                       size_t max_ep_devices, size_t *num_ep_devices) noexcept {
    *num_ep_devices = 0;
    try {
        for (size_t i = 0; i < num_devices && *num_ep_devices < max_ep_devices; ++i) {
            if (ort_api().HardwareDevice_Type(devices[i]) != OrtHardwareDeviceType_CPU) {
                continue;
            }
            OrtEpDevice *ep_device = nullptr;
            check_call(ep_api().CreateEpDevice(factory, devices[i], nullptr, nullptr, &ep_device));
            check_call(ep_api().EpDevice_AddAllocatorInfo(ep_device, meminfo()));
            ep_devices[(*num_ep_devices)++] = ep_device;
        }
        return nullptr;
    } catch (const std::exception &e) {
        return fail_status(e);
    }
}

static OrtStatus *ORT_API_CALL CreateEpImpl(OrtEpFactory * /*this_ptr*/, const OrtHardwareDevice *const * /*devices*/,
                                            const OrtKeyValuePairs *const * /*ep_metadata_pairs*/,
                                            size_t /*num_devices*/, const OrtSessionOptions *session_options,
                                            const OrtLogger * /*logger*/, OrtEp **ep) noexcept {
    try {
        auto *ort_ep = new TtEp{session_options};
        ort_ep->ort_version_supported = ORT_API_VERSION;
        ort_ep->GetName = GetNameEpImpl;
        ort_ep->GetCapability = GetCapabilityImpl;
        ort_ep->Compile = CompileImpl;
        ort_ep->ReleaseNodeComputeInfos = ReleaseNodeComputeInfosImpl;
        ort_ep->IsConcurrentRunSupported = IsConcurrentRunSupportedImpl;

        *ep = ort_ep;

        return nullptr;
    } catch (const std::exception &e) {
        return fail_status(e);
    }
}

static void ORT_API_CALL ReleaseEpImpl(OrtEpFactory * /*this_ptr*/, OrtEp *ep) noexcept {
    delete as<TtEp *>(ep);
}

static void *ORT_API_CALL AllocImpl(OrtAllocator * /*this_ptr*/, size_t /*size*/) noexcept {
    return new (std::nothrow) TensorBox{};
}

static void ORT_API_CALL FreeImpl(OrtAllocator * /*this_ptr*/, void *p) noexcept {
    delete as<TensorBox *>(p);
}

static const OrtMemoryInfo *ORT_API_CALL AllocatorInfoImpl(const OrtAllocator * /*this_ptr*/) noexcept {
    return meminfo();
}

static OrtStatus *ORT_API_CALL CreateAllocatorImpl(OrtEpFactory * /*this_ptr*/, const OrtMemoryInfo *memory_info,
                                                   const OrtKeyValuePairs * /*allocator_options*/,
                                                   OrtAllocator **allocator) noexcept {
    if (memory_info == nullptr) {
        *allocator = nullptr;
        return nullptr;
    }
    auto *tt_allocator = new (std::nothrow) OrtAllocator{};
    if (tt_allocator == nullptr) {
        return ort_api().CreateStatus(ORT_EP_FAIL, "OrtAllocator allocation failed");
    }
    tt_allocator->version = ORT_API_VERSION;
    tt_allocator->Alloc = AllocImpl;
    tt_allocator->Free = FreeImpl;
    tt_allocator->Info = AllocatorInfoImpl;
    tt_allocator->Reserve = AllocImpl;
    *allocator = tt_allocator;
    return nullptr;
}

static void ORT_API_CALL ReleaseAllocatorImpl(OrtEpFactory * /*this_ptr*/, OrtAllocator *allocator) noexcept {
    delete allocator;
}

static void ORT_API_CALL DataTransferReleaseImpl(OrtDataTransferImpl *this_ptr) noexcept {
    delete this_ptr;
}

// Returns whether CopyTensorsImpl will be called for provided tensor devices.
// Call custom copy only when we have tensor(s) with TT memory device.
// For host<->host transfer, let ORT do the copy.
static bool ORT_API_CALL CanCopyImpl(const OrtDataTransferImpl * /*this_ptr*/, const OrtMemoryDevice *src_device,
                                     const OrtMemoryDevice *dst_device) noexcept {
    return is_tt(src_device) || is_tt(dst_device);
}

static OrtStatus *ORT_API_CALL CopyTensorsImpl(OrtDataTransferImpl * /*this_ptr*/, const OrtValue **src_tensors,
                                               OrtValue **dst_tensors, OrtSyncStream ** /*streams*/,
                                               size_t num_tensors) noexcept {
    try {
        for (std::size_t i = 0; i < num_tensors; ++i) {
            copy_tensor(src_tensors[i], dst_tensors[i]);
        }
        return nullptr;
    } catch (const std::exception &e) {
        return fail_status(e);
    }
}

static OrtStatus *ORT_API_CALL CreateDataTransferImpl(OrtEpFactory * /*this_ptr*/,
                                                      OrtDataTransferImpl **data_transfer) noexcept {
    auto *tt_data_transfer = new (std::nothrow) OrtDataTransferImpl{};
    if (tt_data_transfer == nullptr) {
        return ort_api().CreateStatus(ORT_EP_FAIL, "OrtDataTransferImpl allocation failed");
    }
    tt_data_transfer->ort_version_supported = ORT_API_VERSION;
    tt_data_transfer->Release = DataTransferReleaseImpl;
    tt_data_transfer->CanCopy = CanCopyImpl;
    tt_data_transfer->CopyTensors = CopyTensorsImpl;
    *data_transfer = tt_data_transfer;
    return nullptr;
}

static bool ORT_API_CALL IsStreamAwareImpl(const OrtEpFactory * /*this_ptr*/) noexcept {
    return false;
}

static OrtStatus *ORT_API_CALL CreateSyncStreamForDeviceImpl(OrtEpFactory * /*this_ptr*/,
                                                             const OrtMemoryDevice * /*memory_device*/,
                                                             const OrtKeyValuePairs * /*stream_options*/,
                                                             OrtSyncStreamImpl **stream) noexcept {
    *stream = nullptr;
    return nullptr;
}

extern "C" {

__attribute__((visibility("default"))) OrtStatus *CreateEpFactories(const char * /*registered_name*/,
                                                                    const OrtApiBase *ort_api_base,
                                                                    const OrtLogger * /*default_logger*/,
                                                                    OrtEpFactory **factories, size_t /*max_factories*/,
                                                                    size_t *num_factories) {
    ort_api_ptr = ort_api_base->GetApi(ORT_API_VERSION);
    TT_FATAL(ort_api_ptr != nullptr, "Unsupported API version.");

    ep_api_ptr = ort_api().GetEpApi();
    TT_FATAL(ep_api_ptr != nullptr, "Execution provider API null");

    model_editor_ptr = ort_api().GetModelEditorApi();
    TT_FATAL(model_editor_ptr != nullptr, "Model editor API null");

    auto *factory = new OrtEpFactory{}; // NOLINT
    factory->ort_version_supported = ORT_API_VERSION;
    factory->GetName = GetNameImpl;
    factory->GetVendor = GetVendorImpl;
    factory->GetVendorId = GetVendorIdImpl;
    factory->GetVersion = GetVersionImpl;
    factory->GetSupportedDevices = GetSupportedDevicesImpl;
    factory->CreateEp = CreateEpImpl;
    factory->ReleaseEp = ReleaseEpImpl;
    factory->CreateAllocator = CreateAllocatorImpl;
    factory->ReleaseAllocator = ReleaseAllocatorImpl;
    factory->CreateDataTransfer = CreateDataTransferImpl;
    factory->IsStreamAware = IsStreamAwareImpl;
    factory->CreateSyncStreamForDevice = CreateSyncStreamForDeviceImpl;

    check_call(ort_api().CreateMemoryInfo_V2("TTKurbla", OrtMemoryInfoDeviceType_NPU, /*vendor_id=*/0x1E52,
                                             /*device_id=*/0, OrtDeviceMemoryType_DEFAULT, /*alignment=*/0,
                                             OrtDeviceAllocator, &tt_meminfo));

    *factories = factory;
    *num_factories = 1;

    return nullptr;
}

__attribute__((visibility("default"))) OrtStatus *ReleaseEpFactory(OrtEpFactory *factory) {
    ort_api().ReleaseMemoryInfo(meminfo());
    delete factory;
    try {
        close_runtime_device_mesh();
        return nullptr;
    } catch (const std::exception &e) {
        return fail_status(e);
    }
}

} // extern "C"

}; // namespace tt::crank::onnx
