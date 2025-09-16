// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <sstream>

#include "tt/runtime/debug.h"
#include "tt/runtime/perf.h"
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "tt/runtime/workarounds.h"

#include "tt/runtime/detail/python/nanobind_headers.h"

namespace nb = nanobind;

namespace tt::runtime::python {
void registerRuntimeBindings(nb::module_ &m) {
  nb::class_<tt::runtime::MemoryView>(m, "MemoryView")
      .def_ro("num_banks", &tt::runtime::MemoryView::numBanks)
      .def_ro("total_bytes_per_bank",
              &tt::runtime::MemoryView::totalBytesPerBank)
      .def_ro("total_bytes_allocated_per_bank",
              &tt::runtime::MemoryView::totalBytesAllocatedPerBank)
      .def_ro("total_bytes_free_per_bank",
              &tt::runtime::MemoryView::totalBytesFreePerBank)
      .def_ro("largest_contiguous_bytes_free_per_bank",
              &tt::runtime::MemoryView::largestContiguousBytesFreePerBank)
      .def_ro("block_table", &tt::runtime::MemoryView::blockTable);

  nb::class_<tt::runtime::Device>(m, "Device")
      .def("get_mesh_shape", &tt::runtime::getMeshShape)
      .def("get_device_ids", &tt::runtime::getDeviceIds)
      .def("get_num_hw_cqs", &tt::runtime::getNumHwCqs)
      .def("is_program_cache_enabled", &tt::runtime::isProgramCacheEnabled)
      .def("get_l1_small_size", &tt::runtime::getL1SmallSize)
      .def("get_trace_region_size", &tt::runtime::getTraceRegionSize)
      .def("get_num_dram_channels", &tt::runtime::getNumDramChannels)
      .def("get_dram_size_per_channel", &tt::runtime::getDramSizePerChannel)
      .def("get_l1_size_per_core", &tt::runtime::getL1SizePerCore)
      .def("release_trace", &tt::runtime::releaseTrace)
      .def("deallocate_buffers", &tt::runtime::detail::deallocateBuffers)
      .def("dump_memory_report", &tt::runtime::detail::dumpMemoryReport)
      .def("read_device_profiler_results",
           &tt::runtime::detail::readDeviceProfilerResults)
      .def("get_memory_view", &tt::runtime::detail::getMemoryView);

  nb::class_<tt::runtime::Event>(m, "Event");

  nb::class_<tt::runtime::TensorDesc>(m, "TensorDesc")
      .def_ro("shape", &tt::runtime::TensorDesc::shape)
      .def_ro("stride", &tt::runtime::TensorDesc::stride)
      .def_ro("item_size", &tt::runtime::TensorDesc::itemsize)
      .def_ro("dtype", &tt::runtime::TensorDesc::dataType)
      .def_ro("physical_volume", &tt::runtime::TensorDesc::physicalVolume);

  nb::class_<tt::runtime::MeshDeviceOptions>(m, "MeshDeviceOptions")
      .def(nb::init<>())
      .def_rw("mesh_offset", &tt::runtime::MeshDeviceOptions::meshOffset)
      .def_rw("device_ids", &tt::runtime::MeshDeviceOptions::deviceIds)
      .def_rw("num_hw_cqs", &tt::runtime::MeshDeviceOptions::numHWCQs)
      .def_rw("enable_program_cache",
              &tt::runtime::MeshDeviceOptions::enableProgramCache)
      // NOLINTBEGIN(clang-analyzer-core.NullDereference)
      .def_prop_rw(
          "mesh_shape",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.meshShape.has_value() ? nb::cast(o.meshShape.value())
                                           : nb::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, nb::handle value) {
            if (!value.is_none() && !nb::isinstance<nb::list>(value) &&
                !nb::isinstance<nb::tuple>(value)) {
              throw nb::type_error("mesh_shape must be a list, tuple, or None");
            }
            o.meshShape = value.is_none()
                              ? std::nullopt
                              : std::make_optional(
                                    nb::cast<std::vector<uint32_t>>(value));
          })
      // NOLINTEND(clang-analyzer-core.NullDereference)
      .def_prop_rw(
          "l1_small_size",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.l1SmallSize.has_value() ? nb::cast(o.l1SmallSize.value())
                                             : nb::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, nb::handle value) {
            o.l1SmallSize = value.is_none()
                                ? std::nullopt
                                : std::make_optional(nb::cast<size_t>(value));
          })
      .def_prop_rw(
          "trace_region_size",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.traceRegionSize.has_value()
                       ? nb::cast(o.traceRegionSize.value())
                       : nb::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, nb::handle value) {
            o.traceRegionSize =
                value.is_none() ? std::nullopt
                                : std::make_optional(nb::cast<size_t>(value));
          })
      .def_prop_rw(
          "dispatch_core_type",
          [](const tt::runtime::MeshDeviceOptions &o) {
            return o.dispatchCoreType.has_value()
                       ? nb::cast(o.dispatchCoreType.value())
                       : nb::none();
          },
          [](tt::runtime::MeshDeviceOptions &o, nb::handle value) {
            o.dispatchCoreType =
                value.is_none()
                    ? std::nullopt
                    : std::make_optional(
                          nb::cast<tt::runtime::DispatchCoreType>(value));
          });

  nb::class_<tt::runtime::Tensor>(m, "Tensor")
      .def("is_allocated",
           [](tt::runtime::Tensor self) {
             return tt::runtime::isTensorAllocated(self);
           })
      .def("get_retain",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorRetain(self);
           })
      .def("set_retain",
           [](tt::runtime::Tensor self, bool retain) {
             tt::runtime::setTensorRetain(self, retain);
           })
      .def("get_shape",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorShape(self);
           })
      .def("get_stride",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorStride(self);
           })
      .def("get_volume",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorVolume(self);
           })
      .def("get_dtype",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorDataType(self);
           })
      .def("get_element_size",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorElementSize(self);
           })
      .def("get_tensor_desc",
           [](tt::runtime::Tensor self) {
             return tt::runtime::getTensorDesc(self);
           })
      .def(
          "get_data_buffer",
          [](tt::runtime::Tensor self) -> nb::bytearray {
            std::vector<std::byte> vec = tt::runtime::getTensorDataBuffer(self);
            return nb::bytearray(reinterpret_cast<const char *>(vec.data()),
                                 vec.size());
          },
          nb::rv_policy::take_ownership);

  nb::class_<tt::runtime::TensorRef>(m, "TensorRef");
  nb::class_<tt::runtime::Layout>(m, "Layout");
  nb::class_<tt::runtime::OpContext>(m, "OpContext");
  nb::class_<tt::runtime::CallbackContext>(m, "CallbackContext");

  nb::enum_<tt::runtime::MemoryBufferType>(m, "MemoryBufferType")
      .value("DRAM", tt::runtime::MemoryBufferType::DRAM)
      .value("L1", tt::runtime::MemoryBufferType::L1)
      .value("L1_SMALL", tt::runtime::MemoryBufferType::L1_SMALL)
      .value("TRACE", tt::runtime::MemoryBufferType::TRACE);

  nb::enum_<::tt::target::DataType>(m, "DataType")
      .value("Float32", ::tt::target::DataType::Float32)
      .value("Float16", ::tt::target::DataType::Float16)
      .value("BFloat16", ::tt::target::DataType::BFloat16)
      .value("BFP_Float8", ::tt::target::DataType::BFP_Float8)
      .value("BFP_BFloat8", ::tt::target::DataType::BFP_BFloat8)
      .value("BFP_Float4", ::tt::target::DataType::BFP_Float4)
      .value("BFP_BFloat4", ::tt::target::DataType::BFP_BFloat4)
      .value("BFP_Float2", ::tt::target::DataType::BFP_Float2)
      .value("BFP_BFloat2", ::tt::target::DataType::BFP_BFloat2)
      .value("UInt32", ::tt::target::DataType::UInt32)
      .value("UInt16", ::tt::target::DataType::UInt16)
      .value("UInt8", ::tt::target::DataType::UInt8)
      .value("Int32", ::tt::target::DataType::Int32)
      // Unsupported data types
      .value("Float64", ::tt::target::DataType::Float64)
      .value("Int64", ::tt::target::DataType::Int64)
      .value("UInt64", ::tt::target::DataType::UInt64)
      .value("Int16", ::tt::target::DataType::Int16)
      .value("Int8", ::tt::target::DataType::Int8)
      .value("Bool", ::tt::target::DataType::Bool);

  nb::enum_<::tt::runtime::DeviceRuntime>(m, "DeviceRuntime")
      .value("Disabled", ::tt::runtime::DeviceRuntime::Disabled)
      .value("TTNN", ::tt::runtime::DeviceRuntime::TTNN)
      .value("TTMetal", ::tt::runtime::DeviceRuntime::TTMetal)
      .value("CUDA", ::tt::runtime::DeviceRuntime::CUDA);

  nb::enum_<::tt::runtime::DispatchCoreType>(m, "DispatchCoreType")
      .value("WORKER", ::tt::runtime::DispatchCoreType::WORKER)
      .value("ETH", ::tt::runtime::DispatchCoreType::ETH);

  nb::enum_<::tt::runtime::FabricConfig>(m, "FabricConfig")
      .value("DISABLED", ::tt::runtime::FabricConfig::DISABLED)
      .value("FABRIC_1D", ::tt::runtime::FabricConfig::FABRIC_1D)
      .value("FABRIC_1D_RING", ::tt::runtime::FabricConfig::FABRIC_1D_RING)
      .value("FABRIC_2D", ::tt::runtime::FabricConfig::FABRIC_2D)
      .value("FABRIC_2D_TORUS_X",
             ::tt::runtime::FabricConfig::FABRIC_2D_TORUS_X)
      .value("FABRIC_2D_TORUS_Y",
             ::tt::runtime::FabricConfig::FABRIC_2D_TORUS_Y)
      .value("FABRIC_2D_TORUS_XY",
             ::tt::runtime::FabricConfig::FABRIC_2D_TORUS_XY)
      .value("FABRIC_2D_DYNAMIC",
             ::tt::runtime::FabricConfig::FABRIC_2D_DYNAMIC)
      .value("FABRIC_2D_DYNAMIC_TORUS_X",
             ::tt::runtime::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_X)
      .value("FABRIC_2D_DYNAMIC_TORUS_Y",
             ::tt::runtime::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_Y)
      .value("FABRIC_2D_DYNAMIC_TORUS_XY",
             ::tt::runtime::FabricConfig::FABRIC_2D_DYNAMIC_TORUS_XY)
      .value("CUSTOM", ::tt::runtime::FabricConfig::CUSTOM);

  nb::enum_<::tt::runtime::Arch>(m, "Arch")
      .value("GRAYSKULL", ::tt::runtime::Arch::GRAYSKULL)
      .value("WORMHOLE_B0", ::tt::runtime::Arch::WORMHOLE_B0)
      .value("BLACKHOLE", ::tt::runtime::Arch::BLACKHOLE)
      .value("QUASAR", ::tt::runtime::Arch::QUASAR);

  m.def("get_current_runtime", &tt::runtime::getCurrentRuntime,
        "Get the backend device runtime type");
  m.def("get_available_runtimes", &tt::runtime::getAvailableRuntimes,
        "Get the available backend device runtime types");
  m.def("set_compatible_runtime", &tt::runtime::setCompatibleRuntime,
        nb::arg("binary"),
        "Set the backend device runtime type to match the binary");
  m.def("set_current_runtime", &tt::runtime::setCurrentRuntime,
        nb::arg("runtime"), "Set the backend device runtime type");
  m.def("get_current_system_desc", &tt::runtime::getCurrentSystemDesc,
        nb::arg("dispatch_core_type") = nb::none(),
        nb::arg("mesh_device") = nb::none(),
        "Get the current system descriptor");
  m.def(
      "create_borrowed_host_tensor",
      [](std::uintptr_t ptr, const std::vector<std::uint32_t> &shape,
         const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
         ::tt::target::DataType dataType) {
        return tt::runtime::createBorrowedHostTensor(
            reinterpret_cast<void *>(ptr), shape, stride, itemsize, dataType);
      },
      "Create a host tensor with borrowed memory");
  m.def(
      "create_owned_host_tensor",
      [](std::uintptr_t ptr, const std::vector<std::uint32_t> &shape,
         const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
         ::tt::target::DataType dataType) {
        return tt::runtime::createOwnedHostTensor(
            reinterpret_cast<const void *>(ptr), shape, stride, itemsize,
            dataType);
      },
      "Create a tensor with owned memory");
  m.def(
      "create_empty_tensor",
      [](::tt::runtime::Device device, ::tt::runtime::Layout layout,
         const std::vector<std::uint32_t> &shape,
         const std::vector<std::uint32_t> &stride, std::uint32_t itemsize) {
        return tt::runtime::createEmptyTensor(device, layout, shape, stride,
                                              itemsize);
      },
      "Create an empty tensor with the specified layout");
  m.def(
      "create_multi_device_host_tensor",
      [](std::vector<std::uintptr_t> &ptrs,
         const std::vector<std::uint32_t> &shape,
         const std::vector<std::uint32_t> &stride, std::uint32_t itemsize,
         ::tt::target::DataType dataType,
         const std::unordered_map<std::string, std::string> &strategy,
         const std::vector<uint32_t> &meshShape) {
        std::vector<const void *> data;
        data.reserve(ptrs.size());
        std::transform(ptrs.begin(), ptrs.end(), std::back_inserter(data),
                       [](std::uintptr_t ptr) {
                         return reinterpret_cast<const void *>(ptr);
                       });
        return tt::runtime::createMultiDeviceHostTensor(
            data, shape, stride, itemsize, dataType, strategy, meshShape);
      },
      "Create a multi-device host tensor with owned memory");
  m.def("get_arch", &tt::runtime::getArch,
        "Get the architecture of the device");
  m.def("enable_persistent_kernel_cache",
        &tt::runtime::enablePersistentKernelCache,
        "Enable persistent kernel cache, which will cache kernel binaries on "
        "disk usable across runs.");
  m.def("disable_persistent_kernel_cache",
        &tt::runtime::disablePersistentKernelCache,
        "Disable persistent kernel cache, which will disable caching kernel "
        "binaries on disk.");
  m.def("get_num_available_devices", &tt::runtime::getNumAvailableDevices,
        "Get the number of available devices");
  m.def("open_mesh_device", &tt::runtime::openMeshDevice,
        nb::arg("options") = tt::runtime::MeshDeviceOptions(),
        "Open a parent mesh of devices");
  m.def("close_mesh_device", &tt::runtime::closeMeshDevice,
        nb::arg("parent_mesh"), "Close a mesh device");
  m.def("create_sub_mesh_device", &tt::runtime::createSubMeshDevice,
        nb::arg("parent_mesh"), nb::arg("mesh_shape"),
        nb::arg("mesh_offset") = nb::none(),
        "Open a sub mesh of devices from a parent mesh");
  m.def("release_sub_mesh_device", &tt::runtime::releaseSubMeshDevice,
        nb::arg("sub_mesh"), "Release a sub mesh device from the parent");
  m.def("reshape_mesh_device", &tt::runtime::reshapeMeshDevice,
        nb::arg("mesh_device"), nb::arg("mesh_shape"), "Reshape a mesh device");
  m.def("to_host", &tt::runtime::toHost, nb::arg("tensor"),
        nb::arg("untilize") = false, nb::arg("blocking") = true,
        "Copy the tensor to host");
  m.def("to_layout", &tt::runtime::toLayout, nb::arg("tensor"),
        nb::arg("device"), nb::arg("layout"), nb::arg("retain") = nb::none(),
        "Create a copy of the tensor with the specified layout");
  m.def("get_layout", &tt::runtime::getLayout, nb::arg("executable"),
        nb::arg("program_index"), nb::arg("input_index"),
        "Get the layout of the input tensor");
  m.def("set_fabric_config", &tt::runtime::setFabricConfig, nb::arg("config"),
        "Set the fabric config");
  m.def(
      "submit",
      [](::tt::runtime::Device device, ::tt::runtime::Binary &executable,
         std::uint32_t programIndex, std::vector<::tt::runtime::Tensor> &inputs)
          -> std::vector<::tt::runtime::Tensor> {
        return ::tt::runtime::submit(device, executable, programIndex, inputs);
      },
      nb::arg("device"), nb::arg("executable"), nb::arg("program_index"),
      nb::arg("inputs"),
      "Submit a ttnn binary for execution, returns a vector of output tensors."
      "The input tensors will be moved and consumed.");
  m.def(
      "wait", [](::tt::runtime::Event event) { ::tt::runtime::wait(event); },
      nb::arg("event"));
  m.def(
      "wait",
      [](::tt::runtime::Tensor tensor, std::optional<uint8_t> cqId) {
        ::tt::runtime::wait(tensor, cqId);
      },
      nb::arg("tensor"), nb::arg("cq_id") = nb::none());
  m.def(
      "wait",
      [](const std::vector<::tt::runtime::Tensor> &tensors,
         std::optional<uint8_t> cqId) { ::tt::runtime::wait(tensors, cqId); },
      nb::arg("tensors"), nb::arg("cq_id") = nb::none());
  m.def(
      "get_op_output_tensor",
      [](tt::runtime::OpContext &opContextHandle,
         tt::runtime::CallbackContext &programContextHandle) {
        return tt::runtime::getOpOutputTensor(opContextHandle,
                                              programContextHandle);
      },
      "Get the output tensor of the op");
  m.def(
      "get_op_output_ref",
      [](tt::runtime::OpContext &op_context_handle,
         tt::runtime::CallbackContext &program_context_handle) {
        return tt::runtime::getOpOutputRef(op_context_handle,
                                           program_context_handle);
      },
      nb::arg("op_context_handle"), nb::arg("program_context_handle"),
      R"(
    Return a reference to the *output* tensor produced by an operator.

    Parameters
    ----------
    op_context_handle : tt.runtime.OpContext
    program_context_handle : tt.runtime.CallbackContext

    Returns
    -------
    Optional[tt.runtime.TensorRef]
        A reference that uniquely identifies the output tensor, or ``None`` if the
        operator has no outputs.
    )");

  m.def(
      "get_op_input_refs",
      [](tt::runtime::OpContext &op_context_handle,
         tt::runtime::CallbackContext &program_context_handle) {
        return tt::runtime::getOpInputRefs(op_context_handle,
                                           program_context_handle);
      },
      nb::arg("op_context_handle"), nb::arg("program_context_handle"),
      R"(
    Return a list of references to the *input* tensors consumed by an operator.

    Parameters
    ----------
    op_context_handle : ttrt.runtime.OpContext
    program_context_handle : ttrt.runtime.CallbackContext

    Returns
    -------
    List[tt.runtime.TensorRef]
        A possibly empty list of tensor references. The list is empty when the
        operator has no inputs.
    )");

  m.def(
      "retrieve_tensor_from_pool",
      [](tt::runtime::CallbackContext program_context_handle,
         tt::runtime::TensorRef tensor_ref, bool untilize = true) {
        return tt::runtime::retrieveTensorFromPool(program_context_handle,
                                                   tensor_ref, untilize);
      },
      nb::arg("program_context_handle"), nb::arg("tensor_ref"),
      nb::arg("untilize") = true,
      R"(
    Returns tensor from tensor pool to which tensor_ref refers
    For now only supports single device tensors

    Parameters
    ----------
    program_context_handle : ttrt.runtime.CallbackContext
    tensor_ref : ttrt.runtime.TensorRef
        Reference to the tensor of interest (from get_op_output_ref/get_op_input_refs).
    untilize : bool, default ``True``
        If the tensor is stored in a tilized format, de-tilize it before returning. If the untilize flag is ``False``, tensor will be with padding so shape will be different from the original shape

    Returns
    -------
    Optional[tt.runtime.Tensor]
        The tensor corresponding to *tensor_ref*, or ``None`` if the
        tensor is not present in the pool (e.g., it was deallocated).
    )");

  m.def(
      "update_tensor_in_pool",
      [](tt::runtime::CallbackContext program_context_handle,
         tt::runtime::TensorRef tensor_ref_handle,
         tt::runtime::Tensor tensor_handle) {
        tt::runtime::updateTensorInPool(program_context_handle,
                                        tensor_ref_handle, tensor_handle);
      },
      nb::arg("program_context_handle"), nb::arg("tensor_ref_handle"),
      nb::arg("tensor_handle"),
      R"(
    Overwrite the data associated with an existing tensor reference.
    Prefered to be owned tensor to avoid unexpected behavior in case of
    deallocation.

    Parameters
    ----------
    program_context_handle : ttrt.runtime.CallbackContext
    tensor_ref_handle : ttrt.runtime.TensorRef
    tensor_handle : ttrt.runtime.Tensor
        Source tensor which data will tensor_ref refer to.

    Returns
    -------
    None
    )");
  m.def("get_op_debug_str", &tt::runtime::getOpDebugString,
        "Get the debug string of the op");
  m.def("get_op_loc_info", &tt::runtime::getOpLocInfo,
        "Get the location info of the op");
  m.def(
      "memcpy",
      [](std::uintptr_t dst, ::tt::runtime::Tensor src,
         std::optional<::tt::target::DataType> dstDataType) {
        void *dstPtr = reinterpret_cast<void *>(dst);
        ::tt::runtime::memcpy(dstPtr, src, dstDataType);
      },
      nb::arg("dst"), nb::arg("src"), nb::arg("dstDataType") = nb::none(),
      "Copy the data from src tensor to dst pointer");
  m.def(
      "memcpy",
      [](::tt::runtime::Tensor dst, ::tt::runtime::Tensor src) {
        ::tt::runtime::memcpy(dst, src);
      },
      nb::arg("dst"), nb::arg("src"),
      "Copy the data from src tensor to dst tensor");
  m.def("deallocate_tensor", &tt::runtime::deallocateTensor, nb::arg("tensor"),
        nb::arg("force") = false, "Deallocate the tensor memory");

  nb::class_<tt::runtime::debug::Env>(m, "DebugEnv")
      .def_static("get", &tt::runtime::debug::Env::get)
      .def("__str__", [](const tt::runtime::debug::Env &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });

  nb::class_<tt::runtime::perf::Env>(m, "PerfEnv")
      .def_static("get", &tt::runtime::perf::Env::get, nb::rv_policy::reference)
      .def("set_program_metadata", &tt::runtime::perf::Env::setProgramMetadata)
      .def("tracy_log_op_location", &tt::runtime::perf::Env::tracyLogOpLocation)
      .def("tracy_log_const_eval_program",
           &tt::runtime::perf::Env::tracyLogConstEvalProgram)
      .def("tracy_log_program_metadata",
           &tt::runtime::perf::Env::tracyLogProgramMetadata)
      .def("__str__", [](const tt::runtime::perf::Env &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });

  nb::class_<tt::runtime::debug::Hooks>(m, "DebugHooks")
      .def_static(
          "get",
          [](nb::callable pre_op_func, nb::callable post_op_func) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
            return tt::runtime::debug::Hooks::get(
                [pre_op_func](tt::runtime::Binary Binary,
                              tt::runtime::CallbackContext programContext,
                              tt::runtime::OpContext opContext) {
                  pre_op_func(Binary, programContext, opContext);
                },
                [post_op_func](tt::runtime::Binary Binary,
                               tt::runtime::CallbackContext programContext,
                               tt::runtime::OpContext opContext) {
                  post_op_func(Binary, programContext, opContext);
                });
#else
            tt::runtime::debug::Hooks::get();
            return std::nullopt;
#endif
          })
      .def("__str__", [](const tt::runtime::debug::Hooks &hooks) {
        std::stringstream os;
        os << hooks;
        return os.str();
      });

  nb::class_<tt::runtime::debug::Stats>(m, "DebugStats")
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
      .def_static("get", &tt::runtime::debug::Stats::get,
                  nb::rv_policy::reference)
#else
      .def_static("get", &tt::runtime::debug::Stats::get)
#endif
      .def("increment_stat", &tt::runtime::debug::Stats::incrementStat,
           nb::arg("stat"), nb::arg("value") = 1)
      .def("get_stat", &tt::runtime::debug::Stats::getStat, nb::arg("stat"))
      .def("remove_stat", &tt::runtime::debug::Stats::removeStat,
           nb::arg("stat"))
      .def("clear", &tt::runtime::debug::Stats::clear)
      .def("__str__", &tt::runtime::debug::Stats::toString);

  nb::class_<tt::runtime::workaround::Env>(m, "WorkaroundEnv")
      .def_static("get", &tt::runtime::workaround::Env::get)
      .def("__str__", [](const tt::runtime::workaround::Env &env) {
        std::stringstream os;
        os << env;
        return os.str();
      });

  m.def("unregister_hooks",
        []() { ::tt::runtime::debug::Hooks::get().unregisterHooks(); });
}
} // namespace tt::runtime::python
