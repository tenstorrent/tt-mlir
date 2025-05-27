# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.types import *
import hashlib
import inspect
import logging

# Total possible CBs a
TOTAL_CBS = 16


class PyKernelOp:
    """
    Base class for PyKernel Operations, contains a variety of methods to be overriden.
    """

    def __init__(self, ttnn):
        """Initialize the PyKernelOp with an empty kernel selection dictionary. Intakes the `ttnn` module to operate."""
        self.kernel_selection = {}
        self.kernel_cache = {}

        # Keep a mobile statewise reference to the ttnn module
        self.ttnn = ttnn

    def _tile_bytes_from_ttnn_dtype(self, dtype):
        # Retrieved from: https://github.com/tenstorrent/tt-buda/blob/main/pybuda/csrc/balancer/balancer_utils.hpp
        match dtype:
            case 0 | 6:  # BFP16, U16
                return 32 * 32 * 2
            case 1 | 2 | 7:
                return 32 * 32 * 4
            case 3:
                return 32 * 32 + 64
            case 5:
                return 32 * 32
            case 4:
                return 512 + 64
            case _:
                return "Invalid DataType Processed"

    def _get_semaphores(self, tensors, options):
        if hasattr(self, "_made_semaphores"):
            return getattr(self, "_made_semaphores")

        # Define the Sempahores otherwise
        semaphores = self.define_semaphores(tensors, options)
        setattr(self, "_made_semaphores", semaphores)

        return semaphores

    def define_semaphores(self, tensors, options):
        """Define Semaphore Descriptors from tensors and options."""
        # Default implementation - subclasses should override
        return []

    def _get_cbs(self, selected_kernels, tensors, options):
        if hasattr(self, "_made_cbs"):
            return getattr(self, "_made_cbs")

        # Define CBs otherwise, use autoconfigure if define_cbs isn't defined
        cbs = self.define_cbs(tensors, options)
        if cbs is None:
            cbs = self._autoconfigure_cbs(selected_kernels, tensors, options)
            if cbs is None:
                raise ValueError("Please use define_cbs. Autoconfiguration has failed.")

        setattr(self, "_made_cbs", cbs)

        # Define the PyKernel Type CBs as well
        pykernel_cbs = []
        for cb_format in self._cb_formats:
            pykernel_cbs.append(CircularBuffer(cb_format.buffer_index))
        setattr(self, "_made_pykernel_cbs", pykernel_cbs)

        return cbs

    def define_cbs(self, tensors, options):
        """Define Circular Buffers Descriptors from tensors and options."""
        # Default implementation - subclasses should override
        return None

    def _get_core_ranges(self, tensors, options):
        res = self.define_core_ranges(tensors, options)
        if res is None:
            raise ValueError(
                "Please use define_core_ranges, can't automatically resolve"
            )
        return res

    def define_core_ranges(self, tensors, options):
        """Define Core Ranges from tensors and options."""
        # Default implementation - subclasses should override
        return None

    def select_kernels(self, tensors, options):
        """Select appropriate kernels based on input tensors and options."""
        # Default implementation - subclasses should override

        return {}

    def _compute_input_hash(self, tensors, options):
        """Compute a hash of the input tensors and compile-time options."""
        # Simplified implementation - should be enhanced for production
        hash_input = str(
            [(getattr(t, "shape", None), getattr(t, "dtype", None)) for t in tensors]
        )
        hash_input += str(options)
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _autoconfigure_options(self, tensors, options):
        """Populate the Options dict with relevant values based on tensor information"""

        if len(tensors) == 2:
            # Assume I/O Tensors, only greedily applied option for now.
            i, o = tensors
            i_cfg, o_cfg = i.memory_config(), o.memory_config()

            # Populate is_dram_input
            options["is_dram_input"] = i_cfg.buffer_type == self.ttnn.BufferType.DRAM

            # Check case where inputs have same shape
            if list(i.shape) == list(o.shape):
                # Now we can set the tile size greedily for only I/O CBs
                pass

    def _autoconfigure_cbs(self, selected_kernels, tensors, options):
        # Total possible CBs is 16
        cbs = {}
        num_cbs = 0
        # Here we need to ingest the selected kernels as well
        for kernel, config in selected_kernels:
            # Figure out which is which
            params = inspect.signature(kernel).parameters

            # Get all CB params
            kernel_cbs = [
                param
                for param in params.values()
                if param.annotation.__name__ == "CircularBuffer"
            ]

            # Since these are posiitional arguments, the position matters the most.
            # For each kernel where the position matches as well as the name, this will be considered the SAME circular buffer
            # Otherwise we can't autoconfigure this.

            for i, cb in enumerate(kernel_cbs):
                if cb.name in cbs:
                    idx = cbs[cb.name]
                    if i == idx:
                        # Same CB here, move forward
                        continue
                    else:
                        logging.error(
                            "all CBs must share the same position and name to be autoconfigured, please use defined_cbs."
                        )
                        return None
                else:
                    # Populate the cbs dict
                    cbs[cb.name] = i
                    num_cbs += 1

        # Check for index collisions
        if len(cbs.values()) != len(set(cbs.values())):
            logging.error(
                "Index Collisions in Autoconfiguration Logic, please use define_cbs."
            )
            return None

        # Define the CB formats using the largest tensor to define the size, data format, etc..
        dtypes = [t.dtype for t in tensors]
        if len(set(dtypes)) != 1:
            logging.error(
                "Datatypes for all Input Tensors must match for Autoconfiguration, please use define_cbs."
            )
            return None
        dtype = dtypes[0]

        # Page size is just 1 tile x dtype size
        page_size = self._tile_bytes_from_ttnn_dtype(int(dtype))
        if not isinstance(page_size, int):
            raise ValueError("Invalid DataType possessed by tensors.")

        # Get the largest volume
        largest_vol = max(t.volume() for t in tensors)
        # Divide by 32x32 tile shapes
        num_tiles = largest_vol // 1024
        total_size = num_tiles * page_size

        # Construct CBs with resultant buffer_indices using the cbs
        res_cbs = [None for i in range(num_cbs)]
        cb_formats = [None for i in range(num_cbs)]
        current_buffer_index = 0

        # Construct the resultant cbs and cb_formats
        for cb, i in cbs.items():
            core_ranges = self._get_core_ranges(tensors, options)

            cb_formats[i] = self.ttnn.CBFormatDescriptor(
                buffer_index=current_buffer_index,
                data_format=dtype,
                page_size=page_size,
            )

            res_cbs[i] = self.ttnn.CBDescriptor(
                total_size=total_size,
                core_ranges=core_ranges,
                format_descriptors=[cb_formats[i]],
            )

            current_buffer_index += 1

        if any(x is None for x in res_cbs):
            logging.error(
                "Not all CBs were filled, AutoConfiguration failed. Please use define_cbs"
            )
            return None

        self._cb_formats = cb_formats

        return res_cbs

    def __call__(self, *tensors, **options):
        """
        Execute the kernel operation on the given tensors with specified options.

        Args:
            tensors: Input tensors
            options: Additional options for kernel execution

        Returns:
            Result tensor(s) from the operation
        """
        # Autoconfigure Options based on Input
        self._autoconfigure_options(tensors, options)

        # Compute hash for kernel selection and caching
        input_hash = self._compute_input_hash(tensors, options)

        # Check if kernel is already cached
        if input_hash not in self.kernel_cache:
            # Select appropriate kernels based on inputs
            selected_kernels = self.select_kernels(tensors, options)

            # Get cbs and semaphores
            cbs = self._get_cbs(selected_kernels, tensors, options)
            pykernel_cbs = self._made_pykernel_cbs

            semaphores = self._get_semaphores(tensors, options)

            # Prepare arguments need to call using getattr and .__name__ attrs.
            kernel_descriptors = []
            for kernel, config in selected_kernels:
                # Default to just returning a list if it doesn't exist.
                ct_args_fn = lambda a, b: []
                rt_args_fn = lambda a, b: []
                defines = None

                if hasattr(self, f"{kernel.__name__}_CT_ARGS"):
                    ct_args_fn = getattr(self, f"{kernel.__name__}_CT_ARGS")
                if hasattr(self, f"{kernel.__name__}_RT_ARGS"):
                    rt_args_fn = getattr(self, f"{kernel.__name__}_RT_ARGS")
                if hasattr(self, f"{kernel.__name__}_DEFINES"):
                    defines = getattr(self, f"{kernel.__name__}_DEFINES")

                ct_args = ct_args_fn(tensors, options)
                rt_args = rt_args_fn(tensors, options)

                # No default value for defines, so it involves a little more delicacy in handling
                if defines is not None:
                    defines = defines(tensors, options)

                # Compile the module to C++ and store in the generated directory
                kernel_string = kernel(*pykernel_cbs, rt_args, ct_args=ct_args)
                kernel_t = Kernel(kernel.__name__, kernel_string)
                kernel_path = kernel_t.dump_to_file()

                # Define the KernelDescriptor for each Kernel
                kernel_desc_args = {
                    "kernel_source": kernel_path,
                    "core_ranges": self._get_core_ranges(tensors, options),
                    "compile_time_args": ct_args,
                    "runtime_args": [[rt_args]],
                    "config": config(),
                }

                if defines is not None:
                    kernel_desc_args["defines"] = defines

                common_runtime_args = options.get("common_runtime_args")
                if common_runtime_args is not None:
                    kernel_desc_args["common_runtime_args"] = common_runtime_args

                kernel_desc = self.ttnn.KernelDescriptor(**kernel_desc_args)

                kernel_descriptors.append(kernel_desc)

            # Create the program descriptor
            program_descriptor = self.ttnn.ProgramDescriptor(
                kernels=kernel_descriptors, cbs=cbs, semaphores=semaphores
            )

            # Store the compiled kernel in the cache
            self.kernel_cache[input_hash] = {
                "kernels": kernel_descriptors,
                "program_descriptor": program_descriptor,
            }

        # Retrieve cached kernel information and execute
        cached_info = self.kernel_cache[input_hash]
        program = cached_info["program_descriptor"]

        return self.ttnn.generic_op(tensors, program)
