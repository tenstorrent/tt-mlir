# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.types import *
import hashlib


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

    def _get_cbs(self, tensors, options):
        if hasattr(self, "_made_cbs"):
            return getattr(self, "_made_cbs")

        # Define CBs otherwise
        cbs = self.define_cbs(tensors, options)
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
        return []

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

    def __call__(self, *tensors, **options):
        """
        Execute the kernel operation on the given tensors with specified options.

        Args:
            tensors: Input tensors
            options: Additional options for kernel execution

        Returns:
            Result tensor(s) from the operation
        """
        # Compute hash for kernel selection and caching
        input_hash = self._compute_input_hash(tensors, options)

        # Check if kernel is already cached
        if input_hash not in self.kernel_cache:
            # Select appropriate kernels based on inputs
            selected_kernels = self.select_kernels(tensors, options)

            # Get cbs and semaphores
            cbs = self._get_cbs(tensors, options)
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
                    "core_ranges": options["core_ranges"],
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
