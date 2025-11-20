# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN-JIT internal utilities and implementation."""

import sys

# Try wheel installation path first, then fallback to local development path
try:
    # Import ttmlir,then register as top-level module
    # This prevents double initialization when C++ extensions later do "import ttmlir"
    import ttnn_jit.runtime.ttmlir as _ttmlir_module
    import ttnn_jit.runtime.ttmlir.ir as _ttmlir_ir_module
    import ttnn_jit.runtime.ttmlir.passes as _ttmlir_passes_module
    import ttnn_jit.runtime.ttmlir.dialects as _ttmlir_dialects_module
    import ttnn_jit.runtime.ttmlir._mlir_libs as _ttmlir_mlir_libs
    import ttnn_jit.runtime.ttmlir._mlir_libs._mlir as _ttmlir_mlir
    import ttnn_jit.runtime.ttmlir._mlir_libs._ttmlir as _ttmlir_ttmlir
    import ttnn_jit.runtime.ttmlir.dialects.func as _ttmlir_dialect_func
    import ttnn_jit.runtime.ttmlir.dialects.tensor as _ttmlir_dialect_tensor
    import ttnn_jit.runtime.ttmlir.dialects.ttcore as _ttmlir_dialect_ttcore
    import ttnn_jit.runtime.ttmlir.dialects.ttir as _ttmlir_dialect_ttir
    import ttnn_jit.runtime.ttmlir.dialects.ttkernel as _ttmlir_dialect_ttkernel
    import ttnn_jit.runtime.ttmlir.dialects.ttnn as _ttmlir_dialect_ttnn

    # Register as top-level modules so you can still do something like `import ttmlir.ir`
    sys.modules["ttmlir"] = _ttmlir_module
    sys.modules["ttmlir.ir"] = _ttmlir_ir_module
    sys.modules["ttmlir.passes"] = _ttmlir_passes_module
    sys.modules["ttmlir.dialects"] = _ttmlir_dialects_module
    sys.modules["ttmlir._mlir_libs"] = _ttmlir_mlir_libs
    sys.modules["ttmlir._mlir_libs._mlir"] = _ttmlir_mlir
    sys.modules["ttmlir._mlir_libs._ttmlir"] = _ttmlir_ttmlir
    sys.modules["ttmlir.dialects.func"] = _ttmlir_dialect_func
    sys.modules["ttmlir.dialects.tensor"] = _ttmlir_dialect_tensor
    sys.modules["ttmlir.dialects.ttcore"] = _ttmlir_dialect_ttcore
    sys.modules["ttmlir.dialects.ttir"] = _ttmlir_dialect_ttir
    sys.modules["ttmlir.dialects.ttkernel"] = _ttmlir_dialect_ttkernel
    sys.modules["ttmlir.dialects.ttnn"] = _ttmlir_dialect_ttnn

    from ttnn_jit.runtime._ttmlir_runtime.runtime import (
        submit,
        set_compatible_device_runtime,
        get_current_device_runtime,
        get_current_system_desc,
        DispatchCoreType,
    )
    from ttnn_jit.runtime._ttmlir_runtime.binary import (
        load_binary_from_path,
        load_binary_from_capsule,
    )
    from ttnn_jit.runtime._ttmlir_runtime.utils import (
        create_runtime_device_from_ttnn,
        create_runtime_tensor_from_ttnn,
        get_ttnn_tensor_from_runtime_tensor,
    )
    from ttnn_jit.runtime._ttnn_jit import JitCache
except ModuleNotFoundError:
    # Fallback to local development imports
    from _ttmlir_runtime.runtime import (
        submit,
        set_compatible_device_runtime,
        get_current_device_runtime,
        get_current_system_desc,
        DispatchCoreType,
    )
    from _ttmlir_runtime.binary import (
        load_binary_from_path,
        load_binary_from_capsule,
    )
    from _ttmlir_runtime.utils import (
        create_runtime_device_from_ttnn,
        create_runtime_tensor_from_ttnn,
        get_ttnn_tensor_from_runtime_tensor,
    )
    from ttnn_jit._ttnn_jit import JitCache
