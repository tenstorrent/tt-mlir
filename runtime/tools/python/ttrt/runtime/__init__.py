try:
    from ._C import (
        Device,
        Event,
        Tensor,
        DataType,
        get_current_system_desc,
        open_device,
        close_device,
        submit,
        create_tensor,
    )
except ModuleNotFoundError:
    raise ImportError(
        "Error: Project was not built with runtime enabled, rebuild with: -DTTMLIR_ENABLE_RUNTIME=ON"
    )
