try:
    from ._C import get_current_system_desc, open_device, close_device, submit
except ModuleNotFoundError:
    import sys
    print("Error: Project was not built with runtime enabled, rebuild with: -DTTMLIR_ENABLE_RUNTIME=ON", file=sys.stderr)
    sys.exit(1)
