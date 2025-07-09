import functools
import pdb
import traceback

def debug_wrap(*, debug: bool = False):
    """
    Decorator factory: use as @debug_wrap(debug=DEBUG_FLAG).
    Helpful to use with runtime callbacks to get more info about errors.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:              # noqa: BLE001
                if debug:                  # flag decided at *definition* time
                    traceback.print_exc()
                    pdb.set_trace()
                raise
        return wrapper
    return decorator