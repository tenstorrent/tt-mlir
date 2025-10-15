# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttrt
import ttrt.runtime
import json
import platform
from functools import reduce
import operator
import torch
import subprocess
from typing import Any, Dict, List, Tuple, Optional

ALL_BACKENDS = set(["ttnn", "ttmetal", "ttnn-standalone", "emitpy"])
ALL_SYSTEMS = set(["n150", "n300", "llmbox", "tg", "p150", "p300"])


_current_device = None
_current_device_target: Optional[str] = None
_current_device_mesh_shape: Optional[Tuple[int, int]] = None


def _get_device_for_target(target: str, mesh_shape: Tuple[int, int], pytestconfig):
    """Given a `target`, returns a device capable of executing a flatbuffer
    compiled for that `target`.

    For efficiency, this device is reused from the last test if possible via
    the `_current_device`, `_current_device_target` & `_current_device_mesh_shape` caches
    """
    global _current_device, _current_device_target, _current_device_mesh_shape

    if _current_device is not None:

        # Cache hit
        if (
            _current_device_target == target
            and _current_device_mesh_shape == mesh_shape
        ):
            return _current_device
        else:  # Cache miss, need to teardown
            print(
                f"Found new target {target} with mesh shape {mesh_shape}, closing device for {_current_device_target} with {_current_device_mesh_shape}"
            )
            ttrt.runtime.close_mesh_device(_current_device)
            _current_device = None
            _current_device_target = None
            _current_device_mesh_shape = None

    # Open new device for target
    print(f"Opening device for {target} with mesh shape {mesh_shape}")

    mesh_options = ttrt.runtime.MeshDeviceOptions()

    if pytestconfig.getoption("--disable-eth-dispatch"):
        mesh_options.dispatch_core_type = ttrt.runtime.DispatchCoreType.WORKER
    else:
        mesh_options.dispatch_core_type = ttrt.runtime.DispatchCoreType.ETH

    # Start with a small mesh shape that should work for most tests
    # Tests requiring larger meshes will be handled appropriately
    mesh_options.mesh_shape = mesh_shape

    device_runtime_enum = None

    if target == "ttnn":
        device_runtime_enum = ttrt.runtime.DeviceRuntime.TTNN
    elif target == "ttmetal":
        device_runtime_enum = ttrt.runtime.DeviceRuntime.TTMetal
    else:
        raise ValueError(f"Only TTNN and TTMetal devices are supported, got {target}")

    ttrt.runtime.set_current_device_runtime(device_runtime_enum)
    device = ttrt.runtime.open_mesh_device(mesh_options)
    print(
        f"Device opened for test session with target {target} & mesh shape {mesh_options.mesh_shape}."
    )
    _current_device = device
    _current_device_target = target
    _current_device_mesh_shape = mesh_shape
    return _current_device


def is_x86_machine():
    machine = platform.machine().lower()
    return machine in ["x86_64", "amd64", "i386", "i686", "x86"]


x86_only = pytest.mark.skipif(
    not is_x86_machine(),
    reason=f"Test requires x86 architecture, but running on {platform.machine()}",
)


@pytest.fixture(scope="session", autouse=True)
def log_global_env_facts(record_testsuite_property, pytestconfig):
    """Log details about the environment into the XML report

    This autouse fixture logs the following properties:
        - card: from `get_board_id()`, the type of card these tests are running on
        - git_sha: current git commit SHA of the repository
    """
    system_desc = ttrt.binary.fbb_as_dict(
        ttrt.binary.load_system_desc_from_path(pytestconfig.option.sys_desc)
    )["system_desc"]
    record_testsuite_property("card", get_board_id(system_desc))

    # Get current git SHA.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=".",
        )
        git_sha = result.stdout.strip()
        record_testsuite_property("git_sha", git_sha)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If git command fails or git is not available, record as unknown.
        record_testsuite_property("git_sha", "unknown")


@pytest.fixture(scope="function")
def device(request, pytestconfig):
    """Device fixture that is reevaluated for every test to determine if the
    runtime mode needs to be switched from the last test, i.e. the device must
    be reinitialized
    """
    # default target is ttnn elsewhere, if no "target" is supplied it will compile to ttnn
    target = "ttnn"
    mesh_shape = (1, 1)

    if hasattr(request.node, "callspec"):
        target = request.node.callspec.params.get("target", "ttnn")

        # Support for other backends coming soon.
        if target not in ["ttnn", "ttmetal"]:
            return None

        mesh_shape = request.node.callspec.params.get("mesh_shape", (1, 1))
    return _get_device_for_target(target, mesh_shape, pytestconfig)


def pytest_addoption(parser):
    parser.addoption(
        "--path",
        action="store",
        default=".",
        help="Path to store test artifacts (e.g. flatbuffers and .mlir files)",
    )
    parser.addoption(
        "--sys-desc",
        action="store",
        default="ttrt-artifacts/system_desc.ttsys",
        help="Path to system descriptor",
    )
    parser.addoption(
        "--require-exact-mesh",
        action="store_true",
        help="Require exact mesh shape match with the current device (default allows subset)",
    )
    parser.addoption(
        "--require-opmodel",
        action="store_true",
        help="Require tests to run only if build has opmodel enabled",
    )
    parser.addoption(
        "--disable-eth-dispatch",
        action="store_true",
        help="disable putting dispatch on ethernet cores - place it on worker cores instead; necessary on blackhole",
    )


def get_board_id(system_desc) -> str:
    arch = system_desc["chip_descs"][0]["arch"]
    num_chips = len(system_desc["chip_desc_indices"])

    match arch, num_chips:
        case "Blackhole", 1:
            return "p150"
        case "Blackhole", 2:
            return "p300"
        case "Wormhole_b0", 1:
            return "n150"
        case "Wormhole_b0", 2:
            return "n300"
        case "Wormhole_b0", 8:
            return "llmbox"
        case _:
            raise ValueError(f"Unknown architecture/chip# combo: {arch}, {num_chips}")


def filter_valid_mesh_shape(system_desc, params, require_exact_mesh=False):
    num_chips = reduce(operator.mul, params.get("mesh_shape", [1]), 1)
    num_physical_chips = len(system_desc["chip_desc_indices"])
    if require_exact_mesh:
        return num_chips == num_physical_chips
    else:
        return num_chips <= num_physical_chips


def torch_dtype_to_abbrev(dtype):
    """Convert torch dtype to abbreviated string representation"""
    dtype_str = str(dtype)

    # Handle torch.dtype format.
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[6:]  # Remove "torch." prefix.

    # Map common torch dtypes to abbreviations.
    dtype_mapping = {
        "float32": "f32",
        "float16": "f16",
        "bfloat16": "bf16",
        "int32": "i32",
        "int16": "i16",
        "int8": "i8",
        "uint8": "u8",
        "bool": "bool",
    }

    return dtype_mapping.get(dtype_str, dtype_str)


# Utility functions for fault-tolerant metadata extraction.
def _safe_add_property(item: pytest.Item, key: str, value: Any) -> bool:
    """Safely add a property to the test item, returning success status"""
    try:
        if not hasattr(item, "user_properties"):
            return False
        item.user_properties.append((key, str(value)))
        return True
    except Exception:
        return False


def _safe_serialize(value: Any) -> str:
    """Safely serialize a value to string with fallbacks"""
    if isinstance(value, torch.dtype):
        return torch_dtype_to_abbrev(value)

    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        try:
            return repr(value)
        except Exception:
            return f"<serialization_error: {type(value).__name__}>"


def _get_shapes_param(params: Dict[str, Any]) -> Optional[Any]:
    """Get shapes parameter from various possible keys"""
    # TODO(ctod): figure out a better way to detect the input shapes to become
    # robust to tests that construct the shapes within the test itself (#4518)
    shape_keys = ["shapes", "shape", "input_shape", "inputs_shapes"]
    for key in shape_keys:
        if key in params:
            return params[key]
    return None


def _get_dtypes_param(params: Dict[str, Any], num_shapes: int) -> List[Any]:
    """Get dtypes parameter, broadcasting single dtype if needed"""
    dtype_keys = ["dtypes", "dtype", "inputs_dtypes"]
    dtypes_param = torch.float32  # Default.

    for key in dtype_keys:
        if key in params:
            dtypes_param = params[key]
            break

    if not isinstance(dtypes_param, list):
        return [dtypes_param] * num_shapes
    return dtypes_param


def _extract_shapes_and_dtypes(item: pytest.Item, params: Dict[str, Any]) -> None:
    """Extract and record shape and dtype information"""
    shapes_param = _get_shapes_param(params)
    if shapes_param is None:
        return

    if not isinstance(shapes_param, list):
        shapes_param = [shapes_param]

    # Record shapes.
    shapes_success = _safe_add_property(item, "input_shapes", json.dumps(shapes_param))

    if shapes_success:
        # Only extract dtypes if shapes extraction succeeded.
        dtypes_param = _get_dtypes_param(params, len(shapes_param))
        dtypes_list = [torch_dtype_to_abbrev(dtype) for dtype in dtypes_param]
        _safe_add_property(item, "input_dtypes", str(dtypes_list))


def _extract_operation_name(item: pytest.Item, params: Dict[str, Any]) -> None:
    """Extract and record operation name"""
    op_name = None

    # Try test_fn parameter first.
    if "test_fn" in params:
        test_fn = params["test_fn"]
        if hasattr(test_fn, "__name__"):
            op_name = test_fn.__name__

    # Fall back to test function name.
    if not op_name:
        test_name = item.name.split("[")[0]
        if test_name.startswith("test_"):
            op_name = test_name[5:]

    if op_name:
        # Handle hoisted operations.
        if op_name.startswith("hoisted_"):
            op_name = op_name[8:]

        _safe_add_property(item, "op_name", op_name)
        # TODO(ctod): Extract actual framework (torch) operation name in the
        # future, once we have access to it via golden checking. (#4094)
        _safe_add_property(item, "framework_op_name", op_name)


def _extract_backend_and_params(item: pytest.Item, params: Dict[str, Any]) -> None:
    """Extract backend and remaining parameters"""
    # Extract backend. Default to ttnn for now, since that's what
    # `compile_and_execute_ttir` defaults to
    # TODO(ctod): figure out a better way to detect the backend without
    # necessitating a singleton parameter in test cases that will never need to
    # test both ttnn and ttmetal (#4518)
    backend = params.get("target", "ttnn")
    _safe_add_property(item, "backend", backend)

    # Extract remaining parameters.
    covered_params = {
        "shapes",
        "shape",
        "input_shape",
        "inputs_shapes",
        "dtypes",
        "dtype",
        "inputs_dtypes",
        "test_fn",
        "target",
    }

    for key, value in params.items():
        if key not in covered_params:
            value_str = _safe_serialize(value)
            _safe_add_property(item, f"param_{key}", value_str)


def _extract_frontend(item: pytest.Item) -> None:
    """Extracts the type of frontend that this test is starting from (i.e.
    what IR the builder graph represents). Currently possible values are
    `"ttir"` and `"shlo"` This information is encoded as marks applied by file
    to each test, via `pytestmark`"""

    for m in item.iter_markers(name="frontend"):
        _safe_add_property(item, "frontend", m.args[0])
        return

    raise KeyError("No frontend marker found!")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item: pytest.Item):
    """
    Extract test metadata during setup phase for XML reporting.

    Extracts parameter-based metadata from all parametrized tests, including skipped ones.
    This ensures that even tests that don't execute still have their metadata captured.

    Extracted metadata includes:
    - Input tensor shapes and data types (standardized from various parameter names)
    - Operation names (extracted from test_fn parameter or test function name)
    - Backend information (from target parameter, defaults to "ttnn")
    - Ad-hoc operation-specific parameters (any parameters not covered above)

    The metadata is stored as XML properties in two categories:
    1. Standard properties: Direct key-value pairs for metadata common to all tests
       - input_shapes: String representation of tensor shape list
       - input_dtypes: List of abbreviated data type strings (e.g., "f32", "i32")
       - op_name: Name of the operation being tested
       - framework_op_name: Framework-specific operation name (currently same as op_name)
       - backend: Target backend ("ttnn", "ttmetal", or "ttnn-standalone")

    2. Prefixed properties: Operation-specific parameters with "param_" prefix. For `conv2d`, e.g.:
       - param_stride: Convolution stride parameters
       - param_padding: Padding configuration
       - param_dilation: Dilation settings
       - param_groups: Grouping parameters
       - param_*: Any other test-specific parameters

    XML Output Example:
    ===================
    <properties>
        <property name="input_shapes" value="['(1, 32, 32, 64)', '(64, 32, 3, 3)']" />
        <property name="input_dtypes" value="['f32', 'f32']" />
        <property name="op_name" value="conv2d" />
        <property name="framework_op_name" value="conv2d" />
        <property name="backend" value="ttnn" />
        <property name="param_stride" value="[2, 1]" />
        <property name="param_padding" value="[2, 1]" />
        <property name="param_dilation" value="[2, 1]" />
        <property name="param_groups" value="2" />
    </properties>

    Note that the above example excludes the "failure_stage" entry. This is
    handled by the hookwrapper for `pytest_runtest_makereport` below
    """
    yield

    if hasattr(item, "callspec"):
        params = item.callspec.params
        _extract_shapes_and_dtypes(item, params)
        _extract_operation_name(item, params)
        _extract_backend_and_params(item, params)
        _extract_frontend(item)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """
    Extract runtime information from tests that actually execute.

    This includes failure classification and error reporting during the call phase.

    Runtime report data includes:
    - failure_stage: Categorizes where in the compilation pipeline the test failed
      - "success": Test passed completely
      - "compile": Failed during TTBuilder compilation (`TTBuilderCompileException`)
      - "runtime": Failed during execution (`TTBuilderRuntimeException`)
      - "golden": Failed golden result verification (`TTBuilderGoldenException`)
    The "runtime" and "golden" are currently unused, but are needed for
    downstream schemas to support future features once pytest itself
    orchestrates the running of the generated flatbuffers
    """

    TTBUILDER_EXCEPTIONS = {
        "TTBuilderCompileException": "compile",
        "TTBuilderRuntimeException": "runtime",
        "TTBuilderGoldenException": "golden",
    }

    failure_stage = "success"  # Default to success.

    outcome = yield
    try:
        outcome.get_result()
    except Exception as exc:
        exc_type = type(exc)
        exc_name = exc_type.__name__
        if exc_name not in TTBUILDER_EXCEPTIONS.keys():
            pytest.fail(
                f"Unknown failure detected! Please address this or correctly throw a `TTBuilder*` exception instead if this is a compilation issue, runtime error, or golden mismatch. Exception: {exc}:{type(exc)}"
            )
        failure_stage = TTBUILDER_EXCEPTIONS[exc_name]
    finally:
        _safe_add_property(item, "failure_stage", failure_stage)


def pytest_collection_modifyitems(config, items):
    valid_items = []
    deselected = []
    system_desc = ttrt.binary.fbb_as_dict(
        ttrt.binary.load_system_desc_from_path(config.option.sys_desc)
    )["system_desc"]

    skip_opmodel = pytest.mark.skip(reason="Test requires --require-opmodel flag")
    require_opmodel = config.getoption("--require-opmodel")

    for item in items:
        # Skip optimizer tests if opmodel flag is missing
        if not require_opmodel and "optimizer" in str(item.fspath):
            item.add_marker(skip_opmodel)

        # Only check parameterized tests
        if hasattr(item, "callspec"):
            params = item.callspec.params
            if not filter_valid_mesh_shape(
                system_desc, params, require_exact_mesh=config.option.require_exact_mesh
            ):
                # Deselect the test case
                deselected.append(item)
                continue
        valid_items.append(item)

        # Skip specific target / system combinations

        # Fetch the current target of this test, if any
        current_target = None
        for param in item.callspec.params.items():
            if param[0] == "target":
                current_target = param[1]
                break

        for marker in item.iter_markers(name="skip_config"):
            for platform_config in marker.args:

                # All of the operations we need to do on these are set membership based
                platform_config = set(platform_config)

                reason = marker.kwargs.get("reason", "")

                # Verify this is a valid configuration
                if not platform_config <= ALL_BACKENDS.union(ALL_SYSTEMS):
                    outliers = platform_config - ALL_BACKENDS.union(ALL_SYSTEMS)
                    raise ValueError(
                        f"Invalid skip config: {platform_config}, invalid entries: {outliers}. Please ensure that all entries in the config are members of {ALL_SYSTEMS} or {ALL_BACKENDS}"
                    )

                board_id = get_board_id(system_desc)

                if platform_config <= set([current_target, board_id]):
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Operation not supported on following platform/target combination: {platform_config}. {reason}"
                        )
                    )

    # Update the items list (collected tests)
    items[:] = valid_items

    # Sort tests alphabetically by their target and then nodeid to ensure consistent ordering.
    items.sort(key=lambda x: (x.callspec.params.get("target", "ttnn"), x.nodeid))
    items.reverse()

    # Report deselected items to pytest
    if deselected:
        config.hook.pytest_deselected(items=deselected)


def pytest_sessionfinish(session):
    global _current_device, _current_device_target, _current_device_mesh_shape
    if _current_device is not None:
        print("Closing device for end of session")
        ttrt.runtime.close_mesh_device(_current_device)
        _current_device = None
        _current_device_target = None
        _current_device_mesh_shape = None
