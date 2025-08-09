# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import ttrt
import platform
from functools import reduce
import operator
import torch
import subprocess

ALL_BACKENDS = set(["ttnn", "ttmetal", "ttnn-standalone"])
ALL_SYSTEMS = set(["n150", "n300", "llmbox", "tg", "p150", "p300"])


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
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # If git command fails or git is not available, record as unknown.
        record_testsuite_property("git_sha", "unknown")


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

    # Handle torch.dtype format
    if dtype_str.startswith("torch."):
        dtype_str = dtype_str[6:]  # Remove "torch." prefix

    # Map common torch dtypes to abbreviations
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


def pytest_runtest_makereport(item, call):
    """
    Extract test metadata and runtime information for XML reporting.

    This pytest hook runs during multiple phases of test execution to collect
    comprehensive metadata that gets embedded in the junit XML report. The
    purpose of the expressiveness here is to be ingested by some sort of parser
    after CI runs of these tests for easy triaging and organization. The hook
    operates in two distinct phases:

    SETUP PHASE (`call.when == "setup"`):
    =====================================
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

    CALL PHASE (`call.when == "call"`):
    ==================================
    Extracts runtime information from tests that actually execute. This includes
    failure classification and error reporting.

    Runtime report data includes:
    - failure_stage: Categorizes where in the compilation pipeline the test failed
      - "success": Test passed completely
      - "compile": Failed during TTIR compilation (TTIRCompileException)
      - "runtime": Failed during execution (TTIRRuntimeException)
      - "golden": Failed golden result verification (TTIRGoldenException)
    The "runtime" and "golden" are currently unused, but are needed for
    downstream schemas to support future features once pytest itself
    orchestrates the running of the generated flatbuffers


    NOTES:
        - Currently, the `framework_op_name` field is directly copied from op
          name. This is not always correct, but is a best guess for the time
          being


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
        <property name="failure_stage" value="success" />
    </properties>
    """

    # SETUP PHASE: Extract parameter information (for all tests including skipped ones).
    if call.when == "setup" and hasattr(item, "callspec"):
        params = item.callspec.params

        # Extract shapes information.
        shapes_param = None
        if "shapes" in params:
            shapes_param = params["shapes"]
        elif "shape" in params:
            shapes_param = params["shape"]
        elif "input_shape" in params:
            shapes_param = params["input_shape"]
        elif "inputs_shapes" in params:
            shapes_param = params["inputs_shapes"]

        if shapes_param is not None:
            if not isinstance(shapes_param, list):
                shapes_param = [shapes_param]
            # Format shapes as strings for XML.
            item.user_properties.append(("input_shapes", str(shapes_param)))

            # Extract dtypes information.
            # This needs to happen iff the shapes work, since we need to know
            # `len(shapes_param)` to properly broadcast types to a list of that
            # length in the case where only one type is provided.
            dtypes_param = torch.float32  # default to float32.
            if "dtypes" in params:
                dtypes_param = params["dtypes"]
            elif "dtype" in params:
                dtypes_param = params["dtype"]
            elif "inputs_dtypes" in params:
                dtypes_param = params["inputs_dtypes"]

            if not isinstance(dtypes_param, list):
                # Handle single dtype, broadcast it across all inputs if only one is supplied as it is in certain tests.
                dtypes_param = [dtypes_param] * len(shapes_param)
            dtypes_str = [torch_dtype_to_abbrev(dtype) for dtype in dtypes_param]
            item.user_properties.append(("input_dtypes", str(dtypes_str)))

        # Extract operation name from various sources.
        op_name = None

        # First try to get from `test_fn` parameter (for parametrized op tests).
        if "test_fn" in params:
            test_fn = params["test_fn"]
            if hasattr(test_fn, "__name__"):
                op_name = test_fn.__name__

        # If no `test_fn`, try to extract from test function name.
        if not op_name:
            test_name = item.name.split("[")[0]  # Remove parameter part.
            if test_name.startswith("test_"):
                op_name = test_name[5:]  # Remove "test_" prefix.

        if op_name:

            # Handle hoisted operations.
            if op_name.startswith("hoisted_"):
                op_name = op_name[8:]  # Remove "hoisted_" prefix.

            item.user_properties.append(("op_name", op_name))
            # For now, use the same op_name as framework_op_name.

            # TODO(ctod): Extract actual framework (torch) operation name in the
            # future, once we have access to it via golden checking.
            item.user_properties.append(("framework_op_name", op_name))

        # Extract backend from target parameter, default to "ttnn" if not present.
        backend = params.get("target", "ttnn")
        item.user_properties.append(("backend", backend))

        # Add remaining parameters (excluding those already covered) as prefixed properties.
        if params:
            # Parameters already covered by setup-time logging.
            covered_params = {
                "shapes",
                "shape",
                "input_shape",
                "inputs_shapes",  # shape parameters
                "dtypes",
                "dtype",
                "inputs_dtypes",  # dtype parameters
                "test_fn",  # operation name extraction
                "target",  # backend extraction
            }

            # Add uncovered parameters as individual properties with "param_" prefix.
            for key, value in params.items():
                if key not in covered_params:
                    try:
                        value_str = str(value)
                        item.user_properties.append((f"param_{key}", value_str))
                    except:
                        value_str = repr(value)
                        item.user_properties.append((f"param_{key}", value_str))

    # CALL PHASE: Extract runtime information (failure stage, error messages, etc.).
    if call.when == "call":
        # Determine failure stage based on test outcomes and exceptions from `compile_ttir_to_flatbuffer`.
        failure_stage = "success"  # Default to success.

        if hasattr(call, "excinfo") and call.excinfo is not None:
            # Test failed, determine failure stage from exception type only.
            exc_type = call.excinfo.type

            # TODO(ctod): Capture stderr from test execution (to be implemented later).

            # Check for specific TTIR exception types.
            if exc_type and exc_type.__name__ == "TTIRCompileException":
                failure_stage = "compile"
            elif exc_type and exc_type.__name__ == "TTIRRuntimeException":
                failure_stage = "runtime"
            elif exc_type and exc_type.__name__ == "TTIRGoldenException":
                failure_stage = "golden"
            # If no specific TTIR exception, leave as "success" default.

        item.user_properties.append(("failure_stage", failure_stage))


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

    # Sort tests alphabetically by their nodeid to ensure consistent ordering.
    items.sort(key=lambda x: x.nodeid)

    # Report deselected items to pytest
    if deselected:
        config.hook.pytest_deselected(items=deselected)
