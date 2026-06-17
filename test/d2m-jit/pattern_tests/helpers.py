# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Helper functions for creating pattern test configurations.

These helpers reduce boilerplate and make pattern test definitions more
concise while maintaining type safety.
"""

from typing import Callable, Dict, List, Any, Optional
from .config import LitTestCase, E2ETestCase, LayoutConfig, PatternTestConfig


def lit_test(
    name: str,
    module: str,
    checks: List[str],
    description: str = "",
) -> LitTestCase:
    """Create a LIT test case with less boilerplate.

    Args:
        name: Test case name
        module: MLIR module text
        checks: List of CHECK patterns (without "CHECK:" prefix if omitted)
        description: Optional test description

    Returns:
        Configured LitTestCase

    Example:
        >>> lit_test(
        ...     "exp_positive",
        ...     module="module { ... }",
        ...     checks=["func.func @forward", "NOT: ttir.exp", "d2m.generic"],
        ... )
    """
    # Normalize check patterns - add "CHECK:" if not present
    normalized_checks = []
    for check in checks:
        if not check.startswith("CHECK"):
            # Detect CHECK-NOT, CHECK-LABEL patterns
            if check.startswith("NOT:"):
                normalized_checks.append(f"CHECK-NOT:{check[4:]}")
            elif check.startswith("LABEL:"):
                normalized_checks.append(f"CHECK-LABEL:{check[6:]}")
            else:
                normalized_checks.append(f"CHECK: {check}")
        else:
            normalized_checks.append(check)

    return LitTestCase(
        name=name,
        module_text=module,
        file_checks=normalized_checks,
        description=description,
    )


def simple_layout(
    *shape: int,
    dtype: Any = None,
    block_shape: Optional[List[int]] = None,
    grid_shape: Optional[List[int]] = None,
    tiled: bool = True,
) -> LayoutConfig:
    """Create a simple layout config with sensible defaults.

    Args:
        *shape: Shape dimensions (e.g., 32, 32 or 128, 128)
        dtype: d2m dtype (defaults to d2m.float32)
        block_shape: Block shape (defaults to [1, 1])
        grid_shape: Grid shape (defaults to [1, 1])
        tiled: Whether to use tiled layout

    Returns:
        Configured LayoutConfig

    Example:
        >>> simple_layout(32, 32)
        >>> simple_layout(128, 128, block_shape=[2, 2], grid_shape=[2, 2])
    """
    # Import here to avoid circular dependency
    try:
        import d2m_jit as d2m

        default_dtype = d2m.float32
    except ImportError:
        default_dtype = None

    return LayoutConfig(
        shape=tuple(shape),
        dtype=dtype or default_dtype,
        block_shape=block_shape or [1, 1],
        grid_shape=grid_shape or [1, 1],
        tiled=tiled,
    )


def default_kernel_args(
    m_blocks: int = 1,
    n_blocks: int = 1,
    grid: tuple = (1, 1),
    **extra_args,
) -> Dict[str, Any]:
    """Create default kernel arguments.

    Args:
        m_blocks: Number of blocks in M dimension
        n_blocks: Number of blocks in N dimension
        grid: Grid shape tuple
        **extra_args: Additional kernel arguments

    Returns:
        Dictionary of kernel arguments

    Example:
        >>> default_kernel_args()
        >>> default_kernel_args(m_blocks=2, grid=(2, 2))
    """
    args = {
        "m_blocks": m_blocks,
        "n_blocks": n_blocks,
        "grid": grid,
    }
    args.update(extra_args)
    return args


def e2e_test(
    name: str,
    kernel: Callable,
    inputs: Dict[str, Callable[[], Any]],
    reference: Callable,
    layout: LayoutConfig,
    kernel_args: Optional[Dict[str, Any]] = None,
    description: str = "",
    pcc_threshold: float = 0.99,
) -> E2ETestCase:
    """Create an E2E test case with less boilerplate.

    Args:
        name: Test name (will be prefixed with "test_" if not already)
        kernel: The @d2m.kernel function to test
        inputs: Dict of input names to generator lambdas
        reference: Function computing expected output
        layout: Layout configuration
        kernel_args: Kernel arguments (defaults to default_kernel_args())
        description: Optional test description
        pcc_threshold: PCC threshold for assertion

    Returns:
        Configured E2ETestCase

    Example:
        >>> e2e_test(
        ...     "exp_basic",
        ...     kernel=exp_fused,
        ...     inputs={"x": lambda: torch.rand(32, 32)},
        ...     reference=lambda x: torch.exp(x),
        ...     layout=simple_layout(32, 32),
        ... )
    """
    # Ensure name starts with "test_"
    if not name.startswith("test_"):
        name = f"test_{name}"

    # Create input generator that calls all input lambdas
    def input_generator():
        return {k: v() for k, v in inputs.items()}

    return E2ETestCase(
        name=name,
        kernel_fn=kernel,
        input_generator=input_generator,
        reference_fn=reference,
        layout_config=layout,
        kernel_args=kernel_args or default_kernel_args(),
        description=description,
        pcc_threshold=pcc_threshold,
    )


def single_input_e2e(
    name: str,
    kernel: Callable,
    input_shape: tuple,
    reference: Callable,
    dtype: Any = None,
    description: str = "",
    **layout_kwargs,
) -> E2ETestCase:
    """Create a single-input E2E test (common case).

    Args:
        name: Test name
        kernel: The @d2m.kernel function
        input_shape: Shape of the single input tensor
        reference: Function computing expected output (takes single tensor)
        dtype: d2m dtype
        description: Test description
        **layout_kwargs: Additional layout configuration

    Returns:
        Configured E2ETestCase

    Example:
        >>> single_input_e2e(
        ...     "exp_32x32",
        ...     kernel=exp_fused,
        ...     input_shape=(32, 32),
        ...     reference=lambda x: torch.exp(x),
        ... )
    """
    import torch

    try:
        import d2m_jit as d2m

        default_dtype = d2m.float32
    except ImportError:
        default_dtype = None

    layout = LayoutConfig(
        shape=input_shape,
        dtype=dtype or default_dtype,
        block_shape=layout_kwargs.pop("block_shape", [1, 1]),
        grid_shape=layout_kwargs.pop("grid_shape", [1, 1]),
        tiled=layout_kwargs.pop("tiled", True),
    )

    return E2ETestCase(
        name=name if name.startswith("test_") else f"test_{name}",
        kernel_fn=kernel,
        input_generator=lambda: {"x": torch.rand(*input_shape, dtype=torch.float32)},
        reference_fn=lambda x: reference(x),
        layout_config=layout,
        kernel_args=default_kernel_args(),
        description=description,
    )


def multi_input_e2e(
    name: str,
    kernel: Callable,
    input_shapes: Dict[str, tuple],
    reference: Callable,
    dtype: Any = None,
    description: str = "",
    **layout_kwargs,
) -> E2ETestCase:
    """Create a multi-input E2E test.

    Args:
        name: Test name
        kernel: The @d2m.kernel function
        input_shapes: Dict of input names to shapes
        reference: Function computing expected output (takes **inputs)
        dtype: d2m dtype
        description: Test description
        **layout_kwargs: Additional layout configuration

    Returns:
        Configured E2ETestCase

    Example:
        >>> multi_input_e2e(
        ...     "add_exp_32x32",
        ...     kernel=add_exp_fused,
        ...     input_shapes={"a": (32, 32), "b": (32, 32)},
        ...     reference=lambda a, b: torch.exp(a + b),
        ... )
    """
    import torch

    try:
        import d2m_jit as d2m

        default_dtype = d2m.float32
    except ImportError:
        default_dtype = None

    # Use first input shape for layout (assume all same shape)
    first_shape = next(iter(input_shapes.values()))

    layout = LayoutConfig(
        shape=first_shape,
        dtype=dtype or default_dtype,
        block_shape=layout_kwargs.pop("block_shape", [1, 1]),
        grid_shape=layout_kwargs.pop("grid_shape", [1, 1]),
        tiled=layout_kwargs.pop("tiled", True),
    )

    def input_generator():
        return {
            name: torch.rand(*shape, dtype=torch.float32)
            for name, shape in input_shapes.items()
        }

    return E2ETestCase(
        name=name if name.startswith("test_") else f"test_{name}",
        kernel_fn=kernel,
        input_generator=input_generator,
        reference_fn=reference,
        layout_config=layout,
        kernel_args=default_kernel_args(),
        description=description,
    )
