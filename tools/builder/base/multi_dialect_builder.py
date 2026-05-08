# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Dialect Builder for tt-mlir.

Supports creating modules with operations from multiple dialects (TTIR, TTNN,
StableHLO, D2M) in a single module. Uses a delegation pattern with shared state
to avoid synchronization overhead.

Supports both API styles:
- Explicit: builder.ttir.sigmoid(x)  (clear, like Option 1)
- Implicit: builder.sigmoid(x)       (convenient, automatic delegation)
"""

from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from collections import OrderedDict
import torch

from ttmlir.ir import *

from builder.base.builder import Builder
from builder.ttir.ttir_builder import TTIRBuilder
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.stablehlo.stablehlo_builder import StableHLOBuilder
from builder.d2m.d2m_builder import D2MBuilder


class MultiDialectBuilder(Builder):
    """
    A builder supporting operations from multiple dialects in a single module.

    This implementation uses a delegation pattern where all dialect builders
    share the same state (context, goldens, function tracking, etc.) by sharing
    the same __dict__. This ensures:
    - No synchronization overhead
    - Cross-dialect operations work correctly
    - Golden tensors are tracked across all dialects
    - All state is automatically consistent

    Supports both explicit and implicit API styles:

    Examples
    --------
    Explicit API (maximum clarity):
        >>> builder = MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn"])
        >>> @builder.func([(32, 32)], [torch.float32])
        ... def forward(input, builder):
        ...     x = builder.ttir.sigmoid(input)   # Explicitly TTIR
        ...     y = builder.ttnn.add(x, input)    # Explicitly TTNN
        ...     return y

    Implicit API (convenience):
        >>> builder = MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn"])
        >>> @builder.func([(32, 32)], [torch.float32])
        ... def forward(input, builder):
        ...     x = builder.sigmoid(input)        # Automatically finds ttir.sigmoid
        ...     y = builder.add(x, input)         # Automatically resolved
        ...     return y

    Mixed API (pragmatic):
        >>> builder = MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn"])
        >>> @builder.func([(32, 32)], [torch.float32])
        ... def forward(input, builder):
        ...     x = builder.ttir.sigmoid(input)   # Explicit when clarity matters
        ...     y = builder.relu(x)               # Implicit when obvious
        ...     z = builder.ttnn.specialized(y)   # Explicit for special ops
        ...     return z
    """

    def __init__(
        self,
        ctx: Context,
        location: Location,
        dialects: List[str] = ["ttir", "ttnn"],
        mesh_name: Union[List[str], str] = "mesh",
        mesh_dict: Union[
            List[OrderedDict[str, int]], OrderedDict[str, int]
        ] = OrderedDict([("x", 1), ("y", 1)]),
        deallocate_goldens: bool = False,
        deallocated_goldens_dir: Optional[str] = "./deallocated_goldens",
    ):
        """
        Initialize MultiDialectBuilder with specified dialects.

        Parameters
        ----------
        ctx : Context
            MLIR context for all operations.
        location : Location
            Default location for operations.
        dialects : List[str]
            List of dialects to enable. Supported: ["ttir", "ttnn", "stablehlo", "d2m"]
            Default: ["ttir", "ttnn"]
        mesh_name : Union[List[str], str]
            Mesh name(s) for distributed operations.
        mesh_dict : Union[List[OrderedDict[str, int]], OrderedDict[str, int]]
            Mesh shape specification. Example: OrderedDict([("x", 1), ("y", 1)])
        deallocate_goldens : bool
            Whether to deallocate golden tensors to disk to save memory.
        deallocated_goldens_dir : Optional[str]
            Directory for storing deallocated golden tensors.

        Raises
        ------
        ValueError
            If an unknown dialect is specified.
        """
        # Initialize base Builder with shared state
        super().__init__(
            ctx,
            location,
            mesh_name,
            mesh_dict,
            deallocate_goldens=deallocate_goldens,
            deallocated_goldens_dir=deallocated_goldens_dir,
        )

        # Map of dialect names to builder classes
        self._dialect_map = {
            "ttir": TTIRBuilder,
            "ttnn": TTNNBuilder,
            "stablehlo": StableHLOBuilder,
            "d2m": D2MBuilder,
        }

        # Create dialect builders that share this instance's state
        self._dialect_builders: Dict[str, Builder] = {}

        # Initialize each requested dialect builder
        for dialect in dialects:
            if dialect not in self._dialect_map:
                raise ValueError(
                    f"Unknown dialect '{dialect}'. "
                    f"Available dialects: {list(self._dialect_map.keys())}"
                )

            builder_cls = self._dialect_map[dialect]
            # Create instance without calling __init__ to avoid duplicate initialization
            builder = builder_cls.__new__(builder_cls)
            # KEY: Share all state by pointing to the same __dict__
            # This ensures zero synchronization overhead and correct cross-dialect operations
            builder.__dict__ = self.__dict__
            self._dialect_builders[dialect] = builder

        # Set up create_tensor_encoding from available dialect builders
        # Priority: TTIR (returns None) > TTNN (has implementation) > StableHLO > D2M
        if "ttir" in self._dialect_builders:
            # TTIR returns None for tensor encoding
            self.create_tensor_encoding = lambda shape, dtype: None
        elif "ttnn" in self._dialect_builders:
            # TTNN has a specific implementation
            ttnn_builder = self._dialect_builders["ttnn"]
            if hasattr(ttnn_builder, "_create_tensor_encoding"):
                self.create_tensor_encoding = ttnn_builder._create_tensor_encoding
        elif "stablehlo" in self._dialect_builders:
            # StableHLO returns None
            self.create_tensor_encoding = lambda shape, dtype: None
        elif "d2m" in self._dialect_builders:
            # D2M has specific encoding
            d2m_builder = self._dialect_builders["d2m"]
            if hasattr(d2m_builder.__class__, "create_tensor_encoding"):
                self.create_tensor_encoding = d2m_builder.create_tensor_encoding
        else:
            # Fallback
            self.create_tensor_encoding = lambda shape, dtype: None

        # Keep track of which methods belong to which dialect for debugging
        self._method_to_dialect: Dict[str, str] = {}
        for dialect, builder in self._dialect_builders.items():
            for attr_name in dir(builder):
                if not attr_name.startswith("_") and callable(
                    getattr(builder, attr_name)
                ):
                    # Track which dialect provides this method (first one wins)
                    if attr_name not in self._method_to_dialect:
                        self._method_to_dialect[attr_name] = dialect

    # ----- Explicit Dialect Accessors (Option 1 API style) -----

    @property
    def ttir(self) -> TTIRBuilder:
        """
        Access TTIR dialect builder explicitly.

        Returns
        -------
        TTIRBuilder
            The TTIR builder instance with shared state.

        Raises
        ------
        AttributeError
            If TTIR dialect is not enabled.

        Examples
        --------
        >>> x = builder.ttir.sigmoid(input)
        >>> y = builder.ttir.relu(x)
        """
        if "ttir" not in self._dialect_builders:
            raise AttributeError(
                "TTIR dialect not enabled. " "Pass dialects=['ttir', ...] to __init__."
            )
        return self._dialect_builders["ttir"]

    @property
    def ttnn(self) -> TTNNBuilder:
        """
        Access TTNN dialect builder explicitly.

        Returns
        -------
        TTNNBuilder
            The TTNN builder instance with shared state.

        Raises
        ------
        AttributeError
            If TTNN dialect is not enabled.

        Examples
        --------
        >>> x = builder.ttnn.add(a, b)
        >>> y = builder.ttnn.multiply(x, c)
        """
        if "ttnn" not in self._dialect_builders:
            raise AttributeError(
                "TTNN dialect not enabled. " "Pass dialects=['ttnn', ...] to __init__."
            )
        return self._dialect_builders["ttnn"]

    @property
    def stablehlo(self) -> StableHLOBuilder:
        """
        Access StableHLO dialect builder explicitly.

        Returns
        -------
        StableHLOBuilder
            The StableHLO builder instance with shared state.

        Raises
        ------
        AttributeError
            If StableHLO dialect is not enabled.

        Examples
        --------
        >>> x = builder.stablehlo.abs(input)
        >>> y = builder.stablehlo.add(x, input)
        """
        if "stablehlo" not in self._dialect_builders:
            raise AttributeError(
                "StableHLO dialect not enabled. "
                "Pass dialects=['stablehlo', ...] to __init__."
            )
        return self._dialect_builders["stablehlo"]

    @property
    def d2m(self) -> D2MBuilder:
        """
        Access D2M dialect builder explicitly.

        Returns
        -------
        D2MBuilder
            The D2M builder instance with shared state.

        Raises
        ------
        AttributeError
            If D2M dialect is not enabled.

        Examples
        --------
        >>> x = builder.d2m.some_op(input)
        """
        if "d2m" not in self._dialect_builders:
            raise AttributeError(
                "D2M dialect not enabled. " "Pass dialects=['d2m', ...] to __init__."
            )
        return self._dialect_builders["d2m"]

    # ----- Implicit Delegation (Option 6 convenience) -----

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to appropriate dialect builder (implicit API).

        This method is called when an attribute is not found on the instance.
        It searches through all enabled dialect builders to find the requested method.

        Note: This provides automatic delegation for convenience, but you can
        also use explicit dialect access (builder.ttir.op(), builder.ttnn.op())
        for better API clarity.

        Parameters
        ----------
        name : str
            The attribute/method name to find.

        Returns
        -------
        Any
            The method from the dialect builder that provides it.

        Raises
        ------
        AttributeError
            If no dialect builder has the requested attribute.

        Examples
        --------
        >>> # Implicit delegation - automatically finds the right dialect
        >>> x = builder.sigmoid(input)  # Finds ttir.sigmoid
        >>> y = builder.add(x, input)   # Finds add from available dialects
        """
        # Search through all dialect builders
        for dialect, builder in self._dialect_builders.items():
            if hasattr(builder, name):
                attr = getattr(builder, name)
                return attr

        # If not found in any dialect builder, raise AttributeError
        raise AttributeError(
            f"MultiDialectBuilder has no attribute '{name}'. "
            f"Available dialects: {list(self._dialect_builders.keys())}. "
            f"Use get_method_dialect('{name}') to check which dialect provides this method."
        )

    # ----- Helper Methods -----

    def get_method_dialect(self, method_name: str) -> Optional[str]:
        """
        Get which dialect provides a specific method.

        Useful for debugging and understanding method resolution.

        Parameters
        ----------
        method_name : str
            The method name to look up.

        Returns
        -------
        Optional[str]
            The dialect name that provides this method, or None if not found.

        Examples
        --------
        >>> dialect = builder.get_method_dialect('sigmoid')
        >>> print(f"sigmoid comes from: {dialect}")  # Output: "ttir"
        """
        return self._method_to_dialect.get(method_name)

    def list_methods_by_dialect(self) -> Dict[str, List[str]]:
        """
        Return a dictionary mapping dialects to their available methods.

        Useful for debugging and documentation.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping dialect names to sorted lists of method names.

        Examples
        --------
        >>> methods = builder.list_methods_by_dialect()
        >>> for dialect, method_list in methods.items():
        ...     print(f"{dialect}: {len(method_list)} methods")
        """
        result = {}
        for dialect, builder in self._dialect_builders.items():
            methods = [
                name
                for name in dir(builder)
                if not name.startswith("_") and callable(getattr(builder, name))
            ]
            result[dialect] = sorted(methods)
        return result

    def list_enabled_dialects(self) -> List[str]:
        """
        Return list of enabled dialects.

        Returns
        -------
        List[str]
            List of dialect names that are enabled.

        Examples
        --------
        >>> dialects = builder.list_enabled_dialects()
        >>> print(f"Enabled: {dialects}")  # Output: ['ttir', 'ttnn']
        """
        return list(self._dialect_builders.keys())
