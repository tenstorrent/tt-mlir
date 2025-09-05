# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime Golden Validation Handler for TT-MLIR

This script provides golden reference validation capabilities that can be
embedded and executed during program execution in the TTNN runtime.
"""

import os
import sys
import json
from typing import Dict, Any, Optional, Callable
import logging

# Initialize availability flags
TORCH_AVAILABLE = False
GOLDEN_LIBRARY_AVAILABLE = False
TTRT_RUNTIME_AVAILABLE = False

# Import torch and golden functions library
try:
    import torch

    # Import golden functions from packaged ttmlir module
    from ttmlir.golden.golden_funcs import GOLDEN_MAPPINGS, get_golden_function
    from ttmlir.dialects import ttir, stablehlo

    # Import ttrt runtime for tensor extraction
    try:
        import ttrt.runtime

        TTRT_RUNTIME_AVAILABLE = True
    except ImportError:
        TTRT_RUNTIME_AVAILABLE = False
        logging.warning("ttrt.runtime not available - skipping golden validation")

    TORCH_AVAILABLE = True
    GOLDEN_LIBRARY_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    GOLDEN_LIBRARY_AVAILABLE = False
    logging.warning(
        f"Required libraries not available - golden calculations will be disabled: {e}"
    )

# Configure logging - cleaner format for runtime golden validation
logging.basicConfig(level=logging.INFO, format="[GOLDEN] %(message)s")
logger = logging.getLogger(__name__)

# Reduce noise from other loggers
logging.getLogger("ttrt.runtime").setLevel(logging.WARNING)
logging.getLogger("ttrt.common").setLevel(logging.WARNING)


class RuntimeGoldenValidator:
    """
    Runtime Golden Validator for TT-MLIR operations.

    This class provides golden reference validation capabilities that can be
    called from the embedded C++ runtime to validate operation accuracy.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the runtime golden validator.

        Args:
            config_path: Optional path to configuration file
        """
        self.config = self._load_config(config_path)
        self.execution_stats = {
            "total_operations": 0,
            "operations_processed": 0,
            "start_time": None,
            "end_time": None,
            "errors": [],
            "golden_results": {},
        }
        logger.info("RuntimeGoldenValidator initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        # Default configuration
        return {
            "enable_logging": True,
            "log_level": "INFO",
            "enable_stats_collection": True,
            "enable_error_reporting": True,
            "output_dir": "/tmp/ttmlir_runtime",
        }

    def on_program_start(self, program_name: str, program_context: Any) -> None:
        """
        Called when program execution starts.

        Args:
            program_name: Name of the program being executed
            program_context: Runtime program context
        """
        if self.config["enable_logging"]:
            logger.info(f"Program execution started: {program_name}")

        if self.config["enable_stats_collection"]:
            import time

            self.execution_stats["start_time"] = time.time()
            self.execution_stats["program_name"] = program_name

    def on_program_end(self, program_name: str, program_context: Any) -> None:
        """
        Called when program execution ends.

        Args:
            program_name: Name of the program being executed
            program_context: Runtime program context
        """
        if self.config["enable_stats_collection"]:
            import time

            self.execution_stats["end_time"] = time.time()
            duration = (
                self.execution_stats["end_time"] - self.execution_stats["start_time"]
            )

            if self.config["enable_logging"]:
                logger.info(f"Program execution completed: {program_name}")
                logger.info(f"Execution duration: {duration:.3f} seconds")
                logger.info(
                    f"Operations processed: {self.execution_stats['operations_processed']}"
                )

        # Save stats if configured
        if self.config.get("save_stats", False):
            self._save_execution_stats()

    def on_operation_complete(
        self, op_name: str, op_context: Any, program_context: Any
    ) -> None:
        """
        Called after each operation completes.

        Args:
            op_name: Name of the operation
            op_context: Operation context
            program_context: Program context
        """
        self.execution_stats["operations_processed"] += 1

        if (
            self.config["enable_logging"]
            and self.execution_stats["operations_processed"] % 100 == 0
        ):
            logger.info(
                f"Processed {self.execution_stats['operations_processed']} operations"
            )

        # Custom operation analysis can be added here
        # For example: performance monitoring, tensor validation, etc.

    def handle_error(self, error: Exception, op_context: Any = None) -> None:
        """
        Handle runtime errors.

        Args:
            error: The exception that occurred
            op_context: Optional operation context where error occurred
        """
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": getattr(op_context, "debug_info", "unknown")
            if op_context
            else "unknown",
            "timestamp": None,
        }

        if self.config["enable_stats_collection"]:
            import time

            error_info["timestamp"] = time.time()
            self.execution_stats["errors"].append(error_info)

        if self.config["enable_logging"]:
            logger.error(f"Runtime error: {error_info['error_message']}")

    def _save_execution_stats(self) -> None:
        """Save execution statistics to file."""
        output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        stats_file = os.path.join(output_dir, "execution_stats.json")
        try:
            with open(stats_file, "w") as f:
                json.dump(self.execution_stats, f, indent=2, default=str)
            logger.info(f"Execution stats saved to {stats_file}")
        except Exception as e:
            logger.error(f"Failed to save execution stats: {e}")

    def perform_golden_calculation(
        self,
        op_name: str,
        input_tensors: list,
        output_tensor: Any,
        op_context: Any = None,
    ) -> Dict[str, Any]:
        """
        Perform golden reference calculation for an operation using the golden library.

        Args:
            op_name: Name of the operation
            input_tensors: List of input tensors
            output_tensor: Output tensor from the operation
            op_context: Operation context (if available)

        Returns:
            Dictionary containing golden calculation results
        """
        if not TORCH_AVAILABLE or not GOLDEN_LIBRARY_AVAILABLE:
            logger.warning("Golden library not available - skipping golden calculation")
            return {"error": "Golden library not available"}

        try:
            # Try to map operation name to golden function
            golden_func = self._get_golden_function_for_op(op_name)
            if golden_func is None:
                logger.debug(f"No golden function found for operation: {op_name}")
                return {"error": f"No golden function for {op_name}"}

            # Convert input tensors to torch tensors
            torch_inputs = self._convert_to_torch_tensors(input_tensors)

            # Perform golden calculation using library function
            logger.debug(
                f"Calling golden function for {op_name} with {len(torch_inputs)} inputs"
            )
            golden_result = golden_func(*torch_inputs)

            # Convert output tensor to torch for comparison
            actual_output = self._convert_to_torch_tensor(output_tensor)

            # Compare golden vs actual
            return self._compare_tensors(golden_result, actual_output)

        except Exception as e:
            logger.error(f"Error in golden calculation for {op_name}: {e}")
            return {"error": str(e)}

    def _get_golden_function_for_op(self, op_name: str) -> Optional[Callable]:
        """
        Map operation name to golden function from the library.

        Args:
            op_name: Name of the operation

        Returns:
            Golden function or None if not found
        """
        try:
            # Map common operation names to their golden functions
            op_name_lower = op_name.lower()

            # For TTNN operations that start with "ttnn."
            if op_name_lower.startswith("ttnn."):
                op_base_name = op_name_lower[5:]  # Remove "ttnn." prefix

                # Map to appropriate TTIR operations
                if "add" in op_base_name:
                    return get_golden_function(ttir.AddOp)
                elif "multiply" in op_base_name or "mul" in op_base_name:
                    return get_golden_function(ttir.MulOp)
                elif "subtract" in op_base_name or "sub" in op_base_name:
                    return get_golden_function(ttir.SubOp)
                elif "divide" in op_base_name or "div" in op_base_name:
                    return get_golden_function(ttir.DivOp)
                elif "matmul" in op_base_name:
                    return get_golden_function(ttir.MatmulOp)
                elif "softmax" in op_base_name:
                    return get_golden_function(ttir.SoftmaxOp)
                elif "relu" in op_base_name:
                    return get_golden_function(ttir.ReluOp)
                # Add more mappings as needed

            # For direct operation names
            elif "add" in op_name_lower:
                return get_golden_function(stablehlo.AddOp)
            elif "multiply" in op_name_lower or "mul" in op_name_lower:
                return get_golden_function(stablehlo.MulOp)
            elif "subtract" in op_name_lower or "sub" in op_name_lower:
                return get_golden_function(stablehlo.SubOp)
            elif "divide" in op_name_lower or "div" in op_name_lower:
                return get_golden_function(stablehlo.DivOp)

            # Try to find by iterating through mappings
            for op_class, golden_func in GOLDEN_MAPPINGS.items():
                if hasattr(op_class, "__name__"):
                    class_name = op_class.__name__.lower()
                    if op_name_lower in class_name or class_name in op_name_lower:
                        logger.debug(
                            f"Found golden function for {op_name}: {op_class.__name__}"
                        )
                        return golden_func

            logger.debug(f"No golden function mapping found for: {op_name}")
            return None

        except Exception as e:
            logger.error(f"Error mapping operation {op_name} to golden function: {e}")
            return None

    def _convert_to_torch_tensors(self, tensors: list) -> list:
        """Convert a list of tensors to torch tensors."""
        torch_tensors = []
        for tensor in tensors:
            torch_tensors.append(self._convert_to_torch_tensor(tensor))
        return torch_tensors

    def _convert_to_torch_tensor(self, tensor: Any) -> torch.Tensor:
        """Convert a single tensor to torch tensor."""
        if hasattr(tensor, "get_data_buffer"):
            # This is a runtime tensor
            buffer = tensor.get_data_buffer()
            dtype = self._runtime_dtype_to_torch(tensor.get_dtype())
            shape = tensor.get_shape()
            return torch.frombuffer(buffer, dtype=dtype).reshape(shape).clone()
        else:
            # Assume it's already a torch tensor
            return tensor

    def _runtime_dtype_to_torch(self, dtype: Any) -> torch.dtype:
        """Convert runtime dtype to torch dtype."""
        # This is a simplified version - in real implementation you'd use
        # the proper conversion from util.py
        try:
            from ttrt.runtime import DataType

            if dtype == DataType.Float32:
                return torch.float32
            elif dtype == DataType.BFloat16:
                return torch.bfloat16
            elif dtype == DataType.Int32:
                return torch.int32
            elif dtype == DataType.UInt32:
                return torch.uint32
            elif dtype == DataType.UInt16:
                return torch.uint16
            elif dtype == DataType.UInt8:
                return torch.uint8
            else:
                return torch.float32  # Default fallback
        except:
            return torch.float32  # Default fallback

    def _extract_tensors_from_runtime(
        self, op_context: Any, program_context: Any
    ) -> tuple:
        """
        Extract input and output tensors from runtime operation context.

        Args:
            op_context: Runtime operation context
            program_context: Runtime program context

        Returns:
            Tuple of (input_tensors_list, output_tensor) or (None, None) if extraction fails

        Raises:
            RuntimeError: If tensor extraction fails
        """
        if not TTRT_RUNTIME_AVAILABLE:
            raise RuntimeError("ttrt.runtime not available for tensor extraction")

        try:
            # Extract output tensor
            output_tensor_rt = ttrt.runtime.get_op_output_tensor(
                op_context, program_context
            )
            if output_tensor_rt is None:
                raise RuntimeError("Could not extract output tensor from runtime")

            # Convert output tensor to PyTorch
            output_buffer = output_tensor_rt.get_data_buffer()
            output_dtype = self._runtime_dtype_to_torch(output_tensor_rt.get_dtype())
            output_shape = output_tensor_rt.get_shape()
            output_tensor = (
                torch.frombuffer(output_buffer, dtype=output_dtype)
                .reshape(output_shape)
                .clone()
            )

            # Extract input tensors
            input_refs = ttrt.runtime.get_op_input_refs(op_context, program_context)
            input_tensors = []

            for input_ref in input_refs:
                try:
                    input_tensor_rt = ttrt.runtime.retrieve_tensor_from_pool(
                        program_context, input_ref, True
                    )
                    if input_tensor_rt:
                        input_buffer = input_tensor_rt.get_data_buffer()
                        input_dtype = self._runtime_dtype_to_torch(
                            input_tensor_rt.get_dtype()
                        )
                        input_shape = input_tensor_rt.get_shape()
                        input_tensor = (
                            torch.frombuffer(input_buffer, dtype=input_dtype)
                            .reshape(input_shape)
                            .clone()
                        )
                        input_tensors.append(input_tensor)
                    else:
                        raise RuntimeError(
                            f"Could not retrieve input tensor from pool for ref: {input_ref}"
                        )
                except Exception as e:
                    raise RuntimeError(f"Error extracting input tensor: {e}")

            return input_tensors, output_tensor

        except Exception as e:
            raise RuntimeError(f"Error extracting tensors from runtime: {e}")

    def _compare_tensors(
        self, golden: torch.Tensor, actual: torch.Tensor
    ) -> Dict[str, Any]:
        """Compare golden and actual tensors and compute metrics."""
        try:
            # Ensure same shape for comparison
            if golden.shape != actual.shape:
                logger.warning(
                    f"Shape mismatch: golden {golden.shape} vs actual {actual.shape}"
                )
                return {"error": f"Shape mismatch: {golden.shape} vs {actual.shape}"}

            # Calculate metrics
            diff = torch.abs(golden - actual)

            results = {
                "shape_match": True,
                "max_difference": torch.max(diff).item(),
                "mean_difference": torch.mean(diff).item(),
                "root_mean_square_error": torch.sqrt(torch.mean(diff**2)).item(),
                "cosine_similarity": torch.nn.functional.cosine_similarity(
                    golden.flatten().unsqueeze(0), actual.flatten().unsqueeze(0)
                ).item(),
                "allclose_atol_1e-6": torch.allclose(golden, actual, atol=1e-6),
                "allclose_atol_1e-3": torch.allclose(golden, actual, atol=1e-3),
                "allclose_atol_1e-1": torch.allclose(golden, actual, atol=1e-1),
            }

            # Calculate PCC if tensors are not all zeros
            if torch.any(golden) and torch.any(actual):
                golden_flat = golden.flatten()
                actual_flat = actual.flatten()

                # Convert to float for PCC calculation
                if not torch.is_floating_point(golden_flat):
                    golden_flat = golden_flat.float()
                if not torch.is_floating_point(actual_flat):
                    actual_flat = actual_flat.float()

                results["pearson_correlation"] = torch.corrcoef(
                    torch.stack([golden_flat, actual_flat])
                )[0, 1].item()
            else:
                results["pearson_correlation"] = (
                    1.0 if torch.equal(golden, actual) else 0.0
                )

            return results

        except Exception as e:
            logger.error(f"Error comparing tensors: {e}")
            return {"error": str(e)}

    def get_stats(self) -> Dict[str, Any]:
        """Get current execution statistics."""
        return self.execution_stats.copy()


# Global handler instance
_runtime_handler = None


def get_runtime_handler(config_path: Optional[str] = None) -> RuntimeGoldenValidator:
    """
    Get or create the global runtime handler instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        RuntimeGoldenValidator instance
    """
    global _runtime_handler
    if _runtime_handler is None:
        _runtime_handler = RuntimeGoldenValidator(config_path)
    return _runtime_handler


def initialize_runtime_script(
    config_path: Optional[str] = None,
) -> RuntimeGoldenValidator:
    """
    Initialize the runtime golden validation system.

    Args:
        config_path: Optional path to configuration file

    Returns:
        RuntimeGoldenValidator instance
    """
    handler = get_runtime_handler(config_path)
    logger.info("Runtime golden validation system initialized successfully")
    return handler


# Convenience functions for C++ embedding
def program_start_callback(program_name: str, program_context: Any) -> None:
    """Callback for program start (callable from C++)."""
    try:
        handler = get_runtime_handler()
        handler.on_program_start(program_name, program_context)
        logger.info(f"Started golden validation for program: {program_name}")
    except Exception as e:
        logger.error(f"Program start error: {e}")


def program_end_callback(program_name: str, program_context: Any) -> None:
    """Callback for program end (callable from C++)."""
    try:
        handler = get_runtime_handler()
        handler.on_program_end(program_name, program_context)
        stats = handler.get_stats()
        golden_count = len(stats.get("golden_results", {}))
        if golden_count > 0:
            logger.info(f"Program completed - {golden_count} operations validated")
    except Exception as e:
        logger.error(f"Program end error: {e}")


def operation_complete_callback(
    op_name: str, op_context: Any, program_context: Any
) -> None:
    """Callback for operation completion (callable from C++)."""
    try:
        handler = get_runtime_handler()
        handler.on_operation_complete(op_name, op_context, program_context)

        # Perform golden calculation using library functions
        if TORCH_AVAILABLE and GOLDEN_LIBRARY_AVAILABLE and TTRT_RUNTIME_AVAILABLE:
            # Check if context objects are available (not None)
            if op_context is not None and program_context is not None:
                try:
                    # Context objects are already properly typed when passed from C++
                    # Use them directly without trying to construct them
                    (
                        input_tensors,
                        actual_output_tensor,
                    ) = handler._extract_tensors_from_runtime(
                        op_context, program_context
                    )

                    logger.debug(
                        f"Extracted {len(input_tensors)} input tensors and 1 output tensor for {op_name}"
                    )
                    logger.debug(
                        f"Input tensor shapes: {[t.shape for t in input_tensors]}"
                    )
                    logger.debug(f"Output tensor shape: {actual_output_tensor.shape}")

                    golden_results = handler.perform_golden_calculation(
                        op_name, input_tensors, actual_output_tensor, op_context
                    )

                    # Store results and log summary
                    if "error" not in golden_results:
                        handler.execution_stats["golden_results"][
                            op_name
                        ] = golden_results
                        pcc = golden_results.get("pearson_correlation", 0.0)
                        max_diff = golden_results.get("max_difference", 0.0)
                        logger.info(
                            f"✅ Operation '{op_name}' validated - PCC: {pcc:.4f}, MaxDiff: {max_diff:.2e}"
                        )
                    else:
                        logger.warning(
                            f"⚠️  Golden validation failed for '{op_name}': {golden_results['error']}"
                        )

                except RuntimeError as extraction_error:
                    logger.warning(
                        f"⚠️  Skipping golden validation for '{op_name}': {extraction_error}"
                    )
                except Exception as golden_error:
                    logger.error(
                        f"❌ Golden validation error for '{op_name}': {golden_error}"
                    )
            else:
                logger.debug(
                    f"Skipping golden validation for '{op_name}' - context objects not available"
                )
        else:
            logger.debug(
                f"Skipping golden validation for '{op_name}' - required libraries not available"
            )

    except Exception as e:
        logger.error(f"Callback error: {e}")


def error_callback(error: Exception, op_context: Any = None) -> None:
    """Callback for error handling (callable from C++)."""
    try:
        handler = get_runtime_handler()
        handler.handle_error(error, op_context)
    except Exception as e:
        logger.error(f"Error in error_callback: {e}")


if __name__ == "__main__":
    # Standalone execution for testing
    print("Testing RuntimeGoldenValidator...")

    handler = initialize_runtime_script()

    # Test golden calculation
    if TORCH_AVAILABLE:
        test_input1 = torch.randn(4, 4)
        test_input2 = torch.randn(4, 4)
        test_output = test_input1 + test_input2

        results = handler.perform_golden_calculation(
            "add_test", [test_input1, test_input2], test_output
        )

        if "error" not in results:
            print("✅ Golden calculation test PASSED!")
            print(f"   PCC: {results.get('pearson_correlation', 0.0):.6f}")
            print(f"   Max Diff: {results.get('max_difference', 0.0):.2e}")
        else:
            print(f"❌ Test FAILED: {results['error']}")

    print("Runtime golden validation test completed!")
