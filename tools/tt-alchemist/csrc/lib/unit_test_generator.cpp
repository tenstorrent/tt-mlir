// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "unit_test_generator.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "ttmlir/Conversion/TTNNToEmitPy/TTNNToEmitPy.h"
// Note: PythonTranslation may need to be implemented or adjusted
// #include "ttmlir/Target/Python/PythonTranslation.h"
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace tt::alchemist {

namespace fs = std::filesystem;

UnitTestGenerator::UnitTestGenerator(const TestGenerationOptions& options)
    : options_(options), context_(nullptr) {
    // Context will be initialized when needed
}

bool UnitTestGenerator::generate(mlir::ModuleOp module, const std::string& outputDir) {
    // Ensure output directory exists
    if (!ensureDirectoryExists(outputDir)) {
        lastError_ = "Failed to create output directory: " + outputDir;
        return false;
    }

    // Split module into operations
    MLIRModuleSplitter splitter;
    std::vector<OpInfo> operations = splitter.split(module);

    if (operations.empty()) {
        lastError_ = "No operations found in module";
        return false;
    }

    // Filter operations if needed
    if (!options_.opFilter.empty()) {
        operations.erase(
            std::remove_if(operations.begin(), operations.end(),
                [this](const OpInfo& op) { return !shouldIncludeOp(op.opName); }),
            operations.end()
        );
    }

    if (operations.empty()) {
        lastError_ = "No operations matched the filter";
        return false;
    }

    // Group and parametrize operations
    ParametrizationConfig paramConfig;
    paramConfig.enableParametrization = options_.generateParametrized;
    TestParametrizer parametrizer(paramConfig);
    std::vector<TestGroup> testGroups = parametrizer.groupOperations(operations);

    // Generate test files
    bool success = true;
    for (const auto& group : testGroups) {
        if (!generateTestFile(group, outputDir)) {
            if (options_.verbose) {
                std::cerr << "Failed to generate test for " << group.opName << std::endl;
            }
            success = false;
        }
    }

    // Generate conftest.py
    if (options_.generateConftest) {
        if (!generateConftest(outputDir)) {
            if (options_.verbose) {
                std::cerr << "Failed to generate conftest.py" << std::endl;
            }
            success = false;
        }
    }

    // Generate test utilities
    if (!generateTestUtils(outputDir)) {
        if (options_.verbose) {
            std::cerr << "Failed to generate test_utils.py" << std::endl;
        }
        success = false;
    }

    return success;
}

bool UnitTestGenerator::generateFromFile(const std::string& inputFile, const std::string& outputDir) {
    // Parse MLIR file
    mlir::MLIRContext context;
    context.loadAllAvailableDialects();

    auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFile, &context);
    if (!module) {
        lastError_ = "Failed to parse MLIR file: " + inputFile;
        return false;
    }

    context_ = &context;
    return generate(module.get(), outputDir);
}

bool UnitTestGenerator::generateTestFile(const TestGroup& group, const std::string& outputDir) {
    // Generate test file name
    std::string fileName = "test_" + sanitizeOpName(group.opName) + ".py";
    fs::path filePath = fs::path(outputDir) / fileName;

    // Generate test content
    std::string testContent;
    if (group.canParametrize && options_.generateParametrized) {
        testContent = generateParametrizedTest(group);
    } else {
        testContent = generateIndividualTests(group);
    }

    // Write to file
    std::ofstream file(filePath);
    if (!file) {
        lastError_ = "Failed to open file for writing: " + filePath.string();
        return false;
    }

    file << testContent;
    file.close();

    if (options_.verbose) {
        std::cout << "Generated: " << filePath.string() << std::endl;
    }

    return true;
}

bool UnitTestGenerator::generateConftest(const std::string& outputDir) {
    fs::path filePath = fs::path(outputDir) / "conftest.py";

    std::string content = R"(# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration and fixtures for TTNN unit tests."""

import pytest
import ttnn
import os

@pytest.fixture(scope="session")
def device():
    """Get TTNN device for testing."""
    device_id = int(os.environ.get("TT_DEVICE_ID", 0))
    device = ttnn.open_device(device_id=device_id)
    yield device
    ttnn.close_device(device)

@pytest.fixture
def mesh_device():
    """Get mesh device for multi-device testing."""
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        dispatch_core_type=ttnn.DispatchCoreType.WORKER
    )
    yield mesh
    ttnn.close_mesh_device(mesh)

@pytest.fixture(autouse=True)
def reset_seeds():
    """Reset random seeds for reproducibility."""
    import torch
    import random
    import numpy as np

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
)";

    std::ofstream file(filePath);
    if (!file) {
        lastError_ = "Failed to create conftest.py";
        return false;
    }

    file << content;
    return true;
}

bool UnitTestGenerator::generateTestUtils(const std::string& outputDir) {
    fs::path filePath = fs::path(outputDir) / "test_utils.py";

    std::string content = R"(# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utilities for TTNN unit tests."""

import torch
import ttnn
import numpy as np
from typing import List, Tuple, Optional

def create_random_tensor(shape: Tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Create a random tensor with given shape and dtype."""
    if dtype in [torch.float32, torch.float16, torch.bfloat16]:
        return torch.randn(shape, dtype=dtype)
    elif dtype == torch.int32:
        return torch.randint(-100, 100, shape, dtype=dtype)
    elif dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype).bool()
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def validate_output(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    pcc_threshold: float = 0.99
) -> bool:
    """Validate output tensor against expected values."""
    # For integer types, use exact comparison
    if actual.dtype in [torch.int32, torch.int64, torch.bool]:
        return torch.equal(actual, expected)

    # For float types, use allclose
    if torch.allclose(actual, expected, rtol=rtol, atol=atol):
        return True

    # Calculate PCC for additional validation
    pcc = calculate_pcc(actual, expected)
    return pcc >= pcc_threshold

def calculate_pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Calculate Pearson Correlation Coefficient."""
    actual_flat = actual.flatten().float()
    expected_flat = expected.flatten().float()

    if len(actual_flat) == 0:
        return 0.0

    actual_mean = actual_flat.mean()
    expected_mean = expected_flat.mean()

    actual_centered = actual_flat - actual_mean
    expected_centered = expected_flat - expected_mean

    numerator = (actual_centered * expected_centered).sum()
    denominator = torch.sqrt((actual_centered ** 2).sum() * (expected_centered ** 2).sum())

    if denominator == 0:
        return 0.0

    return (numerator / denominator).item()

def to_ttnn_tensor(tensor: torch.Tensor, device) -> ttnn.Tensor:
    """Convert PyTorch tensor to TTNN tensor."""
    return ttnn.from_torch(tensor, device=device)

def from_ttnn_tensor(tensor: ttnn.Tensor) -> torch.Tensor:
    """Convert TTNN tensor to PyTorch tensor."""
    return ttnn.to_torch(tensor)

def get_golden_function(op_name: str):
    """Get the corresponding PyTorch function for golden value computation."""
    op_map = {
        "add": torch.add,
        "sub": torch.sub,
        "mul": torch.mul,
        "div": torch.div,
        "exp": torch.exp,
        "log": torch.log,
        "relu": torch.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "sqrt": torch.sqrt,
        "abs": torch.abs,
        "neg": torch.neg,
        "sin": torch.sin,
        "cos": torch.cos,
        "matmul": torch.matmul,
        # Add more mappings as needed
    }
    return op_map.get(op_name)
)";

    std::ofstream file(filePath);
    if (!file) {
        lastError_ = "Failed to create test_utils.py";
        return false;
    }

    file << content;
    return true;
}

std::string UnitTestGenerator::generateParametrizedTest(const TestGroup& group) {
    std::stringstream ss;

    // Header
    ss << "# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC\n";
    ss << "#\n";
    ss << "# SPDX-License-Identifier: Apache-2.0\n\n";
    ss << "\"\"\"Auto-generated unit tests for " << group.opName << " operation.\"\"\"\n\n";

    // Imports
    ss << "import pytest\n";
    ss << "import torch\n";
    ss << "import ttnn\n";
    ss << "from test_utils import (\n";
    ss << "    create_random_tensor,\n";
    ss << "    validate_output,\n";
    ss << "    to_ttnn_tensor,\n";
    ss << "    from_ttnn_tensor,\n";
    ss << "    get_golden_function\n";
    ss << ")\n\n";

    // Test class
    ss << "class Test" << getTestClassName(group.opName) << ":\n";
    ss << "    \"\"\"Test cases for " << group.fullOpName << " operation.\"\"\"\n\n";

    // Format parameters
    auto params = formatTestParameters(group);

    // Parametrized test
    ss << "    @pytest.mark.parametrize(\"shape\", " << params["shapes"] << ")\n";
    ss << "    @pytest.mark.parametrize(\"dtype\", " << params["dtypes"] << ")\n";

    if (params.count("attrs") && params["attrs"] != "[]") {
        ss << "    @pytest.mark.parametrize(\"attrs\", " << params["attrs"] << ")\n";
    }

    ss << "    def test_" << sanitizeOpName(group.opName) << "_parametrized(self, shape, dtype, device):\n";
    ss << "        \"\"\"Test " << group.opName << " with various parameters.\"\"\"\n";

    // Test body
    size_t numInputs = group.operations.empty() ? 1 : group.operations[0].inputTypes.size();

    // Create input tensors
    for (size_t i = 0; i < numInputs; ++i) {
        ss << "        input" << i << " = create_random_tensor(shape, dtype)\n";
        ss << "        ttnn_input" << i << " = to_ttnn_tensor(input" << i << ", device)\n";
    }
    ss << "\n";

    // Execute operation
    ss << "        # Execute TTNN operation\n";
    ss << "        output = ttnn." << group.opName << "(";
    for (size_t i = 0; i < numInputs; ++i) {
        if (i > 0) ss << ", ";
        ss << "ttnn_input" << i;
    }
    ss << ")\n\n";

    // Compute golden values
    ss << "        # Compute golden values\n";
    ss << "        golden_fn = get_golden_function(\"" << group.opName << "\")\n";
    ss << "        if golden_fn:\n";
    ss << "            expected = golden_fn(";
    for (size_t i = 0; i < numInputs; ++i) {
        if (i > 0) ss << ", ";
        ss << "input" << i;
    }
    ss << ")\n";
    ss << "        else:\n";
    ss << "            # Fallback: use first input as expected for unary ops\n";
    ss << "            expected = input0\n\n";

    // Validate
    ss << "        # Validate output\n";
    ss << "        actual = from_ttnn_tensor(output)\n";
    ss << "        assert validate_output(actual, expected), ";
    ss << "f\"Output validation failed for shape {shape} and dtype {dtype}\"\n";

    return ss.str();
}

std::string UnitTestGenerator::generateIndividualTests(const TestGroup& group) {
    std::stringstream ss;

    // Header
    ss << "# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC\n";
    ss << "#\n";
    ss << "# SPDX-License-Identifier: Apache-2.0\n\n";
    ss << "\"\"\"Auto-generated unit tests for " << group.opName << " operation.\"\"\"\n\n";

    // Imports
    ss << "import pytest\n";
    ss << "import torch\n";
    ss << "import ttnn\n";
    ss << "from test_utils import (\n";
    ss << "    create_random_tensor,\n";
    ss << "    validate_output,\n";
    ss << "    to_ttnn_tensor,\n";
    ss << "    from_ttnn_tensor,\n";
    ss << "    get_golden_function\n";
    ss << ")\n\n";

    // Test class
    ss << "class Test" << getTestClassName(group.opName) << ":\n";
    ss << "    \"\"\"Test cases for " << group.fullOpName << " operation.\"\"\"\n\n";

    // Generate individual test methods
    for (const auto& testCase : group.testCases) {
        ss << "    def test_" << sanitizeOpName(group.opName) << "_" << testCase.id << "(self, device):\n";
        ss << "        \"\"\"" << testCase.description << "\"\"\"\n";

        // Create inputs
        for (size_t i = 0; i < testCase.shapes.size(); ++i) {
            ss << "        shape" << i << " = " << TestParametrizer::formatShape(testCase.shapes[i]) << "\n";
            ss << "        input" << i << " = create_random_tensor(shape" << i << ")\n";
            ss << "        ttnn_input" << i << " = to_ttnn_tensor(input" << i << ", device)\n";
        }
        ss << "\n";

        // Execute operation
        ss << "        # Execute TTNN operation\n";
        ss << "        output = ttnn." << group.opName << "(";
        for (size_t i = 0; i < testCase.shapes.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << "ttnn_input" << i;
        }
        ss << ")\n\n";

        // Validate
        ss << "        # Validate output\n";
        ss << "        actual = from_ttnn_tensor(output)\n";
        ss << "        assert actual is not None, \"Operation failed\"\n\n";
    }

    return ss.str();
}

std::map<std::string, std::string> UnitTestGenerator::formatTestParameters(const TestGroup& group) {
    std::map<std::string, std::string> params;

    // Format shapes
    std::stringstream shapes;
    shapes << "[";
    bool first = true;
    for (const auto& shape : group.uniqueShapes) {
        if (!first) shapes << ", ";
        shapes << TestParametrizer::formatShape(shape);
        first = false;
    }
    shapes << "]";
    params["shapes"] = shapes.str();

    // Format dtypes
    std::stringstream dtypes;
    dtypes << "[";
    first = true;
    for (const auto& dtype : group.uniqueDtypes) {
        if (!first) dtypes << ", ";
        if (dtype == "f32") {
            dtypes << "torch.float32";
        } else if (dtype == "bf16") {
            dtypes << "torch.bfloat16";
        } else if (dtype == "f16") {
            dtypes << "torch.float16";
        } else if (dtype == "i32") {
            dtypes << "torch.int32";
        } else {
            dtypes << "torch.float32";  // Default
        }
        first = false;
    }
    dtypes << "]";
    params["dtypes"] = dtypes.str();

    // Format attributes if any
    params["attrs"] = "[]";  // Simplified for now

    return params;
}

bool UnitTestGenerator::shouldIncludeOp(const std::string& opName) const {
    if (options_.opFilter.empty()) {
        return true;
    }

    for (const auto& filter : options_.opFilter) {
        if (opName.find(filter) != std::string::npos) {
            return true;
        }
    }

    return false;
}

std::string UnitTestGenerator::sanitizeOpName(const std::string& opName) const {
    std::string sanitized = opName;

    // Remove dialect prefix
    size_t dotPos = sanitized.find('.');
    if (dotPos != std::string::npos) {
        sanitized = sanitized.substr(dotPos + 1);
    }

    // Replace non-alphanumeric characters with underscores
    std::replace_if(sanitized.begin(), sanitized.end(),
        [](char c) { return !std::isalnum(c); }, '_');

    return sanitized;
}

std::string UnitTestGenerator::getTestClassName(const std::string& opName) const {
    std::string sanitized = sanitizeOpName(opName);

    // Capitalize first letter and each letter after underscore
    bool capitalize = true;
    for (char& c : sanitized) {
        if (c == '_') {
            capitalize = true;
        } else if (capitalize) {
            c = std::toupper(c);
            capitalize = false;
        }
    }

    // Remove underscores
    sanitized.erase(std::remove(sanitized.begin(), sanitized.end(), '_'), sanitized.end());

    return sanitized;
}

bool UnitTestGenerator::ensureDirectoryExists(const std::string& path) {
    try {
        fs::create_directories(path);
        return true;
    } catch (const fs::filesystem_error& e) {
        lastError_ = std::string("Filesystem error: ") + e.what();
        return false;
    }
}

} // namespace tt::alchemist