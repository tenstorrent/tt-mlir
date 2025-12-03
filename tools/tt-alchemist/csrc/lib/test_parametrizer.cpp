// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "test_parametrizer.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include <sstream>
#include <algorithm>
#include <iomanip>

namespace tt::alchemist {

TestParametrizer::TestParametrizer(const ParametrizationConfig& config)
    : config_(config) {}

std::vector<TestGroup> TestParametrizer::groupOperations(const std::vector<OpInfo>& ops) {
    std::vector<TestGroup> groups;
    std::map<std::string, TestGroup> groupMap;

    // First pass: group operations by name
    for (const auto& op : ops) {
        std::string baseName = MLIRModuleSplitter::getOpBaseName(op.op);

        // Find or create group
        auto& group = groupMap[baseName];
        if (group.opName.empty()) {
            group.opName = baseName;
            group.fullOpName = op.opName;
        }

        group.operations.push_back(op);
    }

    // Second pass: split groups if attributes are incompatible
    if (config_.groupByAttributes) {
        std::map<std::string, TestGroup> refinedGroupMap;

        for (auto& [name, group] : groupMap) {
            // Check if all operations in the group have compatible attributes
            bool needsSplit = false;
            std::vector<TestGroup> subGroups;

            for (const auto& op : group.operations) {
                bool foundCompatible = false;

                for (auto& subGroup : subGroups) {
                    if (!subGroup.operations.empty() &&
                        areOpsSimilar(subGroup.operations[0], op)) {
                        subGroup.operations.push_back(op);
                        foundCompatible = true;
                        break;
                    }
                }

                if (!foundCompatible) {
                    TestGroup newSubGroup;
                    newSubGroup.opName = group.opName;
                    newSubGroup.fullOpName = group.fullOpName;
                    newSubGroup.operations.push_back(op);
                    subGroups.push_back(newSubGroup);
                }
            }

            // Add subgroups to refined map
            for (size_t i = 0; i < subGroups.size(); ++i) {
                std::string key = name;
                if (subGroups.size() > 1) {
                    key += "_" + std::to_string(i);
                }
                refinedGroupMap[key] = std::move(subGroups[i]);
            }
        }

        groupMap = std::move(refinedGroupMap);
    }

    // Third pass: extract unique parameters and generate test cases
    for (auto& [name, group] : groupMap) {
        // Skip groups that are too small
        if (group.operations.size() < config_.minGroupSize && config_.enableParametrization) {
            // Convert to individual test cases
            group.canParametrize = false;
        } else {
            group.canParametrize = config_.enableParametrization;
        }

        extractUniqueParams(group.operations, group);
        generateTestCases(group);
        groups.push_back(std::move(group));
    }

    return groups;
}

TestGroup TestParametrizer::generateTestCases(TestGroup& group) {
    if (shouldParametrize(group)) {
        if (config_.generateCombinations) {
            generateCombinationTests(group);
        } else {
            generateIndividualTests(group);
        }
    } else {
        // Create individual test cases for each operation
        for (const auto& op : group.operations) {
            TestCase testCase;
            testCase.id = createTestId(
                op.inputShapes.empty() ? std::vector<int64_t>{} : op.inputShapes[0],
                extractDtype(op.inputTypes.empty() ? mlir::Type() : op.inputTypes[0]),
                {}
            );
            testCase.description = "Test for " + op.opName;
            testCase.shapes = op.inputShapes;
            testCase.dtypes = op.inputTypes;
            testCase.originalOp = op;

            group.testCases.push_back(testCase);
        }
    }

    return group;
}

bool TestParametrizer::areOpsSimilar(const OpInfo& op1, const OpInfo& op2) {
    // Must have same operation name
    if (op1.opName != op2.opName) {
        return false;
    }

    // Must have same signature (number of inputs/outputs)
    if (!hasSameSignature(op1, op2)) {
        return false;
    }

    // Check attribute compatibility if configured
    if (config_.groupByAttributes && !hasCompatibleAttributes(op1, op2)) {
        return false;
    }

    return true;
}

std::string TestParametrizer::extractDtype(mlir::Type type) {
    if (!type) {
        return "unknown";
    }

    std::string typeStr = "";
    llvm::raw_string_ostream stream(typeStr);
    type.print(stream);

    // Extract element type from tensor type
    if (auto tensorType = type.dyn_cast<mlir::RankedTensorType>()) {
        mlir::Type elemType = tensorType.getElementType();

        if (elemType.isF32()) {
            return "f32";
        } else if (elemType.isBF16()) {
            return "bf16";
        } else if (elemType.isF16()) {
            return "f16";
        } else if (elemType.isInteger(32)) {
            return "i32";
        } else if (elemType.isInteger(8)) {
            return "i8";
        }
    }

    return typeStr;
}

std::string TestParametrizer::attributeToString(mlir::Attribute attr) {
    if (!attr) {
        return "";
    }

    std::string attrStr;
    llvm::raw_string_ostream stream(attrStr);
    attr.print(stream);
    return attrStr;
}

bool TestParametrizer::shouldParametrize(const TestGroup& group) {
    // Don't parametrize if disabled
    if (!group.canParametrize) {
        return false;
    }

    // Need at least minimum group size
    if (group.operations.size() < config_.minGroupSize) {
        return false;
    }

    // Check if there's variation in parameters
    bool hasVariation = group.uniqueShapes.size() > 1 ||
                       group.uniqueDtypes.size() > 1 ||
                       !group.uniqueAttrs.empty();

    return hasVariation;
}

void TestParametrizer::extractUniqueParams(const std::vector<OpInfo>& ops, TestGroup& group) {
    for (const auto& op : ops) {
        // Extract shapes
        for (const auto& shape : op.inputShapes) {
            group.uniqueShapes.insert(shape);
        }

        // Extract dtypes
        for (const auto& type : op.inputTypes) {
            group.uniqueDtypes.insert(extractDtype(type));
        }

        // Extract attributes
        if (op.attributes) {
            for (const auto& namedAttr : op.attributes) {
                std::string attrName = namedAttr.getName().str();
                std::string attrValue = attributeToString(namedAttr.getValue());
                group.uniqueAttrs[attrName].insert(attrValue);
            }
        }
    }
}

void TestParametrizer::generateIndividualTests(TestGroup& group) {
    // Generate test cases based on actual operations
    std::set<std::string> usedIds;

    for (const auto& op : group.operations) {
        TestCase testCase;
        testCase.originalOp = op;
        testCase.shapes = op.inputShapes;
        testCase.dtypes = op.inputTypes;

        // Extract attributes
        if (op.attributes) {
            for (const auto& namedAttr : op.attributes) {
                testCase.attrs[namedAttr.getName().str()] = namedAttr.getValue();
            }
        }

        // Create unique ID
        std::string baseId = createTestId(
            testCase.shapes.empty() ? std::vector<int64_t>{} : testCase.shapes[0],
            extractDtype(testCase.dtypes.empty() ? mlir::Type() : testCase.dtypes[0]),
            {}
        );

        // Ensure uniqueness
        std::string finalId = baseId;
        int counter = 1;
        while (usedIds.count(finalId)) {
            finalId = baseId + "_" + std::to_string(counter++);
        }
        usedIds.insert(finalId);

        testCase.id = finalId;
        testCase.description = "Test " + group.opName + " with " + formatShape(testCase.shapes[0]);

        group.testCases.push_back(testCase);
    }
}

void TestParametrizer::generateCombinationTests(TestGroup& group) {
    // Generate all combinations of parameters (limited by maxTestCases)
    size_t testCount = 0;

    for (const auto& shape : group.uniqueShapes) {
        for (const auto& dtype : group.uniqueDtypes) {
            if (testCount >= config_.maxTestCases) {
                return;
            }

            TestCase testCase;
            testCase.id = createTestId(shape, dtype, {});
            testCase.description = "Test " + group.opName + " with shape " +
                                 formatShape(shape) + " and dtype " + dtype;
            testCase.shapes = {shape};

            // Create dummy type for dtype
            // Note: This is simplified - in real implementation, we'd need to
            // properly construct the MLIR type
            testCase.dtypes = {};

            group.testCases.push_back(testCase);
            testCount++;
        }
    }
}

std::string TestParametrizer::createTestId(
    const std::vector<int64_t>& shape,
    const std::string& dtype,
    const std::map<std::string, std::string>& attrs) {

    std::stringstream ss;

    // Add shape to ID
    if (!shape.empty()) {
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) ss << "x";
            ss << shape[i];
        }
    }

    // Add dtype
    if (!dtype.empty()) {
        ss << "_" << dtype;
    }

    // Add key attributes
    for (const auto& [key, value] : attrs) {
        // Sanitize attribute values for use in test names
        std::string sanitized = value;
        std::replace(sanitized.begin(), sanitized.end(), ' ', '_');
        std::replace(sanitized.begin(), sanitized.end(), '.', '_');
        ss << "_" << key << "_" << sanitized;
    }

    return ss.str();
}

bool TestParametrizer::hasCompatibleAttributes(const OpInfo& op1, const OpInfo& op2) {
    // If one has attributes and the other doesn't, they're incompatible
    if ((op1.attributes && !op2.attributes) || (!op1.attributes && op2.attributes)) {
        return false;
    }

    // If both have no attributes, they're compatible
    if (!op1.attributes && !op2.attributes) {
        return true;
    }

    // Check if they have the same attribute keys
    std::set<std::string> keys1, keys2;
    for (const auto& namedAttr : op1.attributes) {
        keys1.insert(namedAttr.getName().str());
    }
    for (const auto& namedAttr : op2.attributes) {
        keys2.insert(namedAttr.getName().str());
    }

    return keys1 == keys2;
}

bool TestParametrizer::hasSameSignature(const OpInfo& op1, const OpInfo& op2) {
    return op1.inputTypes.size() == op2.inputTypes.size() &&
           op1.outputTypes.size() == op2.outputTypes.size();
}

std::string TestParametrizer::formatShape(const std::vector<int64_t>& shape) {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape[i];
    }
    ss << ")";
    return ss.str();
}

} // namespace tt::alchemist