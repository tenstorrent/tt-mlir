// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_TEST_PARAMETRIZER_HPP
#define TT_ALCHEMIST_TEST_PARAMETRIZER_HPP

#include "mlir_module_splitter.hpp"
#include <vector>
#include <map>
#include <set>
#include <string>

namespace tt::alchemist {

/**
 * @brief Represents a single test case with specific parameter values
 */
struct TestCase {
    std::string id;                              ///< Unique test case identifier
    std::string description;                     ///< Human-readable description
    std::vector<std::vector<int64_t>> shapes;    ///< Input shapes for this test case
    std::vector<mlir::Type> dtypes;              ///< Data types for this test case
    std::map<std::string, mlir::Attribute> attrs;///< Attributes for this test case
    OpInfo originalOp;                           ///< Original operation this test case is based on
};

/**
 * @brief Groups similar operations for parametrized testing
 */
struct TestGroup {
    std::string opName;                                    ///< Base operation name
    std::string fullOpName;                                ///< Full operation name with dialect
    std::set<std::vector<int64_t>> uniqueShapes;          ///< All unique shapes seen
    std::set<std::string> uniqueDtypes;                    ///< All unique data types seen
    std::map<std::string, std::set<std::string>> uniqueAttrs; ///< Unique attribute values by name
    std::vector<OpInfo> operations;                        ///< All operations in this group
    std::vector<TestCase> testCases;                       ///< Generated test cases
    bool canParametrize;                                   ///< Whether this group can be parametrized
};

/**
 * @brief Configuration for test parametrization
 */
struct ParametrizationConfig {
    bool enableParametrization = true;      ///< Enable parametrized test generation
    bool groupByAttributes = true;          ///< Group operations with same attributes
    bool generateCombinations = false;      ///< Generate all combinations of parameters
    size_t maxTestCases = 100;              ///< Maximum test cases per group
    size_t minGroupSize = 2;                ///< Minimum operations to form a group
};

/**
 * @brief Handles grouping and parametrization of operations for test generation
 */
class TestParametrizer {
public:
    /**
     * @brief Constructor with configuration
     */
    explicit TestParametrizer(const ParametrizationConfig& config = {});

    /**
     * @brief Group similar operations for parametrized testing
     *
     * @param ops Vector of operations to group
     * @return Vector of test groups
     */
    std::vector<TestGroup> groupOperations(const std::vector<OpInfo>& ops);

    /**
     * @brief Generate test cases for a group
     *
     * @param group The test group
     * @return Updated test group with generated test cases
     */
    TestGroup generateTestCases(TestGroup& group);

    /**
     * @brief Check if two operations are similar enough to group
     *
     * @param op1 First operation
     * @param op2 Second operation
     * @return true if operations can be grouped
     */
    bool areOpsSimilar(const OpInfo& op1, const OpInfo& op2);

    /**
     * @brief Extract dtype string from MLIR type
     *
     * @param type The MLIR type
     * @return String representation of the dtype
     */
    static std::string extractDtype(mlir::Type type);

    /**
     * @brief Convert attribute to string for comparison
     *
     * @param attr The MLIR attribute
     * @return String representation
     */
    static std::string attributeToString(mlir::Attribute attr);

    /**
     * @brief Check if a group should be parametrized
     *
     * @param group The test group
     * @return true if the group should use parametrization
     */
    bool shouldParametrize(const TestGroup& group);

private:
    /**
     * @brief Extract unique parameters from operations
     *
     * @param ops Vector of operations
     * @param group Group to update with unique parameters
     */
    void extractUniqueParams(const std::vector<OpInfo>& ops, TestGroup& group);

    /**
     * @brief Generate individual test cases from unique parameters
     *
     * @param group The test group
     */
    void generateIndividualTests(TestGroup& group);

    /**
     * @brief Generate combination test cases from parameters
     *
     * @param group The test group
     */
    void generateCombinationTests(TestGroup& group);

    /**
     * @brief Create a test case ID from parameters
     *
     * @param shape The shape vector
     * @param dtype The data type string
     * @param attrs The attributes map
     * @return Unique test case ID
     */
    std::string createTestId(
        const std::vector<int64_t>& shape,
        const std::string& dtype,
        const std::map<std::string, std::string>& attrs);

    /**
     * @brief Check if operations have compatible attributes
     *
     * @param op1 First operation
     * @param op2 Second operation
     * @return true if attributes are compatible
     */
    bool hasCompatibleAttributes(const OpInfo& op1, const OpInfo& op2);

    /**
     * @brief Check if operations have same number of inputs/outputs
     *
     * @param op1 First operation
     * @param op2 Second operation
     * @return true if signatures match
     */
    bool hasSameSignature(const OpInfo& op1, const OpInfo& op2);

    /**
     * @brief Format shape as string for test generation
     *
     * @param shape The shape vector
     * @return String representation (e.g., "(1, 32, 128, 128)")
     */
    static std::string formatShape(const std::vector<int64_t>& shape);

private:
    ParametrizationConfig config_;  ///< Configuration for parametrization
};

} // namespace tt::alchemist

#endif // TT_ALCHEMIST_TEST_PARAMETRIZER_HPP