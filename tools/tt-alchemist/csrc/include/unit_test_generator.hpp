// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_UNIT_TEST_GENERATOR_HPP
#define TT_ALCHEMIST_UNIT_TEST_GENERATOR_HPP

#include "mlir_module_splitter.hpp"
#include "test_parametrizer.hpp"
#include <string>
#include <vector>
#include <map>
#include <filesystem>

namespace tt::alchemist {

/**
 * @brief Configuration for unit test generation
 */
struct TestGenerationOptions {
    std::vector<std::string> opFilter;     ///< Filter for specific operations (empty = all)
    bool generateParametrized = true;      ///< Generate parametrized tests
    std::string testFramework = "pytest";  ///< Test framework to use
    bool useEmitPy = true;                ///< Use EmitPy for Python code generation
    std::string pipelineOptions = "";      ///< Additional MLIR pipeline options
    bool generateConftest = true;          ///< Generate conftest.py with fixtures
    bool verbose = false;                  ///< Verbose output during generation
};

/**
 * @brief Generates unit tests from TTNN MLIR operations
 */
class UnitTestGenerator {
public:
    /**
     * @brief Constructor
     *
     * @param options Generation options
     */
    explicit UnitTestGenerator(const TestGenerationOptions& options = {});

    /**
     * @brief Generate unit tests from MLIR module
     *
     * @param module The MLIR module
     * @param outputDir Output directory for generated tests
     * @return true if generation successful
     */
    bool generate(mlir::ModuleOp module, const std::string& outputDir);

    /**
     * @brief Generate unit tests from MLIR file
     *
     * @param inputFile Path to MLIR file
     * @param outputDir Output directory for generated tests
     * @return true if generation successful
     */
    bool generateFromFile(const std::string& inputFile, const std::string& outputDir);

    /**
     * @brief Get the last error message
     *
     * @return Error message string
     */
    std::string getLastError() const { return lastError_; }

private:
    /**
     * @brief Generate test file for a test group
     *
     * @param group The test group
     * @param outputDir Output directory
     * @return true if successful
     */
    bool generateTestFile(const TestGroup& group, const std::string& outputDir);

    /**
     * @brief Generate conftest.py with common fixtures
     *
     * @param outputDir Output directory
     * @return true if successful
     */
    bool generateConftest(const std::string& outputDir);

    /**
     * @brief Generate test utilities file
     *
     * @param outputDir Output directory
     * @return true if successful
     */
    bool generateTestUtils(const std::string& outputDir);

    /**
     * @brief Convert operation to Python code using EmitPy
     *
     * @param op The operation info
     * @return Generated Python code string
     */
    std::string generateOpPythonCode(const OpInfo& op);

    /**
     * @brief Create a minimal MLIR module containing a single operation
     *
     * @param op The operation to wrap
     * @return MLIR module containing the operation
     */
    mlir::OwningOpRef<mlir::ModuleOp> createTestModule(const OpInfo& op);

    /**
     * @brief Apply a test template with parameters
     *
     * @param templateName Name of the template
     * @param params Template parameters
     * @return Generated test code
     */
    std::string applyTemplate(
        const std::string& templateName,
        const std::map<std::string, std::string>& params);

    /**
     * @brief Generate parametrized test code
     *
     * @param group The test group
     * @return Generated Python test code
     */
    std::string generateParametrizedTest(const TestGroup& group);

    /**
     * @brief Generate individual test code
     *
     * @param group The test group
     * @return Generated Python test code
     */
    std::string generateIndividualTests(const TestGroup& group);

    /**
     * @brief Format test parameters for pytest
     *
     * @param group The test group
     * @return Formatted parameter strings
     */
    std::map<std::string, std::string> formatTestParameters(const TestGroup& group);

    /**
     * @brief Check if operation should be included based on filter
     *
     * @param opName Operation name
     * @return true if operation should be included
     */
    bool shouldIncludeOp(const std::string& opName) const;

    /**
     * @brief Get the Python dtype string for an MLIR type
     *
     * @param type The MLIR type
     * @return Python dtype string (e.g., "torch.float32")
     */
    std::string getPythonDtype(mlir::Type type);

    /**
     * @brief Generate golden value computation code
     *
     * @param opName Operation name
     * @param numInputs Number of inputs
     * @return Python code for golden value computation
     */
    std::string generateGoldenComputation(const std::string& opName, size_t numInputs);

    /**
     * @brief Sanitize operation name for use as Python identifier
     *
     * @param opName Operation name
     * @return Sanitized name
     */
    std::string sanitizeOpName(const std::string& opName) const;

    /**
     * @brief Get test class name from operation name
     *
     * @param opName Operation name
     * @return Test class name
     */
    std::string getTestClassName(const std::string& opName) const;

    /**
     * @brief Load template from file or embedded string
     *
     * @param templateName Template name
     * @return Template content
     */
    std::string loadTemplate(const std::string& templateName);

    /**
     * @brief Replace template variables
     *
     * @param templateContent Template content
     * @param params Parameters to replace
     * @return Processed template
     */
    std::string replaceTemplateVars(
        const std::string& templateContent,
        const std::map<std::string, std::string>& params);

    /**
     * @brief Create output directory if it doesn't exist
     *
     * @param path Directory path
     * @return true if successful
     */
    bool ensureDirectoryExists(const std::string& path);

private:
    TestGenerationOptions options_;     ///< Generation options
    std::string lastError_;             ///< Last error message
    mlir::MLIRContext* context_;        ///< MLIR context
};

} // namespace tt::alchemist

#endif // TT_ALCHEMIST_UNIT_TEST_GENERATOR_HPP