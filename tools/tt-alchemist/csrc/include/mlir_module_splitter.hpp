// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_ALCHEMIST_MLIR_MODULE_SPLITTER_HPP
#define TT_ALCHEMIST_MLIR_MODULE_SPLITTER_HPP

#include <vector>
#include <map>
#include <string>
#include <memory>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace tt::alchemist {

/**
 * @brief Information about a single MLIR operation
 */
struct OpInfo {
    mlir::Operation* op;                    ///< The MLIR operation
    std::string opName;                      ///< Operation name (e.g., "ttnn.add")
    std::vector<mlir::Type> inputTypes;     ///< Input tensor types
    std::vector<mlir::Type> outputTypes;    ///< Output tensor types
    mlir::DictionaryAttr attributes;        ///< Operation attributes
    std::vector<std::vector<int64_t>> inputShapes;  ///< Input tensor shapes
    std::vector<std::vector<int64_t>> outputShapes; ///< Output tensor shapes

    // For tracking operation context
    std::string parentFunc;                 ///< Parent function name
    size_t opIndex;                          ///< Index in execution order
};

/**
 * @brief Splits MLIR modules into constituent operations for unit test generation
 *
 * This class processes TTNN/TTIR MLIR modules and extracts individual operations
 * in execution order, handling nested function calls and device module wrapping.
 */
class MLIRModuleSplitter {
public:
    /**
     * @brief Constructor
     */
    MLIRModuleSplitter() = default;

    /**
     * @brief Destructor
     */
    ~MLIRModuleSplitter() = default;

    /**
     * @brief Split a TTNN module into individual operations
     *
     * @param module The MLIR module to split
     * @return Vector of OpInfo structures representing each operation
     */
    std::vector<OpInfo> split(mlir::ModuleOp module);

    /**
     * @brief Extract operations from a specific function
     *
     * @param func The function to extract operations from
     * @return Vector of OpInfo structures
     */
    std::vector<OpInfo> extractOpsFromFunc(mlir::func::FuncOp func);

    /**
     * @brief Get the main function from the module
     *
     * @param module The MLIR module
     * @return The main function, or nullptr if not found
     */
    mlir::func::FuncOp getMainFunction(mlir::ModuleOp module);

    /**
     * @brief Check if an operation should be included in test generation
     *
     * @param op The operation to check
     * @return true if the operation should be included
     */
    bool shouldIncludeOp(mlir::Operation* op);

    /**
     * @brief Extract shape information from a type
     *
     * @param type The MLIR type
     * @return Vector of dimensions
     */
    static std::vector<int64_t> extractShape(mlir::Type type);

    /**
     * @brief Get operation name without dialect prefix
     *
     * @param op The operation
     * @return Operation name (e.g., "add" instead of "ttnn.add")
     */
    static std::string getOpBaseName(mlir::Operation* op);

private:
    /**
     * @brief Process operations recursively, handling function calls
     *
     * @param op The operation to process
     * @param parentFunc The parent function name
     */
    void processOperation(mlir::Operation* op, const std::string& parentFunc);

    /**
     * @brief Handle function call operations
     *
     * @param callOp The call operation
     * @param parentFunc The parent function name
     */
    void processCallOp(mlir::Operation* callOp, const std::string& parentFunc);

    /**
     * @brief Build a map of function names to function operations
     *
     * @param module The MLIR module
     */
    void buildFunctionMap(mlir::ModuleOp module);

    /**
     * @brief Extract DeviceModuleOp if present
     *
     * @param module The MLIR module
     * @return The inner module if DeviceModuleOp exists, otherwise the original
     */
    mlir::ModuleOp unwrapDeviceModule(mlir::ModuleOp module);

    /**
     * @brief Create OpInfo from an MLIR operation
     *
     * @param op The operation
     * @param parentFunc The parent function name
     * @return OpInfo structure
     */
    OpInfo createOpInfo(mlir::Operation* op, const std::string& parentFunc);

    /**
     * @brief Check if operation is a TTNN dialect operation
     *
     * @param op The operation to check
     * @return true if it's a TTNN operation
     */
    bool isTTNNOp(mlir::Operation* op);

    /**
     * @brief Check if operation is a TTIR dialect operation
     *
     * @param op The operation to check
     * @return true if it's a TTIR operation
     */
    bool isTTIROp(mlir::Operation* op);

private:
    std::vector<OpInfo> operations_;                    ///< Collected operations
    std::map<std::string, mlir::func::FuncOp> funcMap_; ///< Function name to operation map
    std::set<mlir::Operation*> processedOps_;           ///< Track processed operations to avoid duplicates
    size_t currentOpIndex_ = 0;                         ///< Current operation index for ordering
};

} // namespace tt::alchemist

#endif // TT_ALCHEMIST_MLIR_MODULE_SPLITTER_HPP