// // #include "ttmlir/Target/TTNN/program_generated.h"
// #include "ttmlir/Target/TTNN/operations/configs_generated.h"
// #include "ttmlir/Target/TTNN/program_generated.h"
// // #include "types_generated.h"
// #include <concepts>
// #include <cstdint>
// #include "tt/runtime/detail/ttnn/operations/utils.h"
// #include "ttmlir/OpModel/TTNN/Conversion.h"
// #include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
// #include "gtest/gtest.h"

#include "gtest/gtest.h"
#include <cstdio>
#include <optional>
#include "ttmlir/OpModel/TTNN/Conversion.h"
#include "ttmlir/Target/TTNN/operations/configs_generated.h"
#include "ttmlir/Target/Utils/MLIRToFlatbuffer.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"
#include "tt/runtime/detail/ttnn/utils.h"
#include "tt/runtime/detail/ttnn/operations/utils.h"


// namespace mlir {
// namespace tt {
// namespace ttnn{
    // class Conv2dConfigGeneratorTest : public ::testing::Test {
    //     public:
    //       mlir::MLIRContext context;
    //       std::function<bool(const Conv2dConfigAttr &)> filterOutFn;
    //       void SetUp() override {
    //         context.loadDialect<TTNNDialect>();
    //         filterOutFn = [](const Conv2dConfigAttr &config) { return false; };
    //       }
    //     };
        
    //     TEST_F(Conv2dConfigGeneratorTest, ConstructionMinimal) {
    //       Conv2dConfigAttr baseConfig = Conv2dConfigAttr::get(&context);
    //       Conv2dConfigSearchSpace space;
    //       Conv2dConfigGenerator gen(static_cast<Conv2dOp *>(nullptr), baseConfig, space,
    //                                 filterOutFn);
    //       EXPECT_TRUE(gen.searchDone());
    //     }
class ThreeWayConsistencyTest : public ::testing::Test {
    public: 
        mlir::MLIRContext context;
        void SetUp() override {
          context.loadDialect<mlir::tt::ttnn::TTNNDialect>();
        }
    protected:
        void RunTest(mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeConfigAttr) {
            std::optional<::ttnn::DeviceComputeKernelConfig> resultA = pathA(computeConfigAttr);
            std::optional<::ttnn::DeviceComputeKernelConfig> resultB = pathB(computeConfigAttr);
            //std::optional<::ttnn::DeviceComputeKernelConfig> resultB = resultA;
            // std::optional<::ttnn::DeviceComputeKernelConfig> resultC = pathC(computeConfigAttr);

            //if(resultA != std::nullopt || resultB != std::nullopt || resultC != std::nullopt) {
            //    if(resultA == std::nullopt && resultB == std::nullopt && resultC == std::nullopt) SUCCEED();
            //    FAIL();
            //}

            // overload == ? 
            EXPECT_EQ(resultA->math_fidelity, resultB->math_fidelity);
            // std::cout << resultA->math_fidelity << " AAA " << resultB->math_fidelity << std::endl;
            //EXPECT_EQ(resultA->math_fidelity, resultC->math_fidelity);

            EXPECT_EQ(resultA->math_approx_mode, resultB->math_approx_mode);
            // std::cout << resultA->math_approx_mode << " AAA " << resultB->math_approx_mode << std::endl;
            //EXPECT_EQ(resultA->math_approx_mode, resultC->math_approx_mode);

            EXPECT_EQ(resultA->fp32_dest_acc_en, resultB->fp32_dest_acc_en);
            // std::cout << resultA->fp32_dest_acc_en << " AAA " << resultB->fp32_dest_acc_en << std::endl;
            //EXPECT_EQ(resultA->fp32_dest_acc_en, resultC->fp32_dest_acc_en);

            EXPECT_EQ(resultA->packer_l1_acc, resultB->packer_l1_acc);
            // std::cout << resultA->packer_l1_acc << " AAA " << resultB->packer_l1_acc << std::endl;
            //EXPECT_EQ(resultA->packer_l1_acc, resultC->packer_l1_acc);

            EXPECT_EQ(resultA->dst_full_sync_en, resultB->dst_full_sync_en);
            // std::cout << resultA->dst_full_sync_en << " AAA " << resultB->dst_full_sync_en << std::endl;
            //EXPECT_EQ(resultA->dst_full_sync_en, resultC->dst_full_sync_en);

        }

        // pathA - ret DeviceComputeKernelConfig - conversionOpModel
        // pathB - ret DeviceComputeKernelConfig - OpModelWithNative
        // pathC - ret DeviceComputeKernelConfig - runtime
        
        std::optional<::ttnn::DeviceComputeKernelConfig> pathA(mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeConfigAttr) {
            return mlir::tt::ttnn::op_model::conversion::getDeviceComputeKernelConfig(computeConfigAttr); 
        }

        std::optional<::ttnn::DeviceComputeKernelConfig> pathB(mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeConfigAttr) { //DeviceComputeKernelConfigAttr computeConfigAttr
           tt::target::ttnn::DeviceComputeKernelConfigT deviceComputeKernelConfigT = mlir::tt::toNative(computeConfigAttr);
           return tt::runtime::ttnn::operations::utils::createDeviceComputeKernelConfig(deviceComputeKernelConfigT);
        }

//         std::optional<::ttnn::DeviceComputeKernelConfig> pathC(mlir::tt::ttnn::DeviceComputeKernelConfigAttr computeConfigAttr) {
//             std::optional<::ttnn::DeviceComputeKernelConfig> a;
// //             inline ::flatbuffers::Offset<::tt::target::ttnn::DeviceComputeKernelConfig>
// //                          toFlatbuffer(FlatbufferObjectCache &cache, ttnn::DeviceComputeKernelConfigAttr computeConfigAttr)
//             flatbuffers::FlatBufferBuilder fbb;
//             mlir::tt::FlatbufferObjectCache cache(&fbb);
//             ::flatbuffers::Offset<::tt::target::ttnn::DeviceComputeKernelConfig> deviceComputeKernelConfigFB = mlir::tt::toFlatbuffer(cache, computeConfigAttr);
//             // serialize -> deserialize
//             return tt::runtime::ttnn::operations::utils::createDeviceComputeKernelConfig(deviceComputeKernelConfigFB.offset_type); // std::optional<::ttnn::DeviceComputeKernelConfig>
//             // const ::tt::target::ttnn::DeviceComputeKernelConfig * config

//             return a;
//             // return tt::runtime::ttnn::operations::utils::createDeviceComputeKernelConfig(deviceComputeKernelConfig);
//         }


};

TEST_F(ThreeWayConsistencyTest, ComputeKernelConfig) { 
    mlir::tt::ttnn::DeviceComputeKernelConfigAttr baseConfig = mlir::tt::ttnn::DeviceComputeKernelConfigAttr::get(&context);
    RunTest(baseConfig);
}

// } // namespace ttnn
// } // namespace tt
// } // namespace mlir


