#include <gtest/gtest.h>

#include "engine/device.hpp"

namespace {

// Closes the process-wide MeshDevice after all tests run, while the program is
// still live (worker threads + thread-local state intact).
class DeviceTeardownEnvironment : public ::testing::Environment {
public:
    void TearDown() override { ::tt::kurbla::close_runtime_device_mesh(); }
};

// Registered at static-init time, which runs before gtest_main reaches
// RUN_ALL_TESTS — gtest only requires environments be added before then.
const bool s_device_teardown_registered = [] {
    ::testing::AddGlobalTestEnvironment(new DeviceTeardownEnvironment());
    return true;
}();

} // namespace
