// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Support/Logger.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cstdlib>
#include <sstream>

using namespace ttmlir;

TEST(LoggerTest, LogLevels) {
  // Test log level string conversions
  EXPECT_STREQ(getLogLevelStr(LogLevel::Trace), "TRACE");
  EXPECT_STREQ(getLogLevelStr(LogLevel::Debug), "DEBUG");
  EXPECT_STREQ(getLogLevelStr(LogLevel::Fatal), "FATAL");
}

TEST(LoggerTest, LogComponents) {
  // Test log component string conversions
  EXPECT_STREQ(getLogComponentStr(LogComponent::Optimizer), "optimizer");
  EXPECT_STREQ(getLogComponentStr(LogComponent::General), "general");
}

TEST(LoggerTest, LogLevelColors) {
  // Test log level color assignments
  EXPECT_EQ(getLogLevelColor(LogLevel::Trace), llvm::raw_ostream::CYAN);
  EXPECT_EQ(getLogLevelColor(LogLevel::Debug), llvm::raw_ostream::GREEN);
  EXPECT_EQ(getLogLevelColor(LogLevel::Fatal), llvm::raw_ostream::RED);
}

TEST(LoggerTest, Timestamp) {
  std::string timestamp = getCurrentTimestamp();

  // Verify timestamp format: YYYY-MM-DD HH:MM:SS.mmm
  EXPECT_EQ(timestamp.length(), 23);
  EXPECT_EQ(timestamp[4], '-');
  EXPECT_EQ(timestamp[7], '-');
  EXPECT_EQ(timestamp[10], ' ');
  EXPECT_EQ(timestamp[13], ':');
  EXPECT_EQ(timestamp[16], ':');
  EXPECT_EQ(timestamp[19], '.');
}

TEST(LoggerTest, LoggingOutput) {
  // Enable debug logging for both optimizer and general components
  llvm::DebugFlag = true;
  const char *debugTypes[] = {"general", "optimizer", nullptr};
  llvm::setCurrentDebugTypes(debugTypes, 2);

  // Set log level to TRACE to see all messages
  setenv("TTMLIR_LOGGER_LEVEL", "TRACE", 1);

  // Start capturing stderr because that's where the logs go
  testing::internal::CaptureStderr();

  // Test all log levels with the optimizer component
  TTMLIR_TRACE(LogComponent::Optimizer,
               "This is a trace message with value: {0}", 42);
  TTMLIR_DEBUG(LogComponent::Optimizer,
               "Debug message with two values: {0}, {1}", "hello", 3.14);

  // Test the general component
  TTMLIR_DEBUG(LogComponent::General, "Debug with general component");
  TTMLIR_TRACE(LogComponent::General, "Trace with general component");

  // Switch to only general component
  llvm::setCurrentDebugType("general");

  // These should not be visible since optimizer component is disabled
  TTMLIR_TRACE(LogComponent::Optimizer, "This trace should not be visible");
  TTMLIR_DEBUG(LogComponent::Optimizer, "This debug should not be visible");

  // This should be visible since general component is enabled
  TTMLIR_DEBUG(LogComponent::General, "This debug should be visible");

  // Get the captured output
  std::string outputBuffer = testing::internal::GetCapturedStderr();

#ifdef TTMLIR_ENABLE_DEBUG_LOGS
  // Verify output contains expected messages
  EXPECT_TRUE(outputBuffer.find("trace message with value: 42") !=
              std::string::npos);
  EXPECT_TRUE(outputBuffer.find("Debug message with two values: hello, 3.14") !=
              std::string::npos);
  EXPECT_TRUE(outputBuffer.find("Debug with general component") !=
              std::string::npos);
  EXPECT_TRUE(outputBuffer.find("Trace with general component") !=
              std::string::npos);
  EXPECT_TRUE(outputBuffer.find("This debug should be visible") !=
              std::string::npos);

  // Verify messages that should not appear
  EXPECT_TRUE(outputBuffer.find("This trace should not be visible") ==
              std::string::npos);
  EXPECT_TRUE(outputBuffer.find("This debug should not be visible") ==
              std::string::npos);
#else
  EXPECT_TRUE(outputBuffer.empty());
#endif
}
