// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#if !defined(TT_ASSERT_DISABLE_ASSERTS)

#include "llvm/Support/raw_ostream.h"

#include "testing/Utils.h"

#include <memory>
#include <string>

// The tests here check that customizing TT_ASSERT_REPORT_STREAM() and
// TT_ASSERT_FAILURE() can be used to avoid aborting the process on an assert
// failure and/or to re-route failure messaging to a custom stream.
//
// For that reason, we must define a few things before #including "Asserts.h" in
// the translation unit.
//
// See "DeathTestAsserts.cpp" for a version of this test suite using "default"
// assert behavior.

namespace {
struct MockErrStream {

  MockErrStream() : os(std::make_unique<llvm::raw_string_ostream>(msg)) {}

  llvm::raw_ostream &out() { return (*os); }

  void reset() {
    os->flush();
    msg.clear();
    os = std::make_unique<llvm::raw_string_ostream>(msg);
  }

  std::string msg;
  std::unique_ptr<llvm::raw_string_ostream> os;
};
} // namespace

inline MockErrStream &stream() {
  static MockErrStream instance;

  return instance;
}

#define TT_ASSERT_REPORT_STREAM() stream().out()
#define TT_ASSERT_FAILURE() ((void)0)
#include "ttmlir/Asserts.h"

#include <regex>
#include <string_view>

class AssertsTest : public testing::Test {
protected:
  AssertsTest() : mocks(&stream()) {}

  const std::string &msg() { return mocks->msg; }

  void check(const std::string &re) {
    ASSERT_TRUE(std::regex_search(msg(), std::regex(re)))
        << "expected to find [" << re << "] in assertion message [" << msg()
        << "]";
  }

  void reset() { mocks->reset(); }

  MockErrStream *mocks;
};

#define TT_STANDARD_ASSERT_MSG_PREFIX(condition)                               \
  "Assertion `" #condition "` failed"                                          \
  ".*"

TEST_F(AssertsTest, BinaryExprDecomposition) {
  {
    reset();

    int32_t x = 1;
    int32_t y = 2;
    TT_assert(x == y);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(x == y));
    check("1 == 2");
  }
  {
    reset();

    int32_t x = 1;
    int32_t y = 2;
    TT_assert(x > y);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(x > y));
    check("1 > 2");
  }
  {
    reset();

    int32_t x = 1;
    int32_t y = 2;
    TT_assert(y <= x);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(y <= x));
    check("2 <= 1");
  }
  // With custom formatted context.
  {
    reset();

    int32_t x = 1;
    int32_t y = 2;
    TT_assertv(x == y, "x-y was {}", x - y);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(x == y));
    check("x-y was -1");
  }
  // Strings and views.
  {
    reset();

    std::string s = "hello";
    TT_assert(s != "hello");

    check(TT_STANDARD_ASSERT_MSG_PREFIX(s != "hello"));
    // check("s != \"hello\"");
    check("hello != hello");
  }
  {
    reset();

    using namespace std::literals;

    auto sa = "abcdef"sv;
    const std::string sb = "asdf";
    TT_assert(sa == sb);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(sa == sb));
    check("abcdef == asdf");
  }
  // Literals.
  {
    reset();

    // Note that the stringifier macro will capture the source
    // text but the runtime message will show the evaluated
    // literal. This is ok.

    TT_assert(2 + 2 == 3);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(2 \\+ 2 == 3));
    check("4 == 3");
  }
  // Chars.
  {
    reset();

    // TODO(vroubtsov) better consistency handling 1-byte int types (e.g,
    // uint8_t).

    int8_t a = 'A';
    char b = 'B';
    int8_t c = b;
    TT_assertv(a == b,
               "hmm, I think a, b, and c were {}, {}, and {}, respectively", a,
               b, c);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(a == b));
    check("hmm, I think a, b, and c were A, B, and B, respectively");
  }
  // llvm streams have issues printing std::nullptr_t, but TT_assert*()s don't.
  {
    reset();

    int32_t x = 42;
    void *y = &x;
    TT_assert(y == nullptr);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(y == nullptr));
    check("0x.+ == nullptr");
  }
}

TEST_F(AssertsTest, IntegralRangeChecks) {
  {
    reset();

    int32_t y = 20;
    TT_assert_limit(y, 20);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(in_open_range\\(y, 0, 20\\)));
    check(R"(y \(20\) is not in \[0, 20\))");
  }
  {
    reset();

    int32_t y = 20;

    // These don't get triggered.

    TT_assert_open_range(y, 20, 21);
    TT_assert_inclusive_range(y, 20, 20);
    TT_assert_exclusive_range(y, 19, 21);

    ASSERT_TRUE(msg().empty());
  }
  {
    reset();

    int32_t y = 2;
    TT_assert_open_range(y, 0, 2);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(in_open_range\\(y, 0, 2\\)));
    check(R"(y \(2\) is not in \[0, 2\))");
  }
  {
    reset();

    int32_t y = 2;
    TT_assert_inclusive_range(y, 0, 1);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(in_inclusive_range\\(y, 0, 1\\)));
    check(R"(y \(2\) is not in \[0, 1\])");
  }
  {
    reset();

    int32_t y = 2;
    TT_assert_exclusive_range(y, 2, 3);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(in_exclusive_range\\(y, 2, 3\\)));
    check(R"(y \(2\) is not in \(2, 3\))");
  }
}

TEST_F(AssertsTest, ChainedExprRecipes) {
  // Recipe for chained expressions (automatic decomposition isn't supported).
  {
    reset();

    int32_t x = 1;
    int32_t y = 2;
    int32_t z = 3;
    TT_assert((x > y && y < z));

    check(TT_STANDARD_ASSERT_MSG_PREFIX(\\(x > y && y < z\\)));
  }
  // Custom context is another way to see runtime values.
  {
    reset();

    int32_t x = 1;
    int32_t y = 2;
    int32_t z = 3;
    TT_assertv((x > y && y < z), "was {} > {} && {} < {}", x, y, y, z);

    check(TT_STANDARD_ASSERT_MSG_PREFIX(\\(x > y && y < z\\)));
    check("was 1 > 2 && 2 < 3");
  }
}

namespace {

struct PrintableType {
  int32_t x;
  int32_t y;

  friend bool operator<(const PrintableType &lhs, const PrintableType &rhs) {
    return (lhs.x < rhs.x) && (lhs.y < rhs.y);
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const PrintableType &obj) {
    return os << "{x = " << obj.x << ", y = " << obj.y << '}';
  }
};

struct NonprintableType {
  int32_t x;
  int32_t y;

  friend bool operator==(const NonprintableType &lhs,
                         const NonprintableType &rhs) {
    return (lhs.x == rhs.x) && (lhs.y == rhs.y);
  }
};

} // namespace

// TODO(vroubtsov) ideally, implementing either to_string or operator<<
// should make the type usable both for decomposed messages and custom
// context suffixes, but right now that works only for operator<<

TEST_F(AssertsTest, CustomPrintableTypes) {
  {
    reset();

    PrintableType small{123, 456};
    PrintableType big{1230, 4560};
    TT_assert(big < small);

    check(TT_STANDARD_ASSERT_MSG_PREFIX((big < small)));
    check(R"(\{x = 1230, y = 4560\} < \{x = 123, y = 456\})");
  }
  {
    reset();

    PrintableType small{123, 456};
    PrintableType big{1230, 4560};
    TT_assertv(big < small, "something's wrong with {} and {}", big, small);

    check(TT_STANDARD_ASSERT_MSG_PREFIX((big < small)));
    check(
        R"(something's wrong with \{x = 1230, y = 4560\} and \{x = 123, y = 456\})");
  }
}

// Types that can't be printed in a discoverable way should still compile
// and the assert message will still contain the stringified `condition`.

TEST_F(AssertsTest, NonprintableTypes) {
  {
    reset();

    NonprintableType one{123, 456};
    NonprintableType another{1230, 4560};
    TT_assert(one == another);

    check(TT_STANDARD_ASSERT_MSG_PREFIX((one == another)));
  }
}

namespace {
struct NoncopyableType {
  int32_t x;
  int32_t y;

  NoncopyableType(const NoncopyableType &) = delete;
  NoncopyableType &operator=(const NoncopyableType &) = delete;

  NoncopyableType(NoncopyableType &&) = delete;
  NoncopyableType &operator=(NoncopyableType &&) = delete;

  friend bool operator<(const NoncopyableType &lhs,
                        const NoncopyableType &rhs) {
    return (lhs.x < rhs.x) && (lhs.y < rhs.y);
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                       const NoncopyableType &obj) {
    return os << "{x = " << obj.x << ", y = " << obj.y << '}';
  }
};
} // namespace

// Confirm that our lhs/rhs captures don't trigger any extra copies.

TEST_F(AssertsTest, NoncopyableType) {
  {
    reset();

    NoncopyableType small{123, 456};
    NoncopyableType big{1230, 4560};
    TT_assert(big < small);

    check(TT_STANDARD_ASSERT_MSG_PREFIX((big < small)));
  }
}

#undef TT_STANDARD_ASSERT_MSG_PREFIX

#endif // TT_ASSERT_DISABLE_ASSERTS
