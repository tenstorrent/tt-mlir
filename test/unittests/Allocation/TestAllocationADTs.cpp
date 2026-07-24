// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/D2M/Analysis/Allocation/FastIntervalTree.h"

#include "testing/Utils.h"
#include "ttmlir/Dialect/D2M/Analysis/Allocation/Utils.h"

#include <ratio>
#include <unordered_set>

namespace mlir::tt::d2m::allocation {

namespace gtest = ::testing;

using std::int16_t;
using std::int32_t;
using std::int64_t;

namespace {

template <typename Tree>
void validate(const Tree &tree) {
  int64_t count = 0;

#if defined(TT_BUILD_DEBUG)
  tree.check(); // Internal consistency checker.
#endif

  using IntervalT = typename Tree::IntervalT;
  using IntervalSet = std::unordered_set<const IntervalT *>;

  // Collect an independent list of all intervals in `tree`, used
  // by further validation logic:
  IntervalSet all;
  for (const auto &iv : tree.intervals()) {
    ASSERT_TRUE(all.insert(&iv).second);
  }
  ASSERT_EQ(all.size(), static_cast<std::size_t>(tree.size()));

  using SequenceT = typename Tree::SequenceT;

  // Validate that iteration over all intervals is in non-decreasing order
  // of intervals' left endpoints:
  SequenceT leftPrev = -1;
  for (const auto &iv : tree.intervals()) {
    ASSERT_GE(iv.left, leftPrev);
    leftPrev = iv.left;
  }

  // Brute-force sample every domain value (note that we don't want
  // to loop until 2^31 or whatever, so this is done only by tests
  // that set tight domain limits):
  for (typename Tree::SequenceT q = 0; q < tree.limit(); ++q) {
    const auto results = tree.query(q);

    // Validate that the query returned a unique list of correct intervals:
    IntervalSet unique;
    for (const IntervalT *iv : results) {
      ++count;
      ASSERT_TRUE(iv->stabs(q)) << " query returned interval " << *iv
                                << " that doesn't stab point " << q;
      ASSERT_TRUE(unique.insert(iv).second)
          << " query contains duplicate interval objects";
    }

    // Validate that no intervals were missed by the query:
    for (const auto &iv : all) {
      if (!unique.count(iv)) {
        ASSERT_FALSE(iv->stabs(q))
            << " query missed interval " << *iv << " that stabs point " << q;
      }
    }

    // Validate that results are in lexicographic order:
    for (std::size_t i = /* ! */ 1; i < results.size(); ++i) {
      ASSERT_TRUE(!(results[i] < results[i - 1]))
          << "expected " << results[i - 1] << " <=lex " << results[i];
    }
  }

  if constexpr (Tree::hasIntervalIntersectionAPI) {
    // Brute-force sample every interval in the domain:
    for (typename Tree::SequenceT left = 0; left < tree.limit(); ++left) {
      for (typename Tree::SequenceT right = left; right < tree.limit();
           ++right) {
        const auto results = tree.query(left, right);

        // Validate that the query returned a unique list of correct intervals:
        IntervalSet unique;
        for (const IntervalT *iv : results) {
          ++count;
          ASSERT_TRUE(iv->overlaps(left, right))
              << " query returned interval " << *iv
              << " that doesn't intersect interval [" << left << ", " << right
              << ']';
          ASSERT_TRUE(unique.insert(iv).second)
              << " query contains duplicate interval objects";
        }

        // Validate that no intervals were missed by the query:
        for (const auto &iv : all) {
          if (!unique.count(iv)) {
            ASSERT_FALSE(iv->overlaps(left, right))
                << " query missed interval " << *iv
                << " that intersects interval [" << left << ", " << right
                << ']';
          }
        }

        // Validate that results are in lexicographic order:
        for (std::size_t i = /* ! */ 1; i < results.size(); ++i) {
          ASSERT_TRUE(!(results[i] < results[i - 1]))
              << "expected " << results[i - 1] << " <=lex " << results[i];
        }
      }
    }
  }

  // Make sure the test doesn't pass trivially because it failed to retrieve any
  // data.
  if (tree.size() > 0) {
    EXPECT_GT(count, 0) << "expected some query results";
  }
}

} // namespace
//===----------------------------------------------------------------------===//

TEST(FastIntervalTreeTest, Construction) {
  using Tree = FastIntervalTree<>;

  // An empty tree is valid.
  {
    constexpr auto limit = 100;

    Tree tree(limit);
    tree.create();

    ASSERT_EQ(tree.limit(), limit);
    ASSERT_EQ(tree.size(), 0);

    for (Tree::SequenceT q = 0; q < tree.limit(); ++q) {
      const auto results = tree.query(q);
      ASSERT_TRUE(results.empty());
    }
  }
  // Intervals are closed and can be length-one.
  {
    constexpr auto limit = 100;

    Tree tree(limit);
    {
      tree.insert(10, 10);

      tree.insert(49, 50);
      tree.insert(50, 50);
      tree.insert(50, 51);

      tree.insert(limit - 1, limit - 1);
    }
    tree.create();

    ASSERT_EQ(tree.limit(), limit);
    ASSERT_EQ(tree.size(), 5);

    validate(tree);
  }
  // It is valid to insert duplicate intervals.
  {
    constexpr auto limit = 3;

    Tree tree(limit);
    {
      for (int32_t i = 0; i < 100; ++i) {
        tree.insert(0, 1);
        tree.insert(1, 2);
        tree.insert(0, 2);
      }
    }
    tree.create();

    ASSERT_EQ(tree.limit(), limit);
    ASSERT_EQ(tree.size(), 300);

    validate(tree);
  }
  // There is no cost to declaring very large domain limits, as long as
  // actually used interval endpoints are in some compact [0, extent) range.
  {
    constexpr auto limit = std::numeric_limits<Tree::SequenceT>::max() / 2;

    Tree tree(limit);
    {
      tree.insert(10, 11);
      tree.insert(20, 21);
      tree.insert(30, 31);
    }
    tree.create();

    ASSERT_EQ(tree.limit(), limit);
    ASSERT_EQ(tree.size(), 3);

    EXPECT_EQ(tree.query(0).size(), 0u);

    EXPECT_EQ(tree.query(10).size(), 1u);
    EXPECT_EQ(tree.query(20).size(), 1u);
    EXPECT_EQ(tree.query(30).size(), 1u);

    EXPECT_EQ(tree.query(limit - 1).size(), 0u);
  }
}

TEST(FastIntervalTreeTest, PayloadValueT) {
  using Tree = FastIntervalTree<std::string>;

  // Use a tree with some non-empty payload ValueT:
  {
    constexpr auto limit = 10;

    Tree tree(limit);
    {
      tree.insert(0, 1, "this is [0, 1]");
      tree.insert(0, 2, "this is [0, 2]");
    }
    tree.create();

    ASSERT_EQ(tree.size(), 2);

    auto results = tree.query(0);
    ASSERT_EQ(results.size(), 2u);

    for (const auto &iv : results) {
      switch (iv->right) {
      case 1:
        EXPECT_EQ(iv->value, "this is [0, 1]");
        break;
      case 2:
        EXPECT_EQ(iv->value, "this is [0, 2]");
        break;
      }
    }
  }
}
//===----------------------------------------------------------------------===//

struct IntervalGenUnif {

  template <typename T, typename RNG>
  static std::tuple<T, T> generate(T limit, RNG &rng) {
    using U = std::make_unsigned_t<T>;

    auto runif = std::uniform_int_distribution<U>{0, static_cast<U>(limit - 1)};

    const T a = runif(rng);
    const T b = runif(rng);
    if (a <= b) {
      return {a, b};
    }
    return {b, a};
  }
};

template <typename Gamma>
struct IntervalGenExp {

  static constexpr double gamma = static_cast<double>(Gamma::num) / Gamma::den;

  template <typename T, typename RNG>
  static std::tuple<T, T> generate(T limit, RNG &rng) {
    using U = std::make_unsigned_t<T>;

    auto runif = std::uniform_int_distribution<U>{0, static_cast<U>(limit - 1)};
    auto rexp = std::exponential_distribution<double>{gamma};

    const T left = runif(rng);
    const T length = static_cast<T>(rexp(rng));

    return {left, std::min(left + length, limit - 1)};
  }
};

template <
    // clang-format off
    bool IntervalIntersectionAPI,
    typename IntervalGen,
    typename Sequence = int32_t,
    typename Index = int32_t
    // clang-format on
    >
struct FastIntervalTreeTestScenario
    : public FastIntervalTreeTraits<IntervalIntersectionAPI> {
  using IntervalGenT = IntervalGen;
  using SequenceT = Sequence;
  using IndexT = Index;
};

using FastIntervalTreeTestScenarios = gtest::Types<
    // clang-format off
    FastIntervalTreeTestScenario<false, IntervalGenUnif>,
    FastIntervalTreeTestScenario<false, IntervalGenExp<std::ratio<1, 10>>>,
    FastIntervalTreeTestScenario<false, IntervalGenExp<std::ratio<1, 1>>>,

    FastIntervalTreeTestScenario<true,  IntervalGenUnif>,
    FastIntervalTreeTestScenario<true,  IntervalGenExp<std::ratio<1, 10>>>,
    FastIntervalTreeTestScenario<true,  IntervalGenExp<std::ratio<1, 1>>>,

    FastIntervalTreeTestScenario<true,  IntervalGenUnif, int64_t, int16_t>,
    FastIntervalTreeTestScenario<true,  IntervalGenUnif, int16_t, int64_t>
    // clang-format on
    >;

template <typename T>
struct FastIntervalTreeTest : public gtest::Test {};
TYPED_TEST_SUITE(FastIntervalTreeTest, FastIntervalTreeTestScenarios,
                 /* clang variadic macro issue workaround */);

TYPED_TEST(FastIntervalTreeTest, Randomized) {
  using Scenario = TypeParam; // test parameter

  using Traits = Scenario;
  using TreeT = FastIntervalTree<EmptyValue, Traits>;

  using IntervalGenT = typename Scenario::IntervalGenT;
  using SequenceT = typename Traits::SequenceT;
  using IndexT = typename Traits::IndexT;

  static constexpr bool doIntervalIntersectionAPI =
      Scenario::hasIntervalIntersectionAPI;

  auto seed = tt::testing::randomSeed();
  auto rng = tt::testing::createRNG(seed);

  // `doIntervalIntersectionAPI` used to avoid exhaustive O(N^2)
  // interval checks for large N:

  const std::vector<SequenceT> limits = {
      1, 109, (doIntervalIntersectionAPI ? 256 : 1024)};

  for (SequenceT limit : limits) {
    const std::vector<IndexT> sizes = {
        0, 1, 19, static_cast<IndexT>(limit / 3),
        static_cast<IndexT>(doIntervalIntersectionAPI ? (2 * limit)
                                                      : (5 * limit))};

    for (IndexT size : sizes) {
      TT_TEST_DEBUG("testing with domain limit {}, size {} ...", limit, size);

      TreeT tree(limit);
      {
        for (IndexT i = 0; i < size; ++i) {
          const auto iv = IntervalGenT::generate(limit, rng);
          tree.insert(std::get<0>(iv), std::get<1>(iv));
        }
      }
      tree.create();

      ASSERT_EQ(tree.limit(), limit);
      ASSERT_EQ(tree.size(), size);

      validate(tree);
    }
  }
}

} // namespace mlir::tt::d2m::allocation
//----------------------------------------------------------------------------
