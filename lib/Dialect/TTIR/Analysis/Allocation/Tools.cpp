// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TTIR/Analysis/Allocation/Tools.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <sstream>

//===---------------------------------------------------------------------===//
namespace mlir::tt::ttir::allocation {

namespace {

using std::int32_t;
using std::int64_t;
using std::uint64_t;

int64_t readUntilEOF(std::istream &src, std::ostream &dst) {
  int64_t readCount = 0;
  std::array<char, 1024> buf;

  while (src) {
    src.read(buf.data(), buf.size());
    const auto rc = src.gcount();
    dst.write(buf.data(), rc);
    readCount += rc;
  }

  return readCount;
}

std::string readAsString(std::istream &in) {
  std::stringstream dst;
  readUntilEOF(in, dst);
  return dst.str();
}

template <typename IntegerSet>
void parseArrayAsIntegerSet(const llvm::json::Array &array, IntegerSet &out) {
  for (auto &v : array) {
    out.insert(*v.getAsInteger());
  }
}

using Space = PlannerImpl::Space;

std::optional<Space> parseSpace(llvm::StringRef s) {
  if ("scratch" == s) {
    return Space::Scratch;
  }
  if ("spill" == s) {
    return Space::Spill;
  }
  if ("NA" == s) {
    return Space::NA;
  }
  return {};
}

using NodeID = int32_t;

struct Node {

  llvm::SmallSet<NodeID, 2> adj{};
  NodeID ID;

  Node(NodeID ID) : ID(ID) {}

  friend std::ostream &operator<<(std::ostream &os, const Node &obj) {
    os << obj.ID << " {";
    bool first = true;
    for (const auto v : obj.adj) {
      if (first) {
        first = false;
      } else {
        os << ", ";
      }
      os << v;
    }
    return os << '}';
  }
};

struct Graph {
  llvm::SmallVector<std::unique_ptr<Node>> nodes; // A "stable vector".

  NodeID nodeCount() const { return nodes.size(); }

  const Node &node(NodeID ID) const {
    TT_assert_limit(ID, nodeCount());
    return (*nodes[ID]);
  }

  Node &node(NodeID ID) {
    return const_cast<Node &>(const_cast<const Graph *>(this)->node(ID));
  }

  Node &addNode() {
    const NodeID id = nodeCount();
    return *nodes.emplace_back(std::make_unique<Node>(id));
  }

  bool addEdge(Node &u, const Node &v) { return u.adj.insert(v.ID).second; }

  bool addEdge(NodeID u, NodeID v) {
    TT_assertv(u != v, "self-edges are't allowed");
    return addEdge(node(u), node(v));
  }

  Tools::Problem generateProblem(const llvm::SmallVector<NodeID> &order,
                                 double bindFraction, uint64_t seed) {
    llvm::SmallVector<int32_t> mapping(nodeCount());
    for (int32_t i = 0; i < static_cast<int32_t>(order.size()); ++i) {
      mapping[order[i]] = i;
    }

    llvm::DenseMap<NodeID, std::array<int32_t, 2>> liveness{};

    for (const NodeID v : order) {
      const Node &n = node(v);
      int32_t first = mapping[n.ID];
      int32_t last = first;
      for (const NodeID v : n.adj) {
        last = std::max(last, mapping[node(v).ID]);
      }

      liveness[n.ID] = {first, last};
    }

    // Each Node results in a variable conceptually representing the node's
    // "output", a value that is described by two different groups of requests,
    // representing "scratch" and "spill" memory requirements, correspondingly.

    Tools::Problem problem;

    std::mt19937_64 gen{seed};
    auto unif = [&gen](int32_t a, int32_t b) {
      return std::uniform_int_distribution<int32_t>(a, b)(gen);
    };
    auto unifDouble = [&gen]() {
      return std::uniform_real_distribution<>(0.0, 1.0)(gen);
    };

    for (const NodeID v : order) {

      const int32_t streamBufSize = unif(2, 1009) * 32;
      const int32_t tensorDataSize = 16 * streamBufSize;

      problem.def([&](PlannerImpl::VariableBuilder &b) {
        {
          const std::array<int32_t, 2> live = liveness.find(v)->second;

          b.request(Space::Scratch, tensorDataSize, live[0], live[1]);
        }
        {
          for (const NodeID succ : node(v).adj) {
            const std::array<int32_t, 2> live = liveness.find(succ)->second;
            const int32_t t = live[1];

            b.request(Space::Spill, streamBufSize, t, t);
          }
        }
        if (unifDouble() < bindFraction) {
          b.bind(Space::Scratch);
        }
      });
    }

    return problem;
  }

  [[maybe_unused]] friend std::ostream &operator<<(std::ostream &os,
                                                   const Graph &obj) {
    for (const auto &n : obj.nodes) {
      os << "  " << *n << "\n";
    }
    return os;
  }

  llvm::SmallVector<NodeID> reversePostOrder(const Node &start) const {
    llvm::SmallVector<NodeID> rpo;
    rpo.reserve(nodes.size());

    llvm::DenseSet<NodeID> visited{};
    postOrderVisit(start, rpo, visited);

    std::reverse(rpo.begin(), rpo.end());
    return rpo;
  }

  void postOrderVisit(const Node &n, llvm::SmallVector<NodeID> &rpo,
                      llvm::DenseSet<NodeID> &visited) const {
    visited.insert(n.ID);
    for (const NodeID v : n.adj) {
      if (visited.count(v) == 0) {
        postOrderVisit(*nodes[v], rpo, visited);
      }
    }
    rpo.emplace_back(n.ID);
  }
};

} // namespace

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const Tools::GenerateCfg &obj) {
  for (std::size_t i = 0; i < obj.segments.size(); ++i) {
    const auto &s = obj.segments[i];
    os << '(' << s.neckLength << ", " << s.conflictCount << ')';
  }
  os << ", bind fraction " << obj.bindFraction << ", seed " << obj.seed;
  return os;
}

Tools::Problem Tools::generate(const GenerateCfg &cfg) {
  TT_assertv(!cfg.segments.empty(), "expected a non-empty list of segments");

  Graph g;

  Node *start = nullptr;
  Node *end = nullptr;

  for (std::size_t s = 0; s < cfg.segments.size(); ++s) {
    const auto &parms = cfg.segments[s];
    const int32_t h = parms.neckLength;
    TT_assert(h > 0);
    const int32_t c = parms.conflictCount;
    TT_assert(c > 0);

    llvm::SmallVector<std::reference_wrapper<Node>> hNodes{};

    for (int32_t i = 0; i < h; ++i) {
      Node &n = hNodes.emplace_back(g.addNode());
      if (i) {
        g.addEdge(hNodes[i - 1], n);
      }
    }

    llvm::SmallVector<std::reference_wrapper<Node>> cNodes{};

    for (int32_t i = 0; i < 2 * c; ++i) {
      cNodes.emplace_back(g.addNode());
    }
    for (int32_t i = 0; i < c; ++i) {
      g.addEdge(cNodes[i], cNodes[2 * c - 1 - i]); // "vertical"

      g.addEdge(cNodes[i], cNodes[i + 1]);
      if (i < c - 1) {
        g.addEdge(cNodes[2 * c - 2 - i], cNodes[2 * c - 1 - i]);
      }
    }

    // Connect "h" and "c" pieces.

    g.addEdge(hNodes.back(), cNodes.front());

    if (s == 0) {
      // Capture start of 'g'.
      start = &hNodes.front().get();
    } else {
      // Connect previous segment to the current one.
      g.addEdge(*end, hNodes.front());
    }
    end = &cNodes.back().get();
  }

  auto order = g.reversePostOrder(*start);
  return g.generateProblem(order, cfg.bindFraction, cfg.seed);
}

llvm::Expected<Tools::Problem> Tools::read(std::istream &in) {
  TT_assertv(in.good(), "couldn't read input stream");
  const std::string content = readAsString(in);

  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(content);
  if (auto e = parsed.takeError()) {
    return e;
  }

  llvm::json::Object *obj = parsed->getAsObject();
  TT_assert(obj != nullptr);

  llvm::json::Array *variables = obj->get("variables")->getAsArray();
  llvm::json::Array *requests = obj->get("requests")->getAsArray();

  Tools::Problem problem;

  for (llvm::json::Value v : *requests) {
    llvm::json::Array &req = *v.getAsArray();
    TT_assert(req.size() == 5u);

    const PlannerImpl::IndexT varIndex = *req[0].getAsInteger();
    const PlannerImpl::SequenceT first = *req[1].getAsInteger();
    const PlannerImpl::SequenceT last = *req[2].getAsInteger();
    const PlannerImpl::AllocSizeT size = *req[3].getAsInteger();
    const PlannerImpl::AllocSizeT offset = *req[4].getAsInteger();

    problem.requests.emplace_back(
        PlannerImpl::VarRequest{{{first, last}, size, offset}, varIndex});
  }

  for (llvm::json::Value v : *variables) {
    llvm::json::Object *var = v.getAsObject();

    PlannerImpl::Variable variable;

    if (auto placement = parseSpace(*var->get("placement")->getAsString())) {
      variable.placement = *placement;
    } else {
      return llvm::createStringError("failed to parse 'placement'");
    }

    if (auto bound = var->get("bound")->getAsBoolean()) {
      if (*bound) {
        problem.bound.insert(problem.variables.size());
      }
    } else {
      return llvm::createStringError("failed to parse 'bound'");
    }

    if (llvm::json::Object *domain = var->get("domain")->getAsObject()) {
      for (Space space = Space::begin; space < Space::end; ++space) {
        llvm::json::Array *indices =
            domain->get(to_string(space))->getAsArray();
        TT_assert(indices != nullptr);

        parseArrayAsIntegerSet(*indices, variable.domain[ordinal(space)]);
      }
    } else {
      return llvm::createStringError("failed to parse 'domain'");
    }

    problem.variables.emplace_back(std::move(variable));
  }

  TT_assert(problem.valid());
  return problem;
}

void Tools::write(const Tools::Problem &problem, std::ostream &out) {
  TT_assertv(out.good(), "couldn't write to output stream");

  int32_t width = 0;
  auto indent = [&width]() { return std::string(width << 1, ' '); };

  out << indent() << '{' << "\n";
  {
    ++width;
    out << indent() << "\"variables\" : " << '[' << "\n";
    {
      ++width;
      for (std::size_t vi = 0; vi < problem.variables.size(); ++vi) {
        const auto &v = problem.variables[vi];

        out << indent() << '{';
        out << "\"placement\" : " << '"' << to_string(v.placement) << '"';
        out << ", ";
        out << "\"bound\" : "
            << (problem.bound.contains(vi) ? "true" : "false");
        out << ", ";
        out << "\"domain\" : ";

        out << '{';
        {
          for (Space space = Space::begin; space < Space::end; ++space) {
            if (space != Space::begin) {
              out << ", ";
            }
            // TODO(vroubtsov) "inline" requests inside variables instead of
            // using indices into "requests"?
            out << '"' << to_string(space) << '"' << " : ";
            out << '[';
            {
              bool first = true;
              for (const auto ri : v.domain[ordinal(space)]) {
                if (first) {
                  first = false;
                } else {
                  out << ", ";
                }
                out << ri;
              }
            }
            out << ']';
          }
        }
        out << '}';

        out << '}';
        if (vi + 1 < problem.variables.size()) {
          out << ',';
        }
        out << "\n";
      }
      --width;
    }
    out << indent() << "],"
        << "\n";

    out << indent() << "\"requests\" : " << '[' << "\n";
    {
      ++width;
      for (std::size_t ri = 0; ri < problem.requests.size(); ++ri) {
        const auto &r = problem.requests[ri];
        out << indent() << '[' << r.varIndex << ", " << r.first << ", "
            << r.last << ", " << r.size << ", " << r.offset << ']';
        if (ri + 1 < problem.requests.size()) {
          out << ',';
        }
        out << "\n";
      }
      --width;
    }
    out << indent() << ']' << "\n";

    --width;
  }
  out << indent() << '}' << "\n";
}

llvm::Expected<Tools::Problem> Tools::read(const std::filesystem::path &file) {
  std::ifstream in(file);
  TT_assertv(in.is_open(), "couldn't open output file [{}]", file);

  return read(in);
}

void Tools::write(const Tools::Problem &problem,
                  const std::filesystem::path &file) {
  TT_assertv(problem.valid(), "expected a valid problem");

  std::ofstream out(file);
  TT_assertv(out.is_open(), "couldn't open output file [{}]", file);

  return write(problem, out);
}

} // namespace mlir::tt::ttir::allocation
//===---------------------------------------------------------------------===//
