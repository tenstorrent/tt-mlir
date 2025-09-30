// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_DISTRIBUTED_TYPES_SPSC_QUEUE_H
#define TT_RUNTIME_DETAIL_DISTRIBUTED_TYPES_SPSC_QUEUE_H

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace tt::runtime::distributed {

template <typename T>
class SPSCQueue {
public:
  SPSCQueue() = default;
  ~SPSCQueue() = default;

  SPSCQueue(const SPSCQueue &) = delete;
  SPSCQueue &operator=(const SPSCQueue &) = delete;
  SPSCQueue(SPSCQueue &&) = delete;
  SPSCQueue &operator=(SPSCQueue &&) = delete;

  void push(const T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(item);
    cv_.notify_one();
  }

  void push(T &&item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(std::move(item));
    cv_.notify_one();
  }

  T popBlocking() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });

    T result = std::move(queue_.front());
    queue_.pop();

    return result;
  }

  std::optional<T> popWithTimeout(const std::chrono::milliseconds &timeout =
                                      std::chrono::milliseconds(100)) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait_for(lock, timeout, [this] { return !queue_.empty(); });

    if (queue_.empty()) {
      return std::nullopt;
    }

    T result = std::move(queue_.front());
    queue_.pop();

    return result;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
  }

private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<T> queue_;
};

} // namespace tt::runtime::distributed

#endif
