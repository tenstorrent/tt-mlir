// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_TYPES_SPSC_QUEUE_H
#define TT_RUNTIME_DETAIL_TTNN_DISTRIBUTED_TYPES_SPSC_QUEUE_H

#include <condition_variable>
#include <mutex>
#include <queue>

namespace tt::runtime::ttnn::distributed {

template <typename T>
class SPSCQueue {
private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<T> queue_;

public:
  SPSCQueue() = default;
  ~SPSCQueue() = default;

  SPSCQueue(const SPSCQueue &) = delete;
  SPSCQueue &operator=(const SPSCQueue &) = delete;
  SPSCQueue(SPSCQueue &&) = delete;
  SPSCQueue &operator=(SPSCQueue &&) = delete;

  // Producer calls this
  void push(const T &item) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(item);
    cv_.notify_one();
  }

  // Consumer calls this - blocks until item available
  T popBlocking() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });

    T result = std::move(queue_.front());
    queue_.pop();
    return result;
  }
};

} // namespace tt::runtime::ttnn::distributed

#endif
