#pragma once

#include <thread>
#include <vector>
#include <atomic>
#include <functional>
#include <algorithm>
#include <mutex>
#include <condition_variable>

namespace kitteni {

class ThreadPool {
public:
    explicit ThreadPool(int num_threads = 0) {
        if (num_threads <= 0) {
            unsigned int hc = std::thread::hardware_concurrency();
            num_threads = hc > 0 ? static_cast<int>(hc) : 1;
        }
        num_threads_ = num_threads;
        if (num_threads_ > 1) {
            workers_.reserve(num_threads_ - 1);
            for (int t = 1; t < num_threads_; ++t) {
                workers_.emplace_back([this, t]() { worker_loop(t); });
            }
        }
    }

    ~ThreadPool() {
        if (num_threads_ > 1) {
            {
                std::lock_guard<std::mutex> lk(mutex_);
                stop_.store(true, std::memory_order_release);
                generation_.fetch_add(1, std::memory_order_acq_rel);
            }
            cv_start_.notify_all();
            for (auto& w : workers_) {
                if (w.joinable()) w.join();
            }
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    int num_threads() const { return num_threads_; }

    // Execute `fn(int64_t begin, int64_t end, int thread_id)` over `[0, total_items)`
    template <typename Fn>
    void parallel_for(int64_t total_items, Fn&& fn) const {
        if (total_items <= 0) return;
        int nt = static_cast<int>(std::min<int64_t>(num_threads_, total_items));
        if (nt <= 1) {
            fn(0, total_items, 0);
            return;
        }

        const int64_t grain = std::max<int64_t>(1, total_items / (nt * 4));
        next_chunk_.store(0, std::memory_order_relaxed);
        total_items_.store(total_items, std::memory_order_relaxed);
        grain_.store(grain, std::memory_order_relaxed);

        auto task_wrapper = [&](int tid) {
            while (true) {
                int64_t start = next_chunk_.fetch_add(grain, std::memory_order_relaxed);
                if (start >= total_items) break;
                int64_t end = std::min(total_items, start + grain);
                fn(start, end, tid);
            }
        };

        current_task_ = task_wrapper;
        active_workers_.store(num_threads_ - 1, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(mutex_);
            generation_.fetch_add(1, std::memory_order_acq_rel);
        }
        cv_start_.notify_all();

        // Main thread participates as thread 0
        task_wrapper(0);

        // Spin-wait briefly for workers to finish
        while (active_workers_.load(std::memory_order_acquire) > 0) {
            std::this_thread::yield();
        }
        current_task_ = nullptr;
    }

private:
    int num_threads_ = 1;
    mutable std::vector<std::thread> workers_;
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_start_;
    mutable std::atomic<uint64_t> generation_{0};
    mutable std::atomic<int> active_workers_{0};
    mutable std::atomic<bool> stop_{false};

    mutable std::atomic<int64_t> next_chunk_{0};
    mutable std::atomic<int64_t> total_items_{0};
    mutable std::atomic<int64_t> grain_{1};
    mutable std::function<void(int)> current_task_;

    void worker_loop(int tid) {
        uint64_t last_gen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mutex_);
                cv_start_.wait(lk, [&]() {
                    return generation_.load(std::memory_order_acquire) != last_gen ||
                           stop_.load(std::memory_order_acquire);
                });
                if (stop_.load(std::memory_order_acquire)) return;
                last_gen = generation_.load(std::memory_order_relaxed);
            }
            if (current_task_) {
                current_task_(tid);
            }
            active_workers_.fetch_sub(1, std::memory_order_release);
        }
    }
};

} // namespace kitteni

