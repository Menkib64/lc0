/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>

#if defined(__x86_64__)
#include <emmintrin.h>
#endif

#include "utils/cppattributes.h"
#include "utils/spinhelper.h"

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(thread_sanitizer) || defined(__SANITIZE_THREAD__)
#define TSAN_BUILD 1
#include <sanitizer/tsan_interface.h>
#endif

namespace lczero {

// Implementation of reader-preferenced shared mutex. Based on fair shared
// mutex.
class CAPABILITY("mutex") RpSharedMutex {
 public:
  RpSharedMutex() : waiting_readers_(0) {}

  void lock() ACQUIRE() {
    while (true) {
      mutex_.lock();
      if (waiting_readers_ == 0) return;
      mutex_.unlock();
    }
  }
  void unlock() RELEASE() { mutex_.unlock(); }
  void lock_shared() ACQUIRE_SHARED() {
    ++waiting_readers_;
    mutex_.lock_shared();
  }
  void unlock_shared() RELEASE_SHARED() {
    --waiting_readers_;
    mutex_.unlock_shared();
  }

 private:
  std::shared_timed_mutex mutex_;
  std::atomic<int> waiting_readers_;
};

// std::mutex wrapper for clang thread safety annotation.
class CAPABILITY("mutex") Mutex {
 public:
  // std::unique_lock<std::mutex> wrapper.
  class SCOPED_CAPABILITY Lock {
   public:
    Lock(Mutex& m) ACQUIRE(m) : lock_(m.get_raw()) {}
    ~Lock() RELEASE() {}
    std::unique_lock<std::mutex>& get_raw() { return lock_; }

    void lock() ACQUIRE() { lock_.lock(); }
    void unlock() RELEASE() { lock_.unlock(); }
    bool owns_lock() const { return lock_.owns_lock(); }

   private:
    std::unique_lock<std::mutex> lock_;
  };

  void lock() ACQUIRE() { mutex_.lock(); }
  void unlock() RELEASE() { mutex_.unlock(); }
  std::mutex& get_raw() { return mutex_; }

 private:
  std::mutex mutex_;
};

// std::shared_mutex wrapper for clang thread safety annotation.
class CAPABILITY("mutex") SharedMutex {
 public:
  // std::unique_lock<std::shared_mutex> wrapper.
  class SCOPED_CAPABILITY Lock {
   public:
    Lock(SharedMutex& m) ACQUIRE(m) : lock_(m.get_raw()) {}
    ~Lock() RELEASE() {}

   private:
    std::unique_lock<std::shared_timed_mutex> lock_;
  };

  // std::shared_lock<std::shared_mutex> wrapper.
  class SCOPED_CAPABILITY SharedLock {
   public:
    SharedLock(SharedMutex& m) ACQUIRE_SHARED(m) : lock_(m.get_raw()) {}
    ~SharedLock() RELEASE() {}

   private:
    std::shared_lock<std::shared_timed_mutex> lock_;
  };

  void lock() ACQUIRE() { mutex_.lock(); }
  void unlock() RELEASE() { mutex_.unlock(); }
  void lock_shared() ACQUIRE_SHARED() { mutex_.lock_shared(); }
  void unlock_shared() RELEASE_SHARED() { mutex_.unlock_shared(); }

  std::shared_timed_mutex& get_raw() { return mutex_; }

 private:
  std::shared_timed_mutex mutex_;
};

// A very simple spin lock.
class CAPABILITY("mutex") SpinMutex {
 public:
  // std::unique_lock<SpinMutex> wrapper.
  class SCOPED_CAPABILITY Lock {
   public:
    Lock(SpinMutex& m) ACQUIRE(m) : lock_(m) {}
    ~Lock() RELEASE() {}

    void unlock() RELEASE() { lock_.unlock(); }
    void lock() ACQUIRE() { lock_.lock(); }

    bool owns_lock() const { return lock_.owns_lock(); }

   private:
    std::unique_lock<SpinMutex> lock_;
  };

#if TSAN_BUILD
  SpinMutex() { __tsan_mutex_create(&mutex_, 0); }
#endif
#if TSAN_BUILD
  ~SpinMutex() { __tsan_mutex_destroy(&mutex_, 0); }
#endif

#if defined(__clang__) || defined(__GNUC__)
  [[gnu::always_inline]]
#elif defined(_MSC_VER)
  __forceinline
#endif
  void lock() ACQUIRE() {
#if TSAN_BUILD
    __tsan_mutex_pre_lock(&mutex_, 0);
#endif
    int val = 0;
    if (mutex_.compare_exchange_strong(val, 1, std::memory_order_acquire))
        [[likely]] {
#if TSAN_BUILD
      __tsan_mutex_post_lock(&mutex_, 0, 0);
#endif
      return;
    }
    SpinMutexSlowLock();
  }

 private:
  void SpinMutexSlowLock() {
    // Slow contention path. We use random spin count with occasional yield to
    // avoid starvation.
    const auto get_spin_count = []() {
      const auto hash =
          std::hash<std::thread::id>{}(std::this_thread::get_id());
      const unsigned min_spins = 8;
      const unsigned max_spins = min_spins + 16;
      return min_spins + (hash % (max_spins - min_spins));
    };

    static const thread_local unsigned spin_time_us = get_spin_count();

    MonitorHelper monitor(mutex_, spin_time_us);

    while (true) {
      monitor([](int value) { return value != 0; });
      int val = 0;
      if (mutex_.compare_exchange_weak(val, 1, std::memory_order_acquire)) {
#if TSAN_BUILD
        __tsan_mutex_post_lock(&mutex_, 0, 0);
#endif
        break;
      }
    }
  }

 public:
#if defined(__clang__) || defined(__GNUC__)
  [[gnu::always_inline]]
#elif defined(_MSC_VER)
  __forceinline
#endif
  bool try_lock() TRY_ACQUIRE(true) {
#if TSAN_BUILD
    __tsan_mutex_pre_lock(&mutex_, __tsan_mutex_try_lock);
#endif
    int val = 0;
    if (mutex_.compare_exchange_strong(val, 1, std::memory_order_acquire))
        [[likely]] {
#if TSAN_BUILD
      __tsan_mutex_post_lock(&mutex_, __tsan_mutex_try_lock, 0);
#endif
      return true;
    }
#if TSAN_BUILD
    __tsan_mutex_post_lock(&mutex_, __tsan_mutex_try_lock_failed, 0);
#endif
    return false;
  }

#if defined(__clang__) || defined(__GNUC__)
  [[gnu::always_inline]]
#elif defined(_MSC_VER)
  __forceinline
#endif
  void unlock() RELEASE() {
#if TSAN_BUILD
    __tsan_mutex_pre_unlock(&mutex_, 0);
#endif
    mutex_.store(0, std::memory_order_release);
#if TSAN_BUILD
    __tsan_mutex_post_unlock(&mutex_, 0);
#endif
  }

 private:
  std::atomic<int> mutex_{0};
};

}  // namespace lczero
