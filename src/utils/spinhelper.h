/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2023 The LCZero Authors

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
#include <cassert>
#include <chrono>
#include <cstddef>
#include <random>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <x86intrin.h>
#endif

namespace lczero {

#if defined(__clang__) || defined(__GNUC__)
  [[gnu::always_inline]]
#elif defined(_MSC_VER)
  __forceinline
#endif
static inline void SpinloopPause() {
#if defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
  __asm__ __volatile__("yield" ::: "memory");
#elif defined(__arch64__) && defined(_MSC_VER)
  __yield();
#else
  asm volatile("");
#endif
}

#if defined(__clang__) || defined(__GNUC__)
  [[gnu::always_inline]]
#elif defined(_MSC_VER)
  __forceinline
#endif
static inline uint64_t GetCurrentCpuCycles() {
#if defined(__x86_64__) || defined(_MSC_VER)
  return __rdtsc();
#elif defined(__aarch64__)
  uint64_t cntpct_el0;
  __asm__ __volatile__("mrs %0, cntpct_el0" : "=r"(cntpct_el0));
  return cntpct_el0;
#else
  // Fallback for unsupported architectures.
  return std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
}

static inline uint64_t GetCyclesInUs() {
#if defined(__x86_64__)
  int64_t start = GetCurrentCpuCycles();
  auto now = std::chrono::high_resolution_clock::now();
  while (std::chrono::high_resolution_clock::now() - now <
         std::chrono::microseconds(50)) {
  }
  int64_t end = GetCurrentCpuCycles();
  return (end - start) / 50;
#elif defined(__aarch64__)
  uint64_t freq;
  __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(freq));
  return freq / 1000000;
#else
  return 1000;
#endif
}

#if defined(__clang__) || defined(__GNUC__)
  [[gnu::always_inline]]
#elif defined(_MSC_VER)
  __forceinline
#endif
static inline size_t GetMaxCycles(size_t spin_limit_ns) {
  // Approximate number of CPU cycles in a microsecond.
  static const size_t kCyclesInUs = GetCyclesInUs();
  return spin_limit_ns * kCyclesInUs / 1000;
}

#if defined(__clang__) || defined(__GNUC__)
  [[gnu::always_inline]]
#elif defined(_MSC_VER)
  __forceinline
#endif
static inline size_t GetThreadYieldLimit() {
  // Approximate number of nanoseconds to yield a thread.
  const auto init_yield = [] {
    const auto hash = std::hash<std::thread::id>{}(std::this_thread::get_id());
    const unsigned min_spins = 50'000;
    const unsigned max_spins = min_spins * 8;
    return min_spins + (hash % (max_spins - min_spins));
  };
  static thread_local size_t kThreadYieldLimit = init_yield();
  return kThreadYieldLimit;
}

template <typename Atomic>
class MonitorHelper {
  using Type = typename Atomic::value_type;
#if defined(__clang__) || defined(__GNUC__)
  [[gnu::always_inline]]
#elif defined(_MSC_VER)
  __forceinline
#endif
  void PrepareForMonitor() const {
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ __volatile__("sevl" ::: "memory");
#else
    // No preparation needed for x86_64.
#endif
  }
  template <typename Func>
#if defined(__clang__) || defined(__GNUC__)
  [[gnu::always_inline]]
#elif defined(_MSC_VER)
  __forceinline
#endif
  bool Monitor(Func check, uint64_t start, uint64_t now, bool high_power_state,
               uint64_t target_cycles) const {
    std::ignore = start;
    std::ignore = now;
    std::ignore = high_power_state;
    std::ignore = target_cycles;
#if defined(__MWAITX__)
    void* addr = const_cast<void*>(static_cast<const void*>(&atomic_));
    static constexpr unsigned kTimeoutFlag =
        0x2;  // Use timeout hint for mwaitx
    static constexpr unsigned kHighPowerHint = 0xf0;
    const uint64_t elapsed_cycles = now - start;
    assert(elapsed_cycles < target_cycles);
    const uint64_t time_to_wait = target_cycles - elapsed_cycles;
    _mm_monitorx(addr, 0, 0);
    if (!check(atomic_.load(std::memory_order_relaxed))) {
      return false;
    }
    _mm_mwaitx(kTimeoutFlag, high_power_state ? kHighPowerHint : 0,
               time_to_wait);
    return check(atomic_.load(std::memory_order_relaxed));
#elif defined(__WAITPKG__)
    void* addr = const_cast<void*>(static_cast<const void*>(&atomic_));
    static constexpr unsigned kHighPowerHint = 0x1;
    const uint64_t end = start + target_cycles;
    _umonitor(addr);
    if (!check(atomic_.load(std::memory_order_relaxed))) {
      return false;
    }
    _umwait(high_power_state ? kHighPowerHint : 0, end);
    return check(atomic_.load(std::memory_order_relaxed));
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    if (!check(atomic_.load(std::memory_order_relaxed))) {
      return false;
    }
    __asm__ __volatile__("wfe\n\t" : : : "memory");
    return check(atomic_.load(std::memory_order_relaxed));
#else
    for (size_t i = 0; i < 32; ++i) {
      SpinloopPause();
      if (!check(atomic_.load(std::memory_order_relaxed))) {
        return false;
      }
    }
    return true;
#endif
  }

  void Yield() const { std::this_thread::yield(); }

 public:
  MonitorHelper(const Atomic& atomic, int spin_limit_ns)
      : atomic_(atomic), kMaxCycles(GetMaxCycles(spin_limit_ns)) {
    assert(spin_limit_ns > 0);
  }

  // Wait until the atomic variable satisfies the condition specified by the
  // check function. Check function should return true if monitoring should
  // continue, and false if the condition is satisfied.
  template <typename Func>
  void operator()(Func check) const {
    if (!check(atomic_.load(std::memory_order_relaxed))) {
      return;
    }
    MonitorWait(check);
  }
 private:
  template <typename Func>
  void MonitorWait(Func check) const {
    uint64_t start_cycles = GetCurrentCpuCycles();
    uint64_t current_cycles = start_cycles;
    const size_t kBusyWaitCycles = GetMaxCycles(500);
    const size_t kHighPowerStateWait = GetMaxCycles(10'000);
    const size_t kYieldWaitCycles =
        std::max(kMaxCycles, GetMaxCycles(GetThreadYieldLimit()));
#if defined(__WAITPKG__)
    // Intel has consistent low cycle wake ups and higher power states. We want
    // to use umonitor/umwait for all spin limits to help SMT performance.
    const bool kUseWaitPkg = true;
#else
    const bool kUseWaitPkg = false;
#endif
    if (kMaxCycles < kBusyWaitCycles && !kUseWaitPkg) {
      // Use busy waiting for very short spin limits, as the wake-up latency and
      // random timer delays are problem for very short spin limits. Less than
      // about 2000 cycles is likely too short.
      for (size_t i = 0; i < 32; ++i) {
        SpinloopPause();
        if (!check(atomic_.load(std::memory_order_relaxed))) {
          return;
        }
      }
      current_cycles = GetCurrentCpuCycles();
    }
    PrepareForMonitor();
    size_t elapsed_cycles = current_cycles - start_cycles;
    bool high_power_state = !kUseWaitPkg && kMaxCycles < kHighPowerStateWait &&
                            elapsed_cycles < kHighPowerStateWait;
    if (elapsed_cycles >= kYieldWaitCycles) {
      Yield();
      current_cycles = start_cycles = GetCurrentCpuCycles();
    }
    while (Monitor(check, start_cycles, current_cycles, high_power_state,
                   high_power_state ? kHighPowerStateWait : kYieldWaitCycles)) {
      current_cycles = GetCurrentCpuCycles();
      if (high_power_state &&
          current_cycles - start_cycles >= kHighPowerStateWait) {
        high_power_state = false;
      }
      if (current_cycles - start_cycles >= kYieldWaitCycles) {
        // Wait is taking longer that expected, yield to make sure lock holders
        // gets scheduled if it isn't running currently.
        Yield();
        current_cycles = start_cycles = GetCurrentCpuCycles();
      }
    }
  }

  const Atomic& atomic_;
  const size_t kMaxCycles;
};

class SpinHelper {
 public:
  virtual ~SpinHelper() = default;
  virtual void Backoff() {}
  virtual void Wait() {}
};

class ExponentialBackoffSpinHelper : public SpinHelper {
 public:
  ExponentialBackoffSpinHelper()
      : backoff_iters_(kMinBackoffIters), spin_to_sleep_iters_(0) {}

  virtual void Backoff() {
    thread_local std::uniform_int_distribution<size_t> distribution;
    thread_local std::minstd_rand generator(std::random_device{}());
    const size_t spin_count = distribution(
        generator, decltype(distribution)::param_type{0, backoff_iters_});

    for (size_t i = 0; i < spin_count; i++) SpinloopPause();

    backoff_iters_ = std::min(2 * backoff_iters_, kMaxBackoffIters);
    spin_to_sleep_iters_ = 0;
  }

  // Spin to sleep
  virtual void Wait() {
    if (spin_to_sleep_iters_ < kMaxSpinToSleepIters) {
      spin_to_sleep_iters_++;
      SpinloopPause();
    } else {
      spin_to_sleep_iters_ = 0;
      std::this_thread::sleep_for(kSleepDuration);
    }
  }

 private:
  static constexpr size_t kMaxSpinToSleepIters = 0x10000;
  static constexpr size_t kMinBackoffIters = 0x20;
  static constexpr size_t kMaxBackoffIters = 0x400;
  static constexpr std::chrono::microseconds kSleepDuration{1000};

  size_t backoff_iters_;
  size_t spin_to_sleep_iters_;
};

}  // namespace lczero
