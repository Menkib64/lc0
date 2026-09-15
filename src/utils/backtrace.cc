/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include "utils/backtrace.h"

#include "backtrace_config.h"
#include "utils/exception.h"

#if USE_TIMER_BACKTRACE
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <signal.h>

#include <atomic>
#include <system_error>
#endif

namespace lczero {

#if USE_TIMER_BACKTRACE
namespace {

class BacktraceSignalHandler {
 public:
  BacktraceSignalHandler() {
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = &BacktraceSignalHandler::HandleSignal;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGUSR1, &sa, nullptr) == -1) {
      throw Exception("Failed to set signal handler for debug backtraces: " +
                      std::make_error_code(std::errc(errno)).message());
    }
  }

  struct BacktracePointers {
    std::array<uintptr_t, 128> pointers;
    int size;
  };

  static thread_local BacktracePointers backtrace_pointers_;
  static std::atomic<const BacktracePointers*> backtrace_pointers_ptr_;

  class BacktracePrinterThread {
    static constexpr BacktracePointers kEndThreadSignal{};

   public:
    BacktracePrinterThread() : thread_(&BacktracePrinterThread::Run, this) {
      struct sigevent sev;
      sev.sigev_notify = SIGEV_SIGNAL;
      sev.sigev_signo = SIGUSR1;
      sev.sigev_value.sival_ptr = nullptr;
      if (timer_create(CLOCK_MONOTONIC, &sev, &timer_id_) != 0) {
        throw Exception("Failed to create timer for debug backtraces: " +
                        std::make_error_code(std::errc(errno)).message());
      }
    }

    ~BacktracePrinterThread() {
      backtrace_pointers_ptr_.store(&kEndThreadSignal,
                                    std::memory_order_release);
      backtrace_pointers_ptr_.notify_one();
      if (thread_.joinable()) {
        thread_.join();
      }
    }

   private:
    void Run() {
      while (true) {
        backtrace_pointers_ptr_.wait(nullptr, std::memory_order_acquire);
        const BacktracePointers* backtrace_pointers =
            backtrace_pointers_ptr_.load(std::memory_order_acquire);
        if (backtrace_pointers == &kEndThreadSignal) {
          break;
        }
        if (backtrace_pointers == nullptr) {
          continue;
        }
        BacktracePointers backtrace{*backtrace_pointers};
        backtrace_pointers_ptr_.compare_exchange_strong(
            backtrace_pointers, nullptr, std::memory_order_release);
        CERR << "Backtrace (depth " << backtrace.size << "):";
        // Print the backtrace to stderr.
        for (int i = 0; i < backtrace.size; ++i) {
          unw_word_t ip = backtrace.pointers[i];
          unw_word_t offset = 0;
          std::array<char, 256> symbols;

          Dl_info info;
          if (dladdr(reinterpret_cast<void*>(ip), &info) == 0) {
            CERR << "  #" << i << " " << std::hex << ip << ": <unmapped>";
            continue;
          } else if (info.dli_fname != nullptr && info.dli_sname != nullptr) {
            CERR << "  #" << i << " " << std::hex << ip << ": "
                 << info.dli_sname << " + 0x" << std::hex
                 << (ip - reinterpret_cast<uintptr_t>(info.dli_saddr)) << " ("
                 << info.dli_fname << " + 0x" << std::hex
                 << (ip - reinterpret_cast<uintptr_t>(info.dli_fbase)) << ")";
            continue;
          }

          int rv =
              unw_get_proc_name_by_ip(unw_local_addr_space, ip, symbols.data(),
                                      symbols.size(), &offset, nullptr);

          if (rv == 0) {
            if (info.dli_fname != nullptr) {
              CERR << "  #" << i << " " << std::hex << ip << ": "
                   << symbols.data() << " (" << info.dli_fname << " + 0x"
                   << std::hex
                   << (ip - reinterpret_cast<uintptr_t>(info.dli_fbase)) << ")";
            } else {
              CERR << "  #" << i << " " << std::hex << ip << ": "
                   << symbols.data() << " + 0x" << std::hex << offset;
            }
          } else {
            if (info.dli_fname != nullptr) {
              CERR << "  #" << i << " " << std::hex << ip << ": <unknown> ("
                   << info.dli_fname << " + 0x" << std::hex
                   << (ip - reinterpret_cast<uintptr_t>(info.dli_fbase)) << ")";
            } else {
              CERR << "  #" << i << " " << std::hex << ip << ": <unknown>";
            }
          }
        }
        // Use timer for a delayed crash to allow backtrace from all slow
        // threads.
        struct itimerspec its;
        its.it_value.tv_sec = 0;
        its.it_value.tv_nsec = 200'000'000;  // 200ms
        its.it_interval.tv_sec = 0;
        its.it_interval.tv_nsec = 0;
        if (timer_settime(timer_id_, 0, &its, nullptr) == -1) {
          throw Exception("Failed to start timer for debug backtraces: " +
                          std::make_error_code(std::errc(errno)).message());
        }
      }
    }

    std::thread thread_;
    timer_t timer_id_;
  };

  static BacktracePrinterThread backtrace_printer_thread_;

  static void HandleSignal(int, siginfo_t* info, void* context) {
    // Print the backtrace to stderr.
    TimerBacktrace* timer_backtrace =
        reinterpret_cast<TimerBacktrace*>(info->si_value.sival_ptr);
    if (timer_backtrace == nullptr) {
      abort();
    }
    {
      auto message = timer_backtrace->GetMessage();
      write(STDERR_FILENO, message.data(), message.size());
    }
    ucontext_t* ucontext = reinterpret_cast<ucontext_t*>(context);

    uintptr_t first_ip = ucontext->uc_mcontext.gregs[REG_RIP];

    unw_cursor_t cursor;

    if (unw_init_local2(&cursor, ucontext, UNW_INIT_SIGNAL_FRAME) < 0) {
      const char message[] = "Failed to initialize libunwind cursor\n";
      write(STDERR_FILENO, message, sizeof(message) - 1);
      return;
    }

    size_t depth = 0;
    int err = 0;
    backtrace_pointers_.pointers[depth++] = first_ip;
    while ((err = unw_step(&cursor)) > 0 &&
           depth < backtrace_pointers_.pointers.size()) {
      unw_word_t ip;
      if (unw_get_reg(&cursor, UNW_REG_IP, &ip) < 0) {
        const char message[] = "Failed to get instruction pointer\n";
        write(STDERR_FILENO, message, sizeof(message) - 1);
        break;
      }
      backtrace_pointers_.pointers[depth++] = static_cast<uintptr_t>(ip);
    }

    backtrace_pointers_.size = depth;
    std::array<char, 128> message;
    int len = snprintf(message.data(), message.size(),
                       "Captured depth: %zu err: %d\n", depth, err);
    if (len > 0) {
      write(STDERR_FILENO, message.data(), len);
    }
    const BacktracePointers* expected = nullptr;
    if (backtrace_pointers_ptr_.compare_exchange_strong(
            expected, &backtrace_pointers_, std::memory_order_release)) {
      // Notify the printer thread that a backtrace is available.
      backtrace_pointers_ptr_.notify_one();
    }
  }
};

thread_local BacktraceSignalHandler::BacktracePointers
    BacktraceSignalHandler::backtrace_pointers_{};
std::atomic<const BacktraceSignalHandler::BacktracePointers*>
    BacktraceSignalHandler::backtrace_pointers_ptr_{nullptr};
BacktraceSignalHandler::BacktracePrinterThread
    BacktraceSignalHandler::backtrace_printer_thread_{};

}  // namespace

TimerBacktrace::TimerBacktrace(uint64_t delay_ns, std::string_view message)
    : delay_ns_(delay_ns), message_(message) {
  assert(delay_ns_ < 1000000000 && "Delay must be less than 1 second.");
  static BacktraceSignalHandler backtrace_signal_handler;
  struct sigevent sev;
  sev.sigev_notify = SIGEV_THREAD_ID;
  sev.sigev_signo = SIGUSR1;
  sev.sigev_value.sival_ptr = reinterpret_cast<void*>(this);
  sev._sigev_un._tid = gettid();
  if (timer_create(CLOCK_MONOTONIC, &sev, &timer_id_) == -1) {
    throw Exception("Failed to create timer for debug backtraces: " +
                    std::make_error_code(std::errc(errno)).message());
  }
}

TimerBacktrace::~TimerBacktrace() { timer_delete(timer_id_); }

void TimerBacktrace::Start() {
  struct itimerspec its;
  its.it_value.tv_sec = 0;
  its.it_value.tv_nsec = delay_ns_;
  its.it_interval.tv_sec = 0;
  its.it_interval.tv_nsec = 0;
  if (timer_settime(timer_id_, 0, &its, nullptr) == -1) {
    throw Exception("Failed to start timer for debug backtraces: " +
                    std::make_error_code(std::errc(errno)).message());
  }
}

void TimerBacktrace::Stop() {
  struct itimerspec its;
  its.it_value.tv_sec = 0;
  its.it_value.tv_nsec = 0;
  its.it_interval.tv_sec = 0;
  its.it_interval.tv_nsec = 0;
  if (timer_settime(timer_id_, 0, &its, nullptr) == -1) {
    throw Exception("Failed to stop timer for debug backtraces: " +
                    std::make_error_code(std::errc(errno)).message());
  }
}

std::string_view TimerBacktrace::GetMessage() const { return message_; }

#elif !defined(_WIN32)

TimerBacktrace::TimerBacktrace(uint64_t delay_ns, std::string_view message)
    : delay_ns_(delay_ns), message_(message) {
  assert(delay_ns_ < 1000000000 && "Delay must be less than 1 second.");
}

TimerBacktrace::~TimerBacktrace() {
  // No-op on non-Windows platforms without libunwind.
}

void TimerBacktrace::Start() {
  // No-op on non-Windows platforms without libunwind.
}

void TimerBacktrace::Stop() {
  // No-op on non-Windows platforms without libunwind.
}

std::string_view TimerBacktrace::GetMessage() const { return message_; }

#endif

}  // namespace lczero
