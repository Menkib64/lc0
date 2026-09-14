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

#pragma once

#include <string_view>
#include <cstdint>

#ifndef _WIN32
#include <sys/types.h>
#endif

namespace lczero {

#ifndef _WIN32
class TimerBacktrace {
public:
  TimerBacktrace(uint64_t delay_ns, std::string_view message);
  ~TimerBacktrace();

  // Start the timer. If the timer is already running, this will reset it.
  void Start();
  // Stop the timer. If the timer is not running, this will do nothing.
  void Stop();

  // Prints the message and backtrace to stderr.
  std::string_view GetMessage() const;

private:
  uint64_t delay_ns_;
  std::string_view message_;
  timer_t timer_id_;
};
#else
class TimerBacktrace {
public:
  TimerBacktrace(uint64_t, std::string_view) {}
  void Start() {}
  void Stop() {}
  std::string_view GetMessage() const { return ""; };
};
#endif

}  // namespace lczero
