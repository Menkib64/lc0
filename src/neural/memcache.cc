/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "neural/memcache.h"
#include <atomic>

#include "neural/shared_params.h"
#include "utils/atomic.h"
#include "utils/atomic_vector.h"
#include "utils/cache.h"

namespace lczero {
namespace {

// TODO For now it uses the hash of the current position, ignoring repetitions
// and history. We'll likely need to have configurable hash function that we'll
// also reuse as a tree hash key.
uint64_t ComputeEvalPositionHash(const EvalPosition& pos) {
  return pos.pos.back().Hash();
}

struct CachedValue {
  CachedValue() : p{}, state(UNITIALIZED), num_moves(0) {}
  CachedValue(const EvalPosition& pos) {
    num_moves = pos.legal_moves.size();
    p.reset(pos.legal_moves.empty() ? nullptr
                                    : new float[pos.legal_moves.size()]);
  }

  ~CachedValue() { reset(); }

  CachedValue(CachedValue&& other) noexcept
      : p(std::move(other.p)),
        state(other.state.load(std::memory_order_acquire)),
        q(other.q),
        d(other.d),
        m(other.m),
        e(other.e),
        num_moves(other.num_moves) {
    assert(state.load(std::memory_order_acquire) != UNITIALIZED || !p);
    other.state.store(UNITIALIZED, std::memory_order_release);
    assert(!other.p);
  }

  void reset() {
    assert(state.load(std::memory_order_relaxed) == UNITIALIZED ||
           state.load(std::memory_order_relaxed) == NOT_QUEUED ||
           state.load(std::memory_order_relaxed) == READY);
    p.reset();
    state.store(UNITIALIZED, std::memory_order_release);
  }

  explicit operator bool() const {
    return state.load(std::memory_order_acquire) != UNITIALIZED;
  }
  // State transitions happen atomically using release and acquire semantics for
  // dependant reads and writes. The state progresses in order. Each transition
  // must happen only once which requires compare and exchange. Secondary
  // readers must wait for READY state to read the cached value.
  enum State {
    UNITIALIZED,
    NOT_QUEUED,  // Initial state before NN submission.
    PREPARED,    // Prepared to submit to NN. It might be cancelled if minibatch
                 // is full.
    NO_WAITERS,  // One thread has taken this position to be evaluated. None is
                 // yet waiting for the result.
    WAITERS,     // Another thread is waiting for results. Setting READY state
                 // must be followed by notify_all to wake up waiters.
    READY,       // The value is ready. Waiters can read the value and proceed.
  };
  std::unique_ptr<float[]> p;
  WaitableAtomic<State> state = NOT_QUEUED;
  float q;
  float d;
  float m;
  float e;
  uint8_t num_moves;
};

void CachedValueToEvalResult(const CachedValue& cv, const EvalResultPtr& ptr) {
  if (ptr.d) *ptr.d = cv.d;
  if (ptr.q) *ptr.q = cv.q;
  if (ptr.m) *ptr.m = cv.m;
  if (ptr.e) *ptr.e = cv.e;
  assert(cv.num_moves >= ptr.p.size());
  std::copy(cv.p.get(), cv.p.get() + ptr.p.size(), ptr.p.begin());
}

class MemCache : public CachingBackend {
 public:
  MemCache(std::unique_ptr<Backend> wrapped, const OptionsDict& options,
           float max_out_of_order_evals_factor, size_t threads)
      : wrapped_backend_(std::move(wrapped)),
        cache_(2 * options.Get<int>(SharedBackendParams::kNNCacheSizeId) *
                   cache_.GetItemStructSize() / cache_.GetItemsInStructure(),
               threads),
        max_batch_size_(wrapped_backend_->GetAttributes().maximum_batch_size *
                        (1.0f + max_out_of_order_evals_factor)) {}

  BackendAttributes GetAttributes() const override {
    return wrapped_backend_->GetAttributes();
  }
  std::unique_ptr<BackendComputation> CreateComputation() override;
  std::optional<EvalResult> GetCachedEvaluation(const EvalPosition&) override;

  void ClearCache() override { cache_.Clear(); }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    auto ret = wrapped_backend_->UpdateConfiguration(options);
    if (ret == Backend::UPDATE_OK) {
      // Check if we need to clear the cache.
      if (!wrapped_backend_->IsSameConfiguration(options)) {
        cache_.Clear();
      }
    }
    return ret;
  }

  bool IsSameConfiguration(const OptionsDict& options) const override {
    return wrapped_backend_->IsSameConfiguration(options);
  }

  void SetCacheSize(size_t size, size_t threads) override {
    cache_.SetCapacity(
        2 * size * cache_.GetItemStructSize() / cache_.GetItemsInStructure(),
        threads);
  }

 private:
  std::unique_ptr<Backend> wrapped_backend_;
  HashKeyedCache<CachedValue> cache_;
  const size_t max_batch_size_;
  friend class MemCacheComputation;
};

class MemCacheComputation : public BackendComputation {
 public:
  MemCacheComputation(std::unique_ptr<BackendComputation> wrapped_computation,
                      MemCache* memcache)
      : wrapped_computation_(std::move(wrapped_computation)),
        memcache_(memcache),
        entries_(memcache->max_batch_size_) {}

 private:
  size_t UsedBatchSize() const override { return entries_.size(); }
  virtual AddInputResult AddInput(const EvalPosition& pos,
                                  EvalResultPtr result) override {
    assert(pos.legal_moves.size() == result.p.size() || result.p.empty());
    const uint64_t hash = ComputeEvalPositionHash(pos);
    bool to_be_queued = false;
    EvalResultPtr result_ptr;
    std::unique_ptr<CachedValue> value;

    CachedValue* cached = std::get<0>(memcache_->cache_.Emplace(hash, pos));

    // Sometimes search queries NN without passing the legal moves. It is
    // still cached in this case, but in subsequent queries we only return it
    // if legal moves are not passed again. Otherwise check the size to guard
    // against hash collisions.
    if (cached &&
        (pos.legal_moves.empty() ||
         (cached->p && cached->num_moves == pos.legal_moves.size()))) {
      value.reset();
      auto state = cached->state.load(std::memory_order_acquire);
      while (state == CachedValue::NOT_QUEUED || state == CachedValue::PREPARED) {
        if (state == CachedValue::PREPARED) {
          MonitorHelper monitor(cached->state, 1000);
          monitor([](CachedValue::State s) {
              return s == CachedValue::PREPARED;
              });
        }
        state = CachedValue::NOT_QUEUED;
        if (cached->state.compare_exchange_weak(state, CachedValue::PREPARED,
                                                std::memory_order_acq_rel)) {
          to_be_queued = true;
          break;
        }
      }
      if (state == CachedValue::READY) {
        CachedValueToEvalResult(*cached, result);
        return AddInputResult::FETCHED_IMMEDIATELY;
      }
      result_ptr = EvalResultPtr{
          &cached->q, &cached->d, &cached->m, &cached->e,
          cached->p ? std::span<float>{cached->p.get(), pos.legal_moves.size()}
                    : std::span<float>{}};
    } else {
      // No space, hash collision, or value was removed after insert.
      cached = nullptr;
      value = std::make_unique<CachedValue>(pos);
    }
    if (!cached) {
      to_be_queued = true;
      result_ptr = EvalResultPtr{
          &value->q, &value->d, &value->m, &value->e,
          value->p ? std::span<float>{value->p.get(), pos.legal_moves.size()}
                   : std::span<float>{}};
    } else {
      assert(cached->state.load(std::memory_order_acquire) !=
             CachedValue::UNITIALIZED);
    }

    // If another thread is already computing the value, we'll fetch it in
    // ComputeBlocking.
    AddInputResult rv = AddInputResult::FETCHED_DELAYED;
    if (to_be_queued) {
      rv = wrapped_computation_->AddInput(pos, result_ptr);
    }
    if (rv == AddInputResult::MINIBATCH_FULL) {
      // Cancel the queued state if minibatch is full.
      if (cached) {
        assert(cached->state.load(std::memory_order_relaxed) ==
               CachedValue::PREPARED);
        cached->state.store(CachedValue::NOT_QUEUED, std::memory_order_release);
      }
      return rv;
    }
    if (cached && to_be_queued) {
      assert(cached->state.load(std::memory_order_relaxed) ==
             CachedValue::PREPARED);
      cached->state.store(CachedValue::NO_WAITERS, std::memory_order_release);
    }
    entries_.emplace_back(
        Entry{cached, std::move(value), result, to_be_queued});
    return rv;
  }

  virtual void ComputeBlocking(ComputationCallback callback) override {
    if (wrapped_computation_->UsedBatchSize() != 0) {
      wrapped_computation_->ComputeBlocking(callback);
    } else {
      callback(ComputationEvent::FIRST_BACKEND_IDLE);
    }
    // Process results from our branch.
    for (auto& entry : entries_) {
      if (entry.queued_for_eval) {
        if (entry.value) {
          // There is no cache entry.
          assert(!entry.cached);
          CachedValueToEvalResult(*entry.value, entry.result_ptr);
        } else {
          // There is a cache entry.
          auto& lock = entry.cached;
          assert(lock);
          CachedValueToEvalResult(*lock, entry.result_ptr);
          auto state = lock->state.exchange(CachedValue::READY,
                                            std::memory_order_release);
          assert(state == CachedValue::NO_WAITERS ||
                 state == CachedValue::WAITERS);
          // Wake up waiters if there are any,
          if (state == CachedValue::WAITERS) {
            lock->state.notify_all();
          }
        }
      }
    }
    // Process results from other batches which we got through cache before
    // results were ready.
    for (auto& entry : entries_) {
      if (!entry.queued_for_eval) {
        auto& lock = entry.cached;
        assert(lock);
        auto state = lock->state.load(std::memory_order_acquire);
        assert(state == CachedValue::NO_WAITERS ||
               state == CachedValue::WAITERS ||
               state == CachedValue::READY);
        // Make sure writing side knows about waiters
        if (state == CachedValue::NO_WAITERS) {
          lock->state.compare_exchange_strong(state, CachedValue::WAITERS,
                                              std::memory_order_acquire);
        }
        // Wait until the value is ready.
        lock->state.wait(CachedValue::WAITERS, std::memory_order_acquire);
        assert(lock->state.load(std::memory_order_acquire) ==
               CachedValue::READY);
        CachedValueToEvalResult(*lock, entry.result_ptr);
      }
    }
  }

  struct Entry {
    CachedValue* cached;
    std::unique_ptr<CachedValue> value;
    EvalResultPtr result_ptr;
    bool queued_for_eval = false;
  };

  std::unique_ptr<BackendComputation> wrapped_computation_;
  MemCache* memcache_;
  AtomicVector<Entry> entries_;
};

std::unique_ptr<BackendComputation> MemCache::CreateComputation() {
  return std::make_unique<MemCacheComputation>(
      wrapped_backend_->CreateComputation(), this);
}
std::optional<EvalResult> MemCache::GetCachedEvaluation(
    const EvalPosition& pos) {
  const uint64_t hash = ComputeEvalPositionHash(pos);
  auto* cached = cache_.LookupAndPin(hash);
  if (!cached ||
      cached->state.load(std::memory_order_acquire) != CachedValue::READY ||
      (!pos.legal_moves.empty() &&
       !(cached->p && cached->num_moves == pos.legal_moves.size()))) {
    return std::nullopt;
  }
  EvalResult result;
  result.d = cached->d;
  result.q = cached->q;
  result.m = cached->m;
  result.e = cached->e;
  if (cached->p) {
    result.p.reserve(pos.legal_moves.size());
    std::copy(cached->p.get(), cached->p.get() + pos.legal_moves.size(),
              std::back_inserter(result.p));
  }
  return result;
}

}  // namespace

std::unique_ptr<CachingBackend> CreateMemCache(
    std::unique_ptr<Backend> wrapped, const OptionsDict& options,
    float max_out_of_order_evals_factor, size_t threads) {
  return std::make_unique<MemCache>(std::move(wrapped), options,
                                    max_out_of_order_evals_factor, threads);
}

}  // namespace lczero
