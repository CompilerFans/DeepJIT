// 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#pragma once

#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace deep_jit {

// In-memory cache keyed by value type. Concurrent misses for one key run the
// factory once; the other callers wait for its result.
template <typename Key, typename Value>
class MemCache {
    std::unordered_map<Key, std::shared_future<std::shared_ptr<Value>>> pending;
    std::mutex mutex;

public:
    std::unordered_map<Key, std::shared_ptr<Value>> cache;

    template <typename Factory>
    std::shared_ptr<Value> get_or_create(const Key& key, Factory&& factory) {
        std::shared_future<std::shared_ptr<Value>> future;
        std::optional<std::promise<std::shared_ptr<Value>>> promise;
        {
            std::lock_guard lock(mutex);
            if (const auto iterator = cache.find(key); iterator != cache.end())
                return iterator->second;
            if (const auto iterator = pending.find(key); iterator != pending.end()) {
                future = iterator->second;
            } else {
                promise.emplace();
                future = promise->get_future().share();
                pending.emplace(key, future);
            }
        }

        if (not promise)
            return future.get();

        try {
            auto value = factory();
            {
                std::lock_guard lock(mutex);
                cache.emplace(key, value);
                pending.erase(key);
            }
            promise->set_value(value);
            return value;
        } catch (...) {
            {
                std::lock_guard lock(mutex);
                pending.erase(key);
            }
            promise->set_exception(std::current_exception());
            throw;
        }
    }
};

}  // namespace deep_jit
