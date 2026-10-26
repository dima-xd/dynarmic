/* This file is part of the dynarmic project.
 * Copyright (c) 2018 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <dynarmic/common/spin_lock.h>

namespace Dynarmic {

using VAddr = std::uint64_t;
using Vector = std::array<std::uint64_t, 2>;

class ExclusiveMonitor {
public:
    /// @param processor_count Maximum number of processors using this global
    ///                        exclusive monitor. Each processor must have a
    ///                        unique id.
    explicit ExclusiveMonitor(size_t processor_count);

    size_t GetProcessorCount() const;

    /// Marks a region containing [address, address+size) to be exclusive to
    /// processor processor_id.
    template<typename T, typename Function>
    T ReadAndMark(size_t processor_id, VAddr address, Function op) {
        static_assert(std::is_trivially_copyable_v<T>);
        const VAddr masked_address = address & RESERVATION_GRANULE_MASK;

        // The guest's LDREX, and no lock. Each processor owns its slot in these
        // arrays, so nothing here is shared, and the reservation only has to
        // record what this processor read. See CheckAndClear() for why no other
        // processor's slot has to be looked at.
        __atomic_store_n(&exclusive_addresses[processor_id], masked_address, __ATOMIC_RELAXED);
        const T value = op();
        std::memcpy(exclusive_values[processor_id].data(), &value, sizeof(T));
        return value;
    }

    /// Checks to see if processor processor_id has exclusive access to the
    /// specified region and, if it does, executes the operation.
    ///
    /// The store behind `op` is an atomic compare-exchange of the guest word
    /// (the host's MemoryWriteExclusive* callbacks), so a reservation is lost
    /// exactly when that word changed -- the condition the guest's lock-free
    /// code depends on. Upstream walked every processor slot here and held a
    /// process-wide spinlock around both this and ReadAndMark, so with dozens of
    /// guest threads every atomic increment waited for every other thread's
    /// atomics.
    template<typename T, typename Function>
    bool DoExclusiveOperation(size_t processor_id, VAddr address, Function op) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!CheckAndClear(processor_id, address)) {
            return false;
        }

        T saved_value;
        std::memcpy(&saved_value, exclusive_values[processor_id].data(), sizeof(T));
        return op(saved_value);
    }

    /// Unmark everything.
    void Clear();
    /// Unmark processor id
    void ClearProcessor(size_t processor_id);

private:
    bool CheckAndClear(size_t processor_id, VAddr address);

    void Lock();
    void Unlock();

    friend volatile int* GetExclusiveMonitorLockPointer(ExclusiveMonitor*);
    friend size_t GetExclusiveMonitorProcessorCount(ExclusiveMonitor*);
    friend VAddr* GetExclusiveMonitorAddressPointer(ExclusiveMonitor*, size_t index);
    friend Vector* GetExclusiveMonitorValuePointer(ExclusiveMonitor*, size_t index);

    static constexpr VAddr RESERVATION_GRANULE_MASK = 0xFFFF'FFFF'FFFF'FFFFull;
    static constexpr VAddr INVALID_EXCLUSIVE_ADDRESS = 0xDEAD'DEAD'DEAD'DEADull;
    SpinLock lock;
    std::vector<VAddr> exclusive_addresses;
    std::vector<Vector> exclusive_values;
};

}  // namespace Dynarmic
