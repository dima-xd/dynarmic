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
        // Bumping the granule is what upstream did by clearing every other
        // processor's slot for it, and it is the one part a compare-exchange
        // cannot stand in for: two loads of a word are not exclusive, so the
        // first reservation has to be stale by the time the second one exists.
        exclusive_generations[processor_id] = MarkReservation(masked_address);
        const T value = op();
        std::memcpy(exclusive_values[processor_id].data(), &value, sizeof(T));
        return value;
    }

    /// Checks to see if processor processor_id has exclusive access to the
    /// specified region and, if it does, executes the operation.
    ///
    /// The store behind `op` is an atomic compare-exchange of the guest word
    /// (the host's MemoryWriteExclusive* callbacks), so a store that changed the
    /// word loses the reservation there. The per-granule generation closes the
    /// two holes a compare-exchange cannot see: a word that another load has
    /// taken a reservation on since, and a word that went through a write and
    /// back to the value this processor read. On a reference count the second
    /// one is a lost increment -- and a reference count that loses an increment
    /// on a shared string is a use-after-free the guest then touches as text.
    template<typename T, typename Function>
    bool DoExclusiveOperation(size_t processor_id, VAddr address, Function op) {
        static_assert(std::is_trivially_copyable_v<T>);
        if (!CheckAndClear(processor_id, address)) {
            return false;
        }
        const VAddr masked_address = address & RESERVATION_GRANULE_MASK;
        if (GenerationChanged(masked_address, exclusive_generations[processor_id])) {
            return false;
        }
        // Everything already reserved on this granule is stale from here on, no
        // matter what the store writes: a later write that restores the old
        // value must not let an older reservation through.
        BumpGeneration(masked_address);

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

    // A generation counter per reservation granule, hashed into a fixed table.
    // Two things bump it: a reservation being taken (the load is not exclusive,
    // so an older reservation of the same word is stale from then on) and an
    // exclusive store being about to happen. A collision makes a reservation
    // fail needlessly, which the architecture allows and the guest retries.
    //
    // The key is the word index plus one, so no address needs a "belongs to no
    // one" special case.
    static constexpr size_t GENERATION_SLOTS = 8192;

    static size_t GenerationSlot(std::uint32_t key) {
        return static_cast<size_t>((key * 2654435761u) >> 19);
    }

    static std::uint32_t GenerationKey(VAddr address) {
        return static_cast<std::uint32_t>(address >> 2) + 1u;
    }

    std::uint32_t MarkReservation(VAddr address) {
        const std::uint32_t key = GenerationKey(address);
        const size_t slot = GenerationSlot(key);
        if (__atomic_load_n(&generation_keys[slot], __ATOMIC_ACQUIRE) == key) {
            // Someone is holding this word already: the count moves so that
            // reservation can no longer complete.
            __atomic_add_fetch(&generation_values[slot], 1u, __ATOMIC_ACQ_REL);
        } else {
            __atomic_store_n(&generation_keys[slot], key, __ATOMIC_RELEASE);
        }
        return __atomic_load_n(&generation_values[slot], __ATOMIC_ACQUIRE);
    }

    std::uint32_t BumpGeneration(VAddr address) {
        const std::uint32_t key = GenerationKey(address);
        const size_t slot = GenerationSlot(key);
        __atomic_store_n(&generation_keys[slot], key, __ATOMIC_RELEASE);
        // The count is never reset when the slot changes owner: a generation a
        // guest recorded earlier must never come back, and a counter that
        // restarted at zero would let exactly that happen.
        return __atomic_add_fetch(&generation_values[slot], 1u, __ATOMIC_ACQ_REL);
    }

    bool GenerationChanged(VAddr address, std::uint32_t generation) const {
        const std::uint32_t key = GenerationKey(address);
        const size_t slot = GenerationSlot(key);
        if (__atomic_load_n(&generation_keys[slot], __ATOMIC_ACQUIRE) != key) {
            return true;
        }
        return __atomic_load_n(&generation_values[slot], __ATOMIC_ACQUIRE) != generation;
    }

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
    std::vector<std::uint32_t> exclusive_generations;
    std::array<std::uint32_t, GENERATION_SLOTS> generation_keys{};
    std::array<std::uint32_t, GENERATION_SLOTS> generation_values{};
};

}  // namespace Dynarmic
