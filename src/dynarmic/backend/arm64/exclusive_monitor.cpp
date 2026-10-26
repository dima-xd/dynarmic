/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/interface/exclusive_monitor.h"

#include <algorithm>

#include <mcl/assert.hpp>

namespace Dynarmic {

ExclusiveMonitor::ExclusiveMonitor(size_t processor_count)
        : exclusive_addresses(processor_count, INVALID_EXCLUSIVE_ADDRESS), exclusive_values(processor_count) {}

size_t ExclusiveMonitor::GetProcessorCount() const {
    return exclusive_addresses.size();
}

void ExclusiveMonitor::Lock() {
    lock.Lock();
}

void ExclusiveMonitor::Unlock() {
    lock.Unlock();
}

bool ExclusiveMonitor::CheckAndClear(size_t processor_id, VAddr address) {
    const VAddr masked_address = address & RESERVATION_GRANULE_MASK;

    // Only this processor's own slot is examined. Invalidating every other
    // processor's reservation - and taking a process-wide lock to do it - is
    // how upstream models a store to the granule, but the host's
    // MemoryWriteExclusive* callbacks already perform that store as an atomic
    // compare-exchange of the guest word, so a reservation that should be lost
    // fails there instead. What is left for the monitor is the part the guest
    // cannot get from a compare-exchange: a store with no matching load of the
    // same address has to fail.
    const VAddr reserved =
        __atomic_load_n(&exclusive_addresses[processor_id], __ATOMIC_RELAXED);
    if (reserved != masked_address) {
        return false;
    }

    __atomic_store_n(&exclusive_addresses[processor_id], INVALID_EXCLUSIVE_ADDRESS,
                     __ATOMIC_RELAXED);
    return true;
}

void ExclusiveMonitor::Clear() {
    Lock();
    std::fill(exclusive_addresses.begin(), exclusive_addresses.end(), INVALID_EXCLUSIVE_ADDRESS);
    Unlock();
}

void ExclusiveMonitor::ClearProcessor(size_t processor_id) {
    Lock();
    exclusive_addresses[processor_id] = INVALID_EXCLUSIVE_ADDRESS;
    Unlock();
}

}  // namespace Dynarmic
