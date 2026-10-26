/* This file is part of the dynarmic project.
 * Copyright (c) 2022 MerryMage
 * SPDX-License-Identifier: 0BSD
 */

#include "dynarmic/backend/arm64/fpsr_manager.h"

#include <oaknut/oaknut.hpp>

#include "dynarmic/backend/arm64/abi.h"

namespace Dynarmic::Backend::Arm64 {

using namespace oaknut::util;

FpsrManager::FpsrManager(oaknut::CodeGenerator& code, size_t state_fpsr_offset)
        : code{code}, state_fpsr_offset{state_fpsr_offset}, fpsr_loaded{true} {}

void FpsrManager::Spill() {
    // Keep the host FPSR live for the whole run. A32::Jit::Run merges it into
    // the guest state on exit and restores it on entry, so spilling at the end
    // of every block only bought a MRS FPSR (~4 ns per block on this core).
}

void FpsrManager::SaveForCall() {
    // A host call (or a guest FPSCR read) must see the flags in guest memory.
    Load();
    if (!fpsr_loaded)
        return;

    code.LDR(Wscratch0, Xstate, state_fpsr_offset);
    code.MRS(Xscratch1, oaknut::SystemReg::FPSR);
    code.ORR(Wscratch0, Wscratch0, Wscratch1);
    code.STR(Wscratch0, Xstate, state_fpsr_offset);

    fpsr_loaded = false;
}

void FpsrManager::Load() {
    if (fpsr_loaded)
        return;

    // Restore the guest flags after a host call or a guest FPSCR write.
    code.LDR(Xscratch0, Xstate, state_fpsr_offset);
    code.MSR(oaknut::SystemReg::FPSR, Xscratch0);

    fpsr_loaded = true;
}

void FpsrManager::GetFpsr(oaknut::WReg dest) {
    code.LDR(dest, Xstate, state_fpsr_offset);

    if (fpsr_loaded) {
        code.MRS(Xscratch1, oaknut::SystemReg::FPSR);
        code.ORR(dest, dest, Wscratch1);
    }
}

}  // namespace Dynarmic::Backend::Arm64
