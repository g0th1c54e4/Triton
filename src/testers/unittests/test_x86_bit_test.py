#!/usr/bin/env python3
# coding: utf-8
"""Check BT/BTS/BTR/BTC memory bit-string addresses, CF, taint, and LEA."""

import unittest

from triton import ARCH, CALLBACK, EXCEPTION, Instruction, MemoryAccess, MODE, TritonContext


OPCODES = {"bt": 0xa3, "bts": 0xab, "btr": 0xb3, "btc": 0xbb}


def code(name, bits, address_override=False):
    return ((b"\x67" if address_override else b"") +
            (b"\x66" if bits == 16 else b"\x48" if bits == 64 else b"") +
            bytes((0x0f, OPCODES[name], 0x10)))


class TestX86BitTest(unittest.TestCase):

    def test_memory_offsets(self):
        for arch in (ARCH.X86, ARCH.X86_64):
            for bits in ((16, 32) if arch == ARCH.X86 else (16, 32, 64)):
                for override in (False, True):
                    address_bits = (16 if override else 32) if arch == ARCH.X86 else (32 if override else 64)
                    for name in OPCODES:
                        offsets = {-bits - 1, -bits, -1, 0, bits - 1, bits, bits + 8, 40, 64}
                        if bits == 64:
                            offsets.add(1 << 32)
                        for offset in sorted(offsets):
                            with self.subTest(arch=arch, bits=bits, override=override, name=name, offset=offset):
                                ctx = TritonContext(arch)
                                base = 0x2000
                                if arch == ARCH.X86:
                                    ctx.setConcreteRegisterValue(ctx.registers.ebx if override else ctx.registers.eax, base)
                                else:
                                    ctx.setConcreteRegisterValue(ctx.registers.rax, base | (0x100000000 if override else 0))
                                index = getattr(ctx.registers, {16: "dx", 32: "edx", 64: "rdx"}[bits])
                                ctx.setConcreteRegisterValue(index, offset & ((1 << bits) - 1))
                                unit = (base + (offset // bits) * (bits // 8)) & ((1 << address_bits) - 1)
                                target, bit = unit + (offset % bits) // 8, offset % 8
                                before = {addr: 0 for addr in range(base - 16, base + 24)}
                                before.update({addr: 0 for addr in range(unit, unit + bits // 8)})
                                before[target] = 0 if name == "bts" else 1 << bit
                                for addr, value in before.items():
                                    ctx.setConcreteMemoryValue(addr, value)
                                inst = Instruction(0x1000, code(name, bits, override))
                                self.assertEqual(ctx.processing(inst), EXCEPTION.NO_FAULT)
                                self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.cf), int(name != "bts"))
                                after = dict(before)
                                if name != "bt":
                                    after[target] = 1 << bit if name == "bts" else 0
                                self.assertEqual({addr: ctx.getConcreteMemoryValue(addr) for addr in after}, after)
                                self.assertEqual([(m.getAddress(), m.getSize()) for m, _ in inst.getLoadAccess()], [(unit, bits // 8)])
                                self.assertEqual([(m.getAddress(), m.getSize()) for m, _ in inst.getStoreAccess()], [] if name == "bt" else [(unit, bits // 8)])
                                self.assertEqual(inst.getOperands()[0].getLeaAst().evaluate(), unit)

    def test_register_and_immediate_offsets(self):
        for bits in (16, 32, 64):
            for name in OPCODES:
                for immediate in (False, True):
                    for memory in (False, True):
                        if memory and not immediate:
                            continue
                        with self.subTest(bits=bits, name=name, immediate=immediate, memory=memory):
                            ctx = TritonContext(ARCH.X86_64)
                            prefix = b"\x66" if bits == 16 else b"\x48" if bits == 64 else b""
                            mask = (1 << bits) - 1
                            old = 1 << (bits - 1)
                            reg = getattr(ctx.registers, {16: "ax", 32: "eax", 64: "rax"}[bits])
                            mem = MemoryAccess(0x2000, bits // 8)
                            if memory:
                                ctx.setConcreteRegisterValue(ctx.registers.rax, 0x2000)
                                ctx.setConcreteMemoryValue(mem, old)
                            else:
                                ctx.setConcreteRegisterValue(reg, old)
                            ctx.setConcreteRegisterValue(ctx.registers.rdx, mask)
                            if immediate:
                                group = {"bt": 4, "bts": 5, "btr": 6, "btc": 7}[name]
                                opcode = prefix + bytes((0x0f, 0xba, (0 if memory else 0xc0) | (group << 3), 0xff))
                            else:
                                opcode = prefix + bytes((0x0f, OPCODES[name], 0xd0))
                            ctx.processing(Instruction(0x1000, opcode))
                            self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.cf), 1)
                            expected = 0 if name in ("btr", "btc") else old
                            self.assertEqual(ctx.getConcreteMemoryValue(mem) if memory else ctx.getConcreteRegisterValue(reg), expected)

    def test_correct_memory_taint(self):
        for name in OPCODES:
            with self.subTest(name=name):
                ctx = TritonContext(ARCH.X86_64)
                ctx.setConcreteRegisterValue(ctx.registers.rax, 0x2000)
                ctx.setConcreteRegisterValue(ctx.registers.rdx, 64)
                ctx.taintMemory(MemoryAccess(0x2008, 8))
                ctx.processing(Instruction(0x1000, code(name, 64)))
                self.assertTrue(ctx.isRegisterTainted(ctx.registers.cf))
                self.assertFalse(ctx.isMemoryTainted(MemoryAccess(0x2000, 8)))
                self.assertTrue(ctx.isMemoryTainted(MemoryAccess(0x2008, 8)))

    def test_symbolic_lea_and_callbacks(self):
        for array in (False, True):
            for override in (False, True):
                with self.subTest(array=array, override=override):
                    ctx = TritonContext(ARCH.X86_64)
                    ctx.setMode(MODE.MEMORY_ARRAY, array)
                    ctx.setMode(MODE.SYMBOLIZE_LOAD, array)
                    ctx.setConcreteRegisterValue(ctx.registers.rax, 0x2000)
                    ctx.setConcreteRegisterValue(ctx.registers.rdx, 64)
                    ctx.setConcreteMemoryValue(0x2008, 1)
                    variable = ctx.symbolizeRegister(ctx.registers.rdx, "offset")
                    reads = []
                    ctx.addCallback(CALLBACK.GET_CONCRETE_MEMORY_VALUE, lambda api, mem: reads.append(mem.getAddress()))
                    inst = Instruction(0x1000, code("bt", 64, override))
                    ctx.processing(inst)
                    self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.cf), 1)
                    self.assertIn(0x2008, reads)
                    lea = inst.getOperands()[0].getLeaAst()
                    self.assertEqual(lea.evaluate(), 0x2008)
                    ctx.setConcreteVariableValue(variable, (1 << 64) - 1)
                    self.assertEqual(lea.evaluate(), 0x1ff8)

    def test_address_wrap(self):
        ctx = TritonContext(ARCH.X86_64)
        ctx.setConcreteRegisterValue(ctx.registers.rax, 0xfffffffc)
        ctx.setConcreteRegisterValue(ctx.registers.edx, 32)
        ctx.setConcreteMemoryValue(0, 1)
        inst = Instruction(0x1000, code("bt", 32, True))
        ctx.processing(inst)
        self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.cf), 1)
        self.assertEqual(inst.getOperands()[0].getAddress(), 0)

    def test_segment_after_address_wrap(self):
        ctx = TritonContext(ARCH.X86_64)
        ctx.setConcreteRegisterValue(ctx.registers.rax, 0xfffffff8)
        ctx.setConcreteRegisterValue(ctx.registers.rdx, 64)
        ctx.setConcreteRegisterValue(ctx.registers.fs, 0x7000)
        ctx.setConcreteMemoryValue(0x7000, 1)
        inst = Instruction(0x1000, b"\x64" + code("bt", 64, True))
        ctx.processing(inst)
        self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.cf), 1)
        self.assertEqual(inst.getOperands()[0].getAddress(), 0x7000)


if __name__ == "__main__":
    unittest.main()
