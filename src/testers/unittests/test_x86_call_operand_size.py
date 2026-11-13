#!/usr/bin/env python3
# coding: utf-8
"""Check 16-bit near CALL stack, target truncation, and indirect targets in non-64-bit modes."""

import unittest

from triton import ARCH, EXCEPTION, Instruction, MemoryAccess, TritonContext


class TestX86CallOperandSize(unittest.TestCase):

    def test_concrete(self):
        cases = (
            (ARCH.X86, "66e81000", 0x12340000, 2, 0x14),
            (ARCH.X86, "66ffd0", 0x12340000, 2, 0x5678),
            (ARCH.X86, "66ff1424", 0x12340000, 2, 0x3456),
            (ARCH.X86, "6667ff10", 0x12340000, 2, 0x4567),
            (ARCH.X86, "66ffd4", 0x12340000, 2, 0x8000),
            (ARCH.X86, "66e80000", 0xffff, 2, 3),
            (ARCH.X86, "e810000000", 0x12340000, 4, 0x12340015),
            (ARCH.X86, "ffd0", 0x12340000, 4, 0x12345678),
            (ARCH.X86_64, "e810000000", 0x123400000000, 8, 0x123400000015),
            (ARCH.X86_64, "ffd0", 0x123400000000, 8, 0x12345678),
        )
        for arch, encoded, address, size, target in cases:
            with self.subTest(arch=arch, encoded=encoded, address=hex(address)):
                ctx = TritonContext(arch)
                stack = ctx.registers.esp if arch == ARCH.X86 else ctx.registers.rsp
                pc = ctx.registers.eip if arch == ARCH.X86 else ctx.registers.rip
                ctx.setConcreteRegisterValue(stack, 0x8000)
                ctx.setConcreteRegisterValue(ctx.registers.eax, 0x12345678)
                ctx.setConcreteRegisterValue(ctx.registers.ebx, 0x3000)
                ctx.setConcreteMemoryValue(MemoryAccess(0x8000, 4), 0xabcd3456)
                ctx.setConcreteMemoryValue(MemoryAccess(0x3000, 2), 0x4567)
                before = b"\xa5" * 8
                ctx.setConcreteMemoryAreaValue(0x7ff8, before)
                inst = Instruction(address, bytes.fromhex(encoded))
                self.assertEqual(ctx.processing(inst), EXCEPTION.NO_FAULT)
                self.assertEqual(ctx.getConcreteRegisterValue(stack), 0x8000 - size)
                self.assertEqual(ctx.getConcreteRegisterValue(pc), target)
                saved = (address + len(bytes.fromhex(encoded))) & ((1 << (size * 8)) - 1)
                self.assertEqual(bytes(ctx.getConcreteMemoryAreaValue(0x7ff8, 8)), before[:8-size] + saved.to_bytes(size, "little"))
                self.assertEqual([(m.getAddress(), m.getSize()) for m, _ in inst.getStoreAccess()], [(0x8000 - size, size)])
                if encoded == "66ff1424":
                    self.assertEqual([(m.getAddress(), m.getSize()) for m, _ in inst.getLoadAccess()], [(0x8000, 2)])

    def test_symbolic_target_and_stack_taint(self):
        ctx = TritonContext(ARCH.X86)
        ctx.setConcreteRegisterValue(ctx.registers.esp, 0x8000)
        ctx.setConcreteRegisterValue(ctx.registers.ax, 0x5678)
        variable = ctx.symbolizeRegister(ctx.registers.ax, "target")
        ctx.taintRegister(ctx.registers.ax)
        ctx.taintMemory(MemoryAccess(0x7ffc, 4))
        inst = Instruction(0x12340000, b"\x66\xff\xd0")
        ctx.processing(inst)
        target = ctx.getRegisterAst(ctx.registers.eip)
        self.assertEqual(target.evaluate(), 0x5678)
        ctx.setConcreteVariableValue(variable, 0xfedc)
        self.assertEqual(target.evaluate(), 0xfedc)
        self.assertTrue(ctx.isRegisterTainted(ctx.registers.eip))
        self.assertTrue(ctx.isMemoryTainted(MemoryAccess(0x7ffc, 2)))
        self.assertFalse(ctx.isMemoryTainted(MemoryAccess(0x7ffe, 2)))
        self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.esp), 0x7ffe)


if __name__ == "__main__":
    unittest.main()
