#!/usr/bin/env python3
# coding: utf-8
"""Check 16-bit near RET width, zero extension, and stack cleanup in non-64-bit modes."""

import unittest

from triton import ARCH, EXCEPTION, Instruction, MemoryAccess, TritonContext


class TestX86RetOperandSize(unittest.TestCase):

    def test_concrete(self):
        cases = (
            (ARCH.X86, "66c3", 2, 0),
            (ARCH.X86, "66c21000", 2, 16),
            (ARCH.X86, "2e66c3", 2, 0),
            (ARCH.X86, "6766c3", 2, 0),
            (ARCH.X86, "6666c3", 2, 0),
            (ARCH.X86, "66c26600", 2, 0x66),
            (ARCH.X86, "c3", 4, 0),
            (ARCH.X86, "c26600", 4, 0x66),
            (ARCH.X86_64, "c3", 8, 0),
            (ARCH.X86_64, "c26600", 8, 0x66),
        )
        for arch, encoded, size, extra in cases:
            for initial_sp in (0x8000, 0x12348000):
                with self.subTest(arch=arch, encoded=encoded, initial_sp=hex(initial_sp)):
                    ctx = TritonContext(arch)
                    stack = ctx.registers.esp if arch == ARCH.X86 else ctx.registers.rsp
                    pc = ctx.registers.eip if arch == ARCH.X86 else ctx.registers.rip
                    ctx.setConcreteRegisterValue(stack, initial_sp)
                    contents = bytes.fromhex("7856341234120000")
                    ctx.setConcreteMemoryAreaValue(initial_sp, contents)
                    inst = Instruction(0x1000, bytes.fromhex(encoded))
                    self.assertEqual(ctx.processing(inst), EXCEPTION.NO_FAULT)
                    self.assertEqual(ctx.getConcreteRegisterValue(pc), int.from_bytes(contents[:size], "little"))
                    self.assertEqual(ctx.getConcreteRegisterValue(stack), initial_sp + size + extra)
                    self.assertEqual([(m.getAddress(), m.getSize()) for m, _ in inst.getLoadAccess()], [(initial_sp, size)])
                    self.assertEqual(inst.getStoreAccess(), [])
                    self.assertEqual(bytes(ctx.getConcreteMemoryAreaValue(initial_sp, 8)), contents)

    def test_symbolic_return_ip(self):
        ctx = TritonContext(ARCH.X86)
        ctx.setConcreteRegisterValue(ctx.registers.esp, 0x8000)
        ctx.setConcreteMemoryValue(MemoryAccess(0x8000, 4), 0xbeef5678)
        variable = ctx.symbolizeMemory(MemoryAccess(0x8000, 2), "ip")
        ctx.processing(Instruction(0x1000, b"\x66\xc3"))
        ip = ctx.getRegisterAst(ctx.registers.eip)
        for value in (0, 1, 0x8000, 0xffff):
            ctx.setConcreteVariableValue(variable, value)
            self.assertEqual(ip.evaluate(), value)
        self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.esp), 0x8002)
        self.assertEqual(len(ctx.getPathConstraints()), 1)

    def test_taint_only_loaded_word(self):
        for tainted in (False, True):
            with self.subTest(tainted=tainted):
                ctx = TritonContext(ARCH.X86)
                ctx.setConcreteRegisterValue(ctx.registers.esp, 0x8000)
                ctx.setConcreteMemoryValue(MemoryAccess(0x8000, 4), 0x12345678)
                ctx.taintMemory(MemoryAccess(0x8002, 2))
                if tainted:
                    ctx.taintMemory(MemoryAccess(0x8000, 2))
                ctx.processing(Instruction(0x1000, b"\x66\xc3"))
                self.assertEqual(ctx.isRegisterTainted(ctx.registers.eip), tainted)


if __name__ == "__main__":
    unittest.main()
