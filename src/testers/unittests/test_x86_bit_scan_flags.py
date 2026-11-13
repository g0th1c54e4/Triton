#!/usr/bin/env python3
# coding: utf-8
"""Check BSF/BSR full-source parity and defined flags against SDM 093."""

import unittest

from triton import ARCH, EXCEPTION, Instruction, MemoryAccess, TritonContext


class TestX86BitScanFlags(unittest.TestCase):

    def test_concrete_flags(self):
        for arch in (ARCH.X86, ARCH.X86_64):
            for bits in ((16, 32) if arch == ARCH.X86 else (16, 32, 64)):
                for reverse in (False, True):
                    for form in ("reg", "alias", "mem"):
                        for value in (0, 1, 3, 0x100, 0x101, 0x1003, 1 << (bits - 1)):
                            with self.subTest(arch=arch, bits=bits, reverse=reverse, form=form, value=value):
                                ctx = TritonContext(arch)
                                prefix = b"\x66" if bits == 16 else b"\x48" if bits == 64 else b""
                                suffix = {"reg": 0xc3, "alias": 0xc0, "mem": 0x03}[form]
                                opcode = prefix + bytes((0x0f, 0xbd if reverse else 0xbc, suffix))
                                dst = getattr(ctx.registers, {16: "ax", 32: "eax", 64: "rax"}[bits])
                                src = getattr(ctx.registers, {16: "bx", 32: "ebx", 64: "rbx"}[bits])
                                ctx.setConcreteRegisterValue(dst, 0x5a5a)
                                if form == "mem":
                                    base = ctx.registers.ebx if arch == ARCH.X86 else ctx.registers.rbx
                                    ctx.setConcreteRegisterValue(base, 0x2000)
                                    ctx.setConcreteMemoryValue(MemoryAccess(0x2000, bits // 8), value)
                                else:
                                    ctx.setConcreteRegisterValue(dst if form == "alias" else src, value)
                                for flag in ("af", "cf", "of", "sf"):
                                    ctx.setConcreteRegisterValue(getattr(ctx.registers, flag), 1)
                                inst = Instruction(0x1000, opcode)
                                self.assertEqual(ctx.processing(inst), EXCEPTION.NO_FAULT)
                                if value:
                                    index = value.bit_length() - 1 if reverse else (value & -value).bit_length() - 1
                                    self.assertEqual(ctx.getConcreteRegisterValue(dst), index)
                                self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.zf), int(value == 0))
                                self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.pf), int(bin(value).count("1") % 2 == 0))
                                for flag in ("af", "cf", "of", "sf"):
                                    self.assertEqual(ctx.getConcreteRegisterValue(getattr(ctx.registers, flag)), 0)
                                self.assertEqual(inst.getUndefinedRegisters(), [])

    def test_symbolic_source(self):
        for reverse in (False, True):
            for alias in (False, True):
                with self.subTest(reverse=reverse, alias=alias):
                    ctx = TritonContext(ARCH.X86_64)
                    source = ctx.registers.rax if alias else ctx.registers.rbx
                    variable = ctx.symbolizeRegister(source, "source")
                    inst = Instruction(0x1000, bytes((0x48, 0x0f, 0xbd if reverse else 0xbc, 0xc0 if alias else 0xc3)))
                    ctx.processing(inst)
                    pf = ctx.getRegisterAst(ctx.registers.pf)
                    zf = ctx.getRegisterAst(ctx.registers.zf)
                    for value in (0, 1, 0x101, 0x1003, 1 << 63, (1 << 63) | 1):
                        ctx.setConcreteVariableValue(variable, value)
                        self.assertEqual(pf.evaluate(), int(bin(value).count("1") % 2 == 0))
                        self.assertEqual(zf.evaluate(), int(value == 0))

    def test_flag_taint(self):
        for reverse in (False, True):
            for memory in (False, True):
                for tainted in (False, True):
                    with self.subTest(reverse=reverse, memory=memory, tainted=tainted):
                        ctx = TritonContext(ARCH.X86_64)
                        ctx.setConcreteRegisterValue(ctx.registers.rbx, 0x2000 if memory else 0x1003)
                        mem = MemoryAccess(0x2000, 8)
                        ctx.setConcreteMemoryValue(mem, 0x1003)
                        for flag in ("af", "cf", "of", "sf", "pf"):
                            ctx.taintRegister(getattr(ctx.registers, flag))
                        if tainted:
                            if memory:
                                ctx.taintMemory(mem)
                            else:
                                ctx.taintRegister(ctx.registers.rbx)
                        ctx.processing(Instruction(0x1000, bytes((0x48, 0x0f, 0xbd if reverse else 0xbc, 0x03 if memory else 0xc3))))
                        self.assertEqual(ctx.isRegisterTainted(ctx.registers.pf), tainted)
                        for flag in ("af", "cf", "of", "sf"):
                            self.assertFalse(ctx.isRegisterTainted(getattr(ctx.registers, flag)))


if __name__ == "__main__":
    unittest.main()
