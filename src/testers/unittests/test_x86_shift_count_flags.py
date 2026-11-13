#!/usr/bin/env python3
# coding: utf-8
"""Check that SHL/SAL/SHR CF undefined metadata uses the masked count."""

import unittest

from triton import ARCH, EXCEPTION, Instruction, MemoryAccess, TritonContext


class TestX86ShiftCountFlags(unittest.TestCase):

    def test_masked_counts(self):
        for arch in (ARCH.X86, ARCH.X86_64):
            for bits in ((8, 16, 32) if arch == ARCH.X86 else (8, 16, 32, 64)):
                for right in (False, True):
                    for memory in (False, True):
                        for immediate in (False, True):
                            for raw in sorted({0, 1, bits, bits + 1, 32, 33, 64, 65}):
                                with self.subTest(arch=arch, bits=bits, right=right,
                                                  memory=memory, immediate=immediate, raw=raw):
                                    ctx = TritonContext(arch)
                                    prefix = b"\x66" if bits == 16 else b"\x48" if bits == 64 else b""
                                    opcode = (0xc0 if bits == 8 else 0xc1) if immediate else (0xd2 if bits == 8 else 0xd3)
                                    modrm = (0x20 if memory else 0xe0) + (8 if right else 0)
                                    code = prefix + bytes((opcode, modrm)) + (bytes((raw,)) if immediate else b"")
                                    value = (1 << (bits - 1)) | 1
                                    mem = MemoryAccess(0x2000, bits // 8)
                                    reg = getattr(ctx.registers, {8: "al", 16: "ax", 32: "eax", 64: "rax"}[bits])
                                    if memory:
                                        base = ctx.registers.eax if arch == ARCH.X86 else ctx.registers.rax
                                        ctx.setConcreteRegisterValue(base, 0x2000)
                                        ctx.setConcreteMemoryValue(mem, value)
                                    else:
                                        ctx.setConcreteRegisterValue(reg, value)
                                    ctx.setConcreteRegisterValue(ctx.registers.cl, raw)
                                    ctx.setConcreteRegisterValue(ctx.registers.cf, 1)
                                    inst = Instruction(0x1000, code)
                                    self.assertEqual(ctx.processing(inst), EXCEPTION.NO_FAULT)
                                    count = raw & (63 if bits == 64 else 31)
                                    undefined = {r.getId() for r in inst.getUndefinedRegisters()}
                                    self.assertEqual(ctx.registers.cf.getId() in undefined, count >= bits)
                                    expected = value >> count if right else (value << count) & ((1 << bits) - 1)
                                    result = ctx.getConcreteMemoryValue(mem) if memory else ctx.getConcreteRegisterValue(reg)
                                    self.assertEqual(result, expected)
                                    if count == 0:
                                        self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.cf), 1)
                                    elif count < bits:
                                        carry = (value >> (count - 1 if right else bits - count)) & 1
                                        self.assertEqual(ctx.getConcreteRegisterValue(ctx.registers.cf), carry)


if __name__ == "__main__":
    unittest.main()
