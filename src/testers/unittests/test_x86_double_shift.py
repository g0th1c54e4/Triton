#!/usr/bin/env python3
# coding: utf-8
"""Check SHLD/SHRD masked counts, undefined metadata, and three-source taint."""

import unittest

from triton import ARCH, EXCEPTION, Instruction, MemoryAccess, TritonContext


def instruction(bits, right, memory, immediate, count):
    prefix = b"\x66" if bits == 16 else b"\x48" if bits == 64 else b""
    opcode = (0xac if right else 0xa4) + (0 if immediate else 1)
    return prefix + bytes((0x0f, opcode, 0x18 if memory else 0xd8)) + (bytes((count,)) if immediate else b"")


class TestX86DoubleShift(unittest.TestCase):

    def test_masked_counts(self):
        for bits in (16, 32, 64):
            for right in (False, True):
                for memory in (False, True):
                    for immediate in (False, True):
                        for raw in sorted({0, 1, bits, bits + 1, 31, 32, 33, 64, 65}):
                            with self.subTest(bits=bits, right=right, memory=memory, immediate=immediate, raw=raw):
                                ctx = TritonContext(ARCH.X86_64)
                                mask = (1 << bits) - 1
                                old, source = mask ^ 2, 0x1234
                                reg = getattr(ctx.registers, {16: "ax", 32: "eax", 64: "rax"}[bits])
                                mem = MemoryAccess(0x2000, bits // 8)
                                if memory:
                                    ctx.setConcreteRegisterValue(ctx.registers.rax, 0x2000)
                                    ctx.setConcreteMemoryValue(mem, old)
                                else:
                                    ctx.setConcreteRegisterValue(reg, old)
                                ctx.setConcreteRegisterValue(ctx.registers.rbx, source)
                                ctx.setConcreteRegisterValue(ctx.registers.cl, raw)
                                inst = Instruction(0x1000, instruction(bits, right, memory, immediate, raw))
                                self.assertEqual(ctx.processing(inst), EXCEPTION.NO_FAULT)
                                count = raw & (63 if bits == 64 else 31)
                                expected_undefined = set()
                                if count:
                                    expected_undefined.add(ctx.registers.af.getId())
                                if count > 1:
                                    expected_undefined.add(ctx.registers.of.getId())
                                if count > bits:
                                    expected_undefined.update(getattr(ctx.registers, f).getId() for f in ("cf", "pf", "sf", "zf"))
                                    if not memory:
                                        expected_undefined.add(reg.getId())
                                self.assertEqual({r.getId() for r in inst.getUndefinedRegisters()}, expected_undefined)
                                if count <= bits:
                                    result = old if count == 0 else (((old >> count) | (source << (bits - count))) if right else ((old << count) | (source >> (bits - count)))) & mask
                                    actual = ctx.getConcreteMemoryValue(mem) if memory else ctx.getConcreteRegisterValue(reg)
                                    self.assertEqual(actual, result)

    def test_taint(self):
        for right in (False, True):
            for memory in (False, True):
                for tainted in (None, "dst", "src", "count"):
                    with self.subTest(right=right, memory=memory, tainted=tainted):
                        ctx = TritonContext(ARCH.X86_64)
                        mem = MemoryAccess(0x2000, 4)
                        ctx.setConcreteRegisterValue(ctx.registers.rax, 0x2000 if memory else 0x12345678)
                        ctx.setConcreteMemoryValue(mem, 0x12345678)
                        ctx.setConcreteRegisterValue(ctx.registers.ebx, 0x87654321)
                        ctx.setConcreteRegisterValue(ctx.registers.cl, 1)
                        if tainted == "dst":
                            if memory:
                                ctx.taintMemory(mem)
                            else:
                                ctx.taintRegister(ctx.registers.eax)
                        elif tainted == "src":
                            ctx.taintRegister(ctx.registers.ebx)
                        elif tainted == "count":
                            ctx.taintRegister(ctx.registers.cl)
                        inst = Instruction(0x1000, instruction(32, right, memory, False, 1))
                        ctx.processing(inst)
                        actual = ctx.isMemoryTainted(mem) if memory else ctx.isRegisterTainted(ctx.registers.eax)
                        self.assertEqual(actual, tainted is not None)
                        self.assertEqual(inst.getSymbolicExpressions()[0].isTainted(), tainted is not None)


if __name__ == "__main__":
    unittest.main()
