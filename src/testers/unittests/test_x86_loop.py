#!/usr/bin/env python3
# coding: utf-8
"""Test x86 LOOPE and LOOPNE."""

import unittest

from triton import ARCH, Instruction, TritonContext


class TestX86LoopCondition(unittest.TestCase):
    """Check that LOOPE and LOOPNE decrement the counter and branch on it and ZF."""

    def test_concrete(self):
        cases = (
            # opcode, counter, zf, taken
            (b"\xe1\x10", 5, 1, True),    # loope
            (b"\xe1\x10", 5, 0, False),
            (b"\xe1\x10", 1, 1, False),
            (b"\xe0\x10", 5, 0, True),    # loopne
            (b"\xe0\x10", 5, 1, False),
            (b"\xe0\x10", 1, 0, False),
        )
        for arch, counter, pc in ((ARCH.X86, "ecx", "eip"), (ARCH.X86_64, "rcx", "rip")):
            for opcode, count, zf, taken in cases:
                with self.subTest(arch=arch, opcode=opcode, count=count, zf=zf):
                    ctx = TritonContext(arch)
                    ctx.setConcreteRegisterValue(getattr(ctx.registers, counter), count)
                    ctx.setConcreteRegisterValue(ctx.registers.zf, zf)
                    inst = Instruction(0x1000, opcode)
                    ctx.processing(inst)
                    self.assertEqual(ctx.getConcreteRegisterValue(getattr(ctx.registers, counter)), count - 1)
                    self.assertEqual(ctx.getConcreteRegisterValue(getattr(ctx.registers, pc)), 0x1012 if taken else 0x1002)
                    self.assertEqual(inst.isConditionTaken(), taken)

    def test_symbolic(self):
        for opcode, zf in ((b"\xe1\x10", 1), (b"\xe0\x10", 0)):
            with self.subTest(opcode=opcode):
                ctx = TritonContext(ARCH.X86_64)
                ctx.setConcreteRegisterValue(ctx.registers.rcx, 5)
                ctx.symbolizeRegister(ctx.registers.zf, "zf")
                ctx.processing(Instruction(0x1000, opcode))
                constraint = ctx.getPathConstraints()[0]
                self.assertTrue(constraint.isMultipleBranches())
                ast = ctx.getAstContext()
                for target, value in ((0x1012, zf), (0x1002, 1 - zf)):
                    branch = next(b for b in constraint.getBranchConstraints() if b["dstAddr"] == target)
                    model = ctx.getModel(branch["constraint"])
                    self.assertEqual(list(model.values())[0].getValue(), value)


    def test_address_size(self):
        for arch in (ARCH.X86, ARCH.X86_64):
            parent_bits = 32 if arch == ARCH.X86 else 64
            for prefix in ((b"", b"\x67", b"\x2e\x67") if arch == ARCH.X86 else
                           (b"", b"\x67", b"\x2e\x67", b"\x48", b"\x67\x48")):
                bits = (16 if arch == ARCH.X86 else 32) if b"\x67" in prefix else parent_bits
                for opcode in (0xe2, 0xe1, 0xe0):
                    for zf in (0, 1):
                        for value in (0, 1, 2, (1 << bits) | 1, (1 << bits)):
                            if value >= 1 << parent_bits:
                                continue
                            with self.subTest(arch=arch, prefix=prefix.hex(), opcode=opcode, zf=zf, value=value):
                                ctx = TritonContext(arch)
                                parent = ctx.registers.ecx if arch == ARCH.X86 else ctx.registers.rcx
                                pc = ctx.registers.eip if arch == ARCH.X86 else ctx.registers.rip
                                ctx.setConcreteRegisterValue(parent, value)
                                flags = {"af": 1, "cf": 1, "of": 1, "pf": 1, "sf": 1, "zf": zf}
                                for name, flag in flags.items():
                                    ctx.setConcreteRegisterValue(getattr(ctx.registers, name), flag)
                                # The 0x67 rel8 value is a displacement, not another address-size prefix.
                                code = prefix + bytes((opcode, 0x67))
                                inst = Instruction(0x1000, code)
                                ctx.processing(inst)
                                count = (value - 1) & ((1 << bits) - 1)
                                expected = ((value & ~0xffff) | count) if bits == 16 else count
                                taken = count != 0 and (opcode == 0xe2 or zf == (opcode == 0xe1))
                                self.assertEqual(ctx.getConcreteRegisterValue(parent), expected)
                                self.assertEqual(ctx.getConcreteRegisterValue(pc), 0x1000 + len(code) + (0x67 if taken else 0))
                                self.assertEqual(inst.isConditionTaken(), taken)
                                for name, flag in flags.items():
                                    self.assertEqual(ctx.getConcreteRegisterValue(getattr(ctx.registers, name)), flag)

    def test_symbolic_address_size(self):
        for arch in (ARCH.X86, ARCH.X86_64):
            for opcode in (0xe2, 0xe1, 0xe0):
                with self.subTest(arch=arch, opcode=opcode):
                    ctx = TritonContext(arch)
                    parent = ctx.registers.ecx if arch == ARCH.X86 else ctx.registers.rcx
                    pc = ctx.registers.eip if arch == ARCH.X86 else ctx.registers.rip
                    bits = 16 if arch == ARCH.X86 else 32
                    ctx.setConcreteRegisterValue(ctx.registers.zf, int(opcode != 0xe0))
                    variable = ctx.symbolizeRegister(parent, "count")
                    inst = Instruction(0x1000, bytes((0x67, opcode, 0x10)))
                    ctx.processing(inst)
                    count_ast, pc_ast = ctx.getRegisterAst(parent), ctx.getRegisterAst(pc)
                    constraint = ctx.getPathConstraints()[0]
                    self.assertTrue(constraint.isMultipleBranches())
                    for value in (0, 1, 2, (1 << bits) | 1, 1 << bits):
                        ctx.setConcreteVariableValue(variable, value)
                        count = (value - 1) & ((1 << bits) - 1)
                        expected = ((value & ~0xffff) | count) if bits == 16 else count
                        self.assertEqual(count_ast.evaluate(), expected)
                        self.assertEqual(pc_ast.evaluate(), 0x1013 if count else 0x1003)
                        taken = next(b for b in constraint.getBranchConstraints() if b["dstAddr"] == 0x1013)
                        self.assertEqual(bool(taken["constraint"].evaluate()), count != 0)


if __name__ == '__main__':
    unittest.main()
