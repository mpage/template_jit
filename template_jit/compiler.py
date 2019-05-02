import dis

from peachpy import *
from peachpy.x86_64 import *
from peachpy.x86_64.registers import rsp
from template_jit.cfg import build_cfg
from template_jit.runtime import CAPI, JITFunction
from types import FunctionType

PTR_SIZE = 8

ARG_REGISTERS = [rdi, rsi, rdx, rcx, r8, r9]
MAX_ARGS = len(ARG_REGISTERS)


def incref(pyobj):
    """Increment the reference count of a PyObject.

    Args:
        pyobj: A register storing a pointer to the PyObject.
    """
    INC([pyobj])


def decref(pyobj, tmp):
    """Decrement the reference count of a PyObject.

    Args:
        pyobj: A register storing a poiner to the PyObject.
        tmp: A temporary register
    """
    done = Label()
    # Move refcount into register tmp
    MOV(tmp, [pyobj])
    DEC(tmp)
    # Update refcount
    MOV([pyobj], tmp)
    # Check if we need to invoke the destructor
    CMP(tmp, 0)
    JNE(done)
    # Save rdi
    PUSH(rdi)
    # Invoke the destructor
    MOV(rdi, pyobj)
    MOV(tmp, CAPI._Py_Dealloc)
    CALL(tmp)
    # Restore rdi
    POP(rdi)
    LABEL(done)


def xdecref(pyobj, tmp):
    """Decrement the reference count of a PyObject if its not null.

    Args:
        pyobj: A register storing a poiner to the PyObject.
        tmp: A temporary register
    """
    done = Label()
    CMP(pyobj, 0)
    JE(done)
    # Move refcount into register tmp
    MOV(tmp, [pyobj])
    DEC(tmp)
    # Update refcount
    MOV([pyobj], tmp)
    # Check if we need to invoke the destructor
    CMP(tmp, 0)
    JNE(done)
    # Save rdi
    PUSH(rdi)
    # Invoke the destructor
    MOV(rdi, pyobj)
    MOV(tmp, CAPI._Py_Dealloc)
    CALL(tmp)
    # Restore rdi
    POP(rdi)
    LABEL(done)


class Compiler:
    def get_signature(self, func: FunctionType):
        return tuple(), uint64_t

    def emit_BINARY_ADD(self, func: FunctionType, instr: dis.Instruction) -> None:
        MOV(rsi, [rsp])
        MOV(rdi, [rsp + 8])
        MOV(rax, CAPI.PyNumber_Add)
        CALL(rax)
        POP(rdi)
        decref(rdi, rdx)
        POP(rdi)
        decref(rdi, rdx)
        PUSH(rax)
        self.emit_error_check()

    def emit_COMPARE_OP(self, func: FunctionType, instr: dis.Instruction) -> None:
        if instr.argval != "is":
            raise ValueError(f"Cannot handle {instr.argval} comparisons")
        true_obj = id(True)
        false_obj = id(False)
        true_br = Label()
        done = Label()
        POP(rdi)
        POP(rsi)
        MOV(rax, true_obj)
        CMP(rdi, rsi)
        JE(true_br)
        MOV(rax, false_obj)
        LABEL(true_br)
        incref(rax)
        PUSH(rax)
        decref(rdi, rdx)
        decref(rsi, rdx)

    def emit_LOAD_CONST(self, func: FunctionType, instr: dis.Instruction) -> None:
        consts = func.__code__.co_consts
        MOV(rax, id(consts[instr.arg]))
        incref(rax)
        PUSH(rax)

    def emit_LOAD_FAST(self, func: FunctionType, instr: dis.Instruction) -> None:
        MOV(rax, [rbp - (instr.arg + 1) * PTR_SIZE])
        incref(rax)
        PUSH(rax)

    def emit_POP_JUMP_IF_FALSE(self, func: FunctionType, instr: dis.Instruction) -> None:
        false_lbl = Label()
        POP(rax)
        MOV(rdi, id(False))
        CMP(rax, rdi)
        JE(false_lbl)
        decref(rax, rdi)
        JMP(self.block_labels[instr.true_branch])
        LABEL(false_lbl)
        decref(rax, rdi)
        JMP(self.block_labels[instr.false_branch])

    def emit_POP_TOP(self, func: FunctionType, instr: dis.Instruction) -> None:
        POP(rax)
        decref(rax, rsi)

    def emit_RETURN_VALUE(self, func: FunctionType, instr: dis.Instruction) -> None:
        POP(rax)
        JMP(self.epilogue)

    def emit_instruction(self, func: FunctionType, instr: dis.Instruction) -> None:
        emitter = getattr(self, f"emit_{instr.opname}", None)
        if emitter is not None:
            emitter(func, instr)
        else:
            raise ValueError(f"Cannot compile {instr.opname}")

    def emit_error_check(self):
        """Check if an error occurred, and, if so return from the function"""
        CMP(rax, 0)
        JE(self.epilogue)

    def emit_prologue(self, func: FunctionType):
        """Emit code to reserve space on the stack for locals"""
        # Standard prologue - peachpy has some magic to save and restore rbp
        MOV(rbp, rsp)

        # Reserve space on the stack for locals (args + local variables)
        nlocals = func.__code__.co_nlocals
        if nlocals > 0:
            SUB(rsp, nlocals * PTR_SIZE)

        # Copy any arguments onto the stack
        nargs = func.__code__.co_argcount
        for i in range(nargs):
            MOV([rbp - (i + 1) * PTR_SIZE], ARG_REGISTERS[i])

        # Null out any local variables
        nvars = nlocals - nargs
        for i in range(nargs, nvars):
            MOV([rbp - (i + 1) * PTR_SIZE], 0)

    def emit_epilogue(self):
        """Emit code to clean up the stack and return from the function"""
        LABEL(self.epilogue)
        # Decref anything left on the stack
        done = Label()
        loop = Label()
        LABEL(loop)
        CMP(rsp, rbp)
        JE(done)
        POP(rdi)
        xdecref(rdi, rsi)
        JMP(loop)
        LABEL(done)
        RETURN(rax)

    def compile(self, func: FunctionType) -> JITFunction:
        num_args = func.__code__.co_argcount
        if num_args > MAX_ARGS:
            raise ValueError("Cannot compile functions with more than {MAX_ARGS}")
        name = f"jit_{func.__name__}"
        graph = build_cfg(func)
        blocks = list(graph)
        arg_types, ret_type = self.get_signature(func)
        with Function(name, arg_types, ret_type) as asm_func:
            self.emit_prologue(func)
            self.epilogue = Label("epilogue")
            self.block_labels = {block.label: Label(block.label) for block in blocks}
            for block in blocks:
                LABEL(self.block_labels[block.label])
                for instr in block.instructions:
                    # Switch on ins.opcode and emit assembly
                    self.emit_instruction(func, instr)
            self.emit_epilogue()
        loaded = asm_func.finalize(abi.detect()).encode()
        return JITFunction(num_args, loaded)
