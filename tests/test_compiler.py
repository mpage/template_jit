import ctypes

from peachpy import *
from peachpy.x86_64 import *
from template_jit.compiler import Compiler
from template_jit.runtime import CAPI, JITFunction


def test_peachpy_works():
    with Function("meaning_of_life", tuple(), int64_t) as jit_func:
        MOV(rax, 42)
        RETURN(rax)
    meaning_of_life = jit_func.finalize(abi.detect()).encode().load()
    assert meaning_of_life() == 42


def test_simple_function():
    """Check that we can compile and execute a function that does:

    LOAD_CONST
    RETURN_VALUE
    """
    def func():
        return 42
    jitfunc = Compiler().compile(func)
    assert jitfunc() == 42


def test_local_variables():
    """Check that we can load local variables"""
    def identity(x):
        return x
    jit_identity = Compiler().compile(identity)
    arg = "testing 123"
    assert jit_identity(arg) is arg


def test_binary_add():
    """Check that we can call into the runtime and invoke PyNumber_Add"""
    def add(a, b):
        return a + b
    jit_add = Compiler().compile(add)
    assert jit_add(1, 2) == 3
    assert jit_add("foo", "bar") == "foobar"


def test_compare_op():
    """Check that we can successfully invoke compare op"""
    def identical(a, b):
        return a is b
    jit_gt = Compiler().compile(identical)
    x = 1
    y = 2
    assert jit_gt(x, y) is False
    assert jit_gt(x, x) is True


def test_control_flow():
    """Check that we can successfully invoke compare op"""
    def identical(a, b):
        if a is b:
            return True
        return False
    jit_gt = Compiler().compile(identical)
    x = 1
    y = 2
    assert jit_gt(x, y) is False
    assert jit_gt(x, x) is True
