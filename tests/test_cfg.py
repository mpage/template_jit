import pytest

from template_jit import cfg


def max(a, b):
    if a > b:
        return a
    return b


def add(a, b):
    return a + b


def test_compute_block_boundaries():
    boundaries = cfg.compute_block_boundaries(max)
    assert boundaries == [(0, 8), (8, 12), (12, 16)]


@pytest.mark.parametrize("func,expected", [
    (max, """entry:
bb0:
  LOAD_FAST            0    (a)
  LOAD_FAST            1    (b)
  COMPARE_OP           4    (>)
  POP_JUMP_IF_FALSE    12
bb1:
  LOAD_FAST            0    (a)
  RETURN_VALUE
bb2:
  LOAD_FAST            1    (b)
  RETURN_VALUE"""),

    (add, """entry:
bb0:
  LOAD_FAST            0    (a)
  LOAD_FAST            1    (b)
  BINARY_ADD
  RETURN_VALUE"""),
])
def test_build_cfg(func, expected):
    g = cfg.build_cfg(func)
    assert str(g) == expected
