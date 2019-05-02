import ctypes

from ctypes import pythonapi

dllib = ctypes.CDLL(None)
dllib.dlsym.restype = ctypes.c_void_p


def pysym(name):
    """Look up a symbol exposed by CPython"""
    result = dllib.dlsym(pythonapi._handle, name)
    if result is None:
        raise RuntimeError(f"resolved {name} to {result}")
    return result


class CAPI:
    """Addresses of functions exposed by the C-API"""
    _Py_Dealloc = pysym(b"_Py_Dealloc")
    PyNumber_Add = pysym(b"PyNumber_Add")
    PyObject_RichCompare = pysym(b"PyObject_RichCompare")


class JITFunction:
    def __init__(self, num_args, encoded_func) -> None:
        self.encoded_func = encoded_func
        self.loaded_func = encoded_func.load()
        result_type = ctypes.c_uint64
        argument_types = [ctypes.c_uint64 for _ in range(num_args)]
        self.function_type = ctypes.CFUNCTYPE(result_type, *argument_types)
        self.function_pointer = self.function_type(self.loaded_func.loader.code_address)

    def __call__(self, *args):
        addr = self.function_pointer(*map(id, args))
        box = ctypes.cast(addr, ctypes.py_object)
        return box.value

    def format(self):
        return self.encoded_func.format()
