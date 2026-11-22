"""The cuBLASLt FP16 baseline used for the original Windows GTX benchmark."""

import ctypes as ct
import os
from contextlib import ExitStack
from pathlib import Path
import tempfile


Ptr = ct.c_void_p
Int = ct.c_int
Size = ct.c_size_t


class _Algo(ct.Structure):
    _fields_ = [("data", ct.c_uint64 * 8)]


class _Heuristic(ct.Structure):
    _fields_ = [("algo", _Algo), ("workspace", Size), ("state", Int),
                ("waves", ct.c_float), ("reserved", Int * 4)]


def _check(status, name):
    if status:
        raise RuntimeError(f"{name} returned status {status}")


def _pointer(value):
    return Ptr(getattr(value, "value", value))


class DenseLt:
    """One top-ranked FP32-reduction heuristic, in the current CUDA context."""

    def __init__(self, workspace_bytes=32 << 20, dll_dir=None):
        default_dir = Path(tempfile.gettempdir()) / "task3_nvblas_dlls"
        dll_dir = Path(dll_dir or os.environ.get("TASK3_CUBLAS_DIR", default_dir)).resolve()
        self._resources = ExitStack()
        self.ws_size = int(workspace_bytes)
        try:
            self._resources.enter_context(os.add_dll_directory(str(dll_dir)))
            self.cuda = ct.WinDLL("nvcuda.dll")
            self.lt = ct.WinDLL(str(dll_dir / "cublasLt64_12.dll"))
            self.cuda.cuMemAlloc_v2.argtypes = [ct.POINTER(Ptr), Size]
            self.cuda.cuMemFree_v2.argtypes = [Ptr]
            for name, signature in {
                "Create": [ct.POINTER(Ptr)],
                "Destroy": [Ptr],
                "MatmulDescCreate": [ct.POINTER(Ptr), Int, Int],
                "MatmulDescDestroy": [Ptr],
                "MatmulDescSetAttribute": [Ptr, Int, Ptr, Size],
                "MatrixLayoutCreate": [ct.POINTER(Ptr), Int, ct.c_uint64, ct.c_uint64, ct.c_int64],
                "MatrixLayoutDestroy": [Ptr],
                "MatmulPreferenceCreate": [ct.POINTER(Ptr)],
                "MatmulPreferenceDestroy": [Ptr],
                "MatmulPreferenceSetAttribute": [Ptr, Int, Ptr, Size],
                "MatmulAlgoGetHeuristic": [Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Int,
                                           ct.POINTER(_Heuristic), ct.POINTER(Int)],
                "Matmul": [Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Size, Ptr],
            }.items():
                getattr(self.lt, "cublasLt" + name).argtypes = signature
            self.lt.cublasLtGetVersion.restype = Size
            self.lt.cublasLtGetVersion.argtypes = []
            self.version = self.lt.cublasLtGetVersion()
            self.ws = Ptr()
            _check(self.cuda.cuMemAlloc_v2(ct.byref(self.ws), self.ws_size), "cuMemAlloc")
            self._resources.callback(self._free_workspace)
            self.handle = self._create("Create", "Destroy")
            # CUBLAS_COMPUTE_32F, CUDA_R_32F, CUBLASLT_MATMUL_DESC_TRANSA.
            self.desc = self._create("MatmulDescCreate", "MatmulDescDestroy", 68, 0)
            self._set("MatmulDescSetAttribute", self.desc, 3, Int(1))
            self.pref = self._create("MatmulPreferenceCreate", "MatmulPreferenceDestroy")
            self._set("MatmulPreferenceSetAttribute", self.pref, 1, ct.c_uint64(self.ws_size))
            # CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE uses FP32 for split-K reduction.
            self._set("MatmulPreferenceSetAttribute", self.pref, 3, ct.c_uint32(2))
        except BaseException:
            self.close()
            raise

    def _call(self, name, *args):
        _check(getattr(self.lt, "cublasLt" + name)(*args), "cublasLt" + name)

    def _set(self, name, handle, attribute, value):
        self._call(name, handle, attribute, ct.byref(value), ct.sizeof(value))

    def _create(self, name, destroy, *args):
        handle = Ptr()
        self._call(name, ct.byref(handle), *args)
        self._resources.callback(self._call, destroy, handle)
        return handle

    def _free_workspace(self):
        _check(self.cuda.cuMemFree_v2(self.ws), "cuMemFree")

    def prepare(self, dx, dw, dy, m, n, k):
        # Column-major Y.T[N,M] = transpose(W.T[K,N]) @ X.T[K,M].
        a = self._create("MatrixLayoutCreate", "MatrixLayoutDestroy", 2, k, n, k)
        b = self._create("MatrixLayoutCreate", "MatrixLayoutDestroy", 2, k, m, k)
        c = self._create("MatrixLayoutCreate", "MatrixLayoutDestroy", 2, n, m, n)
        hint, returned = _Heuristic(), Int()
        self._call("MatmulAlgoGetHeuristic", self.handle, self.desc, a, b, c, c, self.pref,
                   1, ct.byref(hint), ct.byref(returned))
        if returned.value != 1 or hint.state or hint.workspace > self.ws_size:
            raise RuntimeError(f"No cuBLASLt FP32 heuristic candidate for {(m, n, k)}")
        alpha, beta = ct.c_float(1), ct.c_float(0)
        args = (self.handle, self.desc, ct.byref(alpha), _pointer(dw), a, _pointer(dx), b,
                ct.byref(beta), _pointer(dy), c, _pointer(dy), c, ct.byref(hint.algo),
                self.ws, self.ws_size, None)

        def launch():
            self._call("Matmul", *args)

        launch.workspace_bytes = int(hint.workspace)
        return launch

    def close(self):
        self._resources.close()
