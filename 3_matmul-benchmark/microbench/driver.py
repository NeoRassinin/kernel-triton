"""Small CUDA Driver API helpers for the SM75 microbenchmark (no CUDA PyTorch required)."""

from contextlib import contextmanager, ExitStack
import ctypes as ct
from functools import cache
import sys


@cache
def _driver():
    if sys.platform == "win32":
        return ct.WinDLL("nvcuda.dll")
    return ct.CDLL("libcuda.so.1")


def call(name, *args):
    driver = _driver()
    code = getattr(driver, name)(*args)
    if code:
        message = ct.c_char_p()
        driver.cuGetErrorString(code, ct.byref(message))
        raise RuntimeError(f"{name} failed ({code}): {message.value!r}")


def alloc(nbytes):
    device_ptr = ct.c_uint64()
    call("cuMemAlloc_v2", ct.byref(device_ptr), ct.c_size_t(nbytes))
    return device_ptr


def free(*device_ptrs):
    for ptr in device_ptrs:
        call("cuMemFree_v2", ptr)


def address(device_ptr, offset=0):
    return ct.c_uint64(device_ptr.value + offset)


def hcopy(device_ptr, array, offset=0):
    if not array.flags.c_contiguous:
        raise ValueError("Expected a contiguous source array")
    call("cuMemcpyHtoD_v2", address(device_ptr, offset),
         ct.c_void_p(array.ctypes.data), ct.c_size_t(array.nbytes))


def download(device_ptr, array, offset=0):
    if not array.flags.c_contiguous or not array.flags.writeable:
        raise ValueError("Expected a writable contiguous destination array")
    call("cuMemcpyDtoH_v2", ct.c_void_p(array.ctypes.data),
         address(device_ptr, offset), ct.c_size_t(array.nbytes))


def kernel(obj, grid, params):
    """Load the compiled Triton kernel and return (launch, module to unload)."""
    module = ct.c_void_p()
    call("cuModuleLoadData", ct.byref(module), ct.c_char_p(obj.asm["cubin"]))
    try:
        function = ct.c_void_p()
        call("cuModuleGetFunction", ct.byref(function), module, obj.metadata.name.encode())
        shared = obj.metadata.shared
        if shared >= 48 * 1024:
            call("cuFuncSetAttribute", function, 8, shared)
        # Triton may append NULL scratch pointers to the user parameters.
        entry = obj.asm["ptx"].split(".entry", 1)[1].split(")", 1)[0]
        extra = entry.count(".param") - len(params)
        if not 0 <= extra <= 2:
            raise ValueError("Triton ABI differs from the provided parameter list")
        params = list(params) + [ct.c_void_p() for _ in range(extra)]
        for index, parameter in enumerate(params):
            offset, size = ct.c_size_t(), ct.c_size_t()
            call("cuFuncGetParamInfo", function, ct.c_size_t(index), ct.byref(offset), ct.byref(size))
            if ct.sizeof(parameter) != size.value:
                raise ValueError(f"Incorrect argument width at position {index}: expected {size.value}")
        # Casting pointers, rather than integer addresses, keeps param storage alive.
        argv = (ct.c_void_p * len(params))(*(ct.cast(ct.pointer(p), ct.c_void_p) for p in params))
        threads = obj.metadata.num_warps * 32

        def launch():
            call("cuLaunchKernel", function, *grid, threads, 1, 1, shared, ct.c_void_p(), argv, None)

        return launch, module
    except BaseException:
        call("cuModuleUnload", module)
        raise


@contextmanager
def device_session():
    """Bind device 0 on this thread and yield a synchronous CUDA-event timer in ms."""
    call("cuInit", 0)
    major, minor = ct.c_int(), ct.c_int()
    call("cuDeviceGetAttribute", ct.byref(major), 75, 0)
    call("cuDeviceGetAttribute", ct.byref(minor), 76, 0)
    if (major.value, minor.value) != (7, 5):
        raise RuntimeError(f"The compiled kernels require SM75, found SM{major.value}{minor.value}")
    previous, primary = ct.c_void_p(), ct.c_void_p()
    call("cuCtxGetCurrent", ct.byref(previous))
    with ExitStack() as cleanup:
        call("cuDevicePrimaryCtxRetain", ct.byref(primary), 0)
        cleanup.callback(call, "cuDevicePrimaryCtxRelease_v2", 0)
        call("cuCtxSetCurrent", primary)
        cleanup.callback(call, "cuCtxSetCurrent", previous)
        begin, end = ct.c_void_p(), ct.c_void_p()
        for event in (begin, end):
            call("cuEventCreate", ct.byref(event), 0)
            cleanup.callback(call, "cuEventDestroy_v2", event)

        def elapsed(launch):
            call("cuEventRecord", begin, ct.c_void_p())
            launch()
            call("cuEventRecord", end, ct.c_void_p())
            call("cuEventSynchronize", end)
            duration = ct.c_float()
            call("cuEventElapsedTime", ct.byref(duration), begin, end)
            return duration.value

        yield elapsed
