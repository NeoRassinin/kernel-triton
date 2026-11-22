"""Turing FP16 tensor-core probe for contiguous, aligned benchmark shapes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mma16x8(a0, a1, b, c0, c1, c2, c3):
    return tl.inline_asm_elementwise(
        asm="mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5}, {$6}, {$7,$8,$9,$10};",
        constraints="=f,=f,=f,=f,r,r,r,f,f,f,f",
        args=[a0, a1, b, c0, c1, c2, c3],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=True, pack=1,
    )


@triton.jit
def _packed_mma75(X, P, S, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # Each of four warps computes a 16-by-8 output fragment using the PTX
    # lane-to-fragment mapping. Four neighboring lanes own each 8-element K slice.
    tid = tl.inline_asm_elementwise(
        "mov.u32 $0,%tid.x;", "=r,r", [tl.arange(0, 128)],
        dtype=tl.int32, is_pure=True, pack=1,
    )
    warp = tid // 32
    lane = tid % 32
    m = tl.program_id(0) * 32 + (warp // 2) * 16
    n = tl.program_id(1) * 16 + (warp % 2) * 8
    group = lane // 4
    pair = lane % 4
    xpair = X.to(tl.pointer_type(tl.uint32))
    a_ptr = xpair + (m + group) * (K // 2) + pair
    b_ptr = P + (n + group) * (K // 2) + pair
    c0 = tl.full((128,), 0, tl.float32)
    c1 = tl.full((128,), 0, tl.float32)
    c2 = tl.full((128,), 0, tl.float32)
    c3 = tl.full((128,), 0, tl.float32)
    for tile in range(K // 32):
        for step in tl.static_range(4):
            off = tile * 16 + step * 4
            a0 = tl.load(a_ptr + off)
            a1 = tl.load(a_ptr + off + 8 * (K // 2))
            p = tl.load(b_ptr + off).to(tl.uint8).to(tl.int32)
            lo = (((p & 15) ^ 8) - 8).to(tl.float16).to(tl.uint16, bitcast=True)
            hi = ((p >> 4 ^ 8) - 8).to(tl.float16).to(tl.uint16, bitcast=True)
            b = lo.to(tl.uint32) | (hi.to(tl.uint32) << 16)
            c0, c1, c2, c3 = _mma16x8(a0, a1, b, c0, c1, c2, c3)
    col = n + pair * 2
    out = Y + (m + group) * N + col
    scale0 = tl.load(S + col)
    scale1 = tl.load(S + col + 1)
    tl.store(out, c0 * scale0)
    tl.store(out + 1, c1 * scale1)
    tl.store(out + 8 * N, c2 * scale0)
    tl.store(out + 8 * N + 1, c3 * scale1)


def matmul_fp16_int4_mma75(x, packed, scales):
    """SM75 diagnostic for the multiples-of-32/16/32 shapes used in the benchmark."""
    if x.ndim != 2 or packed.ndim != 2 or scales.ndim != 1:
        raise ValueError("Expected x[M,K], packed[N,K/2] and scales[N].")
    if not x.is_cuda or packed.device != x.device or scales.device != x.device:
        raise ValueError("All inputs must be on one CUDA device.")
    if x.dtype != torch.float16 or packed.dtype != torch.int8 or scales.dtype != torch.float32:
        raise TypeError("Expected FP16 activations, INT8 packed weights and FP32 scales.")
    m, k = x.shape
    n = packed.size(0)
    if min(m, n, k) <= 0 or m % 32 or n % 16 or k % 32:
        raise ValueError("This diagnostic kernel requires M, N, K divisible by 32, 16, 32 respectively.")
    if packed.size(1) != k // 2 or scales.size(0) != n:
        raise ValueError("Incompatible packed weight or scale shape.")
    if not all(t.is_contiguous() for t in (x, packed, scales)) or x.data_ptr() % 4:
        raise ValueError("Expected contiguous tensors and a 4-byte aligned activation pointer.")
    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    with torch.cuda.device(x.device):
        _packed_mma75[(m // 32, n // 16)](x, packed, scales, y, m, n, k, num_warps=4, num_stages=1)
    return y
