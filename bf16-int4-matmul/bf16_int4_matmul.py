"""Inference-only X_bf16 @ W_int4.T with signed, per-row INT4 weights.

Storage contract (shared with the quantization kernel):
    packed: int8 [N, K // 2], K must be even.
    q[n, 2*j] is the low nibble; q[n, 2*j+1] is the high nibble.
    Nibbles use two's complement; the quantizer emits values in [-7, 7].
    scales: float32 [N], W[n, k] = q[n, k] * scales[n].

No full dequantized weight matrix is allocated. Because scales are per-row,
we compute (X @ q.T) * scales in FP32, then cast to the requested output type.
All INT4 integers are exactly representable in BF16. Applying scales after
the reduction avoids rounding every dequantized weight to BF16.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_int4_matmul_kernel(
    x_ptr,
    packed_ptr,
    scales_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_pn,
    stride_pk,
    stride_s,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.arange(0, BLOCK_K)
    packed_k = tl.arange(0, BLOCK_K // 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for tile in range(tl.cdiv(K, BLOCK_K)):
        x_k = tile * BLOCK_K + k
        x = tl.load(
            x_ptr + rows[:, None] * stride_xm + x_k[None, :] * stride_xk,
            mask=(rows[:, None] < M) & (x_k[None, :] < K),
            other=0.0,
        )

        # Load every byte once: a row-major [BLOCK_N, BLOCK_K // 2] tile.
        p_k = tile * (BLOCK_K // 2) + packed_k
        packed = tl.load(
            packed_ptr + cols[:, None] * stride_pn + p_k[None, :] * stride_pk,
            mask=(cols[:, None] < N) & (p_k[None, :] < K // 2),
            other=0,
        ).to(tl.int32)

        low = packed & 0xF
        high = (packed >> 4) & 0xF
        # Two's complement: 0..7 stay positive; 8..15 become -8..-1.
        low = (low ^ 8) - 8
        high = (high ^ 8) - 8
        # join adds a final axis [low, high]; reshape interleaves each pair.
        q = tl.reshape(tl.join(low, high), (BLOCK_N, BLOCK_K))
        weights_t = tl.trans(q.to(tl.bfloat16))
        acc = tl.dot(x, weights_t, acc)

    scale = tl.load(scales_ptr + cols * stride_s, mask=cols < N, other=0.0)
    result = acc * scale[None, :]
    tl.store(
        y_ptr + rows[:, None] * N + cols[None, :],
        result,
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def matmul_bf16_int4(
    x: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Return X @ dequantize(packed, scales).T on the current CUDA stream.

    Args:
        x: BF16 CUDA tensor [M, K].
        packed: INT8 CUDA tensor [N, K // 2], low nibble first.
        scales: FP32 CUDA tensor [N], one finite scale per weight row.
        out_dtype: BF16 (default) or FP32, useful for numerical validation.

    Strided input views are supported; output is contiguous [M, N]. K must
    be even, matching the quantizer's format. This function is for inference
    and does not implement autograd. BF16 dot requires a compatible GPU
    (NVIDIA Ampere or newer); a T4 has no native BF16 support.

    This computes FP32-accumulated X @ q.T followed by FP32 scaling, rather
    than first rounding q * scale to BF16. A BF16-dequantized PyTorch matmul
    therefore has an additional source of rounding error.
    """
    for name, tensor in (("x", x), ("packed", packed), ("scales", scales)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 2 or packed.ndim != 2 or scales.ndim != 1:
        raise ValueError("Expected x [M, K], packed [N, K // 2], scales [N]")
    if x.dtype != torch.bfloat16:
        raise TypeError("x must have dtype torch.bfloat16")
    if packed.dtype != torch.int8:
        raise TypeError("packed must have dtype torch.int8")
    if scales.dtype != torch.float32:
        raise TypeError("scales must have dtype torch.float32")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("out_dtype must be torch.bfloat16 or torch.float32")

    m, k = x.shape
    n = packed.shape[0]
    if k % 2:
        raise ValueError("K (in_features) must be even for two INT4 weights per byte")
    if packed.shape[1] != k // 2:
        raise ValueError("packed.shape must be [N, x.shape[1] // 2]")
    if scales.shape[0] != n:
        raise ValueError("scales must contain exactly one scale per weight row")
    if not (x.is_cuda and packed.is_cuda and scales.is_cuda):
        raise ValueError("x, packed and scales must all be CUDA tensors")
    if x.device != packed.device or x.device != scales.device:
        raise ValueError("x, packed and scales must be on the same CUDA device")

    y = torch.empty((m, n), dtype=out_dtype, device=x.device)
    if m == 0 or n == 0:
        return y
    if k == 0:
        return y.zero_()

    block_m = 16 if m <= 16 else 32 if m <= 32 else 64
    block_n, block_k = 64, 64
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    with torch.cuda.device(x.device):
        _bf16_int4_matmul_kernel[grid](
            x, packed, scales, y,
            m, n, k,
            x.stride(0), x.stride(1),
            packed.stride(0), packed.stride(1), scales.stride(0),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=3,
        )
    return y
