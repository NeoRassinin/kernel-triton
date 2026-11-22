"""GPU quantization of FP16/BF16 weight matrices into signed packed INT4."""

import torch
import triton
import triton.language as tl


@triton.jit
def row_max_kernel(X, RowMax, N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    row_max = tl.full((), 0.0, tl.float32)
    for start in range(tl.cdiv(N, BLOCK_N)):
        j = start * BLOCK_N + cols
        x = tl.load(X + row * N + j, mask=j < N, other=0.0).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(tl.abs(x), 0))
    tl.store(RowMax + row, row_max)


@triton.jit
def quantize_pack_kernel(X, RowMax, Packed, Scales,
                         N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    max_val = tl.load(RowMax + row)

    scale = tl.where(max_val > 0.0, tl.div_rn(max_val, 7.0), 1.0)
    if block == 0:
        tl.store(Scales + row, scale)

    j = block * BLOCK + tl.arange(0, BLOCK)
    even = 2 * j
    odd = even + 1
    x0 = tl.load(X + row * N + even, mask=even < N, other=0.0).to(tl.float32)
    x1 = tl.load(X + row * N + odd, mask=odd < N, other=0.0).to(tl.float32)

    a0 = tl.abs(x0) * 14.0
    a1 = tl.abs(x1) * 14.0
    q0_abs = tl.full((BLOCK,), 0, tl.int32)
    q1_abs = tl.full((BLOCK,), 0, tl.int32)
    for i in tl.static_range(0, 7):
        threshold = max_val * (2 * i + 1)
        q0_abs += ((max_val > 0.0) & (a0 >= threshold)).to(tl.int32)
        q1_abs += ((max_val > 0.0) & (a1 >= threshold)).to(tl.int32)

    q0 = tl.where(x0 < 0.0, -q0_abs, q0_abs)
    q1 = tl.where(x1 < 0.0, -q1_abs, q1_abs)
    byte = ((q0 & 15) | ((q1 & 15) << 4)).to(tl.int8)
    tl.store(Packed + row * (N // 2) + j, byte, mask=j < N // 2)


@triton.jit
def fused_quantize_pack_kernel(X, Packed, Scales,
                               N: tl.constexpr, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, BLOCK_N)
    x = tl.load(X + row * N + j, mask=j < N, other=0.0).to(tl.float32)
    max_val = tl.max(tl.abs(x), 0)
    scale = tl.where(max_val > 0.0, tl.div_rn(max_val, 7.0), 1.0)
    tl.store(Scales + row, scale)

    # Квантизация выполняется над всеми элементами строки (пороговая
    # арифметика как в baseline), затем упаковываются пары соседей.
    a = tl.abs(x) * 14.0
    q_abs = tl.full((BLOCK_N,), 0, tl.int32)
    for k in tl.static_range(0, 7):
        threshold = max_val * (2 * k + 1)
        q_abs += ((max_val > 0.0) & (a >= threshold)).to(tl.int32)
    q = tl.where(x < 0.0, -q_abs, q_abs)

    # Gather по четным и нечетным позициям; по одной паре на байт.
    pairs = tl.arange(0, BLOCK_N // 2)
    q0 = tl.gather(q, 2 * pairs, axis=0)
    q1 = tl.gather(q, 2 * pairs + 1, axis=0)
    packed = ((q0 & 15) | ((q1 & 15) << 4)).to(tl.int8)
    tl.store(Packed + row * (N // 2) + pairs, packed, mask=pairs < N // 2)


RATIOS = (1.0, 0.99, 0.975, 0.95, 0.925, 0.90, 0.875, 0.85, 0.825,
          0.80, 0.775, 0.75, 0.725, 0.70, 0.65, 0.60, 0.55, 0.50)


@triton.jit
def _abs_code(x, clip):
    # Same threshold comparisons as task 1, including ties and the zero row.
    a14 = tl.abs(x) * 14.0
    code = tl.full(x.shape, 0, tl.int32)
    for i in tl.static_range(7):
        code += (a14 >= clip * (2 * i + 1)).to(tl.int32)
    return tl.where(clip > 0.0, code, 0)


@triton.jit
def _scale(clip):
    return tl.where(clip > 0.0, tl.div_rn(clip, 7.0), 1.0)


@triton.jit
def _row_max(W, Maxima, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    maximum = tl.full((BLOCK,), 0.0, tl.float32)
    for tile in range(tl.cdiv(K, BLOCK)):
        col = tile * BLOCK + lane
        x = tl.load(W + row * K + col, col < K, other=0.0).to(tl.float32)
        maximum = tl.maximum(maximum, tl.abs(x))
    tl.store(Maxima + row, tl.max(maximum, 0))


@triton.jit
def _score(W, Maxima, Ratios, Scores, K: tl.constexpr, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    row, candidate = tl.program_id(0), tl.program_id(1)
    clip = tl.load(Maxima + row) * tl.load(Ratios + candidate)
    scale = _scale(clip)
    lane = tl.arange(0, BLOCK)
    sums = tl.full((BLOCK,), 0.0, tl.float64)
    for tile in range(tl.cdiv(K, BLOCK)):
        col = tile * BLOCK + lane
        x = tl.load(W + row * K + col, col < K, other=0.0).to(tl.float32)
        reconstructed_magnitude = _abs_code(x, clip).to(tl.float32) * scale
        error = tl.abs(x).to(tl.float64) - reconstructed_magnitude.to(tl.float64)
        sums += error * error
    tl.store(Scores + row * COUNT + candidate, tl.sum(sums, 0))


@triton.jit
def _pack_best(W, Maxima, Ratios, Scores, Packed, Scales, K: tl.constexpr,
               COUNT: tl.constexpr, PADDED_COUNT: tl.constexpr, BLOCK: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    candidate = tl.arange(0, PADDED_COUNT)
    row_scores = Scores + row * COUNT
    scores = tl.load(row_scores + candidate, candidate < COUNT, other=float("inf"))
    # The first minimum prefers the largest retained range on equal scores.
    best = tl.argmin(scores, axis=0, tie_break_left=True)
    # Keep the canonical baseline when improvement is within FP64 reduction noise.
    best = tl.where(tl.min(scores, 0) < tl.load(row_scores) * (1.0 - 1e-12), best, 0)
    clip = tl.load(Maxima + row) * tl.load(Ratios + best)
    if tile == 0:
        tl.store(Scales + row, _scale(clip))
    j = tile * BLOCK + tl.arange(0, BLOCK)
    even = tl.load(W + row * K + 2 * j, j < K // 2, other=0.0).to(tl.float32)
    odd = tl.load(W + row * K + 2 * j + 1, j < K // 2, other=0.0).to(tl.float32)
    a, b = _abs_code(even, clip), _abs_code(odd, clip)
    a = tl.where(even < 0, -a, a)
    b = tl.where(odd < 0, -b, b)
    byte = ((a & 15) | ((b & 15) << 4)).to(tl.int8)
    tl.store(Packed + row * (K // 2) + j, byte, j < K // 2)


@torch.no_grad()
def quantize_mse(w):
    """Return int8 packed weights [rows, K/2] and one FP32 scale per row.

    Takes a CUDA FP16/BF16 matrix of finite model weights with even positive K.
    Candidate 1.0 reproduces the project's absmax quantizer. Each row selects
    a scale by SSE between the original weights and the FP32 reconstruction.
    """
    if w.ndim != 2 or w.shape[1] == 0 or w.shape[1] % 2:
        raise ValueError("Expected a matrix with positive, even width")
    if not w.is_cuda or w.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Expected CUDA FP16 or BF16 weights")
    w = w.contiguous()
    rows, width = w.shape
    packed = torch.empty((rows, width // 2), device=w.device, dtype=torch.int8)
    scales = torch.empty(rows, device=w.device, dtype=torch.float32)
    if rows == 0:
        return packed, scales
    count = len(RATIOS)
    tile = min(512, triton.next_power_of_2(width))
    pack_tile = min(256, triton.next_power_of_2(width // 2))
    chunk = min(rows, 2048)
    # Reuse the scratch on the current CUDA stream, including for the LM head.
    with torch.cuda.device(w.device):
        ratios = torch.tensor(RATIOS, device=w.device, dtype=torch.float32)
        maxima = torch.empty(chunk, device=w.device, dtype=torch.float32)
        scores = torch.empty((chunk, count), device=w.device, dtype=torch.float64)
        for first in range(0, rows, chunk):
            n = min(rows - first, chunk)
            part = w[first:first + n]
            _row_max[(n,)](part, maxima, width, tile)
            _score[(n, count)](part, maxima, ratios, scores, width, count, tile, enable_fp_fusion=False)
            _pack_best[(n, triton.cdiv(width // 2, pack_tile))](
                part, maxima, ratios, scores, packed[first:first + n], scales[first:first + n],
                width, count, triton.next_power_of_2(count), pack_tile)
    return packed, scales


def quantize_fp16_to_int4_packed(W, method="two_pass"):
    """Симметричная per-row FP16/BF16 -> packed INT4 + FP32 scales.

    method="mse" selects the least-error scale from 18 candidates per row.
    "two_pass" (default) and "fused" use the original absmax scale.
    Контракт входа: CUDA, конечные числа, 2D, N чётное.
    Проверка конечности не выполняется внутри, чтобы не добавлять GPU sync.
    """
    assert W.is_cuda and W.ndim == 2
    assert W.dtype in (torch.float16, torch.bfloat16)
    M, N = W.shape
    assert M > 0 and N > 0 and N % 2 == 0
    W = W.contiguous()
    if method == "mse":
        return quantize_mse(W)
    if method not in ("two_pass", "fused"):
        raise ValueError("method must be 'mse', 'two_pass' or 'fused'")
    packed = torch.empty((M, N // 2), device=W.device, dtype=torch.int8)
    scales = torch.empty((M,), device=W.device, dtype=torch.float32)
    if method == "two_pass":
        row_max = torch.empty((M,), device=W.device, dtype=torch.float32)
        row_max_kernel[(M,)](W, row_max, N, BLOCK_N=1024, num_warps=4)
        B = 256
        quantize_pack_kernel[(M, triton.cdiv(N // 2, B))](
            W, row_max, packed, scales, N, B, num_warps=4
        )
    else:
        fused_quantize_pack_kernel[(M,)](
            W, packed, scales, N, BLOCK_N=triton.next_power_of_2(N), num_warps=4
        )
    return packed, scales
