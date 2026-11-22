"""AOT launch of both project quantizers and an independent NumPy packing check."""

from contextlib import ExitStack
from functools import cache
from pathlib import Path
import sys

import numpy as np
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


TASK1 = Path(__file__).resolve().parents[2] / "1_int4-quantization"
sys.path.insert(0, str(TASK1))
from quantize_int4 import RATIOS, _pack_best, _row_max, _score, quantize_pack_kernel, row_max_kernel


def _compile(fn, pointers, constants, **options):
    signature = dict(zip(fn.arg_names, pointers))
    signature.update(dict.fromkeys(constants, "constexpr"))
    return triton.compile(
        ASTSource(fn, signature, constants), target=GPUTarget("cuda", 75, 32),
        options={"num_warps": 4, "num_stages": 3, **options},
    )


@cache
def _quantizers(k):
    count = len(RATIOS)
    return (
        _compile(row_max_kernel, ["*fp16", "*fp32"], {"N": k, "BLOCK_N": 1024}),
        _compile(quantize_pack_kernel, ["*fp16", "*fp32", "*i8", "*fp32"],
                 {"N": k, "BLOCK": 256}),
        _compile(_row_max, ["*fp16", "*fp32"], {"K": k, "BLOCK": 512}),
        _compile(_score, ["*fp16", "*fp32", "*fp32", "*fp64"],
                 {"K": k, "COUNT": count, "BLOCK": 512}, enable_fp_fusion=False),
        _compile(_pack_best, ["*fp16", "*fp32", "*fp32", "*fp64", "*i8", "*fp32"],
                 {"K": k, "COUNT": count, "PADDED_COUNT": triton.next_power_of_2(count), "BLOCK": 256}),
    )


def _reference_row(row):
    weights = row.astype(np.float32)
    abs14 = np.abs(weights) * np.float32(14)
    clips = np.asarray(RATIOS, dtype=np.float32) * np.max(np.abs(weights))
    scales = np.where(clips > 0, clips / np.float32(7), np.float32(1))
    candidates, scores = [], []
    odd = np.arange(1, 14, 2, dtype=np.float32)
    for clip, scale in zip(clips, scales):
        # searchsorted(right) counts thresholds reached, including an exact tie.
        magnitude = np.searchsorted(clip * odd, abs14, side="right") if clip else np.zeros_like(weights)
        codes = np.where(weights < 0, -magnitude, magnitude).astype(np.int8)
        reconstructed = codes.astype(np.float32) * scale
        scores.append(float(np.sum((weights.astype(np.float64) - reconstructed) ** 2)))
        candidates.append(codes)
    idx = int(np.argmin(scores))
    best = idx if scores[idx] < scores[0] * (1 - 1e-12) else 0
    return [(scales[i], candidates[i], scores[i]) for i in (0, best)]


def quantize_both(driver, w, n, k, wcheck, cols):
    """Return absmax and MSE (packed, scales, reference rows), plus sampled SSE."""
    if wcheck.shape != (len(cols), k) or k % 2 or len(cols) == 0:
        raise ValueError("Expected sampled rows of a matrix of even width")
    old_max, old_pack, new_max, score, new_pack = _quantizers(k)
    count = len(RATIOS)
    with ExitStack() as owned, ExitStack() as scratch:
        def allocate(stack, size):
            ptr = driver.alloc(size)
            stack.callback(driver.free, ptr)
            return ptr

        def run(code, grid, params):
            launch, mod = driver.kernel(code, grid, params)
            scratch.callback(driver.call, "cuModuleUnload", mod)
            launch()

        packed = [allocate(owned, n*k//2) for _ in range(2)]
        scales = [allocate(owned, n*4) for _ in range(2)]
        maxima = allocate(scratch, n*4)
        ratios = allocate(scratch, count*4)
        errors = allocate(scratch, n*count*8)
        driver.hcopy(ratios, np.asarray(RATIOS, np.float32))
        blocks = triton.cdiv(k // 2, 256)
        run(old_max, (n, 1, 1), [w, maxima])
        run(old_pack, (n, blocks, 1), [w, maxima, packed[0], scales[0]])
        # Every row is independent. Even the largest case has <18 MiB of scores.
        run(new_max, (n, 1, 1), [w, maxima])
        run(score, (n, count, 1), [w, maxima, ratios, errors])
        run(new_pack, (n, blocks, 1), [w, maxima, ratios, errors, packed[1], scales[1]])
        driver.call("cuCtxSynchronize")

        decoded = [np.empty_like(wcheck, dtype=np.float64) for _ in range(2)]
        sse = [0., 0.]
        changed = 0
        for i, (col, sampled) in enumerate(zip(cols, wcheck)):
            results = _reference_row(sampled)
            changed += int(results[1][0] != results[0][0])
            for scheme, (expected_scale, codes, error) in enumerate(results):
                actual_scale = np.empty(1, np.float32)
                actual_bytes = np.empty(k//2, np.int8)
                driver.download(scales[scheme], actual_scale, int(col)*4)
                driver.download(packed[scheme], actual_bytes, int(col)*(k//2))
                expected_bytes = (codes[::2] & 15) | ((codes[1::2] & 15) << 4)
                np.testing.assert_array_equal(actual_scale, [expected_scale])
                np.testing.assert_array_equal(actual_bytes, expected_bytes)
                decoded[scheme][i] = codes.astype(np.float64) * float(expected_scale)
                sse[scheme] += error
        audit = {"sampled_rows": list(map(int, cols)), "checked_codes_per_scheme": wcheck.size,
                 "changed_scales_in_sample": changed, "sample_absmax_sse": sse[0], "sample_mse_sse": sse[1]}
        owned.pop_all()
        return (packed[0], scales[0], decoded[0]), (packed[1], scales[1], decoded[1]), audit
