"""Paired SM75 projection microbenchmark, reusing the original point-3 inputs."""
import argparse
from contextlib import ExitStack
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import statistics
from datetime import datetime, timezone

import numpy as np
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROOT = Path(__file__).resolve().parents[2]
import driver
from blaslt import DenseLt
from quantize_driver import quantize_both
from t4_int4_mma import _packed_mma75


SHAPES = [("q_proj / o_proj", 2048, 2048), ("k_proj / v_proj", 512, 2048),
          ("gate_proj / up_proj", 8192, 2048), ("down_proj", 2048, 8192),
          ("lm_head", 128256, 2048)]
ARCHIVE = ROOT / "3_matmul-benchmark/results/gtx1660ti_fp16_int4.csv"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def owned(stack, size):
    ptr = driver.alloc(size)
    stack.callback(driver.free, ptr)
    return ptr


def make_weights(rng, n, k, w, columns):
    samples = np.empty((len(columns), k), dtype=np.float16)
    digest = hashlib.sha256()
    for start in range(0, n, 1024):
        stop = min(start + 1024, n)
        chunk = rng.standard_normal((stop-start, k), dtype=np.float32).astype(np.float16)
        digest.update(memoryview(chunk).cast("B"))
        driver.hcopy(w, chunk, start * k * 2)
        mask = (columns >= start) & (columns < stop)
        samples[mask] = chunk[columns[mask] - start]
    return samples, digest.hexdigest()


def check_result(ptr, n, rows, columns, expected):
    buffer = np.empty(n, dtype=np.float16)
    got = np.empty_like(expected)
    for i, row in enumerate(rows):
        driver.download(ptr, buffer, int(row) * n * 2)
        got[i] = buffer[columns]
    np.testing.assert_allclose(got, expected, rtol=0.005, atol=0.01)
    return float(np.abs(got - expected).max())


def measure(stack, lt, elapsed, rng, n, k, m, w, dense_reference, columns, quant, compiled):
    x = rng.standard_normal((m, k), dtype=np.float32).astype(np.float16)
    x_digest = sha(memoryview(x).cast("B"))
    rows = np.linspace(0, m-1, 16).round().astype(int)
    x_reference = x[rows].astype(np.float64)
    dx, dy = owned(stack, x.nbytes), owned(stack, m*n*2)
    driver.hcopy(dx, x)
    functions = {}
    expected = {"fp16": x_reference @ dense_reference.T}
    for name, (weights, scales, matrix) in quant.items():
        launch, module = driver.kernel(compiled, (m//32, n//16, 1), [dx, weights, scales, dy])
        stack.callback(driver.call, "cuModuleUnload", module)
        functions[name] = launch
        expected[name] = x_reference @ matrix.T
    functions["fp16"] = lt.prepare(dx.value, w.value, dy.value, m, n, k)
    errors = {}
    for _ in range(4):
        for fn in functions.values():
            fn()
            driver.call("cuCtxSynchronize")
    for name, fn in functions.items():
        fn()
        driver.call("cuCtxSynchronize")
        errors[name] = check_result(dy, n, rows, columns, expected[name])
    # All six orders balance first/last position and both directions for each pair.
    orders = list(itertools.permutations(functions))
    blocks = orders if (n, m) == (128256, 2048) else orders + list(reversed(orders))
    samples = {name: [] for name in functions}
    for order in blocks:
        for name in order:
            samples[name].append(elapsed(functions[name]))
    return dict(M=m, N=n, K=k, samples_ms=samples, median_ms={name: statistics.median(v)
                for name, v in samples.items()}, max_abs_error=errors, checked_per_method=256,
                method_orders=blocks, x_sha256=x_digest, matmul_cubin_sha256=sha(compiled.asm["cubin"]),
                cublaslt_workspace_bytes=functions["fp16"].workspace_bytes)


def write_csv(path, rows):
    with ARCHIVE.open(encoding="utf8", newline="") as f:
        historical = {(int(r["M"]), int(r["N"]), int(r["K"])): r for r in csv.DictReader(f)}
    values = []
    for r in rows:
        key = (r["M"], r["N"], r["K"])
        old = historical[key]
        a, b, f = [r["median_ms"][name] for name in ("absmax", "mse", "fp16")]
        past = float(old["int4_mma_ms"])
        row = dict(projection=r["projection"], M=key[0], N=key[1], K=key[2],
                   archived_absmax_ms=past, current_absmax_ms=a, current_mse_ms=b,
                   change_ms_vs_archive_pct=100*(b/past-1), paired_change_ms_pct=100*(b/a-1),
                   archived_fp16_lt_ms=float(old["fp16_cublaslt_ms"]), current_fp16_lt_ms=f,
                   archived_int4_speedup=float(old["speedup_vs_cublaslt"]),
                   current_absmax_speedup=f/a, current_mse_speedup=f/b,
                   trials=len(r["samples_ms"]["mse"]))
        for name, v in r["samples_ms"].items():
            row[name+"_p25_ms"], row[name+"_p75_ms"] = np.percentile(v, [25, 75])
        values.append(row)
    with path.open("x", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    raw = args.output / "measurements.json"
    summary = args.output / "comparison.csv"
    if raw.exists() or summary.exists():
        p.error("Choose a fresh output directory.")
    target = GPUTarget("cuda", 75, 32)
    kernels = {}
    for _, n, k in SHAPES:
        for m in (128, 512, 2048):
            constants = dict(M=m, N=n, K=k)
            sig = dict(X="*fp16", P="*i8", S="*fp32", Y="*fp16", **dict.fromkeys(constants, "constexpr"))
            kernels[m,n,k] = triton.compile(ASTSource(_packed_mma75, sig, constants), target=target,
                                            options={"num_warps": 4, "num_stages": 1})
    result = dict(started_utc=datetime.now(timezone.utc).isoformat(), seed=42, numpy=np.__version__,
                  triton=triton.__version__, python=platform.python_version(),
                  historical_csv_sha256=sha(ARCHIVE.read_bytes()),
                  matmul_source_sha256=sha(Path(__file__).with_name("t4_int4_mma.py").read_bytes()),
                  quantizer_sha256=sha((ROOT / "1_int4-quantization/quantize_int4.py").read_bytes()),
                  shapes=[], rows=[])
    with driver.device_session() as elapsed, ExitStack() as resources:
        import ctypes as ct
        gpu = ct.create_string_buffer(256)
        driver.call("cuDeviceGetName", gpu, len(gpu), 0)
        result["gpu"] = gpu.value.decode()
        driver_api = ct.c_int()
        driver.call("cuDriverGetVersion", ct.byref(driver_api))
        result["cuda_driver_api"] = driver_api.value
        lt = DenseLt()
        resources.callback(lt.close)
        result["cublaslt_version"] = lt.version
        for i, (label, n, k) in enumerate(SHAPES):
            with ExitStack() as weights:
                w = owned(weights, 2*n*k)
                rng = np.random.default_rng(np.random.SeedSequence([42, i]))
                cols = np.linspace(0, n-1, 16).round().astype(int)
                wcheck, digest = make_weights(rng, n, k, w, cols)
                old, new, audit = quantize_both(driver, w, n, k, wcheck, cols)
                quant = dict(absmax=old, mse=new)
                for d, s, _ in quant.values():
                    weights.callback(driver.free, d, s)
                result["shapes"].append(dict(projection=label, W_sha256=digest, audit=audit))
                for m in (128, 512, 2048):
                    with ExitStack() as scope:
                        r = measure(scope, lt, elapsed, rng, n, k, m, w, wcheck.astype(np.float64),
                                    cols, quant, kernels[m,n,k])
                    r["projection"] = label
                    result["rows"].append(r)
                    print(f"{len(result['rows']):2}/15 {label} M={m}: {r['median_ms']}", flush=True)
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    raw.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf8")
    write_csv(summary, result["rows"])


if __name__ == "__main__":
    main()
