"""
Debug script to actually run tests marked with skipIfXPU, bypassing the skip.
This helps determine if the underlying issue is still present.
"""

from __future__ import annotations

import sys
import traceback
import unittest
from unittest.mock import patch

sys.path.insert(0, "/home/stonepia/helion")

import torch

# Patch XPU check to False so skipIfXPU decorators don't activate
# We do this by monkey-patching torch.xpu.is_available BEFORE importing tests
original_is_available = torch.xpu.is_available


def _force_no_xpu():
    return False


# Now patch skipIfXPU to not skip anything
import helion._testing as _testing_mod

original_skipIfXPU = _testing_mod.skipIfXPU


def noop_skipIfXPU(reason: str):
    """Don't skip - we want to test these."""
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    return decorator


_testing_mod.skipIfXPU = noop_skipIfXPU  # type: ignore

# Now import the test modules (they'll use the patched skipIfXPU)
# We do targeted test runs for key categories
DEVICE = "xpu"
results: dict[str, str] = {}


def run_test_method(module_name: str, class_name: str, test_name: str) -> str:
    """Run a specific test method and return pass/fail."""
    import importlib

    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        suite = unittest.TestLoader().loadTestsFromName(test_name, cls)
        runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=0)
        result = runner.run(suite)
        if result.wasSuccessful():
            return "PASSED"
        else:
            errors = result.errors + result.failures
            return f"FAILED: {errors[0][1][:200] if errors else 'unknown'}"
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


print("=" * 70)
print("XPU SKIP ANALYSIS - Running skipped tests to find actual failures")
print("=" * 70)

# Test key XPU-skipped tests by importing them directly

import os

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

# ─────────────────────────────────────────────────────────────
# test_examples.py
# ─────────────────────────────────────────────────────────────
print("\n[test_examples.py]")

# Imports done after patch
import helion._compat as _compat
from helion._testing import check_example, code_and_output
import helion
import helion.language as hl


def try_test(name: str, fn):
    try:
        fn()
        results[name] = "PASSED"
        print(f"  PASS  {name}")
    except unittest.SkipTest as e:
        results[name] = f"SKIPPED: {e}"
        print(f"  SKIP  {name}: {e}")
    except Exception as e:
        results[name] = f"FAILED: {type(e).__name__}: {str(e)[:150]}"
        print(f"  FAIL  {name}")
        print(f"        {type(e).__name__}: {str(e)[:150]}")


# test_split_k_barrier
def t_split_k_barrier():
    m, k, n = 64, 512, 64
    a = torch.randn([m, k], device=DEVICE, dtype=torch.float32)
    b = torch.randn([k, n], device=DEVICE, dtype=torch.float32)
    check_example(
        "split_k_barrier",
        (a, b),
        a @ b,
        fn_name="split_k_matmul",
        block_sizes=[16, 8, 16, 16, 16],
        pid_type="persistent_blocked",
        split_k=64,
    )


try_test("test_split_k_barrier", t_split_k_barrier)

HALF_DTYPE = torch.float16


# test_template_via_closure1 (issue #795)
def t_template_closure1():
    bias = torch.randn([1, 512], device=DEVICE, dtype=HALF_DTYPE)
    args = (
        torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
        torch.randn([512, 512], device=DEVICE, dtype=HALF_DTYPE),
        lambda acc, tile: torch.relu(acc + bias[tile]),
    )
    with patch.object(_compat, "_supports_tensor_descriptor", lambda: False):
        check_example(
            "matmul",
            args,
            torch.relu(args[0] @ args[1] + bias),
            fn_name="matmul",
            emit_code=False,
            block_sizes=[64, 64, 16],
            loop_orders=[[0, 1]],
            num_warps=2,
            num_stages=4,
            indexing="block_ptr",
            l2_grouping=64,
        )


try_test("test_template_via_closure1 (issue #795)", t_template_closure1)


# test_bf16xint16
def t_bf16xint16():
    from examples.bf16xint16_gemm import reference_bf16xint16_pytorch  # type: ignore

    m, k, n = 65536, 1024, 1280
    x = torch.randn([m, k], device=DEVICE, dtype=torch.bfloat16)
    w = torch.randint(-(2**15), 2**15 - 1, (k, n), device=DEVICE, dtype=torch.int16)
    check_example(
        "bf16xint16_gemm",
        (x, w),
        reference_bf16xint16_pytorch(x, w, False),
        fn_name="_bf16xint16_gemm",
    )


try_test("test_bf16xint16 (precision diff)", t_bf16xint16)


# test_attention_block_pointer
def t_attn_block_ptr():
    with patch.object(_compat, "_supports_tensor_descriptor", lambda: False):
        args = (
            torch.randn(2, 32, 1024, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 32, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            torch.nn.functional.scaled_dot_product_attention(*args),
            block_sizes=[16, 32, 16],
            num_stages=1,
            indexing="block_ptr",
        )


try_test("test_attention_block_pointer", t_attn_block_ptr)


# test_attention_persistent_interleaved_l2_grouping
def t_attn_persistent():
    with patch.object(_compat, "_supports_tensor_descriptor", lambda: False):
        args = (
            torch.randn(2, 16, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 16, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
            torch.randn(2, 16, 512, 64, dtype=HALF_DTYPE, device=DEVICE),
        )
        check_example(
            "attention",
            args,
            torch.nn.functional.scaled_dot_product_attention(*args),
            block_sizes=[16, 32, 16],
            num_stages=1,
            pid_type="persistent_interleaved",
            l2_grouping=4,
            indexing="block_ptr",
        )


try_test("test_attention_persistent_interleaved_l2_grouping", t_attn_persistent)


# test_python_float_promotion
def t_float_promo():
    with patch.object(_compat, "_supports_tensor_descriptor", lambda: False):

        @helion.kernel(config={"block_size": 16, "indexing": "block_ptr"})
        def fn(a, beta):
            for tile0 in hl.tile(a.shape[0]):
                b = a[tile0]
                a[tile0] = (1 - beta) * b
            return a

        a = torch.randn(1024, device=DEVICE)
        expected = (1 - 1.5) * a
        _, out = code_and_output(fn, (a, 1.5))
        torch.testing.assert_close(out, expected)


try_test("test_python_float_promotion", t_float_promo)


# test_gather_gemv (timeout)
def t_gather_gemv():
    args = (
        torch.randn([4, 512, 512], device=DEVICE, dtype=torch.float32),
        torch.randint(0, 4, [2], device=DEVICE, dtype=torch.int32),
        torch.randn([512], device=DEVICE, dtype=torch.float32),
    )

    def expected(w, idx, x):
        return w[idx].to(x.dtype) @ x

    check_example(
        "gather_gemv",
        args,
        expected(*args),
        fn_name="gather_gemv",
        emit_code=False,
        block_sizes=[16, 16],
        num_warps=8,
        num_stages=1,
    )


try_test("test_gather_gemv (timeout)", t_gather_gemv)


# ─────────────────────────────────────────────────────────────
# test_loops.py - worker crash tests
# ─────────────────────────────────────────────────────────────
print("\n[test_loops.py - worker crash]")


@helion.kernel
def device_loop_3d(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_0, tile_1, tile_2, tile_3 in hl.tile(x.shape):
        out[tile_0, tile_1, tile_2, tile_3] = torch.sin(
            x[tile_0, tile_1, tile_2, tile_3]
        )
    return out


def t_3d_loop():
    args = (torch.randn([16, 16, 16, 16], device=DEVICE),)
    _, result = code_and_output(device_loop_3d, args, block_sizes=[1, 8, 8, 8])
    torch.testing.assert_close(result, torch.sin(args[0]))


try_test("test_3d_device_loop (worker crash)", t_3d_loop)


# ─────────────────────────────────────────────────────────────
# test_loops.py - accuracy issue
# ─────────────────────────────────────────────────────────────
def t_unroll_with_pipelining():
    with patch.object(_compat, "_supports_tensor_descriptor", lambda: False):

        @helion.kernel(static_shapes=True)
        def matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, k = x.size()
            k2, n = y.size()
            assert k == k2
            out = torch.empty(
                [m, n],
                dtype=torch.promote_types(x.dtype, y.dtype),
                device=x.device,
            )
            for tile_m, tile_n in hl.tile([m, n]):
                acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
                out[tile_m, tile_n] = acc
            return out

        x = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.float32)
        _, result = code_and_output(
            matmul,
            (x, y),
            block_sizes=[32, 32, 32],
            range_unroll_factors=[2],
            num_stages=2,
        )
        torch.testing.assert_close(result, x @ y, atol=1e-3, rtol=1e-3)


try_test("test_unroll_with_pipelining (accuracy)", t_unroll_with_pipelining)

# ─────────────────────────────────────────────────────────────
# test_config_api.py - cuda device direct
# ─────────────────────────────────────────────────────────────
print("\n[test_config_api.py]")


def t_distributed_pid_types():
    from helion._compiler.compile_environment import CompileEnvironment

    settings = helion.Settings()
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("helion._dist_utils.max_num_blocks_for_symm_mem", return_value=10000),
    ):
        # Changed 'cuda' to DEVICE to fix the XPU issue
        try:
            env = CompileEnvironment(torch.device("cuda", 0), settings)
            print(
                "  NOTE: torch.device('cuda') works even on XPU system - test logic issue"
            )
        except Exception as e:
            # Try with xpu device
            env = CompileEnvironment(torch.device(DEVICE, 0), settings)
    assert env.config_spec.allowed_pid_types == (
        "persistent_blocked",
        "persistent_interleaved",
    )


try_test("test_distributed_limits_pid_types (cuda hardcode)", t_distributed_pid_types)


# ─────────────────────────────────────────────────────────────
# test_autotuner.py - maxnreg
# ─────────────────────────────────────────────────────────────
print("\n[test_autotuner.py - maxnreg]")


def t_maxnreg_config_fragment():
    """Test config fragment with maxnreg - does it work at all on XPU?"""
    from helion.autotuner.config_generation import ConfigGeneration
    from helion.autotuner.loops import _supports_warp_specialize
    from helion._testing import check_example

    # The test patches _supports_maxnreg to True and _supports_tensor_descriptor to True
    # But on XPU, the actual config might differ
    with (
        patch.object(_compat, "_supports_tensor_descriptor", lambda: True),
        patch.object(_compat, "_min_dot_size", lambda *args: (16, 16, 16)),
        patch.object(_compat, "_supports_maxnreg", lambda: True),
    ):
        import sys

        sys.path.insert(0, "test")
        from test_autotuner import _get_examples_matmul  # type: ignore

        args = (
            torch.randn([512, 512], device=DEVICE),
            torch.randn([512, 512], device=DEVICE),
        )
        spec = _get_examples_matmul().bind(args).config_spec
        configs = ConfigGeneration(spec).random_population(10)
        print(f"  Generated {len(configs)} configs")


try_test(
    "test_config_fragment0 (maxnreg CUDA-specific query)", t_maxnreg_config_fragment
)

# ─────────────────────────────────────────────────────────────
# test_type_propagation.py - CUDA-only
# ─────────────────────────────────────────────────────────────
print("\n[test_type_propagation.py - CUDA-only]")


def t_cuda_device_properties():
    """Tests using torch.cuda.get_device_properties - should fail on XPU."""

    @helion.kernel
    def use_device_properties(x: torch.Tensor) -> torch.Tensor:
        device = x.device
        props = torch.cuda.get_device_properties(device)
        sm_count = props.multi_processor_count
        n = x.shape[0]
        out = torch.zeros_like(x)
        for worker_id in hl.grid(sm_count):
            for i in hl.grid(n):
                idx = worker_id + i * sm_count
                if idx < n:
                    out[idx] = x[idx]
        return out

    x = torch.ones([128], device="cuda")  # noqa - deliberately uses cuda
    from helion._testing import type_propagation_report

    output = type_propagation_report(use_device_properties, x)
    print(f"  Output generated: {len(output)} chars")


try_test(
    "test_cuda_device_properties (CUDA-only, uses cuda device)",
    t_cuda_device_properties,
)

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
passed = [k for k, v in results.items() if v == "PASSED"]
failed = [k for k, v in results.items() if v.startswith("FAILED")]
skipped = [k for k, v in results.items() if v.startswith("SKIPPED")]
print(f"\nPASSED  ({len(passed)}):")
for k in passed:
    print(f"  - {k}")
print(f"\nFAILED  ({len(failed)}):")
for k in failed:
    print(f"  - {k}")
    print(f"    {results[k][:200]}")
print(f"\nSKIPPED ({len(skipped)}):")
for k in skipped:
    print(f"  - {k}: {results[k]}")
