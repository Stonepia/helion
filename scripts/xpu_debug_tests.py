"""Debug script to test XPU-skipped tests without the skip decorators."""

from __future__ import annotations

import sys
import traceback
import torch
from unittest.mock import patch

sys.path.insert(0, "/home/stonepia/helion")

import helion
import helion.language as hl
import helion._compat as _compat
from helion._testing import check_example, code_and_output

DEVICE = "xpu"
HALF_DTYPE = torch.float16

results: dict[str, str] = {}


def run_test(name: str, fn):
    try:
        fn()
        results[name] = "PASSED"
        print(f"PASSED: {name}")
    except Exception as e:
        results[name] = f"FAILED: {type(e).__name__}: {e}"
        print(f"FAILED: {name}")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        print()


# ─────────────────────────────────────────────────────────────────────────────
# 1. test_attention_block_pointer / test_attention_persistent_interleaved
# ─────────────────────────────────────────────────────────────────────────────
def _test_attention_block_pointer():
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


run_test("test_attention_block_pointer", _test_attention_block_pointer)


def _test_attention_persistent():
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


run_test(
    "test_attention_persistent_interleaved_l2_grouping", _test_attention_persistent
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. test_template_via_closure1 (issue #795)
# ─────────────────────────────────────────────────────────────────────────────
def _test_template_via_closure1():
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


run_test("test_template_via_closure1 (issue #795)", _test_template_via_closure1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. test_bf16xint16 – precision issue
# ─────────────────────────────────────────────────────────────────────────────
def _test_bf16xint16():
    from examples.bf16xint16_gemm import reference_bf16xint16_pytorch  # type: ignore

    m, k, n = 4096, 512, 640  # smaller shape to avoid OOM
    x = torch.randn([m, k], device=DEVICE, dtype=torch.bfloat16)
    w = torch.randint(-(2**15), 2**15 - 1, (k, n), device=DEVICE, dtype=torch.int16)
    check_example(
        "bf16xint16_gemm",
        (x, w),
        reference_bf16xint16_pytorch(x, w, False),
        fn_name="_bf16xint16_gemm",
    )


run_test("test_bf16xint16 (precision)", _test_bf16xint16)


# ─────────────────────────────────────────────────────────────────────────────
# 4. test_python_float_promotion – type promotion issue
# ─────────────────────────────────────────────────────────────────────────────
def _test_python_float_promotion():
    with patch.object(_compat, "_supports_tensor_descriptor", lambda: False):

        @helion.kernel(config={"block_size": 16, "indexing": "block_ptr"})
        def fn(a, beta):
            for tile0 in hl.tile(a.shape[0]):
                b = a[tile0]
                a[tile0] = (1 - beta) * b
            return a

        a = torch.randn(1024, device=DEVICE)
        beta = 1.5
        expected = (1 - beta) * a
        code, out = code_and_output(fn, (a, beta))
        torch.testing.assert_close(out, expected)


run_test("test_python_float_promotion (type promotion)", _test_python_float_promotion)


# ─────────────────────────────────────────────────────────────────────────────
# 5. test_baddbmm_pipeline_debug_dtype_asserts (issue #772)
# ─────────────────────────────────────────────────────────────────────────────
def _test_baddbmm():
    from helion._testing import get_test_dot_precision

    @helion.kernel(
        autotune_effort="none",
        static_shapes=True,
        dot_precision=get_test_dot_precision(),
        debug_dtype_asserts=True,
    )
    def repro_baddbmm_kernel(
        q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor
    ) -> torch.Tensor:
        b_dim = hl.specialize(q_in.size(0))
        m_dim = hl.specialize(q_in.size(1))
        n_dim = hl.specialize(k_in.size(1))
        head_dim = hl.specialize(q_in.size(2))
        assert n_dim == v_in.size(1)
        assert head_dim == k_in.size(2) == v_in.size(2)

        q = q_in
        k = k_in.transpose(1, 2)
        v = v_in

        out = torch.empty_like(q)
        for tile_m, tile_n in hl.tile([m_dim, n_dim]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(head_dim):
                acc = torch.baddbmm(
                    acc, q[:b_dim, tile_m, tile_k], k[:b_dim, tile_k, tile_n]
                )
            out[:b_dim, tile_m, tile_n] = acc
        return out

    B, M, N, H = 2, 64, 64, 32
    q = torch.randn(B, M, H, device=DEVICE, dtype=torch.float16)
    k = torch.randn(B, N, H, device=DEVICE, dtype=torch.float16)
    v = torch.randn(B, N, H, device=DEVICE, dtype=torch.float16)
    code_and_output(repro_baddbmm_kernel, (q, k, v), block_sizes=[16, 16, 16])


run_test("test_baddbmm_debug_dtype_asserts (issue #772)", _test_baddbmm)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Dot accuracy - small M dim
# ─────────────────────────────────────────────────────────────────────────────
def _test_dot_small_m():
    @helion.kernel(autotune_effort="none")
    def matmul_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        m, k = x.size()
        k2, n = y.size()
        out = torch.zeros([m, n], dtype=torch.float32, device=x.device)
        for tile_m, tile_n in hl.tile([m, n]):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = acc + torch.mm(x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = acc
        return out

    x = torch.randn([2, 32], device=DEVICE, dtype=torch.float32)
    y = torch.randn([32, 64], device=DEVICE, dtype=torch.float32)
    expected = x @ y
    code, result = code_and_output(matmul_fn, (x, y), block_sizes=[2, 64, 32])
    torch.testing.assert_close(result, expected, atol=6e-2, rtol=1e-2)


run_test("test_dot_small_m_dim (accuracy)", _test_dot_small_m)


# ─────────────────────────────────────────────────────────────────────────────
# 7. test_worker_crash – 3d device loop
# ─────────────────────────────────────────────────────────────────────────────
def _test_3d_device_loop():
    @helion.kernel
    def device_loop_3d(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        for tile_0, tile_1, tile_2, tile_3 in hl.tile(x.shape):
            out[tile_0, tile_1, tile_2, tile_3] = torch.sin(
                x[tile_0, tile_1, tile_2, tile_3]
            )
        return out

    args = (torch.randn([16, 16, 16, 16], device=DEVICE),)
    code, result = code_and_output(device_loop_3d, args, block_sizes=[1, 8, 8, 8])
    torch.testing.assert_close(result, torch.sin(args[0]))


run_test("test_3d_device_loop (worker crash)", _test_3d_device_loop)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for name, status in results.items():
    print(f"  {status[:6]:6s}  {name}")
