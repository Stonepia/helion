"""
Benchmark script for comparing GEMM + One-Shot All-Reduce implementations
========================================================================
This script compares the performance of three implementations:
1. Helion kernel (using Helion JIT)
2. Triton kernel (from Kraken)
3. Baseline PyTorch (separate GEMM and all-reduce)
"""

import os
import sys
import time
import argparse
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

# Add Kraken to path for Triton implementation
sys.path.append("/data/users/willfeng/kraken")

import helion
from gemm_one_shot_all_reduce_fixed import helion_gemm_one_shot_all_reduce_fixed as helion_gemm_one_shot_all_reduce
from gemm_one_shot_all_reduce import _reference_gemm_one_shot_all_reduce


# Create a wrapper for Triton implementation to handle API differences
def triton_gemm_one_shot_all_reduce_wrapper(a: torch.Tensor, b: torch.Tensor, **kwargs) -> torch.Tensor:
    """Wrapper to adapt the Triton implementation to our benchmark interface."""
    try:
        from kraken.fused.gemm_one_shot_all_reduce_fused import gemm_one_shot_all_reduce as triton_impl
        return triton_impl(a, b, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Triton kernel failed: {e}")


def benchmark_function(
    func: Callable,
    a: torch.Tensor,
    b: torch.Tensor,
    warmup_iters: int = 5,
    benchmark_iters: int = 20,
    func_name: str = "function",
    **kwargs
) -> Dict[str, float]:
    """Benchmark a function and return timing statistics."""

    # Synchronize before starting
    torch.cuda.synchronize()
    dist.barrier()

    # Warmup
    for i in range(warmup_iters):
        try:
            output = func(a.clone(), b.clone(), **kwargs)
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(f"Warmup iteration {i} failed for {func_name}: {e}")

    # Additional barrier between warmup and benchmark
    dist.barrier()

    # Benchmark
    times = []
    for i in range(benchmark_iters):
        # Clone tensors to avoid any state issues
        a_clone = a.clone()
        b_clone = b.clone()

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        try:
            output = func(a_clone, b_clone, **kwargs)
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(f"Benchmark iteration {i} failed for {func_name}: {e}")

        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)  # Convert to ms

    # Calculate statistics
    times_sorted = sorted(times)
    return {
        "name": func_name,
        "mean_ms": sum(times) / len(times),
        "median_ms": times_sorted[len(times) // 2],
        "min_ms": times_sorted[0],
        "max_ms": times_sorted[-1],
        "p95_ms": times_sorted[int(len(times) * 0.95)],
        "p99_ms": times_sorted[int(len(times) * 0.99)],
    }


def calculate_tflops(M: int, N: int, K: int, time_ms: float, world_size: int) -> float:
    """Calculate TFLOPS for GEMM operation."""
    # GEMM is 2*M*N*K operations (multiply + add)
    # All-reduce adds (world_size - 1) * M * N operations
    flops = 2 * M * N * K + (world_size - 1) * M * N
    tflops = flops / (time_ms / 1000) / 1e12
    return tflops


def run_benchmarks(M: int, N: int, K: int, dtype: torch.dtype = torch.float32) -> None:
    """Run benchmarks for all three implementations."""

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    # Generate random input tensors
    torch.manual_seed(42 + rank)
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)

    if rank == 0:
        print(f"\nBenchmarking GEMM + One-Shot All-Reduce")
        print(f"Matrix sizes: A[{M}x{K}] @ B[{K}x{N}] = C[{M}x{N}]")
        print(f"World size: {world_size}")
        print(f"Data type: {dtype}")
        print("=" * 80)

    results = []

    # Benchmark baseline PyTorch
    try:
        if rank == 0:
            print("Benchmarking PyTorch baseline...")
        baseline_stats = benchmark_function(
            _reference_gemm_one_shot_all_reduce,
            a, b,
            func_name="PyTorch Baseline",
            warmup_iters=5,
            benchmark_iters=20
        )
        results.append(baseline_stats)
        if rank == 0:
            print(f"  Completed: {baseline_stats['mean_ms']:.3f} ms average")
    except Exception as e:
        if rank == 0:
            print(f"  Failed: {e}")

    # Small delay between implementations
    dist.barrier()

    # Benchmark Triton kernel
    try:
        if rank == 0:
            print("Benchmarking Triton kernel...")
        triton_stats = benchmark_function(
            triton_gemm_one_shot_all_reduce_wrapper,
            a, b,
            func_name="Triton Kernel",
            warmup_iters=5,
            benchmark_iters=20
        )
        results.append(triton_stats)
        if rank == 0:
            print(f"  Completed: {triton_stats['mean_ms']:.3f} ms average")
    except Exception as e:
        if rank == 0:
            print(f"  Failed: {e}")

    # Small delay between implementations
    dist.barrier()

    # Benchmark Helion kernel - do this last as it might be more prone to hangs
    try:
        if rank == 0:
            print("Benchmarking Helion kernel...")
        helion_stats = benchmark_function(
            helion_gemm_one_shot_all_reduce,
            a, b,
            func_name="Helion Kernel",
            warmup_iters=3,  # Fewer warmup iterations for Helion
            benchmark_iters=10  # Fewer benchmark iterations for Helion
        )
        results.append(helion_stats)
        if rank == 0:
            print(f"  Completed: {helion_stats['mean_ms']:.3f} ms average")
    except Exception as e:
        if rank == 0:
            print(f"  Failed: {e}")

    # Print results
    if rank == 0 and results:
        print("\nPerformance Results:")
        print("-" * 80)
        print(f"{'Implementation':<20} {'Mean (ms)':<12} {'Median (ms)':<12} {'TFLOPS':<12} {'Speedup':<12}")
        print("-" * 80)

        baseline_time = None
        for i, stats in enumerate(results):
            if stats['name'] == "PyTorch Baseline":
                baseline_time = stats["mean_ms"]

            tflops = calculate_tflops(M, N, K, stats["mean_ms"], world_size)
            speedup = baseline_time / stats["mean_ms"] if baseline_time else 1.0

            print(f"{stats['name']:<20} {stats['mean_ms']:<12.3f} {stats['median_ms']:<12.3f} "
                  f"{tflops:<12.2f} {speedup:<12.2f}x")

        print("\nDetailed Statistics:")
        print("-" * 80)
        for stats in results:
            print(f"\n{stats['name']}:")
            print(f"  Min: {stats['min_ms']:.3f} ms")
            print(f"  Max: {stats['max_ms']:.3f} ms")
            print(f"  P95: {stats['p95_ms']:.3f} ms")
            print(f"  P99: {stats['p99_ms']:.3f} ms")


def verify_correctness(M: int, N: int, K: int, dtype: torch.dtype = torch.float32) -> None:
    """Verify that all implementations produce the same result."""

    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # Generate random input tensors
    torch.manual_seed(42 + rank)
    a = torch.randn((M, K), dtype=dtype, device=device)
    b = torch.randn((K, N), dtype=dtype, device=device)

    # Compute reference result
    ref_result = _reference_gemm_one_shot_all_reduce(a.clone(), b.clone())

    # Test Helion kernel
    try:
        helion_result = helion_gemm_one_shot_all_reduce(a.clone(), b.clone())
        torch.testing.assert_close(helion_result, ref_result, rtol=1e-1, atol=1e-1)
        if rank == 0:
            print("✓ Helion kernel correctness verified")
    except Exception as e:
        if rank == 0:
            print(f"✗ Helion kernel correctness check failed: {e}")

    # Test Triton kernel
    try:
        triton_result = triton_gemm_one_shot_all_reduce_wrapper(a.clone(), b.clone())
        torch.testing.assert_close(triton_result, ref_result, rtol=1e-1, atol=1e-1)
        if rank == 0:
            print("✓ Triton kernel correctness verified")
    except Exception as e:
        if rank == 0:
            print(f"✗ Triton kernel correctness check failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GEMM + One-Shot All-Reduce implementations")
    parser.add_argument("--verify", action="store_true", help="Verify correctness before benchmarking")
    parser.add_argument("--sizes", nargs="+", type=int, default=[256, 128, 64],
                        help="Matrix sizes as M N K (default: 256 128 64)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"],
                        default="float32", help="Data type (default: float32)")
    parser.add_argument("--benchmark-sizes", action="store_true",
                        help="Benchmark multiple predefined sizes")

    args = parser.parse_args()

    # Initialize distributed
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Matrix sizes to benchmark
    if args.benchmark_sizes:
        # Predefined sizes for comprehensive benchmarking
        sizes = [
            (128, 128, 128),    # Very small
            (256, 128, 64),     # Small
            (512, 256, 128),    # Medium
            (1024, 512, 256),   # Large
        ]
    else:
        # User-specified size
        if len(args.sizes) != 3:
            raise ValueError("Must specify exactly 3 values for M, N, K")
        sizes = [(args.sizes[0], args.sizes[1], args.sizes[2])]

    # Run benchmarks for each size
    for M, N, K in sizes:
        if args.verify:
            if rank == 0:
                print(f"\nVerifying correctness for size [{M}x{K}] @ [{K}x{N}]...")
            verify_correctness(M, N, K, dtype)

        try:
            run_benchmarks(M, N, K, dtype)
        except Exception as e:
            if rank == 0:
                print(f"\nBenchmark failed for size [{M}x{K}] @ [{K}x{N}]: {e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
        --nproc-per-node 4 \
        --no_python python3 examples/benchmark_gemm_one_shot_all_reduce.py

    Options:
    --verify: Run correctness verification before benchmarking
    --sizes M N K: Specify custom matrix sizes (default: 256 128 64)
    --dtype: Choose float32, float16, or bfloat16
    --benchmark-sizes: Run benchmarks on multiple predefined sizes

    Examples:
    # Run with default small size
    python -m torch.distributed.run --standalone --nproc-per-node 4 --no_python python3 examples/benchmark_gemm_one_shot_all_reduce.py

    # Run with verification and custom size
    python -m torch.distributed.run --standalone --nproc-per-node 4 --no_python python3 examples/benchmark_gemm_one_shot_all_reduce.py --verify --sizes 512 256 128

    # Run comprehensive benchmark
    python -m torch.distributed.run --standalone --nproc-per-node 4 --no_python python3 examples/benchmark_gemm_one_shot_all_reduce.py --benchmark-sizes
    """
    main()