"""
Fused GEMM + One-Shot All-Reduce Example - Fixed Version
========================================================
This example demonstrates how to fuse a matrix multiplication with a symmetric-memory
one-shot all-reduce using Helion. Each rank computes its local GEMM tile, stages the
result in symmetric memory, and then performs a cross-device reduction by coordinating
through signal pads.

This version fixes the issues identified in the TODO:
1. Uses hl.inline_triton for symm_mem_sync
2. Uses hl.static_range for dynamic buffer tuple handling
3. Adds missing APIs like debug_barrier when needed
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.utils.cpp_extension import load_inline

import helion
from helion._testing import DEVICE
import helion.language as hl


_FROM_BLOB_CPP = """
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>


at::Tensor from_blob(uint64_t data_ptr, c10::IntArrayRef sizes, py::object dtype) {

    at::Tensor tensor = at::for_blob((void*)data_ptr, sizes)
             .deleter([](void *ptr) {
               ;
             })
             .options(at::device(at::kCUDA).dtype(((THPDtype*)dtype.ptr())->scalar_type))
             .make_tensor();

    return tensor;
}
"""

_FROM_BLOB_MOD = load_inline(
    name="helion_symm_mem_from_blob",
    cpp_sources=_FROM_BLOB_CPP,
    functions=["from_blob"],
    with_cuda=True,
)


def _dev_array_to_tensor(
    dev_array_ptr: int, shape: tuple[int, ...], dtype: torch.dtype
) -> torch.Tensor:
    """Creates a tensor view over a device array of pointers."""

    return _FROM_BLOB_MOD.from_blob(dev_array_ptr, shape, dtype)


@helion.jit(
    config=helion.Config(
        block_sizes=[64, 64],
        num_warps=8,
        num_stages=3,
    ),
    static_shapes=True,
)
def _gemm_one_shot_all_reduce_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    buf_tuple: tuple[torch.Tensor, ...],
    signal_pad_ptrs: torch.Tensor,
    output: torch.Tensor,
    BLOCK_SIZE_K: hl.constexpr,
    RANK: hl.constexpr,
    WORLD_SIZE: hl.constexpr,
) -> torch.Tensor:
    M, K = a.size()
    K2, N = b.size()
    assert K2 == K, f"size mismatch {K2} != {K}"

    # Get local buffer
    local_buf = buf_tuple[RANK]

    # Compute local GEMM and write to shared buffer
    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(K, block_size=BLOCK_SIZE_K):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])

        local_buf[tile_m, tile_n] = acc

        # Synchronization using hl.inline_triton for symm_mem_sync
        # First, synchronize to ensure all writes are visible before reads
        sync_id = tile_m.id * helion.cdiv(N, 64) + tile_n.id

        # Define the symm_mem_sync implementation directly in the inline_triton code
        hl.inline_triton(
            """
            import triton.language as tl

            # Get thread ID components
            tid_x = tl.inline_asm_elementwise("mov.u32 $0, %tid.x;", "=r", [], dtype=tl.uint32, is_pure=True, pack=1)
            tid_y = tl.inline_asm_elementwise("mov.u32 $0, %tid.y;", "=r", [], dtype=tl.uint32, is_pure=True, pack=1)
            tid_z = tl.inline_asm_elementwise("mov.u32 $0, %tid.z;", "=r", [], dtype=tl.uint32, is_pure=True, pack=1)
            ntid_x = tl.inline_asm_elementwise("mov.u32 $0, %ntid.x;", "=r", [], dtype=tl.uint32, is_pure=True, pack=1)
            ntid_y = tl.inline_asm_elementwise("mov.u32 $0, %ntid.y;", "=r", [], dtype=tl.uint32, is_pure=True, pack=1)

            flat_tid = tid_z * ntid_y * ntid_x + tid_y * ntid_x + tid_x

            # Set up addresses
            remote_ranks = tl.arange(0, {world_size})
            signal_pad_ptrs_typed = {signal_pad_ptrs}.to(tl.pointer_type(tl.uint64))
            remote_signal_pad_addrs = tl.load(signal_pad_ptrs_typed + remote_ranks).to(
                tl.pointer_type(tl.uint32)
            )
            send_addrs = remote_signal_pad_addrs + {block_id} * {world_size} + {rank}

            local_signal_pad_addr = tl.load(signal_pad_ptrs_typed + {rank}).to(
                tl.pointer_type(tl.uint32)
            )
            wait_addrs = local_signal_pad_addr + {block_id} * {world_size} + remote_ranks

            # First barrier for previous memory access
            tl.debug_barrier()

            if flat_tid < {world_size}:
                # Send signal with release semantics
                tl.inline_asm_elementwise(
                    '''
                    {{
                        .reg .u32   %tmp32_<1>;
                        .reg .pred  %p<1>;

                        send_signal:
                            atom.global.release.sys.cas.b32 %tmp32_0, [$1], 0, 1;
                            setp.eq.u32 %p0, %tmp32_0, 0;
                            @!%p0 bra send_signal;
                    }}
                    ''',
                    "=r, l",
                    [send_addrs],
                    dtype=send_addrs.dtype,
                    is_pure=False,
                    pack=1,
                )

                # Wait for signal with acquire semantics
                tl.inline_asm_elementwise(
                    '''
                    {{
                        .reg .u32   %tmp32_<1>;
                        .reg .pred  %p<1>;

                        wait_signal:
                            atom.global.sys.acquire.cas.b32 %tmp32_0, [$1], 1, 0;
                            setp.eq.u32 %p0, %tmp32_0, 1;
                            @!%p0 bra wait_signal;
                    }}
                    ''',
                    "=r, l",
                    [wait_addrs],
                    dtype=tl.int32,
                    is_pure=False,
                    pack=1,
                )

            # Second barrier for subsequent memory access
            tl.debug_barrier()
            """,
            args={
                "signal_pad_ptrs": signal_pad_ptrs,
                "block_id": sync_id,
                "rank": RANK,
                "world_size": WORLD_SIZE
            },
            output_like=None
        )

        # Perform reduction using hl.static_range for dynamic buffer tuple handling
        reduced = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for i in hl.static_range(WORLD_SIZE):
            shard_buf = buf_tuple[i]
            reduced += shard_buf[tile_m, tile_n]
        output[tile_m, tile_n] = reduced

    return output


def _prepare_signal_pad(
    symm_mem_hdl,
    tiles_m: int,
    tiles_n: int,
) -> torch.Tensor:
    pad = symm_mem_hdl.get_signal_pad(symm_mem_hdl.rank, dtype=torch.int32)
    pad = pad.view(-1, symm_mem_hdl.world_size)
    required = tiles_m * tiles_n
    if required > pad.size(0):
        msg = (
            "Signal pad has insufficient capacity for requested tiling: "
            f"need {required}, have {pad.size(0)}"
        )
        raise RuntimeError(msg)
    pad = pad[:required].contiguous()
    pad.zero_()
    return pad


def helion_gemm_one_shot_all_reduce(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    block_size_k: int = 64,
) -> torch.Tensor:
    """Runs the fused GEMM + one-shot all-reduce kernel."""

    if a.dtype != torch.float32 or b.dtype != torch.float32:
        raise ValueError("Example currently supports float32 inputs")
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match for matrix multiplication")

    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    M, K = a.shape
    _, N = b.shape

    # Allocate shared symmetric memory buffers
    partial = symm_mem.empty((M, N), dtype=torch.float32, device=a.device)
    symm_mem_hdl = symm_mem.rendezvous(partial, group=group)

    buf_tuple = tuple(
        symm_mem_hdl.get_buffer(rank, (M, N), torch.float32)
        for rank in range(symm_mem_hdl.world_size)
    )

    # Prepare signal pad for synchronization
    local_signal_pad = symm_mem_hdl.get_signal_pad(symm_mem_hdl.rank, dtype=torch.int32)

    # Calculate required signal pad size
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    num_tiles_m = helion.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = helion.cdiv(N, BLOCK_SIZE_N)
    num_blocks = num_tiles_m * num_tiles_n

    # Reshape signal pad and verify capacity
    local_signal_pad = local_signal_pad.view(-1, symm_mem_hdl.world_size)
    if num_blocks > local_signal_pad.size(0):
        # If signal pad is too small, we can either:
        # 1. Use a subset of tiles per launch
        # 2. Reuse signal pad locations with proper synchronization
        # For now, we'll error out to make the issue visible
        raise RuntimeError(
            f"Signal pad has insufficient capacity: need {num_blocks}, have {local_signal_pad.size(0)}. "
            f"This typically happens with large matrices. Consider using smaller tile sizes or "
            f"splitting the computation."
        )

    # Use only what we need and ensure it's zeroed
    local_signal_pad = local_signal_pad[:num_blocks].contiguous()
    local_signal_pad.zero_()

    signal_pad_addrs = _dev_array_to_tensor(
        symm_mem_hdl.signal_pad_ptrs_dev,
        (symm_mem_hdl.world_size,),
        torch.uint64,
    )

    output = torch.empty((M, N), dtype=torch.float32, device=a.device)

    return _gemm_one_shot_all_reduce_kernel(
        a,
        b,
        buf_tuple,
        signal_pad_addrs,
        output,
        BLOCK_SIZE_K=block_size_k,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
    )


def _reference_gemm_one_shot_all_reduce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.matmul(a, b)
    dist.all_reduce(out)
    return out


def test(M: int, N: int, K: int, device: torch.device) -> None:
    group = dist.group.WORLD
    if group is None:
        raise RuntimeError("Distributed group is not initialized")

    rank = dist.get_rank()
    a = torch.randn((M, K), dtype=torch.float32, device=device)
    b = torch.randn((K, N), dtype=torch.float32, device=device)

    helion_result = helion_gemm_one_shot_all_reduce(a, b)
    reference = _reference_gemm_one_shot_all_reduce(a, b)

    torch.testing.assert_close(helion_result, reference, rtol=1e-1, atol=1e-1)


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(42 + rank)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    test(512, 256, 128, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    """
    Run with:
    python -m torch.distributed.run --standalone \
    --nproc-per-node 4 \
    --no_python python3 examples/gemm_one_shot_all_reduce_fixed.py
    """

    assert DEVICE.type == "cuda", "Requires CUDA device"
    main()