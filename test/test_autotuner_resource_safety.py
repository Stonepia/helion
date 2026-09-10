from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from helion import exc
from helion.autotuner.base_cache import AutotuneCacheBase
from helion.autotuner.benchmark_provider import LocalBenchmarkProvider


def _make_provider(device: torch.device, jobs: int = 8) -> LocalBenchmarkProvider:
    provider = object.__new__(LocalBenchmarkProvider)
    provider.settings = SimpleNamespace(
        autotune_precompile="spawn",
        autotune_precompile_jobs=jobs,
    )
    provider.args = (torch.empty(1024, dtype=torch.uint8),)
    provider._baseline_output = torch.empty(1024, dtype=torch.uint8)
    provider.kernel = SimpleNamespace(env=SimpleNamespace(device=device))
    provider.log = Mock()
    return provider


def test_xpu_spawn_jobs_are_capped_by_available_memory() -> None:
    provider = _make_provider(torch.device("xpu", 3))
    with patch.object(
        torch.xpu, "mem_get_info", return_value=(3 * 4096, 1 << 30)
    ) as mem:
        jobs = provider._decide_num_jobs()

    assert jobs == 3
    mem.assert_called_once_with(torch.device("xpu", 3))
    provider.log.warning.assert_called_once()


def test_xpu_spawn_jobs_raise_when_one_job_does_not_fit() -> None:
    provider = _make_provider(torch.device("xpu", 2))
    with (
        patch.object(torch.xpu, "mem_get_info", return_value=(4095, 1 << 30)),
        pytest.raises(exc.AutotuneError, match="requires at least one job"),
    ):
        provider._decide_num_jobs()


def test_cuda_spawn_job_cap_is_unchanged() -> None:
    provider = _make_provider(torch.device("cuda", 1))
    with (
        patch.object(
            torch.cuda, "mem_get_info", return_value=(2 * 4096, 1 << 30)
        ) as cuda_mem,
        patch.object(torch.xpu, "mem_get_info") as xpu_mem,
    ):
        jobs = provider._decide_num_jobs()

    assert jobs == 2
    cuda_mem.assert_called_once_with(torch.device("cuda", 1))
    xpu_mem.assert_not_called()


def test_release_trial_state_cleans_current_accelerator() -> None:
    with (
        patch("helion.autotuner.base_cache.gc.collect") as collect,
        patch.object(torch.accelerator, "is_available", return_value=True),
        patch.object(torch.accelerator, "synchronize") as synchronize,
        patch.object(torch.accelerator.memory, "empty_cache") as empty_cache,
        patch.object(torch.cuda, "is_available", return_value=False),
    ):
        AutotuneCacheBase._release_trial_state(object())

    collect.assert_called_once_with()
    synchronize.assert_called_once_with()
    empty_cache.assert_called_once_with()
