from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import call
from unittest.mock import patch

import pytest
import torch

from helion import _compat
from helion._hardware import get_hardware_info
from helion.autotuner.aot_cache import clear_heuristic_cache
from helion.autotuner.aot_cache import find_heuristic_file
from helion.autotuner.external import _ExternalKernelAdapter
from helion.autotuner.external import create_user_config_spec


def test_implicit_hardware_discovery_falls_back_to_current_xpu() -> None:
    props = SimpleNamespace(
        name="Intel Test XPU",
        driver_version="1.2.3",
    )
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.xpu, "is_available", return_value=True),
        patch.object(torch.xpu, "current_device", return_value=2),
        patch.object(
            torch.xpu, "get_device_properties", return_value=props
        ) as get_props,
    ):
        get_hardware_info.cache_clear()
        info = get_hardware_info()

    get_hardware_info.cache_clear()
    assert info.device_kind == "xpu"
    assert info.hardware_name == props.name
    get_props.assert_called_once_with(torch.device("xpu", 2))


def test_implicit_hardware_discovery_tracks_current_xpu() -> None:
    props = [
        SimpleNamespace(name="Intel Test XPU 0", driver_version="1.0"),
        SimpleNamespace(name="Intel Test XPU 1", driver_version="1.1"),
    ]
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.xpu, "is_available", return_value=True),
        patch.object(torch.xpu, "current_device", side_effect=(0, 1)),
        patch.object(
            torch.xpu, "get_device_properties", side_effect=props
        ) as get_props,
    ):
        get_hardware_info.cache_clear()
        first = get_hardware_info()
        second = get_hardware_info()

    get_hardware_info.cache_clear()
    assert first is not second
    assert first.hardware_name == props[0].name
    assert second.hardware_name == props[1].name
    assert get_props.call_args_list == [
        call(torch.device("xpu", 0)),
        call(torch.device("xpu", 1)),
    ]


def test_implicit_hardware_discovery_preserves_cuda_priority() -> None:
    props = SimpleNamespace(name="NVIDIA Test GPU", major=9, minor=0)
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "current_device", return_value=1),
        patch.object(
            torch.cuda, "get_device_properties", return_value=props
        ) as get_props,
        patch.object(torch.xpu, "is_available", return_value=True),
        patch.object(torch.version, "cuda", "12.8"),
        patch.object(torch.version, "hip", None),
    ):
        get_hardware_info.cache_clear()
        info = get_hardware_info()

    get_hardware_info.cache_clear()
    assert info.device_kind == "cuda"
    get_props.assert_called_once_with(torch.device("cuda", 0))


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
def test_external_adapter_detects_xpu_tensor_device() -> None:
    arg = torch.ones(1, device="xpu")
    adapter = _ExternalKernelAdapter(
        create_user_config_spec({}),
        lambda config: lambda tensor: tensor,
        (arg,),
    )
    assert adapter.env.device == arg.device

    explicit = _ExternalKernelAdapter(
        create_user_config_spec({}),
        lambda config: lambda tensor: tensor,
        (arg,),
        device=torch.device("cpu"),
    )
    assert explicit.env.device == torch.device("cpu")


def test_external_adapter_detects_bare_device_argument() -> None:
    target = torch.device("xpu", 3)
    adapter = _ExternalKernelAdapter(
        create_user_config_spec({}),
        lambda config: lambda device: device,
        (target,),
    )
    assert adapter.env.device == target


def test_min_dot_size_queries_requested_xpu() -> None:
    device = torch.device("xpu", 3)
    props = SimpleNamespace(name="Intel Test XPU")
    fake_min_dot_size = Mock(return_value=lambda lhs, rhs: (8, 16, 16))
    with (
        patch.object(torch.xpu, "is_available", return_value=True),
        patch.object(
            torch.xpu, "get_device_properties", return_value=props
        ) as get_props,
        patch("triton.backends.intel.compiler.min_dot_size", fake_min_dot_size),
    ):
        _compat._min_dot_size.cache_clear()
        result = _compat.min_dot_size(device, torch.bfloat16, torch.bfloat16)

    _compat._min_dot_size.cache_clear()
    assert result == (8, 16, 16)
    get_props.assert_called_once_with(device)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU is required")
def test_implicit_xpu_hardware_discovery_supports_aot_lookup() -> None:
    get_hardware_info.cache_clear()
    clear_heuristic_cache()
    info = get_hardware_info()
    heuristic = find_heuristic_file(Path("xpu_aot_smoke.py"))
    assert info.device_kind == "xpu"
    assert heuristic is None


@pytest.mark.skipif(torch.xpu.device_count() < 2, reason="Two XPUs are required")
def test_min_dot_size_accepts_noncurrent_xpu() -> None:
    current = torch.xpu.current_device()
    target = 1 if current != 1 else 0
    _compat._min_dot_size.cache_clear()
    result = _compat.min_dot_size(
        torch.device("xpu", target), torch.bfloat16, torch.bfloat16
    )
    assert len(result) == 3
