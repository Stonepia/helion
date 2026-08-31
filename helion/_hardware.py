from __future__ import annotations

import dataclasses
import functools
import re
from typing import cast

import torch

# Linear compute-capability fallback lists (newest to oldest). ``sm103`` is not
# a safe generic fallback for arbitrary future CUDA architectures, so it stays
# outside this chain: an exact sm103 target still falls back to sm100 through
# the unknown-current path, without offering sm103 artifacts to future targets.
_CUDA_COMPUTE_CAPS: list[str] = [
    "sm100",
    "sm90",
    "sm89",
    "sm87",
    "sm86",
    "sm80",
    "sm75",
    "sm72",
    "sm70",
]

_ROCM_ARCHS: list[str] = [
    "gfx950",
    "gfx942",
    "gfx941",
    "gfx940",
    "gfx90a",
    "gfx908",
    "gfx906",
    "gfx900",
]


@dataclasses.dataclass(frozen=True)
class HardwareInfo:
    """
    Hardware information for cache keys and heuristic selection.

    Attributes:
        device_kind: Device type ('cuda', 'rocm', 'xpu')
        hardware_name: Device name (e.g., 'NVIDIA H100', 'gfx90a')
        runtime_version: Runtime version (e.g., '12.4', 'gfx90a')
        compute_capability: Compute capability for heuristics (e.g., 'sm90', 'gfx90a')
    """

    device_kind: str
    hardware_name: str
    runtime_version: str
    compute_capability: str

    @property
    def hardware_id(self) -> str:
        """Get a unique identifier string for this hardware."""
        safe_name = self.hardware_name.replace(" ", "_")
        return f"{self.device_kind}_{safe_name}_{self.runtime_version}"

    def get_compatible_compute_ids(self) -> list[str]:
        """
        Get a list of compatible compute IDs for fallback, ordered from current to oldest.

        For CUDA/ROCm, returns the current compute capability followed by all older
        compatible architectures. This allows using heuristics tuned on older hardware
        when newer hardware-specific heuristics aren't available.
        """
        if self.device_kind == "cuda":
            arch_list = _CUDA_COMPUTE_CAPS
        elif self.device_kind == "rocm":
            arch_list = _ROCM_ARCHS
        else:
            return [self.compute_capability]

        try:
            current_idx = arch_list.index(self.compute_capability)
            return arch_list[current_idx:]
        except ValueError:
            return [self.compute_capability, *arch_list]


def _xpu_arch_name(props: object) -> str:
    """Short, stable architecture token for an Intel GPU (e.g. ``pvc``).

    Mirrors the granularity of CUDA's ``sm90`` / ROCm's ``gfx942``: one token
    per architecture rather than per SKU, so a heuristic tuned on a Data Center
    GPU Max 1550 also applies to a 1100.  Without Triton (e.g. a Pallas-only
    install) fall back to a slugified device name, which is still stable and
    filesystem safe.
    """
    from ._compat import xpu_arch_name

    arch = xpu_arch_name()
    if arch is not None:
        return arch
    name = cast("str", props.name)  # pyrefly: ignore [missing-attribute]
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _xpu_hardware_info(device: torch.device | None) -> HardwareInfo:
    props = torch.xpu.get_device_properties(device)
    return HardwareInfo(
        device_kind="xpu",
        hardware_name=props.name,
        runtime_version=props.driver_version,
        # XPU has no CUDA-style compute capability; use the architecture token.
        compute_capability=_xpu_arch_name(props),
    )


@functools.cache
def get_hardware_info(device: torch.device | None = None) -> HardwareInfo:
    """
    Get hardware information for the current or specified device.

    Args:
        device: Optional device to get info for. If None, uses first available GPU or CPU.

    Returns:
        HardwareInfo with device details for caching and heuristic lookup.
    """
    # XPU (Intel) path
    if device is not None and device.type == "xpu" and torch.xpu.is_available():
        return _xpu_hardware_info(device)

    # CUDA/ROCm path
    if torch.cuda.is_available():
        dev = (
            device
            if device is not None and device.type == "cuda"
            else torch.device("cuda:0")
        )
        props = torch.cuda.get_device_properties(dev)

        if torch.version.cuda is not None:
            return HardwareInfo(
                device_kind="cuda",
                hardware_name=props.name,
                runtime_version=str(torch.version.cuda),
                compute_capability=f"sm{props.major}{props.minor}",
            )
        if torch.version.hip is not None:
            return HardwareInfo(
                device_kind="rocm",
                hardware_name=props.gcnArchName,
                runtime_version=torch.version.hip,
                compute_capability=props.gcnArchName,
            )

    # Unqualified XPU path: no CUDA/ROCm device, so fall back to Intel GPUs
    # before giving up.  Callers such as the AOT cache and heuristic generator
    # invoke ``get_hardware_info()`` with no device at all.
    if torch.xpu.is_available():
        return _xpu_hardware_info(None)

    # TPU / Pallas path
    try:
        import jax

        tpu_devices = [d for d in jax.devices() if d.platform == "tpu"]
        if tpu_devices:
            first_tpu = tpu_devices[0]
            return HardwareInfo(
                device_kind="tpu",
                hardware_name=first_tpu.device_kind,
                runtime_version=jax.__version__,
                compute_capability=first_tpu.device_kind,
            )
    except ImportError:
        pass

    raise RuntimeError(
        "No supported GPU or TPU device found. Helion requires CUDA, ROCm, XPU, or TPU."
    )
