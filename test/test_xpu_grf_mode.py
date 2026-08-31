from __future__ import annotations

from unittest.mock import patch

import torch

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipUnlessXPUGrfMode
from helion.autotuner.config_spec import DEFAULT_GRF_MODE
from helion.autotuner.config_spec import VALID_GRF_MODES
import helion.language as hl


@onlyBackends(["triton"])
class TestXPUGrfMode(TestCase):
    """Coverage for ``grf_mode``, the Intel/XPU per-thread register-budget knob.

    ``grf_mode`` is the XPU analog of NVIDIA's ``maxnreg``.  Triton's Intel
    backend lowers it to an IGC build flag rather than a PTX directive, so it is
    a distinct config key that is only offered on Intel GPUs.
    """

    @skipUnlessXPUGrfMode("Test requires an Intel GPU (XPU)")
    def test_grf_mode_in_kernel(self) -> None:
        """``grf_mode`` is accepted and forwarded to the Triton launcher."""

        @helion.kernel(
            autotune_effort="none",
            config=helion.Config(block_sizes=[32, 32], grf_mode="256"),
        )
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.shape):
                result[tile] = x[tile] + y[tile]
            return result

        x = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)
        y = torch.randn(128, 128, device=DEVICE, dtype=torch.float32)

        code, result = code_and_output(add_kernel, (x, y))
        torch.testing.assert_close(result, x + y)
        self.assertIn("grf_mode='256'", code)

    @skipUnlessXPUGrfMode("Test requires an Intel GPU (XPU)")
    def test_grf_mode_reaches_triton_compiler(self) -> None:
        """Each ``grf_mode`` yields distinct IGC build flags and a distinct binary.

        This is the anti-no-op check: it asserts the knob actually changes what
        the Intel compiler produces, not merely what Helion emits.
        """
        import triton.compiler.compiler as tcc

        @helion.kernel(autotune_effort="none", static_shapes=True)
        def add_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            result = torch.empty_like(x)
            for tile in hl.tile(x.size(0)):
                result[tile] = x[tile] + y[tile]
            return result

        x = torch.randn(4096, device=DEVICE, dtype=torch.float32)
        y = torch.randn(4096, device=DEVICE, dtype=torch.float32)

        seen: list[tuple[str, str]] = []
        original_init = tcc.CompiledKernel.__init__

        def spy(self: object, *args: object, **kwargs: object) -> None:
            original_init(self, *args, **kwargs)
            metadata = self.metadata  # pyrefly: ignore [missing-attribute]
            seen.append((metadata.build_flags, metadata.hash))

        build_flags: set[str] = set()
        binaries: set[str] = set()
        bound = add_kernel.bind((x, y))
        with patch.object(tcc.CompiledKernel, "__init__", spy):
            for mode in VALID_GRF_MODES:
                seen.clear()
                config = helion.Config(block_sizes=[1024], num_warps=4, grf_mode=mode)
                result = bound.compile_config(config)(x, y)
                torch.testing.assert_close(result, x + y)
                self.assertTrue(seen, f"no kernel compiled for grf_mode={mode!r}")
                flags, digest = seen[-1]
                build_flags.add(flags)
                binaries.add(digest)

        self.assertEqual(len(build_flags), len(VALID_GRF_MODES))
        self.assertEqual(len(binaries), len(VALID_GRF_MODES))
        self.assertTrue(any("128-GRF-per-thread" in f for f in build_flags))
        self.assertTrue(any("256-GRF-per-thread" in f for f in build_flags))
        self.assertTrue(any("auto-large-GRF-mode" in f for f in build_flags))

    @skipUnlessXPUGrfMode("Test requires an Intel GPU (XPU)")
    def test_grf_mode_in_tunable_fragments(self) -> None:
        """``grf_mode`` is offered to the autotuner search space on XPU."""
        env = CompileEnvironment(DEVICE, helion.Settings(backend="triton"))
        fragments = env.config_spec.backend_tunable_fragments
        self.assertIn("grf_mode", fragments)
        self.assertEqual(tuple(fragments["grf_mode"].choices), VALID_GRF_MODES)

    @skipUnlessXPUGrfMode("Test requires an Intel GPU (XPU)")
    def test_grf_mode_normalize_defaults_and_rejects(self) -> None:
        """Normalization fills the default and rejects out-of-range values."""
        env = CompileEnvironment(DEVICE, helion.Settings(backend="triton"))

        config = helion.Config()
        env.config_spec.normalize(config)
        self.assertEqual(config["grf_mode"], DEFAULT_GRF_MODE)

        # "512" is a real Triton grf_mode but is rejected by Data Center GPU Max
        # hardware at build time, so Helion does not offer it.
        bad = helion.Config(grf_mode="512")
        with self.assertRaisesRegex(helion.exc.InvalidConfig, "grf_mode"):
            env.config_spec.normalize(bad)

    def test_grf_mode_round_trips_through_config(self) -> None:
        """``grf_mode`` survives JSON serialization (configs are cached/persisted)."""
        config = helion.Config(block_sizes=[64], grf_mode="256")
        self.assertEqual(config.grf_mode, "256")
        restored = helion.Config.from_json(config.to_json())
        self.assertEqual(restored.grf_mode, "256")
        self.assertEqual(restored.config, config.config)

    def test_grf_mode_defaults_when_absent(self) -> None:
        """The ``Config.grf_mode`` property defaults without touching the device."""
        self.assertEqual(helion.Config(block_sizes=[64]).grf_mode, DEFAULT_GRF_MODE)

    def test_grf_mode_absent_when_unsupported(self) -> None:
        """On non-Intel hardware ``grf_mode`` is not a valid config key.

        This guards the CUDA/ROCm config space: the key must never leak into a
        search space or a launcher call on those backends, and an explicitly
        supplied ``grf_mode`` is rejected rather than silently ignored (same
        contract as the AMD-only ``matrix_instr_nonkdim``).
        """
        with patch("helion._compat._supports_grf_mode", return_value=False):
            env = CompileEnvironment(DEVICE, helion.Settings(backend="triton"))
            self.assertNotIn("grf_mode", env.config_spec.backend_tunable_fragments)
            self.assertFalse(env.config_spec.supports_config_key("grf_mode"))

            config = helion.Config(grf_mode="256")
            with self.assertRaisesRegex(
                helion.exc.InvalidConfig,
                r"Unsupported config keys for backend 'triton': \['grf_mode'\]",
            ):
                env.config_spec.normalize(config)

    def test_grf_mode_excluded_from_default_search_space(self) -> None:
        """Without Intel support the autotuner search space has no ``grf_mode``."""
        with patch("helion._compat._supports_grf_mode", return_value=False):
            env = CompileEnvironment(DEVICE, helion.Settings(backend="triton"))
            flat = env.config_spec.flat_config(lambda fragment: fragment.default())
            self.assertNotIn("grf_mode", flat.config)


if __name__ == "__main__":
    import unittest

    unittest.main()
