"""Tests: device resolution (utils.devices) — cuda -> xpu -> cpu ordering.

Torch-gated where mocking is needed; the no-torch branch is exercised by
import-block monkeypatching. No checkpoints, no network.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import importlib
import sys
import unittest
from unittest import mock


class TestResolveDevice(unittest.TestCase):
    def test_explicit_wins(self):
        from bioacoustic_pipeline.utils.devices import resolve_device

        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertEqual(resolve_device("xpu:0"), "xpu:0")

    def test_no_torch_gives_cpu(self):
        real = sys.modules.pop("torch", None)
        try:
            sys.modules["torch"] = None  # forces the lazy import to fail
            devices = importlib.import_module("bioacoustic_pipeline.utils.devices")
            importlib.reload(devices)
            self.assertEqual(devices.resolve_device(), "cpu")
        finally:
            if real is not None:
                sys.modules["torch"] = real

    def _torch(self):
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            self.skipTest("needs torch")
        return torch

    def test_cuda_preferred_over_xpu(self):
        torch = self._torch()
        from bioacoustic_pipeline.utils.devices import resolve_device

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.xpu, "is_available", return_value=True),
        ):
            self.assertEqual(resolve_device(), "cuda")

    def test_xpu_when_no_cuda(self):
        torch = self._torch()
        from bioacoustic_pipeline.utils.devices import resolve_device

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.xpu, "is_available", return_value=True),
        ):
            self.assertEqual(resolve_device(), "xpu")

    def test_cpu_when_nothing_available(self):
        torch = self._torch()
        from bioacoustic_pipeline.utils.devices import resolve_device

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.xpu, "is_available", return_value=False),
        ):
            self.assertEqual(resolve_device(), "cpu")


if __name__ == "__main__":
    unittest.main()


class TestExplicitValidation(unittest.TestCase):
    def _torch(self):
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            self.skipTest("needs torch")
        return torch

    def test_explicit_xpu_fails_hard_when_unavailable(self):
        torch = self._torch()
        from bioacoustic_pipeline.utils.devices import resolve_device

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(torch.xpu, "is_available", return_value=False),
        ):
            with self.assertRaises(RuntimeError):
                resolve_device("xpu:0")

    def test_explicit_available_xpu_passes(self):
        torch = self._torch()
        from bioacoustic_pipeline.utils.devices import resolve_device

        with mock.patch.object(torch.xpu, "is_available", return_value=True):
            self.assertEqual(resolve_device("xpu:0"), "xpu:0")

    def test_cpu_always_valid(self):
        from bioacoustic_pipeline.utils.devices import resolve_device

        self.assertEqual(resolve_device("cpu"), "cpu")
