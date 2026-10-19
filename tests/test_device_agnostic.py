"""Device-agnostic behaviour of the inference path (XPU, CUDA and CPU).

Everything except the last class runs without any accelerator: backend detection,
the CUDA-only attention choice, autocast dispatch and the server device flags.
The last class compares a tiny random-weight flow-matching action head on CPU and
on XPU, and is skipped when no XPU is present.

Run with ``python -m unittest discover -s tests``.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from starVLA.model import device as dev


def _patch_availability(*, xpu: bool, cuda: bool):
    def fake_is_available(device_type: str) -> bool:
        if device_type == "xpu":
            return xpu
        if device_type == "cuda":
            return cuda
        return device_type == "cpu"

    return mock.patch.object(dev, "is_available", side_effect=fake_is_available)


class DetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")

    def test_cuda_without_xpu(self):
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")

    def test_cpu_without_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.resolve_device(), torch.device("cpu"))
            # A CPU has no index, so torch.device("cpu:1") must not appear.
            self.assertEqual(dev.resolve_device(index=1), torch.device("cpu"))

    def test_index_selects_card_on_detected_backend(self):
        with _patch_availability(xpu=True, cuda=False):
            self.assertEqual(dev.resolve_device(index=1), torch.device("xpu:1"))
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.resolve_device(index=0), torch.device("cuda:0"))

    def test_explicit_device_wins(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cuda:1", index=0), torch.device("cuda:1"))
            self.assertEqual(dev.resolve_device(torch.device("cpu")), torch.device("cpu"))

    def test_device_type_of_tensor_string_and_device(self):
        self.assertEqual(dev.device_type("xpu:0"), "xpu")
        self.assertEqual(dev.device_type(torch.device("cuda")), "cuda")
        self.assertEqual(dev.device_type(torch.zeros(1)), "cpu")


class SynchronizeTest(unittest.TestCase):
    """torch.cuda.synchronize raises on a wheel built without CUDA."""

    def test_dispatches_to_xpu(self):
        with _patch_availability(xpu=True, cuda=True):
            with mock.patch.object(torch.xpu, "synchronize") as xpu_sync, mock.patch.object(
                torch.cuda, "synchronize"
            ) as cuda_sync:
                dev.synchronize()
        xpu_sync.assert_called_once()
        cuda_sync.assert_not_called()

    def test_explicit_cuda_overrides_detection(self):
        with _patch_availability(xpu=True, cuda=True):
            with mock.patch.object(torch.xpu, "synchronize") as xpu_sync, mock.patch.object(
                torch.cuda, "synchronize"
            ) as cuda_sync:
                dev.synchronize("cuda:0")
        cuda_sync.assert_called_once()
        xpu_sync.assert_not_called()

    def test_noop_on_cpu(self):
        with _patch_availability(xpu=False, cuda=False):
            with mock.patch.object(torch.cuda, "synchronize") as cuda_sync:
                dev.synchronize()
        cuda_sync.assert_not_called()


class AttentionAndAutocastTest(unittest.TestCase):
    def test_flash_attention_stays_default_on_cuda(self):
        self.assertEqual(dev.attn_implementation("cuda:0"), "flash_attention_2")

    def test_sdpa_off_cuda(self):
        self.assertEqual(dev.attn_implementation("xpu:0"), "sdpa")
        self.assertEqual(dev.attn_implementation(torch.device("cpu")), "sdpa")

    def test_autocast_uses_backend_of_device(self):
        with mock.patch.object(torch, "autocast") as autocast:
            dev.autocast("xpu:0", dtype=torch.bfloat16)
        autocast.assert_called_once_with("xpu", dtype=torch.bfloat16, enabled=True)
        # XPU autocast rejects float32, and CUDA is the only backend that casts with it.
        with mock.patch.object(torch, "autocast") as autocast:
            dev.autocast("xpu:0", dtype=torch.float32)
        autocast.assert_called_once_with("xpu", dtype=torch.float32, enabled=False)
        with mock.patch.object(torch, "autocast") as autocast:
            dev.autocast("cuda:0", dtype=torch.float32)
        autocast.assert_called_once_with("cuda", dtype=torch.float32, enabled=True)
        self.assertTrue(dev.casts_to_float32("cuda:0"))
        self.assertFalse(dev.casts_to_float32("xpu:0"))
        with mock.patch.object(torch, "autocast") as autocast:
            dev.autocast(torch.zeros(1), dtype=torch.float16)
        autocast.assert_called_once_with("cpu", dtype=torch.float16, enabled=False)

    def test_autocast_is_a_context_manager_on_cpu(self):
        x = torch.ones(2, 2)
        with dev.autocast(x.device):
            self.assertEqual((x @ x).dtype, torch.float32)


class ServerArgumentsTest(unittest.TestCase):
    def setUp(self):
        try:
            from deployment.model_server import server_policy
        except ImportError as exc:  # websocket or msgpack stack not installed
            self.skipTest(str(exc))
        self.parser = server_policy.build_argparser()

    def test_device_defaults_to_detection(self):
        args = self.parser.parse_args([])
        self.assertIsNone(args.device)
        self.assertEqual(args.cuda, 0)

    def test_index_and_explicit_device(self):
        args = self.parser.parse_args(["--cuda", "1", "--device", "xpu:1"])
        self.assertEqual(dev.resolve_device(args.device, index=args.cuda), torch.device("xpu:1"))
        with _patch_availability(xpu=False, cuda=True):
            args = self.parser.parse_args(["--cuda", "1"])
            self.assertEqual(dev.resolve_device(args.device, index=args.cuda), torch.device("cuda:1"))


def _tiny_head_config():
    action_model = SimpleNamespace(
        action_model_type="DiT-B",
        hidden_size=64,
        add_pos_embed=True,
        max_seq_len=64,
        action_dim=7,
        state_dim=8,
        future_action_window_size=3,
        action_horizon=4,
        num_inference_timesteps=2,
        num_target_vision_tokens=4,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        num_timestep_buckets=1000,
        diffusion_model_cfg={
            "cross_attention_dim": 32,
            "dropout": 0.0,
            "final_dropout": False,
            "interleave_self_attention": True,
            "norm_type": "ada_norm",
            "num_layers": 2,
            "output_dim": 64,
            "positional_embeddings": None,
        },
    )
    return SimpleNamespace(framework=SimpleNamespace(action_model=action_model))


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an Intel XPU")
class XpuActionHeadParityTest(unittest.TestCase):
    def test_predict_action_matches_cpu(self):
        from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead

        torch.manual_seed(0)
        head = FlowmatchingActionHead(_tiny_head_config()).eval()
        # Several layers start at zero, which would make the comparison trivial.
        with torch.no_grad():
            for param in head.parameters():
                param.copy_(torch.randn_like(param) * 0.05)
        vl_embs = torch.randn(1, 6, 32)
        state = torch.randn(1, 1, 8)

        real_randn = torch.randn

        def run(device):
            # Same noise on both devices: always draw on CPU, then move.
            def cpu_randn(*args, **kwargs):
                target = kwargs.pop("device", None)
                torch.manual_seed(1234)
                return real_randn(*args, **kwargs).to(target or "cpu")

            model = head.to(device)
            with mock.patch.object(torch, "randn", cpu_randn):
                return model.predict_action(vl_embs.to(device), state.to(device)).float().cpu()

        reference = run("cpu")
        actual = run("xpu:0")
        self.assertEqual(reference.shape, (1, 4, 7))
        self.assertGreater(reference.abs().max().item(), 1e-3)
        torch.testing.assert_close(actual, reference, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    unittest.main()
