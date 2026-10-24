"""Tests: streaming consistency of the AR context nets (§45 Phase-3 contract).

The repo's context nets (TCN/Mamba) are feed-forward and stateless — nothing
carries hidden state across chunks; ``streaming_step`` recomputes from an
explicit history window of LATENTS (what ``StreamingContextBuffer`` rolls,
despite its name). The honest long-context contract is CONSISTENCY:
processing a long sequence incrementally (growing/rolling history +
``streaming_step``) must match the one-shot ``forward`` up to float
ordering, as long as the history covers the receptive field. NO
continuous-PAM claim is attached: the bat corpus is 1.5-3 s bouts.

# Copyright (c) 2025 Sheel Morjaria. All rights reserved.
"""

import unittest

try:
    import torch

    _TORCH = True
except ImportError:
    _TORCH = False


@unittest.skipUnless(_TORCH, "needs torch")
class TestStreamingConsistency(unittest.TestCase):
    def test_tcn_streaming_step_matches_forward(self):
        from bioacoustic_pipeline.embeddings.cpc_autoregressive import TCNAutoregressive

        torch.manual_seed(0)
        tcn = TCNAutoregressive(d_model=16, num_layers=4).eval()
        seq = torch.randn(1, 256, 16)
        warmup = tcn.receptive_field * 2
        with torch.no_grad():
            full = tcn(seq)
            history = seq[:, :warmup]  # growing latent history
            streamed = []
            for i in range(warmup, seq.shape[1]):
                step = tcn.streaming_step(seq[:, i : i + 1], history)
                streamed.append(step)
                history = torch.cat([history, seq[:, i : i + 1]], dim=1)
            tail = torch.cat(streamed, dim=1)
        diff = (full[:, warmup:] - tail).abs().max()
        self.assertLess(float(diff), 1e-4)

    def test_rolling_buffer_streaming_matches_forward(self):
        from bioacoustic_pipeline.embeddings.cpc_autoregressive import (
            StreamingContextBuffer,
            TCNAutoregressive,
        )

        torch.manual_seed(1)
        tcn = TCNAutoregressive(d_model=8, num_layers=3).eval()
        buffer_size = tcn.receptive_field + 8
        buf = StreamingContextBuffer(d_model=8, buffer_size=buffer_size)
        seq = torch.randn(1, 128, 8)
        with torch.no_grad():
            full = tcn(seq)
            warmup = buffer_size
            streamed = []
            for i in range(seq.shape[1]):
                window = buf.update(seq[:, i : i + 1])
                if i >= warmup:  # window is full: last frame has full RF context
                    # streaming_step concatenates z_t itself — the history arg
                    # must EXCLUDE the current frame (the buffer's last slot
                    # already holds it)
                    history = window[:, :-1, :]
                    streamed.append(tcn.streaming_step(seq[:, i : i + 1], history))
            tail = torch.cat(streamed, dim=1)
        diff = (full[:, warmup:] - tail).abs().max()
        self.assertLess(float(diff), 1e-4)

    def test_factory_forward_shape(self):
        from bioacoustic_pipeline.embeddings.cpc_autoregressive import create_autoregressive

        model = create_autoregressive(d_model=8)  # auto -> TCN when mamba absent
        out = model(torch.randn(2, 32, 8))
        self.assertEqual(tuple(out.shape), (2, 32, 8))


if __name__ == "__main__":
    unittest.main()
