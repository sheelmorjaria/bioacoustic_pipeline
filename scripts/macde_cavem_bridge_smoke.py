"""Synthetic mechanical integration check; no interaction training or biological endpoint."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from bioacoustic_pipeline.contextual.macde_bridge import (
    AcousticBridge, AudioSource, VocalEvent, to_cavem_sequence,
)
from bioacoustic_pipeline.contextual.cavem import CausalGRU


def main():
    torch.set_num_threads(2)
    checkpoint = Path('checkpoints/bfsongrepo_development_R2.pt')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    bridge = AcousticBridge.from_checkpoint(checkpoint, expected_sha256=digest)
    sr = 32000
    t = np.arange(sr * 3) / sr
    audio = .1 * np.sin(2 * np.pi * 2100 * t) * (1 + .2 * np.sin(2 * np.pi * 17 * t))
    source = AudioSource('synthetic-tone', 'synthetic-tone:native-samples', sr, 0, audio)
    exports = [bridge.export_event(source, VocalEvent(str(i), source.source_id, a, b))
               for i, (a, b) in enumerate(((0, sr), (sr, 2 * sr)))]
    sequence = to_cavem_sequence(exports)
    torch.manual_seed(0)
    consumer = CausalGRU(event_dim=sequence[0].acoustic.shape[1], hidden=4)
    states = consumer.causal_states(sequence)
    np.testing.assert_allclose(consumer.causal_states(sequence[:1])[0], states[0], atol=1e-6)
    altered = audio.copy()
    altered[sr:] = 0
    other = bridge.export_event(AudioSource(source.source_id, source.coordinate_reference, sr, 0, altered),
                                exports[0].event)
    np.testing.assert_allclose(exports[0].summary, other.summary, equal_nan=True)
    report = dict(mechanical_only=True, stage2_ready=False, model_fitting=False,
                  checkpoint_sha256=digest, synthetic_audio=True,
                  finite_consumer_states=bool(np.isfinite(states).all()),
                  future_audio_exclusion_passed=True, consumer_prefix_causality_passed=True,
                  event_metadata=[x.metadata() for x in exports],
                  consumer_dimension=int(sequence[0].acoustic.shape[1]))
    Path('outputs/macde_cavem_bridge_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print({k: v for k, v in report.items() if k != 'event_metadata'})


if __name__ == '__main__':
    main()
