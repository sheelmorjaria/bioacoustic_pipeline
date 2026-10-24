"""Post-call mechanical MACDE adapter. This module never enables Stage 2."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ObservedMetadata:
    value: object
    available_sample: int
    evidence: str


@dataclass(frozen=True)
class AudioSource:
    source_id: str
    coordinate_reference: str
    sample_rate: float
    sample_origin: int
    audio: np.ndarray
    sample_valid: np.ndarray | None = None

    def __post_init__(self):
        if not self.source_id or not self.coordinate_reference:
            raise ValueError('Explicit recording identity and coordinates required')
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError('Invalid sample rate')
        if int(self.sample_origin) != self.sample_origin:
            raise ValueError('Sample origin must be integral')
        if np.asarray(self.audio).ndim != 1:
            raise ValueError('Native mono audio required')
        if self.sample_valid is not None and np.shape(self.sample_valid) != np.shape(self.audio):
            raise ValueError('Sample mask shape mismatch')


@dataclass(frozen=True)
class VocalEvent:
    event_id: str
    source_id: str
    start_sample: int
    end_sample: int
    caller: ObservedMetadata | None = None
    recipient: ObservedMetadata | None = None
    context: ObservedMetadata | None = None


@dataclass
class AcousticExport:
    event: VocalEvent
    coordinate_reference: str
    source_origin_sample: int
    sample_rate: float
    encoder_id: str
    trajectory: dict
    summary: np.ndarray
    summary_valid: np.ndarray
    summary_names: list
    quality: np.ndarray
    caller: object
    recipient: object
    context: object

    def metadata(self):
        return dict(event_id=self.event.event_id, source_id=self.event.source_id,
                    coordinate_reference=self.coordinate_reference,
                    source_origin_sample=int(self.source_origin_sample),
                    start_sample=int(self.event.start_sample), end_sample=int(self.event.end_sample),
                    sample_rate=float(self.sample_rate), encoder_id=self.encoder_id,
                    prediction_policy='post-call', mechanical_only=True, stage2_ready=False,
                    acoustic_supported=bool(self.summary_valid.any()),
                    supported_summary_features=int(self.summary_valid.sum()),
                    summary_names=self.summary_names,
                    metadata_evidence={k: None if getattr(self.event, k) is None else
                        dict(available_sample=int(getattr(self.event, k).available_sample),
                             evidence=getattr(self.event, k).evidence)
                        for k in ('caller', 'recipient', 'context')})


class AcousticBridge:
    def __init__(self, encoder, *, encoder_id):
        if hasattr(encoder, 'model'):
            if encoder.model.training or any(p.requires_grad for p in encoder.model.parameters()):
                raise ValueError('Encoder must be frozen and in evaluation mode')
        self.encoder, self.encoder_id = encoder, encoder_id

    @classmethod
    def from_checkpoint(cls, path, *, expected_sha256):
        import hashlib
        import torch
        from bioacoustic_pipeline.embeddings.macde_bundle import FrozenMACDE
        digest = hashlib.sha256(open(path, 'rb').read()).hexdigest()
        if digest != expected_sha256:
            raise ValueError('Checkpoint checksum mismatch')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('arm_policy', 'combined') != 'combined':
            raise ValueError('Only ordered combined checkpoints support continuous export')
        return cls(FrozenMACDE(payload), encoder_id=digest)

    def export_event(self, source, event, *, prediction_sample=None):
        cutoff = event.end_sample if prediction_sample is None else prediction_sample
        if cutoff != event.end_sample:
            raise ValueError('This interface requires post-call prediction at event completion')
        if event.source_id != source.source_id:
            raise ValueError('Event recording mismatch')
        if any(int(x) != x for x in (event.start_sample, event.end_sample)):
            raise ValueError('Event samples must be integral')
        origin = source.sample_origin
        if not origin <= event.start_sample < event.end_sample <= origin + len(source.audio):
            raise ValueError('Event outside supplied source coordinates')
        metadata = []
        for field in ('caller', 'recipient', 'context'):
            item = getattr(event, field)
            if item is not None and (not isinstance(item, ObservedMetadata) or
                    item.available_sample > cutoff or not item.evidence):
                raise ValueError('Metadata must have evidence and be available at prediction time')
            metadata.append(None if item is None else item.value)
        stop = int(cutoff - origin)
        hop = max(1, round(.01 * source.sample_rate))
        if stop < 4 * hop:
            raise ValueError('Source prefix is shorter than one acoustic frame')
        mask = None if source.sample_valid is None else np.asarray(source.sample_valid[:stop], bool)
        out = self.encoder.encode_audio(np.asarray(source.audio[:stop]), source.sample_rate, sample_valid=mask)
        ends = np.rint(np.asarray(out['frame_end_seconds']) * source.sample_rate).astype(np.int64) + origin
        starts = ends - 4 * hop
        if np.any(np.diff(ends) <= 0) or np.any(ends > cutoff):
            raise ValueError('Encoder returned invalid causal frame coordinates')
        keep = (starts >= event.start_sample) & (ends <= cutoff)
        state = np.asarray(out['state'])[keep].copy()
        valid = np.asarray(out['state_valid'], bool)[keep].copy()
        changes = np.asarray(out['changes'])[keep].copy()
        cv = np.asarray(out['change_valid'], bool)[keep].copy()
        starts, ends = starts[keep], ends[keep]
        for j, lag in enumerate((2, 5, 10, 20)):
            cv[:, j] &= starts - lag * hop >= event.start_sample
        if not np.isfinite(state[valid]).all() or not np.isfinite(changes[cv]).all():
            raise ValueError('Nonfinite supported encoder output')
        state[~valid] = np.nan
        changes[~cv] = np.nan
        dim = state.shape[1]
        values, names = [], []

        def append(label, value):
            values.extend(np.asarray(value).reshape(-1).tolist())
            names.extend(f'{label}:{i}' for i in range(np.size(value)))

        selected = state[valid]
        missing = np.full(dim, np.nan)
        for label, fn in [('mean', lambda x: x.mean(0)), ('std', lambda x: x.std(0)),
                          ('first', lambda x: x[0]), ('last', lambda x: x[-1])]:
            append(label, fn(selected) if len(selected) else missing)
        times = ends[valid] / source.sample_rate
        centered = times - times.mean() if len(times) else times
        append('slope', (centered[:, None] * selected).sum(0) / (centered @ centered)
               if len(times) > 1 else missing)
        phase = (ends - event.start_sample) / (event.end_sample - event.start_sample)
        for j in range(4):
            chosen = valid & (phase > j/4) & (phase <= (j+1)/4)
            append(f'quarter{j}', state[chosen].mean(0) if chosen.any() else missing)
        for j, lag in enumerate((20, 50, 100, 200)):
            x = changes[cv[:, j], j]
            append(f'change{lag}_mean', x.mean(0) if len(x) else missing)
            append(f'change{lag}_magnitude', np.abs(x).mean(0) if len(x) else missing)
        summary = np.asarray(values, float)
        quality = np.r_[valid.mean() if len(valid) else 0,
                        cv.mean(0) if len(cv) else np.zeros(4),
                        np.log1p(valid.sum()), np.log1p(len(valid))]
        return AcousticExport(event, source.coordinate_reference, origin, source.sample_rate,
                              self.encoder_id, dict(state=state, state_valid=valid, changes=changes,
                              change_valid=cv, frame_start_sample=starts, frame_end_sample=ends),
                              summary, np.isfinite(summary), names, quality, *metadata)


def to_cavem_sequence(exports):
    """Finite vectors plus explicit support masks; preserve unknown and overlapping events."""
    from .cavem import EventRecord
    result = []
    previous = None
    for item in exports:
        gap = 0.
        if previous is not None:
            if (item.event.source_id, item.coordinate_reference, item.sample_rate, item.encoder_id,
                item.summary_names) != (previous.event.source_id, previous.coordinate_reference,
                previous.sample_rate, previous.encoder_id, previous.summary_names):
                raise ValueError('Cannot concatenate different recording sources or export schemas')
            if item.event.end_sample <= previous.event.end_sample:
                raise ValueError('Events require strictly increasing completion times')
            gap = (item.event.start_sample - previous.event.end_sample) / item.sample_rate
        vector = np.r_[np.where(item.summary_valid, item.summary, 0.), item.summary_valid.astype(float),
                       item.quality, float(previous is not None), float(gap < 0),
                       np.sign(gap) * np.log1p(abs(gap))]
        provenance = item.metadata()
        provenance['overlap_with_previous'] = bool(gap < 0)
        result.append(EventRecord(acoustic=vector[None], valid=np.array([True]),
                      dt_prev_ms=gap * 1000,
                      duration_ms=(item.event.end_sample-item.event.start_sample)/item.sample_rate*1000,
                      caller=item.caller, recipient=item.recipient, context=item.context,
                      provenance=provenance))
        previous = item
    return result
