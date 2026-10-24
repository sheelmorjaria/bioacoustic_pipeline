"""CPU-only synthetic integration smoke; no corpus fitting or scientific gate."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from bioacoustic_pipeline.representation.rclat_rtpe import (
    HistoryRidge,
    TemporalPredictionEngine,
    TrainingScaler,
    forecast_windows,
    masked_mse,
    permute_history,
    validate_group_split,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(17)
    source_paths = [
        Path(__file__),
        Path("src/bioacoustic_pipeline/representation/rclat_rtpe.py"),
        Path("tests/test_rclat_rtpe.py"),
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    configuration = dict(
        seed=17,
        history=8,
        horizons=[1, 3],
        scales=[1, 4, 8],
        hidden=12,
        updates=40,
        learning_rate=0.003,
        ridge_alpha=1.0,
        device="cpu",
        purpose="Software integration smoke only; no scientific performance criterion",
        dependencies=hashes,
    )
    (args.output / "configuration.json").write_text(json.dumps(configuration, indent=2))
    records = [
        dict(
            recording=f"synthetic-{i}",
            individual=f"individual-{i}",
            split="train" if i < 6 else ("validation" if i < 8 else "test"),
        )
        for i in range(10)
    ]
    validate_group_split(records)
    raw = torch.zeros(10, 48, 3)
    noise = torch.randn_like(raw) * 0.2
    for t in range(1, 48):
        raw[:, t] = 0.8 * raw[:, t - 1] + noise[:, t]
    valid = torch.ones_like(raw, dtype=torch.bool)
    valid[:, 13, 1] = False
    raw[~valid] = float("nan")
    scaler = TrainingScaler().fit(raw[:6], valid[:6])
    values = scaler.transform(raw, valid)
    windows = [forecast_windows(values[i], valid[i], history=8, horizons=(1, 3)) for i in range(10)]

    def combine(indices):
        return {
            k: torch.cat([windows[i][k] for i in indices])
            for k in ("inputs", "input_valid", "targets", "target_valid")
        }

    train = combine(range(6))
    ridge = HistoryRidge(alpha=1.0).fit(
        train["inputs"], train["input_valid"], train["targets"], train["target_valid"]
    )
    model = TemporalPredictionEngine(3, hidden=12, scales=(1, 4, 8), horizons=(1, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    losses = []
    for _ in range(40):
        optimizer.zero_grad()
        predicted = model(train["inputs"], train["input_valid"])["forecast"][:, -1]
        loss = masked_mse(predicted, train["targets"], train["target_valid"])
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError("Nonfinite smoke gradients")
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    scores = []
    with torch.no_grad():
        for metadata, w in zip(records, windows):
            original = model(w["inputs"], w["input_valid"])["forecast"][:, -1]
            baseline = ridge.predict(w["inputs"], w["input_valid"])
            x, m = permute_history(w["inputs"], w["input_valid"], torch.arange(7, -1, -1))
            disrupted = model(x, m)["forecast"][:, -1]
            scores.append(
                {
                    **metadata,
                    "gru_mse": masked_mse(original, w["targets"], w["target_valid"]).item(),
                    "ridge_mse": masked_mse(baseline, w["targets"], w["target_valid"]).item(),
                    "reversed_history_mse": masked_mse(
                        disrupted, w["targets"], w["target_valid"]
                    ).item(),
                }
            )
    report = dict(
        completed=True,
        synthetic_only=True,
        scientific_models_fitted=0,
        synthetic_smoke_model_fitted=True,
        updates=40,
        training_loss=losses,
        per_recording=scores,
        order_auxiliary_active=False,
        behavioral_head_active=False,
        RTPE_1_frozen=False,
        scientific_execution_ready=False,
        limitations=[
            "No real waveform features, reference certification or scientific comparison",
            "Reversal is a plumbing control, not artifact-matched evidence",
            "No uncertainty certification or superiority claim",
            "No final-test or running-experiment access",
        ],
    )
    (args.output / "results.json").write_text(json.dumps(report, indent=2))
    torch.save(
        dict(model=model.state_dict(), scaler_mean=scaler.mean, scaler_scale=scaler.scale),
        args.output / "smoke_state.pt",
    )
    manifest = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.iterdir()
        if p.is_file()
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Synthetic RTPE smoke completed. Scientific RTPE-1 remains unfrozen.")


if __name__ == "__main__":
    main()
