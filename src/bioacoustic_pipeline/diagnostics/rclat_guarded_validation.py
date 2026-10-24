"""Fixed engineering fault-injection schedule; no scientific-data tuning."""

import torch

LAYOUTS = ["contiguous", "strided", "offset7"]
KINDS = ["nan", "posinf", "neginf"]
VALUES = {"nan": float("nan"), "posinf": float("inf"), "neginf": float("-inf")}


def fault_schedule():
    rows = []

    def add(**r):
        r["name"] = ".".join(
            str(r[k])
            for k in ["batch", "layout", "family", "role", "kind"]
            + [k for k in ["location"] if k in r]
        )
        rows.append(r)

    for batch in [4, 128]:
        for layout in LAYOUTS:
            for location in ["first", "middle", "last"]:
                for role in ["prediction", "target"]:
                    for kind in KINDS:
                        add(
                            batch=batch,
                            layout=layout,
                            family="loss",
                            role=role,
                            kind=kind,
                            location=location,
                            expected_accept=False,
                        )
                for kind in KINDS:
                    add(
                        batch=batch,
                        layout=layout,
                        family="loss",
                        role="masked_pair",
                        kind=kind,
                        location=location,
                        expected_accept=True,
                    )
                for role in ["observed", "masked"]:
                    for kind in KINDS:
                        add(
                            batch=batch,
                            layout=layout,
                            family="input",
                            role=role,
                            kind=kind,
                            location=location,
                            expected_accept=role == "masked",
                        )
            for kind in ["empty_first", "empty_all"]:
                add(
                    batch=batch,
                    layout=layout,
                    family="loss",
                    role="support",
                    kind=kind,
                    expected_accept=False,
                )
            add(
                batch=batch,
                layout=layout,
                family="input",
                role="support",
                kind="empty_frame",
                expected_accept=False,
            )
    return rows


def fault_fixture(row):
    batch = row["batch"]
    family = row["family"]
    shape = (batch, 32, 64) if family == "input" else (batch, 3, 64)
    mask = torch.ones(shape, dtype=torch.bool)
    if family == "input":
        x = (torch.arange(batch * 32 * 64, dtype=torch.float32) % 17).reshape(shape) / 16
        if row["kind"] == "empty_frame":
            mask[-1, -1] = False
            x[-1, -1] = float("nan")
        else:
            position = {"first": 0, "middle": x.numel() // 2, "last": x.numel() - 1}[
                row["location"]
            ]
            x.flatten()[position] = VALUES[row["kind"]]
            if row["role"] == "masked":
                mask.flatten()[position] = False
        return dict(values=x, mask=mask)
    p = torch.ones(shape)
    y = torch.zeros(shape)
    if row["role"] == "support":
        if row["kind"] == "empty_first":
            mask[0] = False
            p[0] = float("inf")
            y[0] = float("nan")
        else:
            mask.fill_(False)
            p.fill_(float("inf"))
            y.fill_(float("nan"))
    else:
        position = {"first": 0, "middle": p.numel() // 2, "last": p.numel() - 1}[row["location"]]
        value = VALUES[row["kind"]]
        if row["role"] == "prediction":
            p.flatten()[position] = value
        elif row["role"] == "target":
            y.flatten()[position] = value
        else:
            p.flatten()[position] = value
            y.flatten()[position] = -value
            mask.flatten()[position] = False
    return dict(prediction=p, target=y, mask=mask)
