"""Render Plotly blog embeds from the metadata-free W&B metric snapshot.

Run: uv run --with plotly==7.0.0 python source/visuals/kda/render.py
Generation is offline; the exported HTML loads Plotly and IBM Plex fonts from CDNs.
"""

import json
import math
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[2]
DATA = json.loads(Path(__file__).with_name("metrics.json").read_text())["series"]
OUTPUT = ROOT / "content" / "media" / "kda"


def save(fig: go.Figure, name: str, title: str) -> None:
    fig.update_layout(
        template="none",
        autosize=True,
        margin={"l": 64, "r": 20, "t": 92, "b": 65},
        paper_bgcolor="#161616",
        plot_bgcolor="#161616",
        font={"family": "IBM Plex Sans, sans-serif", "size": 12, "color": "#d4d4d4"},
        hoverlabel={"font": {"family": "IBM Plex Sans, sans-serif", "size": 12}},
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.03,
            "yanchor": "bottom",
            "font": {"size": 10},
        },
        hovermode="closest",
    )
    fig.update_xaxes(
        automargin=True,
        gridcolor="rgba(153,153,153,0.18)",
        zerolinecolor="rgba(153,153,153,0.32)",
        title_font={"family": "IBM Plex Mono, monospace", "size": 11},
        tickfont={"size": 11},
    )
    fig.update_yaxes(
        automargin=True,
        gridcolor="rgba(153,153,153,0.18)",
        zerolinecolor="rgba(153,153,153,0.32)",
        title_font={"family": "IBM Plex Mono, monospace", "size": 11},
        tickfont={"size": 11},
    )
    plot = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        div_id=name,
        default_height="100%",
        config={"responsive": True, "displaylogo": False, "scrollZoom": False},
    )
    html = Path(__file__).with_name("frame.html").read_text()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / f"{name}.html").write_text(
        html.replace("$TITLE$", title).replace("$PLOT$", plot)
    )
    print(OUTPUT / f"{name}.html")


def validate() -> None:
    expected_steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 7600]
    for name, series in DATA.items():
        rows = series["rows"]
        steps = [row["_step"] for row in rows]
        expected = [2000, 4000] if name.startswith("pilot") else expected_steps
        if name == "scaled_paired_1024":
            expected = [4000, 7600]
        assert steps == expected, (name, steps)
        assert series["num_sequences"] == (
            1024 if "1024" in name or "pilot" in name else 64
        )
        for row in rows:
            assert all(math.isfinite(value) for value in row.values())
            if "eval/all/gap_nats" in row:
                assert math.isclose(
                    row["eval/all/gap_nats"],
                    row["eval/all/autoregressive_nll"] - row["eval/all/parallel_nll"],
                    abs_tol=1e-12,
                )
                assert row["eval/all/gap_stderr_by_sequence"] >= 0
            else:
                assert row["paired/gap_did_se"] >= 0
                assert abs(row["paired/gap_did"]) <= 1.96 * row["paired/gap_did_se"]
    for causal, midpoint, paired in zip(
        DATA["scaled_causal"]["rows"],
        DATA["scaled_midpoint"]["rows"],
        DATA["scaled_paired_64"]["rows"],
        strict=True,
    ):
        assert math.isclose(
            paired["paired/gap_did"],
            midpoint["eval/all/gap_nats"] - causal["eval/all/gap_nats"],
            abs_tol=1e-12,
        )


def scaled_curves() -> None:
    loss, gap = go.Figure(), go.Figure()
    for color, arm in enumerate(("causal", "midpoint")):
        rows = DATA[f"scaled_{arm}"]["rows"]
        steps = [row["_step"] for row in rows]
        for mode, dash, label in (
            ("parallel", "solid", "parallel"),
            ("autoregressive", "dash", "AR"),
        ):
            loss.add_scatter(
                x=steps,
                y=[row[f"eval/all/{mode}_nll"] for row in rows],
                mode="lines+markers",
                name=f"{arm.capitalize()} / {label}",
                line={"dash": dash, "width": 2},
                marker={"size": 7},
                meta={"color": color},
                hovertemplate="Step %{x:,}<br>NLL %{y:.6f} nats/token<extra>%{fullData.name}</extra>",
            )
        gap.add_scatter(
            x=steps,
            y=[row["eval/all/gap_nats"] for row in rows],
            error_y={
                "array": [1.96 * row["eval/all/gap_stderr_by_sequence"] for row in rows]
            },
            customdata=[row["eval/all/gap_stderr_by_sequence"] for row in rows],
            mode="lines+markers",
            name=f"{arm.capitalize()} reference",
            line={"width": 2},
            marker={"size": 7},
            meta={"color": color},
            hovertemplate="Step %{x:,}<br>Gap %{y:.7f} nats/token<br>SE %{customdata:.7f}<extra>%{fullData.name}</extra>",
        )
    for fig in (loss, gap):
        fig.update_xaxes(title="Training step", tickformat=",d")
    loss.update_yaxes(title="NLL (nats/token; lower is better)")
    gap.update_yaxes(
        title="AR − parallel (nats/token)", tickformat=".1e", zeroline=True
    )
    save(loss, "scaled-loss", "Scaled model: held-out NLL")
    save(gap, "scaled-gap", "Scaled model: autoregressive minus parallel NLL")


def paired_gaps() -> None:
    sweep, final = go.Figure(), go.Figure()
    for size, color in ((64, 2), (1024, 5)):
        rows = DATA[f"scaled_paired_{size}"]["rows"]
        sweep.add_scatter(
            x=[row["_step"] for row in rows],
            y=[row["paired/gap_did"] for row in rows],
            error_y={"array": [1.96 * row["paired/gap_did_se"] for row in rows]},
            customdata=[row["paired/gap_did_se"] for row in rows],
            mode="lines+markers" if size == 64 else "markers",
            name=f"{size:,} sequences",
            marker={"size": 8, "symbol": "circle" if size == 64 else "diamond"},
            line={"width": 2},
            meta={"color": color},
            hovertemplate="Step %{x:,}<br>ΔG %{y:.7f} nats/token<br>Paired SE %{customdata:.7f}<extra>%{fullData.name}</extra>",
        )
    names = ["pilot_seed42", "pilot_seed11", "pilot_seed23", "scaled_paired_1024"]
    rows = [DATA[name]["rows"][-1] for name in names]
    final.add_scatter(
        x=[row["paired/gap_did"] for row in rows],
        y=["520M · seed 42", "520M · seed 11", "520M · seed 23", "1.45B · seed 42"],
        error_x={"array": [1.96 * row["paired/gap_did_se"] for row in rows]},
        customdata=[[row["_step"], row["paired/gap_did_se"]] for row in rows],
        mode="markers",
        marker={"size": 9},
        meta={"color": 5},
        showlegend=False,
        hovertemplate="%{y}<br>Step %{customdata[0]:,}<br>ΔG %{x:.7f} nats/token<br>Paired SE %{customdata[1]:.7f}<extra></extra>",
    )
    sweep.update_xaxes(title="Training step", tickformat=",d")
    sweep.update_yaxes(title="ΔG (nats/token)", tickformat=".1e", zeroline=True)
    final.update_xaxes(title="ΔG (nats/token)", tickformat=".1e", zeroline=True)
    final.update_yaxes(autorange="reversed", showgrid=False)
    save(sweep, "paired-checkpoints", "Scaled model: excess autoregressive penalty")
    save(
        final, "paired-seeds", "Final checkpoints: paired excess autoregressive penalty"
    )


if __name__ == "__main__":
    validate()
    scaled_curves()
    paired_gaps()
