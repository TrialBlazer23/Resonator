"""
Resonator Framework — Visualization
Publication-quality telemetry plots for the demo and paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Optional
from .core import StepTelemetry


# ── Color palette ────────────────────────────────────────────────────────────
COLORS = {
    "ar1": "#4FC3F7",           # cyan
    "velocity": "#81C784",      # green
    "stress": "#FF7043",        # orange-red
    "svd": "#CE93D8",           # purple
    "surprise": "#FFF176",      # yellow
    "threshold": "#EF5350",     # red
    "intervention": "#FF6D00",  # amber
    "healthy": "#2E7D32",
    "warning": "#F57F17",
    "critical": "#B71C1C",
    "background": "#0A0A0A",
    "panel": "#111111",
    "grid": "#1E1E1E",
    "text": "#E0E0E0",
    "subtext": "#9E9E9E",
}

STATE_COLORS = {
    "HEALTHY": "#2E7D32",
    "CSD_WARNING": "#F57F17",
    "POINT_ATTRACTOR": "#B71C1C",
    "LIMIT_CYCLE": "#AD1457",
    "ENTROPY_COLLAPSE": "#4527A0",
}


def _apply_theme(fig, axes):
    fig.patch.set_facecolor(COLORS["background"])
    for ax in axes:
        ax.set_facecolor(COLORS["panel"])
        ax.tick_params(colors=COLORS["subtext"], labelsize=8)
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(True, color=COLORS["grid"], linewidth=0.5, alpha=0.6)


def plot_telemetry_dashboard(
    telemetry: List[StepTelemetry],
    title: str = "Chaos Core — Generation Telemetry",
    ar1_threshold: float = 0.80,
    velocity_threshold: float = 0.30,
    figsize=(16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Full-panel telemetry dashboard:
      Row 1: AR(1) + Integrated Stress
      Row 2: Latent Velocity + SVD Entropy
      Row 3: Token Surprise
      Row 4: Attractor State (categorical)
    """
    steps = [t.step for t in telemetry]
    ar1 = [t.ar1 for t in telemetry]
    stress = [t.integrated_stress for t in telemetry]
    velocity = [t.latent_velocity for t in telemetry]
    svd = [t.svd_entropy for t in telemetry]
    surprise = [t.surprise for t in telemetry]
    states = [t.state for t in telemetry]
    interventions = [t.intervention_active for t in telemetry]

    fig = plt.figure(figsize=figsize, facecolor=COLORS["background"])
    fig.suptitle(title, color=COLORS["text"], fontsize=14, fontweight="bold", y=0.98)
    gs = GridSpec(4, 1, figure=fig, hspace=0.45, top=0.93, bottom=0.07)

    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # ── Panel 1: AR(1) and Integrated Stress ─────────────────────────────────
    ax = axes[0]
    ax.plot(steps, ar1, color=COLORS["ar1"], linewidth=1.5, label="AR(1) Autocorrelation", zorder=3)
    ax.axhline(ar1_threshold, color=COLORS["threshold"], linestyle="--", linewidth=1, alpha=0.8, label=f"AR(1) threshold ({ar1_threshold})")
    ax2 = ax.twinx()
    ax2.plot(steps, stress, color=COLORS["stress"], linewidth=1.5, alpha=0.7, label="Integrated Stress F(t)")
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(colors=COLORS["subtext"], labelsize=8)
    ax2.yaxis.label.set_color(COLORS["stress"])
    ax2.set_ylabel("Stress F(t)", color=COLORS["stress"], fontsize=8)
    ax2.set_facecolor(COLORS["panel"])
    for spine in ax2.spines.values():
        spine.set_color(COLORS["grid"])
    ax.set_ylabel("AR(1)", fontsize=8)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Critical Slowing Down Signal", fontsize=9, pad=4)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7,
              facecolor=COLORS["panel"], labelcolor=COLORS["text"], framealpha=0.8)

    # ── Panel 2: Latent Velocity and SVD Entropy ──────────────────────────────
    ax = axes[1]
    ax.plot(steps, velocity, color=COLORS["velocity"], linewidth=1.5, label="Latent Velocity V(t)", zorder=3)
    ax.axhline(velocity_threshold, color=COLORS["threshold"], linestyle="--", linewidth=1, alpha=0.8, label=f"Velocity threshold ({velocity_threshold})")
    ax3 = ax.twinx()
    ax3.plot(steps, svd, color=COLORS["svd"], linewidth=1.5, alpha=0.7, label="SVD Entropy")
    ax3.set_ylim(0, 1.1)
    ax3.tick_params(colors=COLORS["subtext"], labelsize=8)
    ax3.yaxis.label.set_color(COLORS["svd"])
    ax3.set_ylabel("SVD Entropy", color=COLORS["svd"], fontsize=8)
    ax3.set_facecolor(COLORS["panel"])
    for spine in ax3.spines.values():
        spine.set_color(COLORS["grid"])
    ax.set_ylabel("Velocity", fontsize=8)
    ax.set_title("Latent Trajectory Geometry", fontsize=9, pad=4)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(lines1 + lines3, labels1 + labels3, loc="upper right", fontsize=7,
              facecolor=COLORS["panel"], labelcolor=COLORS["text"], framealpha=0.8)

    # ── Panel 3: Token Surprise ───────────────────────────────────────────────
    ax = axes[2]
    ax.plot(steps, surprise, color=COLORS["surprise"], linewidth=1.0, alpha=0.8, label="Token Surprise S(t)")
    # Shade intervention regions
    in_block = False
    start = 0
    for i, (s, intv) in enumerate(zip(steps, interventions)):
        if intv and not in_block:
            start = s
            in_block = True
        elif not intv and in_block:
            ax.axvspan(start, s, alpha=0.15, color=COLORS["intervention"], zorder=1)
            in_block = False
    if in_block:
        ax.axvspan(start, steps[-1], alpha=0.15, color=COLORS["intervention"], zorder=1)
    ax.set_ylabel("Surprise", fontsize=8)
    ax.set_title("Token Surprise  [shaded = intervention active]", fontsize=9, pad=4)
    ax.legend(loc="upper right", fontsize=7, facecolor=COLORS["panel"],
              labelcolor=COLORS["text"], framealpha=0.8)

    # ── Panel 4: Attractor State ──────────────────────────────────────────────
    ax = axes[3]
    unique_states = list(STATE_COLORS.keys())
    state_to_y = {s: i for i, s in enumerate(unique_states)}
    state_y = [state_to_y.get(s, 0) for s in states]
    state_colors_plot = [STATE_COLORS.get(s, "#666666") for s in states]

    ax.scatter(steps, state_y, c=state_colors_plot, s=12, zorder=3, marker="s")
    ax.set_yticks(range(len(unique_states)))
    ax.set_yticklabels(unique_states, fontsize=7)
    ax.set_xlabel("Generation Step", fontsize=8)
    ax.set_title("Attractor State Classification", fontsize=9, pad=4)

    patches = [mpatches.Patch(color=c, label=s) for s, c in STATE_COLORS.items()]
    ax.legend(handles=patches, loc="upper right", fontsize=6,
              facecolor=COLORS["panel"], labelcolor=COLORS["text"],
              framealpha=0.8, ncol=3)

    _apply_theme(fig, axes)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
        print(f"Saved: {save_path}")

    return fig


def plot_comparison(
    baseline_telemetry: List[StepTelemetry],
    monitored_telemetry: List[StepTelemetry],
    baseline_text: str,
    monitored_text: str,
    figsize=(16, 7),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side AR(1) + stress comparison: baseline (no Chaos Core) vs. monitored.
    The key result figure: shows detection firing before output degrades.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor=COLORS["background"])
    fig.suptitle("Chaos Core: Baseline vs. Monitored Generation", 
                 color=COLORS["text"], fontsize=13, fontweight="bold")

    datasets = [(baseline_telemetry, "Baseline (No Monitor)", axes[:, 0]),
                (monitored_telemetry, "Monitored (Chaos Core Active)", axes[:, 1])]

    for telemetry, label, axcol in datasets:
        steps = [t.step for t in telemetry]
        ar1 = [t.ar1 for t in telemetry]
        stress = [t.integrated_stress for t in telemetry]
        states = [t.state for t in telemetry]

        # AR(1) panel
        axcol[0].plot(steps, ar1, color=COLORS["ar1"], linewidth=1.5)
        axcol[0].axhline(0.80, color=COLORS["threshold"], linestyle="--", linewidth=1, alpha=0.7)
        axcol[0].fill_between(steps, ar1, 0, alpha=0.2, color=COLORS["ar1"])
        axcol[0].set_ylim(0, 1.1)
        axcol[0].set_title(f"{label}\nAR(1) Autocorrelation", fontsize=9, color=COLORS["text"])
        axcol[0].set_ylabel("AR(1)", fontsize=8)

        # State density panel
        state_counts = {s: states.count(s) for s in STATE_COLORS}
        bars = axcol[1].barh(
            list(state_counts.keys()),
            list(state_counts.values()),
            color=[STATE_COLORS[s] for s in state_counts],
            alpha=0.85,
        )
        axcol[1].set_title(f"{label}\nStep Distribution by State", fontsize=9, color=COLORS["text"])
        axcol[1].set_xlabel("Steps", fontsize=8)

    _apply_theme(fig, [ax for row in axes for ax in row])

    # Annotate output texts
    fig.text(0.25, 0.01, f'Output: "{baseline_text[:80]}..."',
             ha="center", fontsize=7, color=COLORS["subtext"], style="italic")
    fig.text(0.75, 0.01, f'Output: "{monitored_text[:80]}..."',
             ha="center", fontsize=7, color=COLORS["text"], style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
        print(f"Saved: {save_path}")

    return fig


def print_telemetry_table(telemetry: List[StepTelemetry], max_rows: int = 30):
    """ASCII table for notebook inline display."""
    rows = telemetry[:max_rows]
    header = f"{'Step':>4} {'Token':>12} {'AR1':>6} {'Vel':>6} {'SVD':>6} {'Stress':>7} {'State':<18} {'⚡':>4}"
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for t in rows:
        intv = "⚡" if t.intervention_active else " "
        orbit = f"[O{t.orbit_period}]" if t.orbit_period else ""
        state_str = f"{t.state}{orbit}"
        print(
            f"{t.step:>4} {repr(t.token_str):>12} "
            f"{t.ar1:>6.3f} {t.latent_velocity:>6.3f} "
            f"{t.svd_entropy:>6.3f} {t.integrated_stress:>7.4f} "
            f"{state_str:<18} {intv:>4}"
        )
    print("─" * len(header))
    if len(telemetry) > max_rows:
        print(f"  ... {len(telemetry) - max_rows} more steps not shown")
