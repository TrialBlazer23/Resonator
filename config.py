"""
Resonator Framework — Configuration
All thresholds, decay constants, and behavioral parameters in one place.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChaosConfig:
    # ── Detection window ────────────────────────────────────────────────────
    window_size: int = 20          # rolling window for AR(1) and variance
    min_history: int = 5           # minimum steps before detection activates

    # ── AR(1) / Critical Slowing Down ───────────────────────────────────────
    ar1_threshold: float = 0.80    # lag-1 autocorrelation trigger
    variance_inflation_factor: float = 2.5  # sigma^2 relative to baseline

    # ── Latent Velocity (Point Attractor) ───────────────────────────────────
    velocity_threshold: float = 0.30       # below this → freezing zone
    velocity_window: int = 5               # steps to confirm low velocity

    # ── Orbit Resonance (Limit Cycle) ───────────────────────────────────────
    orbit_distance_threshold: float = 0.15  # normalized distance for match
    orbit_min_lag: int = 3                  # minimum period to consider
    orbit_max_lag: int = 25                 # maximum period to scan

    # ── Integrated Stress (Neuromodulation) ─────────────────────────────────
    stress_rise_rate: float = 0.25          # how fast stress accumulates
    stress_decay_rate: float = 0.92         # exponential decay per step
    stress_trigger_threshold: float = 0.60  # continuous intervention onset
    stress_critical_threshold: float = 0.85 # hard override onset

    # ── Intervention: Temperature ────────────────────────────────────────────
    base_temperature: float = 1.0
    max_temperature: float = 1.6            # ceiling during stress

    # ── Intervention: Logit Penalties ────────────────────────────────────────
    orbit_penalty_strength: float = 6.0    # subtracted from orbit tokens
    point_attractor_penalty: float = 4.0   # subtracted from frozen tokens

    # ── Semantic Anchoring ───────────────────────────────────────────────────
    anchor_boost_strength: float = 3.0     # logit boost for discourse markers
    anchor_tokens: List[str] = field(default_factory=lambda: [
        "however", "suddenly", "instead", "meanwhile", "although",
        "then", "but", "yet", "so", "now", "still", "next",
        "finally", "therefore", "because", "while", "after"
    ])

    # ── SVD Entropy ─────────────────────────────────────────────────────────
    svd_window: int = 15            # states to include in SVD computation
    svd_entropy_threshold: float = 0.4  # below this → dimensionality collapse

    # ── Schmitt Trigger (hysteresis) ─────────────────────────────────────────
    schmitt_upper: float = 0.85     # engage threshold
    schmitt_lower: float = 0.40     # release threshold
