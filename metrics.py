"""
Resonator Framework — Metrics
All signal computation: AR(1), latent velocity, orbit resonance, SVD entropy.
These are pure functions operating on numpy arrays — no model dependency here.
"""

import numpy as np
from typing import Optional, Tuple, List


def compute_ar1(series: np.ndarray) -> float:
    """
    Lag-1 autocorrelation of a 1D time series.
    
    As an LLM approaches a repetition trap (point attractor or limit cycle),
    the token-surprise series loses independence — successive values become
    increasingly correlated. AR(1) → 1.0 is the Critical Slowing Down signal.
    
    Returns 1.0 (maximum warning) for degenerate cases (zero variance = static trap).
    """
    if len(series) < 3:
        return 0.0
    s = series - series.mean()
    std = s.std()
    if std < 1e-8:
        # Zero variance → model is generating identical tokens → maximum warning
        return 1.0
    ar1 = np.corrcoef(s[:-1], s[1:])[0, 1]
    return float(np.clip(ar1, -1.0, 1.0))


def compute_variance_ratio(series: np.ndarray, baseline_window: int = 10) -> float:
    """
    Ratio of current rolling variance to baseline variance.
    
    Variance inflation is the second CSD signature: as resilience decays
    toward zero in the OU process, variance diverges. Ratio >> 1.0 signals
    the system is losing its restoring force.
    """
    if len(series) < baseline_window + 3:
        return 1.0
    baseline_var = np.var(series[:baseline_window]) + 1e-10
    current_var = np.var(series[-5:]) + 1e-10
    return float(current_var / baseline_var)


def compute_latent_velocity(hidden_states: List[np.ndarray]) -> float:
    """
    Euclidean distance between consecutive hidden state vectors (L2 norm).
    
    In healthy generation, the model traverses a broad semantic manifold (high V).
    In a point attractor trap, hidden states collapse to a fixed point → V → 0.
    
    Note: this metric is BLIND to limit cycles where V stays high despite looping.
    Use orbit_resonance() to catch those.
    """
    if len(hidden_states) < 2:
        return 1.0
    return float(np.linalg.norm(hidden_states[-1] - hidden_states[-2]))


def compute_orbit_resonance(
    hidden_states: List[np.ndarray],
    min_lag: int = 3,
    max_lag: int = 25,
    threshold: float = 0.15
) -> Tuple[Optional[int], Optional[float]]:
    """
    Detects limit cycles by scanning lagged self-similarity.
    
    A model generating the same N-token phrase repeatedly will have
    h_t ≈ h_{t-N} — high step velocity but low lagged distance.
    This is the orbit trap that defeats naive velocity monitoring.
    
    Returns: (period, normalized_distance) if cycle detected, else (None, None).
    
    Distance is normalized by product of norms to be scale-invariant.
    """
    if len(hidden_states) < min_lag + 1:
        return None, None

    h_curr = hidden_states[-1]
    norm_curr = np.linalg.norm(h_curr) + 1e-10

    for k in range(min_lag, min(max_lag + 1, len(hidden_states))):
        h_lag = hidden_states[-1 - k]
        norm_lag = np.linalg.norm(h_lag) + 1e-10
        dist = np.linalg.norm(h_curr - h_lag) / (norm_curr * norm_lag) ** 0.5
        if dist < threshold:
            return k, float(dist)

    return None, None


def compute_svd_entropy(hidden_states: List[np.ndarray], window: int = 15) -> float:
    """
    Normalized Shannon entropy of the singular value spectrum of the
    recent hidden-state trajectory matrix.
    
    Healthy generation: broad, multi-dimensional trajectory → entropy ≈ 1.0
    Attractor collapse: low-rank structure → entropy drops toward 0.0
    
    This catches dimensionality collapse even when individual step metrics
    look normal (e.g., slow drift into a semantic dead-end).
    """
    recent = hidden_states[-window:] if len(hidden_states) >= window else hidden_states
    if len(recent) < 3:
        return 1.0

    M = np.stack(recent, axis=0)  # [T, D]
    # Normalize rows to isolate directional variance
    row_norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-10
    M_normalized = M / row_norms

    try:
        _, singular_values, _ = np.linalg.svd(M_normalized, full_matrices=False)
        sv_sq = singular_values ** 2
        sv_sq = sv_sq / (sv_sq.sum() + 1e-10)
        entropy = -np.sum(sv_sq * np.log(sv_sq + 1e-10))
        max_entropy = np.log(len(sv_sq) + 1e-10)
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))
    except np.linalg.LinAlgError:
        return 1.0


def compute_integrated_stress(
    current_stress: float,
    threat_signal: float,
    rise_rate: float = 0.25,
    decay_rate: float = 0.92
) -> float:
    """
    Continuous neuromodulator: integrates threat signals with hysteresis.
    
    Stress rises fast on threat, decays slowly — creating the persistence
    needed to sustain intervention through transient quiet periods
    (the inter-pole valley problem from the GRU experiments).
    
    Rise: F_t = F_{t-1} + rise_rate * threat_signal
    Decay: F_t = F_{t-1} * decay_rate (when no threat)
    """
    if threat_signal > 0:
        new_stress = current_stress + rise_rate * threat_signal
    else:
        new_stress = current_stress * decay_rate
    return float(np.clip(new_stress, 0.0, 1.0))


def attractor_state(
    ar1: float,
    velocity: float,
    orbit_period: Optional[int],
    svd_entropy: float,
    config
) -> str:
    """
    Classify current latent attractor state.
    
    Returns one of:
      'HEALTHY'        — broad chaotic manifold, no intervention needed
      'POINT_ATTRACTOR'— velocity collapse, static trap
      'LIMIT_CYCLE'    — orbit resonance detected, periodic trap
      'CSD_WARNING'    — AR(1) elevated, approaching transition
      'ENTROPY_COLLAPSE'— SVD entropy dropping, dimensionality loss
    """
    if velocity < config.velocity_threshold:
        return "POINT_ATTRACTOR"
    if orbit_period is not None:
        return "LIMIT_CYCLE"
    if ar1 >= config.ar1_threshold:
        return "CSD_WARNING"
    if svd_entropy < config.svd_entropy_threshold:
        return "ENTROPY_COLLAPSE"
    return "HEALTHY"
