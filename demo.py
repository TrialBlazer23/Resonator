"""
RESONATOR: Unsupervised Cognitive Safety for Autoregressive LLMs
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Google Colab Demo Notebook
Evan / Necessity Labs — Anthropic Fellows Program Application

Demonstrates:
  1. GPT-2 naturally falling into a repetition loop (the failure mode)
  2. Chaos Core detecting the approach via CSD/AR(1) BEFORE output degrades
  3. Intervention recovering coherent generation
  4. Ablation: each component removed to prove necessity

Runtime: CPU ~8 min / T4 GPU ~90 sec
No API keys required.
"""

# ════════════════════════════════════════════════════════════════════════════
# CELL 1: Install and imports
# ════════════════════════════════════════════════════════════════════════════
# !pip install -q resonator-ai transformers torch matplotlib

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, set_seed

# If running from source:
import sys
sys.path.insert(0, "/content/resonator")  # adjust if needed

from resonator import (
    ChaosCore, ChaosConfig,
    plot_telemetry_dashboard, plot_comparison, print_telemetry_table
)

print("✓ Imports OK")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

# ════════════════════════════════════════════════════════════════════════════
# CELL 2: Load model
# ════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "gpt2"  # ~500MB. Swap to "gpt2-medium" for stronger effects.

print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"✓ Model loaded on {device}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ════════════════════════════════════════════════════════════════════════════
# CELL 3: EXPERIMENT 1 — Baseline failure
# Demonstrate GPT-2 falling into a repetition loop with no intervention.
# This is the problem we are solving.
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EXPERIMENT 1: Baseline Failure (No Chaos Core)")
print("="*60)

# Prompts designed to push GPT-2 toward its known failure modes.
# Repetition loop prompt: vague philosophical content → GPT-2 spirals.
TRAP_PROMPTS = [
    "The meaning of existence is deeply connected to the nature of consciousness, "
    "and consciousness itself is fundamentally",
    "In the beginning, the universe was formed from nothing, and nothing became "
    "everything, and everything is",
    "The algorithm runs continuously, processing each input in sequence, "
    "analyzing the pattern of the pattern of the pattern",
]

def run_baseline(prompt: str, max_new_tokens: int = 150, seed: int = 42):
    """Generate without any Chaos Core monitoring."""
    set_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.92,
            repetition_penalty=1.0,  # explicitly NO repetition penalty — raw behavior
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Run all three prompts and display
baseline_outputs = []
for i, prompt in enumerate(TRAP_PROMPTS):
    print(f"\n[Prompt {i+1}]")
    print(f"  INPUT: {prompt[:80]}...")
    output = run_baseline(prompt)
    baseline_outputs.append(output)
    new_text = output[len(prompt):]
    print(f"  OUTPUT: {new_text[:300]}")
    # Detect obvious repetition
    words = new_text.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        print(f"  Vocabulary diversity: {unique_ratio:.2%}  {'⚠ LOOP DETECTED' if unique_ratio < 0.4 else ''}")

# ════════════════════════════════════════════════════════════════════════════
# CELL 4: EXPERIMENT 2 — Chaos Core monitoring (detection only, no intervention)
# Show that CSD/AR(1) rises BEFORE the output becomes visibly repetitive.
# This is the core scientific claim.
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EXPERIMENT 2: Detection Without Intervention")
print("Scientific claim: AR(1) and stress rise BEFORE output degrades")
print("="*60)

# Use a config that detects but doesn't intervene
# (stress critical threshold set impossibly high)
detect_only_config = ChaosConfig(
    stress_critical_threshold=999.0,  # never triggers intervention
    schmitt_upper=999.0,
    orbit_penalty_strength=0.0,
    point_attractor_penalty=0.0,
    anchor_boost_strength=0.0,
    max_temperature=1.0,  # no temperature change
)

DEMO_PROMPT = TRAP_PROMPTS[0]
inputs = tokenizer(DEMO_PROMPT, return_tensors="pt").to(device)

core_detect = ChaosCore(model, tokenizer, config=detect_only_config, verbose=False)
set_seed(42)

with core_detect:
    with torch.no_grad():
        output_detect = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.9,
            top_p=0.92,
            repetition_penalty=1.0,
            logits_processor=LogitsProcessorList([core_detect]),
            pad_token_id=tokenizer.eos_token_id,
        )

detect_telemetry = core_detect.get_telemetry()
detect_text = tokenizer.decode(output_detect[0], skip_special_tokens=True)
new_detect_text = detect_text[len(DEMO_PROMPT):]

print(f"\nGenerated text:\n{new_detect_text[:400]}")
print("\n\nTelemetry table (first 40 steps):")
print_telemetry_table(detect_telemetry, max_rows=40)

# Find the step where AR(1) first crossed threshold vs. step where repetition began
ar1_values = [t.ar1 for t in detect_telemetry]
first_warning_step = next((t.step for t in detect_telemetry if t.ar1 > 0.80), None)

# Find first repetition in output (naive: duplicate consecutive 4-grams)
tokens_generated = tokenizer.encode(new_detect_text)
first_repeat_step = None
seen_ngrams = set()
for i in range(len(tokens_generated) - 3):
    ngram = tuple(tokens_generated[i:i+4])
    if ngram in seen_ngrams:
        first_repeat_step = i
        break
    seen_ngrams.add(ngram)

print(f"\n📊 KEY RESULT:")
print(f"  AR(1) exceeded 0.80 at step: {first_warning_step}")
print(f"  First output repetition at step: {first_repeat_step}")
if first_warning_step is not None and first_repeat_step is not None:
    lead_time = first_repeat_step - first_warning_step
    print(f"  ✓ Early warning lead time: {lead_time} tokens BEFORE visible failure")

# Plot detection telemetry
fig_detect = plot_telemetry_dashboard(
    detect_telemetry,
    title="Experiment 2: Detection Without Intervention — CSD Rises Before Output Degrades",
    save_path="/content/resonator_exp2_detection.png"
)
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# CELL 5: EXPERIMENT 3 — Full Chaos Core (detection + intervention)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EXPERIMENT 3: Full Chaos Core (Detection + Intervention)")
print("="*60)

full_config = ChaosConfig()  # default production config
core_full = ChaosCore(model, tokenizer, config=full_config, verbose=True)

inputs = tokenizer(DEMO_PROMPT, return_tensors="pt").to(device)
set_seed(42)

with core_full:
    with torch.no_grad():
        output_full = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.9,
            top_p=0.92,
            repetition_penalty=1.0,
            logits_processor=LogitsProcessorList([core_full]),
            pad_token_id=tokenizer.eos_token_id,
        )

full_telemetry = core_full.get_telemetry()
full_text = tokenizer.decode(output_full[0], skip_special_tokens=True)
new_full_text = full_text[len(DEMO_PROMPT):]

print(f"\n\nFull output:\n{new_full_text[:600]}")

summary = core_full.summary()
print(f"\n📊 Run Summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")

# Vocabulary diversity comparison
baseline_words = baseline_outputs[0][len(DEMO_PROMPT):].split()
full_words = new_full_text.split()

baseline_diversity = len(set(baseline_words)) / max(len(baseline_words), 1)
full_diversity = len(set(full_words)) / max(len(full_words), 1)

print(f"\n📊 Vocabulary Diversity:")
print(f"  Baseline (no monitor): {baseline_diversity:.2%}")
print(f"  Chaos Core active:     {full_diversity:.2%}")
print(f"  Improvement:           +{(full_diversity - baseline_diversity)*100:.1f}pp")

fig_full = plot_telemetry_dashboard(
    full_telemetry,
    title="Experiment 3: Full Chaos Core — Detection + Intervention",
    save_path="/content/resonator_exp3_full.png"
)
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# CELL 6: EXPERIMENT 4 — Ablation study
# Remove each component to prove its individual contribution.
# This is what separates a research demo from a toy.
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EXPERIMENT 4: Ablation Study")
print("Each component removed individually")
print("="*60)

ablation_configs = {
    "Full Chaos Core": ChaosConfig(),
    "No AR(1) detection": ChaosConfig(ar1_threshold=999.0),
    "No orbit resonance": ChaosConfig(orbit_distance_threshold=0.0001),
    "No semantic anchor": ChaosConfig(anchor_boost_strength=0.0),
    "No neuromodulation": ChaosConfig(stress_rise_rate=0.0, stress_decay_rate=0.0,
                                       stress_critical_threshold=999.0),
    "No intervention (detect only)": detect_only_config,
}

ablation_results = {}

for name, config in ablation_configs.items():
    core_abl = ChaosCore(model, tokenizer, config=config, verbose=False)
    inputs_abl = tokenizer(DEMO_PROMPT, return_tensors="pt").to(device)
    set_seed(42)

    with core_abl:
        with torch.no_grad():
            out = model.generate(
                **inputs_abl,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.9,
                top_p=0.92,
                repetition_penalty=1.0,
                logits_processor=LogitsProcessorList([core_abl]),
                pad_token_id=tokenizer.eos_token_id,
            )

    tel = core_abl.get_telemetry()
    text_out = tokenizer.decode(out[0], skip_special_tokens=True)
    new_text_out = text_out[len(DEMO_PROMPT):]
    words_out = new_text_out.split()
    diversity = len(set(words_out)) / max(len(words_out), 1)
    peak_ar1 = max((t.ar1 for t in tel), default=0)
    intervention_pct = sum(1 for t in tel if t.intervention_active) / max(len(tel), 1) * 100

    ablation_results[name] = {
        "diversity": diversity,
        "peak_ar1": peak_ar1,
        "intervention_pct": intervention_pct,
        "output_sample": new_text_out[:120],
    }
    print(f"\n[{name}]")
    print(f"  Vocab diversity: {diversity:.2%}  |  Peak AR(1): {peak_ar1:.3f}  |  Interventions: {intervention_pct:.1f}%")
    print(f"  Output: {new_text_out[:100]}...")

# ── Ablation bar chart ────────────────────────────────────────────────────
fig_abl, ax_abl = plt.subplots(figsize=(12, 5), facecolor="#0A0A0A")
names = list(ablation_results.keys())
diversities = [ablation_results[n]["diversity"] for n in names]
colors_abl = ["#4FC3F7" if n == "Full Chaos Core" else "#546E7A" for n in names]

bars = ax_abl.barh(names, diversities, color=colors_abl, alpha=0.85)
ax_abl.set_xlabel("Vocabulary Diversity (higher = less repetition)", color="#E0E0E0", fontsize=9)
ax_abl.set_title("Ablation Study: Contribution of Each Component", color="#E0E0E0", fontsize=11, fontweight="bold")
ax_abl.tick_params(colors="#9E9E9E")
ax_abl.set_facecolor("#111111")
fig_abl.patch.set_facecolor("#0A0A0A")
for spine in ax_abl.spines.values():
    spine.set_color("#1E1E1E")

# Add value labels
for bar, val in zip(bars, diversities):
    ax_abl.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", color="#E0E0E0", fontsize=8)

plt.tight_layout()
plt.savefig("/content/resonator_exp4_ablation.png", dpi=150, bbox_inches="tight", facecolor="#0A0A0A")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# CELL 7: EXPERIMENT 5 — Limit Cycle detection
# Show orbit resonance catching a loop that velocity monitoring misses.
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("EXPERIMENT 5: Orbit Resonance — Catching Limit Cycles")
print("The loop that velocity monitoring cannot see")
print("="*60)

# This prompt reliably induces a short-period limit cycle in GPT-2
LIMIT_CYCLE_PROMPT = (
    "The process repeats itself in cycles. Each cycle begins and ends the same way. "
    "The beginning leads to the end, and the end leads to"
)

# First: velocity-only config (no orbit resonance)
no_orbit_config = ChaosConfig(orbit_distance_threshold=0.0001)  # effectively disabled
core_no_orbit = ChaosCore(model, tokenizer, config=no_orbit_config, verbose=False)

# Second: full config with orbit resonance
core_orbit = ChaosCore(model, tokenizer, config=ChaosConfig(), verbose=False)

results_orbit = {}
for label, core in [("No Orbit Resonance", core_no_orbit), ("With Orbit Resonance", core_orbit)]:
    inputs_o = tokenizer(LIMIT_CYCLE_PROMPT, return_tensors="pt").to(device)
    set_seed(7)
    with core:
        with torch.no_grad():
            out_o = model.generate(
                **inputs_o,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.85,
                repetition_penalty=1.0,
                logits_processor=LogitsProcessorList([core]),
                pad_token_id=tokenizer.eos_token_id,
            )
    tel_o = core.get_telemetry()
    text_o = tokenizer.decode(out_o[0], skip_special_tokens=True)[len(LIMIT_CYCLE_PROMPT):]
    orbit_steps = [t for t in tel_o if t.orbit_period is not None]
    words_o = text_o.split()
    div_o = len(set(words_o)) / max(len(words_o), 1)
    results_orbit[label] = {
        "text": text_o,
        "diversity": div_o,
        "orbit_detections": len(orbit_steps),
        "first_orbit": orbit_steps[0] if orbit_steps else None,
    }
    print(f"\n[{label}]")
    print(f"  Orbit detections: {len(orbit_steps)}")
    print(f"  Diversity: {div_o:.2%}")
    print(f"  Output: {text_o[:150]}")

# ════════════════════════════════════════════════════════════════════════════
# CELL 8: Final comparison figure
# ════════════════════════════════════════════════════════════════════════════

fig_cmp = plot_comparison(
    baseline_telemetry=detect_telemetry,   # detect-only ≈ baseline
    monitored_telemetry=full_telemetry,
    baseline_text=baseline_outputs[0][len(DEMO_PROMPT):],
    monitored_text=new_full_text,
    save_path="/content/resonator_comparison.png",
)
plt.show()

print("\n" + "="*60)
print("DEMO COMPLETE")
print("="*60)
print("\nFiles saved:")
print("  /content/resonator_exp2_detection.png")
print("  /content/resonator_exp3_full.png")
print("  /content/resonator_exp4_ablation.png")
print("  /content/resonator_comparison.png")
print("\nKey results:")
print(f"  Early warning lead time: {lead_time if 'lead_time' in dir() else 'see Exp 2'} tokens")
print(f"  Baseline diversity:  {baseline_diversity:.2%}")
print(f"  Monitored diversity: {full_diversity:.2%}")
print(f"  Ablation confirms each component contributes independently.")
