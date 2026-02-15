# Reaction Optimizer: Action Space Changes

This document summarizes the changes made to the RL action space in `reaction_optimizer.py` to make the policy more robust.

---

## Overview

The action space was expanded from **3** to **5** continuous dimensions, each normalized to `[0, 1]`. The following table defines the current action vector and its physical meaning.

| Index | Action variable        | Category   | Normalized range | Physical range        | Notes |
|-------|------------------------|------------|------------------|------------------------|--------|
| 0     | Temperature (T)        | Thermal    | [0, 1]           | 20°C to **39.6°C**     | Upper bound set to DCM boiling point (dataset solvent). |
| 1     | Conc. diene            | —          | [0, 1]           | 0.01 M to 5.0 M        | Unchanged. |
| 2     | Conc. dienophile       | —          | [0, 1]           | 0.01 M to 5.0 M        | Unchanged. |
| 3     | Residence time (t)     | Kinetic    | [0, 1]           | 1 min to 24 h          | **Added.** Reward uses rate × time (conversion proxy). |
| 4     | Lewis acid equiv.      | Catalytic  | [0, 1]           | 0.0 to 0.5 eq          | **Added.** Phenomenological barrier lowering (4 kcal/mol per eq). |

---

## Temperature

- **Previous:** `[0, 1]` → 20°C to 200°C (293.15 K–473.15 K).
- **Current:** `[0, 1]` → 20°C to **39.6°C** (293.15 K–312.75 K).
- **Reason:** The dataset was calculated in **dichloromethane (DCM)**; 39.6°C is the boiling point of DCM, so the maximum temperature is capped to stay within the solvent’s liquid range.

---

## Residence time (new)

- **Mapping:** `[0, 1]` → 60 s (1 min) to 86,400 s (24 h).
- **Reward:** Reward is based on **rate × time** via `_normalized_log_reward_time_weighted(rate, t_sec, rate_ref, t_ref_s)` with reference time `t_ref_s = 3600` s (1 h). This favors both higher rate and longer reaction time (conversion proxy).
- **Info:** `t_sec` is reported in the step `info` dict.

---

## Lewis acid equiv. (new)

- **Mapping:** `[0, 1]` → 0.0 to 0.5 equiv.
- **Model effect:** Effective barrier is `delta_g_eff = predicted_delta_g - 4.0 * lewis_equiv` (kcal/mol). The rate constant uses `delta_g_eff` in the Eyring equation, so more catalyst lowers the barrier and increases rate.
- **Info:** `lewis_equiv` and `delta_g_eff` are in the step `info` dict.

---

## Not implemented

- **Pressure (P):** [0, 1] → 1 atm to 10,000 atm was not added. It would require activation volume (ΔV‡) or pressure-dependent data; the current model has no pressure dependence.

---

## Code touchpoints

| Component | Change |
|-----------|--------|
| `DielsAlderOptEnv.action_space` | `shape=(3,)` → `shape=(5,)`. |
| `DielsAlderOptEnv.step()` | Parses 5 actions; builds 5-D condition vector; applies Lewis correction and time-weighted reward; adds `t_sec`, `lewis_equiv`, `delta_g_eff` to `info`. |
| `ConditionAwareTransformer` | Default `condition_dim` 3 → 5. |
| `plot_optimization_landscape()` | Uses 5-D conditions (fixed `t_norm=0.5`, `lewis_norm=0` for the 3D slice). |

Existing checkpoints typically contain only the **base** transformer; the conditioned MLP is built with `condition_dim=5`. For the new actions to be used, train or run PPO with the updated 5-D action space.
