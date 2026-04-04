# Version 11: Exact Paper Match

**Notebook:** `version11.ipynb`
**Accuracy:** Lower leaderboard score than Version 10
**Paper alignment:** EXACTLY matches the original paper (`CVPR Paper/main.pdf`)

Every equation, hyperparameter, and implementation detail in this notebook corresponds 1:1 with the published paper.

---

## Paper-to-Code Verification

### Module 1: Temporal Peak Detection

| Paper Reference | Paper Value | Notebook (Cell 48) | Match |
|---|---|---|---|
| Eq. 1 | Mean absolute frame difference, H=180, W=320 | `np.mean(np.abs(gray - prev_gray))`, resize 320x180 | Yes |
| Eq. 2 | Rolling mean, w=5 | `smooth_window: int = 5` | Yes |
| Eq. 3 | Z-score, epsilon=1e-8 | `sigma + 1e-8` | Yes |
| Eq. 4 | `min{t: z_t > tau}`, tau=1.5 | `candidates[0]`, `z_threshold: float = 1.5` | Yes |
| Eq. 4 | Fallback: `argmax_t z_t` | `int(np.argmax(anomaly))` | Yes |

### Module 2: Spatial Impact Localization

| Paper Reference | Paper Value | Notebook (Cells 41, 49) | Match |
|---|---|---|---|
| Sec. 2.3 | Start at floor(N/3) | `start_frame = max(0, total // 3)` | Yes |
| Sec. 2.3 | 30-frame context window | `n_frames_context: int = 30` | Yes |
| Sec. 2.3 | Farneback: scale=0.5, levels=3, win=15, iter=3, n=5, sigma=1.2 | All parameters match | Yes |
| Eq. 5 | Magnitude summation | `mag_accum += mag` | Yes |
| Eq. 6 | Weighted centroid, normalized | `(xs * mag_map).sum() / total_mag / RESIZE_W` | Yes |
| Eq. 6 | Fallback: (0.5, 0.5) if sum(M) < 1e-6 | `return 0.5, 0.5` | Yes |
| (none) | No thresholding on flow map | No percentile code present | Yes |

### Module 3: Collision Type Classification

| Paper Reference | Paper Value | Notebook (Cells 45, 47, 50) | Match |
|---|---|---|---|
| Sec. 2.1 | K = {head-on, rear-end, sideswipe, single, t-bone} | All 5 types present, no extras | Yes |
| Table 1 | 3 prompts per class | 3 prompts per type in COLLISION_PROMPTS | Yes |
| Eq. 7 | Text encoding: L2-norm then mean-pool over 3 | `features / features.norm(...); .mean(dim=0)` | Yes |
| Table 2 | CLIP backbone: ViT-B/32 | `clip.load('ViT-B/32', ...)` | Yes |
| Eq. 8 | 4 peak-region frames, L2-norm, mean-pool | `n_context_frames: int = 4` | Yes |
| Eq. 9 | `argmax_{k} v . t_k` | `max(scores, key=scores.get)` | Yes |

### Pipeline Architecture

| Paper Reference | Paper Value | Notebook (Cell 51) | Match |
|---|---|---|---|
| Sec. 2.1 | Three independent modules | Temporal, spatial, classification run independently | Yes |
| (implied) | Spatial does NOT receive temporal output | `predict_impact_location(video_path)` (no accident_time) | Yes |

---

## Only Fix Applied: Collision Type Label Space

The original notebook (before Version 11) had a bug where `COLLISION_PROMPTS` included `rollover` and `pedestrian` (not valid competition types) and was missing `single`. This was a code bug, not a paper discrepancy: the paper already correctly defined K = {head-on, rear-end, sideswipe, single, t-bone} in Section 2.1.

Version 11 fixes this by:
- Removing `rollover` and `pedestrian` from `COLLISION_PROMPTS`
- Adding `single` with 3 descriptive prompts
- Improving prompt wording for all 5 types within the 3-per-class constraint

This fix directly impacts the classification score C: videos with single-vehicle accidents can now be classified correctly, and cosine similarity is no longer diluted across invalid categories.

---

## Complete Hyperparameter Table (matches Paper Table 2)

| Component | Parameter | Value |
|---|---|---|
| Temporal | Smoothing window w | 5 |
| Temporal | Z-score threshold tau | 1.5 |
| Spatial | Start frame | floor(N/3) |
| Spatial | Context window T | 30 frames |
| Spatial | Pyramid scale | 0.5 |
| Spatial | Pyramid levels | 3 |
| Spatial | Window size | 15 |
| Classification | CLIP backbone | ViT-B/32 |
| Classification | Peak-region frames | 4 |
| Classification | Prompts per class | 3 |
