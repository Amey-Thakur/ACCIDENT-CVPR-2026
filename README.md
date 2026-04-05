# ACCIDENT @ CVPR 2026: Zero-Shot Accident Detection Pipeline

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/zero-shot-cctv-traffic-accident-understanding/)
[![Paper](https://img.shields.io/badge/Paper-CVPR%202026-blue)](paper/)
[![Preprint](https://img.shields.io/badge/Preprint-arXiv-b31b1b)](preprint/)

A modular zero-shot pipeline for detecting, localizing, and classifying traffic accidents in CCTV surveillance video. Built for the [ACCIDENT @ CVPR 2026](https://kaggle.com/competitions/accident) competition hosted at the [AUTOPILOT Workshop](https://wad.vision/).

**Authors:** [Amey Thakur](https://orcid.org/0000-0001-5644-1575) · [Sarvesh Talele](https://orcid.org/0009-0002-0818-461X)

**Public Leaderboard Score:** `0.25230`

---

## Overview

Given a surveillance video, the pipeline predicts three things without any labeled real-world training data:

| Task | Method | Output |
|------|--------|--------|
| **When** did the accident happen? | Z-score peak detection on frame differences | Accident time in seconds |
| **Where** did the impact occur? | Weighted centroid of thresholded Farneback optical flow | Normalized (x, y) coordinates |
| **What type** of collision? | CLIP cosine similarity with multi-prompt text embeddings | One of 5 collision types |

The three modules are independent — each can be swapped or tuned without touching the others. No model weights are fine-tuned; the pipeline runs entirely on pre-trained CLIP (ViT-B/32) and classical computer vision.

<p align="center">
  <img src="figure/sampled_frames.png" width="100%" alt="Sampled frames from a synthetic CARLA traffic incident"/>
</p>

---

## Pipeline Architecture

### 1. Temporal Localization

Computes mean absolute frame differences, smooths with a rolling window (w=5), normalizes to z-scores, and selects the frame with the highest anomaly score above threshold (τ=1.5).

<p align="center">
  <img src="figure/frame_diff_zscore.png" width="80%" alt="Frame difference series and z-score temporal signal"/>
</p>

### 2. Spatial Impact Localization

Centers a 30-frame window on the detected accident time, accumulates Farneback dense optical flow magnitudes, applies 90th-percentile thresholding, and computes the weighted centroid of the remaining high-motion region.

<p align="center">
  <img src="figure/heatmap.png" width="80%" alt="Optical flow magnitude heatmap showing impact localization"/>
</p>

### 3. Collision Type Classification

Extracts 8 frames around the detected peak, encodes them with CLIP ViT-B/32, and compares the averaged image embedding against 5 text embeddings (one per collision type), each built from 5 prompt templates.

**Collision types:** `head-on` · `rear-end` · `sideswipe` · `single-vehicle` · `t-bone`

---

## Dataset

The competition provides two splits:

| Split | Source | Videos | Annotations |
|-------|--------|--------|-------------|
| Development | CARLA simulator (synthetic) | 2,211 | Accident time, impact coordinates, collision type |
| Test | Real CCTV footage | 2,027 | Hidden (evaluated on Kaggle) |

<p align="center">
  <img src="figure/collision_type_freq.png" width="45%" alt="Collision type distribution"/>
  &nbsp;&nbsp;
  <img src="figure/accident_time_dist.png" width="45%" alt="Accident time distribution"/>
</p>

<p align="center">
  <img src="figure/impact_scatter.png" width="45%" alt="Ground-truth impact point distribution"/>
  &nbsp;&nbsp;
  <img src="figure/impact_kde.png" width="45%" alt="Impact point density"/>
</p>

---

## Results

**Public leaderboard score: 0.25230** (computed on ~25% of test data)

Calibration on a 10-video synthetic subset:

| Component | Mean Score | Best Individual |
|-----------|-----------|-----------------|
| Temporal (𝒯) | 0.438 | 0.94 |
| Spatial (𝒮) | 0.168 | 0.96 |
| Classification (𝒞) | 0.0 | 0.0 |

The classification bottleneck is the primary limitation: CLIP predicts `t-bone` for all calibration videos (which are all `head-on`), driven by the domain gap between internet imagery and CCTV stills.

<p align="center">
  <img src="figure/pred_collision_dist.png" width="45%" alt="Predicted collision type distribution on test set"/>
  &nbsp;&nbsp;
  <img src="figure/score_distributions.png" width="45%" alt="Component score distributions"/>
</p>

<p align="center">
  <img src="figure/temporal_score_curve.png" width="45%" alt="Temporal score vs time error"/>
  &nbsp;&nbsp;
  <img src="figure/spatial_score_curve.png" width="45%" alt="Spatial score vs location error"/>
</p>

---

## Repository Structure

```
.
├── paper/                  # CVPR 2026 submission (LaTeX, two-column)
│   ├── main.tex
│   ├── main.bib
│   ├── sec/                # Abstract, intro, method, experiments, conclusion
│   └── fig/                # Figures used in the paper
├── preprint/               # arXiv preprint (LaTeX, single-column)
│   ├── main.tex
│   ├── references.bib
│   ├── arxiv.sty
│   └── images/             # Figures used in the preprint
├── Notebook/               # Kaggle notebook (exported .ipynb)
├── workspace/              # Development notebooks and iterations
├── figure/                 # All diagnostic and analysis figures
└── .github/workflows/      # CI: LaTeX → PDF compilation
```

---

## Quick Start

### Run on Kaggle

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/zero-shot-cctv-traffic-accident-understanding/)

The notebook runs end-to-end on a single NVIDIA T4 GPU and processes all 2,027 test videos in approximately 4 hours.

### Build the Paper

The GitHub Actions workflow automatically compiles both the CVPR paper and arXiv preprint on every push. You can also build locally:

```bash
cd paper && latexmk -pdf main.tex
cd preprint && latexmk -pdf main.tex
```

---

## Scoring

The competition uses the harmonic mean of three components:

$$\mathcal{H} = \frac{3}{\frac{1}{\mathcal{T}} + \frac{1}{\mathcal{S}} + \frac{1}{\mathcal{C}}}$$

- **Temporal** (𝒯): Gaussian similarity with σ = 2.0 seconds
- **Spatial** (𝒮): Gaussian similarity with σ = 0.1 (normalized coordinates)
- **Classification** (𝒞): Top-1 accuracy

A zero in any component forces the composite score to zero.

---

## Additional Figures

<details>
<summary>Click to expand all diagnostic figures</summary>

### Bounding Box Statistics
<p align="center">
  <img src="figure/bbox_area.png" width="30%" alt="Bounding box area"/>
  <img src="figure/bbox_height.png" width="30%" alt="Bounding box height"/>
  <img src="figure/bbox_width.png" width="30%" alt="Bounding box width"/>
</p>

### Temporal Analysis
<p align="center">
  <img src="figure/accident_time_by_type.png" width="45%" alt="Accident time by collision type"/>
  &nbsp;&nbsp;
  <img src="figure/accident_time_frac.png" width="45%" alt="Accident time as fraction of clip duration"/>
</p>

### Test Set Predictions
<p align="center">
  <img src="figure/pred_time_dist.png" width="45%" alt="Predicted accident time distribution"/>
  &nbsp;&nbsp;
  <img src="figure/pred_impact_scatter.png" width="45%" alt="Predicted impact locations"/>
</p>

### Dataset Features
<p align="center">
  <img src="figure/weather_dist.png" width="45%" alt="Weather distribution"/>
  &nbsp;&nbsp;
  <img src="figure/correlation_matrix.png" width="45%" alt="Feature correlation matrix"/>
</p>

</details>

---

## Citation

```bibtex
@misc{thakur2026zershot,
  title={A Modular Zero-Shot Pipeline for Accident Detection, Localization, 
         and Classification in Traffic Surveillance Video},
  author={Thakur, Amey and Talele, Sarvesh},
  year={2026},
  howpublished={ACCIDENT @ CVPR 2026 Workshop},
  note={\url{https://www.kaggle.com/code/ameythakur20/zero-shot-cctv-traffic-accident-understanding}}
}
```

---

## References

- Picek, L., Čermák, V., et al. [ACCIDENT @ CVPR 2026](https://kaggle.com/competitions/accident). Kaggle Competition.
- [AUTOPILOT Workshop at CVPR 2026](https://wad.vision/)
- Radford, A., et al. (2021). [Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020). ICML.
- Farnebäck, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion. SCIA.

---

## License

This repository is released for academic research purposes in conjunction with the ACCIDENT @ CVPR 2026 competition.
