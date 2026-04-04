# ACCIDENT @ CVPR: Zero-Shot Accident Detection and Localization

**Competition:** ACCIDENT @ CVPR: Zero-Shot Accident Detection from Traffic Surveillance Videos

**Notebook Focus:** Zero-shot temporal-spatial localization and classification of traffic accidents in surveillance footage.

**Author:** [Amey Thakur](https://www.kaggle.com/ameythakur20)

---

## Table of Contents

1. [Data Acquisition](#1-data-acquisition)
2. [Data Inspection](#2-data-inspection)
3. [Data Cleaning](#3-data-cleaning)
4. [EDA](#4-eda)
5. [Feature Engineering](#5-feature-engineering)
6. [Modeling](#6-modeling)
7. [Evaluation](#7-evaluation)
8. [Conclusion](#8-conclusion)
9. [AUTOPILOT Workshop](#9-autopilot-workshop)
10. [CVPR 2026 Author Guidelines](#10-cvpr-2026-author-guidelines)
11. [References](#11-references)

---

## 1. Data Acquisition

The primary objective of this challenge is to evaluate models on real-world traffic accident data without localized training on the target domain. The dataset architecture is bifurcated into a synthetic development set and a real-world test set. The synthetic data, generated via the CARLA simulator, provides high-fidelity CCTV-style viewpoints with comprehensive ground-truth annotations for pre-training and validation. The test set consists of genuine CCTV clips sourced from public traffic feeds, representing the authentic environmental challenges of surveillance monitoring.

## 2. Data Inspection

Initial data analysis involves navigating two distinct metadata structures. For the synthetic dataset, `labels.csv` serves as the central index, containing temporal indices, spatial coordinates (normalized to $[0, 1]$), and semantic classification for each collision scenario. Accompanying these are compressed JSON files providing per-frame simulator telemetry. The real-world test set is documented in `test_metadata.csv`, which provides coarse scene attributes such as lighting conditions and weather states to facilitate post-hoc performance stratification across diverse environmental contexts.

## 3. Data Cleaning

Data preparation for surveillance footage requires addressing systematic noise inherent to fixed-view CCTV systems. Common artifacts include low spatial resolution, temporal jitter from compression codecs, and significant occlusions from infrastructure or other vehicles. The cleaning protocol must account for these degradations to ensure that the localization algorithms remain robust. In the synthetic domain, alignment between simulator offsets and video timestamps is critical for precise temporal ground-truth synchronization.

## 4. EDA

Exploratory Data Analysis focuses on the distribution of accident types and environmental variables. The synthetic set contains a variety of scenarios such as head-on collisions, rear-end impacts, and T-bone accidents. Analysis of the `test_metadata.csv` tags reveals the diversity in camera layouts and illumination levels, which are critical for evaluating model generalization. Understanding the variance in vehicle speeds and impact angles across different CARLA maps provides a foundation for designing robust spatial-temporal priors.

## 5. Feature Engineering

Feature extraction for this competition is centered on three primary dimensions: temporal event boundaries, spatial impact points, and semantic collision attributes. Effective features must capture the pre-collision trajectory dynamics and the sudden kinetic changes at the moment of impact. Given the zero-shot requirement, features derived from foundation models or general-purpose video encoders are preferred over those requiring extensive fine-tuning. Spatial localization relies on normalized coordinate systems to maintain consistency across varying camera aspect ratios.

## 6. Modeling

The modeling strategy is constrained by the training-free / zero-shot nature of the benchmark on real CCTV footage. Approaches involve the use of pre-trained visual-language models or temporal action localization architectures that can generalize from synthetic data to real-world distributions. The objective is to design a pipeline that can identify "when," "where," and "what type" of accident occurred without domain-specific supervised tuning on the real test set. Robustness to visual clutter and wide fields of view is a prerequisite for successful deployment.

## 7. Evaluation

The evaluation metric is the harmonic mean of three component scores, emphasizing balanced performance across all tasks. The temporal score ($\mathcal{T}$) and spatial score ($\mathcal{S}$) both utilize Gaussian-style similarity functions. These functions ensure that small errors in predicted time or location coordinates result in scores near 1.0, with a smooth decay towards zero as the deviation from ground truth increases. The classification score ($\mathcal{C}$) is based on Top-1 accuracy for the impact type. A significant deficiency in any single metric results in a substantial reduction of the final composite score.

## 8. Conclusion

*   The ACCIDENT @ CVPR challenge addresses the high-impact problem of automated accident detection in resource-constrained environments.
*   Zero-shot generalization from synthetic simulations to real CCTV footage is the primary technical hurdle.
*   The use of harmonic mean as a scoring metric necessitates high precision across temporal, spatial, and semantic predictions.
*   Robustness to low-resolution and occluded video inputs is essential for viable surveillance-based incident response.

## 9. AUTOPILOT Workshop

The ACCIDENT competition is hosted in conjunction with the 3rd edition of the AUTOPILOT workshop at CVPR 2026. This venue focuses on safety-critical autonomous driving, specifically emphasizing robust perception, trajectory forecasting, and the deployment of distilled foundation models for on-vehicle applications. A central theme of the workshop is open-world learning, which seeks to identify and mitigate out-of-distribution (OOD) hazards and events that fall outside standard taxonomic definitions.

### Submission Tracks

The workshop provides two primary tracks for research dissemination. Full papers containing significant technical contributions may be submitted to the Archival Track, with accepted works appearing in the official CVPR 2026 workshop proceedings. The archival submission deadline is March 04, 2026. For preliminary findings or work-in-progress, the Non-Archival Track accepts extended abstracts and position papers, with a deadline of April 15, 2026. Accepted participants are required to provide a 5-minute video presentation, poster materials, and slides.

### Core Research Topics

Research topics prioritized for the workshop include multimodal reasoning through vision-language models, embodied AI for decision-making, and open-vocabulary learning for robust hazard detection. Additional areas of interest involve multimodal sensor fusion (LiDAR, radar, and camera), spatio-temporal representation learning, and synthetic-to-real transfer protocols. The workshop aims to bridge the gap between academic research and industrial implementation while maintaining a focus on reproducible evaluation and safety-centric metrics.

## 10. CVPR 2026 Author Guidelines

Submissions to CVPR 2026 must comply with rigorous technical and ethical standards. Authors are responsible for ensuring that all manuscripts are self-contained and adhere to the zero-tolerance policy regarding prompt injection or hidden instructions designed to influence automated tools or reviewers. Failure to comply with these formatting or ethical requirements may result in immediate desk rejection.

### Formatting and Anonymization

The standard paper length is limited to eight pages, including all figures and tables, with additional pages permitted only for cited references. All submissions must be properly anonymized to support the double-blind review process. Authors are prohibited from including external links (e.g., to code repositories, videos, or project websites) that expand content beyond the submitted PDF and supplementary files. Identifying information in acknowledgments, grant IDs, or demo videos is strictly forbidden during the review phase.

### New Initiatives for 2026

*   **Compute Reporting**: CVPR 2026 introduces an experimental initiative requiring authors to report the computational resources used during research. These reports are intended for community benchmarking and do not influence the review process.
*   **Findings Track**: A new publication venue is available for papers characterized by solid experimental validation and technical soundness, even if the novelty is incremental. Area Chairs recommend papers to this track based on review outcomes.
*   **Prompt Injection Policy**: Any attempt to hide instructions within the text to manipulate reviewers or LLM-based analysis tools is classified as an ethics violation.

### General Policies

The conference enforces a strict dual submission policy, where no manuscript with more than 20 percent overlap may be under review at another peer-reviewed venue simultaneously. ArXiv preprints are permitted but must not be cited in a way that compromises anonymity. Authors are also expected to serve as reviewers unless they are new to the computer vision community or otherwise exempt. Each individual is limited to a maximum of 25 paper submissions.

## 11. References

- Picek, L., et al. (2026). *ACCIDENT @ CVPR: Zero-Shot Accident Detection from Traffic Surveillance Videos*. Kaggle Competition. [Link](https://kaggle.com/competitions/accident)
- Picek, L., Čermák, V., et al. (2025). *Zero-shot hazard identification in autonomous driving: A case study on the COOCOL benchmark*. WACV 2025.
- AUTOPILOT Workshop at CVPR 2026. [Workshop Site](https://wad.vision/)
