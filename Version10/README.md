# Version 10: Higher Accuracy, Modified Mathematical Model

**Notebook:** `version10.ipynb`
**Accuracy:** Higher leaderboard score than Version 11
**Paper alignment:** Does NOT match the original paper (`CVPR Paper/main.pdf`)

This version deviates from the paper's mathematical model in 6 places. All deviations are accuracy improvements that change implementation behavior while preserving the same algorithmic families (z-scores, weighted centroids, cosine similarity).

---

## Mathematical Deviations from Paper

### Deviation 1: Temporal Peak Selection (Paper Eq. 4)

**Paper defines:**
```
t*_frame = min{t : z_t > tau}       if exists t s.t. z_t > tau
         = argmax_t z_t             otherwise
```
The paper selects the FIRST frame crossing the threshold (earliest detection).

**Version 10 implements:**
```
t*_frame = argmax_{t : z_t > tau} z_t    if exists t s.t. z_t > tau
         = argmax_t z_t                  otherwise
```
Version 10 selects the STRONGEST peak among all frames exceeding the threshold.

**Code (Cell 48):**
```python
candidates = np.where(anomaly > z_threshold)[0]
if len(candidates) == 0:
    peak_frame = int(np.argmax(anomaly))
else:
    # DEVIATION: argmax of candidates, not candidates[0]
    peak_frame = int(candidates[np.argmax(anomaly[candidates])])
```

**Why this helps:** In real CCTV footage, early non-accident events (passing trucks, camera vibration) can produce moderate z-scores that cross tau=1.5 before the actual collision. The strongest peak is more reliably the collision than the first crossing.

---

### Deviation 2: Flow Window Start Position (Paper Sec. 2.3, Eq. 5)

**Paper defines:**
```
Processing begins at frame floor(N/3)
```
The paper uses a fixed start position independent of the temporal prediction.

**Version 10 implements:**
```
start = max(0, t*_frame - floor(T/2))
```
Version 10 centers the 30-frame flow window on the detected accident frame from Module 1.

**Code (Cell 41):**
```python
if center_frame is not None:
    half_window = n_frames_context // 2
    start_frame = max(0, center_frame - half_window)
else:
    start_frame = max(0, total // 3)
```

**Code (Cell 49) -- wiring:**
```python
def predict_impact_location(video_path, accident_time=None, n_frames_context=30):
    center_frame = None
    if accident_time is not None:
        cap_tmp = cv2.VideoCapture(str(video_path))
        fps = cap_tmp.get(cv2.CAP_PROP_FPS)
        cap_tmp.release()
        if fps > 0:
            center_frame = int(accident_time * fps)
```

**Code (Cell 51) -- pipeline wiring:**
```python
cx, cy = predict_impact_location(video_path, accident_time=acc_time)
```

**Why this helps:** Using floor(N/3) means the spatial module may analyze a temporal region far from the actual collision. Centering on the detected peak ensures the flow map captures collision motion, not unrelated traffic.

---

### Deviation 3: Percentile Thresholding on Flow Map (Not in Paper)

**Paper defines:** No thresholding. The weighted centroid (Eq. 6) operates on the raw accumulated magnitude map M(u,v).

**Version 10 adds:**
```
M_tilde(u,v) = M(u,v)   if M(u,v) >= q_90
             = 0         otherwise
```
where q_90 is the 90th percentile of all values in M. Only the top 10% of flow magnitudes are retained before centroid computation.

**Code (Cell 41):**
```python
# Percentile thresholding: zero out low-magnitude regions
if mag_accum.max() > 0:
    thresh = np.percentile(mag_accum, flow_percentile)
    mag_accum[mag_accum < thresh] = 0.0
```

**Why this helps:** In wide-angle CCTV scenes, background vehicles produce diffuse motion spread across the frame. This biases the raw centroid toward the frame center. Zeroing the bottom 90% concentrates the centroid on the localized high-activity collision region.

---

### Deviation 4: CLIP Backbone (Paper Table 2)

**Paper specifies:** ViT-B/32 (512-dimensional embeddings)

**Version 10 uses:** ViT-L/14 (768-dimensional embeddings)

**Code (Cell 47):**
```python
clip_model, clip_preprocess = clip.load('ViT-L/14', device=DEVICE)
```

**Why this helps:** ViT-L/14 has substantially stronger visual representations than ViT-B/32. This is critical for surveillance frames where image quality is low, viewpoints are oblique, and collision cues are subtle. Standard CLIP benchmarks show ViT-L/14 outperforms ViT-B/32 by approximately 10 points on zero-shot ImageNet.

---

### Deviation 5: Prompts Per Class (Paper Sec. 2.4, Eq. 7, Table 1)

**Paper defines (Eq. 7):**
```
t_k = (1/3) * sum_{j=1}^{3} phi_text(p_k^j) / ||phi_text(p_k^j)||
```
3 prompts per collision type.

**Version 10 uses:**
```
t_k = (1/5) * sum_{j=1}^{5} phi_text(p_k^j) / ||phi_text(p_k^j)||
```
5 prompts per collision type.

**Code (Cell 45) -- example for rear-end:**
```python
'rear-end' : [
    'a car colliding into the back of another car',
    'rear-end collision between two vehicles on a road',
    'vehicle hitting the back of a stationary car from behind',
    'one car rear-ending another car at a traffic light',         # <-- extra
    'a vehicle crashing into the tail of the car ahead',          # <-- extra
],
```

**Why this helps:** More prompts add lexical diversity (varying prepositions, scene context, vehicle descriptions). This reduces the chance that any single phrasing biases the text embedding away from relevant visual features.

---

### Deviation 6: Peak-Region Frames (Paper Table 2, Eq. 8)

**Paper defines (Eq. 8):**
```
v = (1/4) * sum_{i=1}^{4} phi_img(I_{t_i}) / ||phi_img(I_{t_i})||
```
4 frames centered on the predicted accident time.

**Version 10 uses:**
```
v = (1/8) * sum_{i=1}^{8} phi_img(I_{t_i}) / ||phi_img(I_{t_i})||
```
8 frames centered on the predicted accident time.

**Code (Cell 50):**
```python
def extract_frames_around_peak(video_path, peak_time_s, n_context_frames=8, ...):
```
```python
pil_frames = extract_frames_around_peak(video_path, peak_time_s, n_context_frames=8)
```

**Why this helps:** A wider frame window captures the collision from multiple phases (approach, impact, aftermath). Mean-pooling over 8 frames makes the image embedding more robust to motion blur and compression artifacts that can corrupt individual frames.

---

## What Matches the Paper

The following are identical between Version 10 and the paper:

| Paper Reference | Parameter | Value |
|---|---|---|
| Eq. 1 | Frame differencing formula | Mean absolute difference, H=180, W=320 |
| Eq. 2 | Smoothing window w | 5 |
| Eq. 3 | Z-score normalization | (d_bar - mu) / (sigma + 1e-8) |
| Eq. 4 | Z-score threshold tau | 1.5 |
| Sec. 2.3 | Farneback pyramid scale | 0.5 |
| Sec. 2.3 | Farneback pyramid levels | 3 |
| Sec. 2.3 | Farneback window size | 15 |
| Sec. 2.3 | Farneback iterations | 3 |
| Sec. 2.3 | Farneback poly_n, poly_sigma | 5, 1.2 |
| Sec. 2.3 | Flow context window T | 30 frames |
| Eq. 6 | Weighted centroid formula | Preserved (applied to thresholded map) |
| Sec. 2.1 | Label space K | {head-on, rear-end, sideswipe, single, t-bone} |
| Eq. 9 | Cosine similarity classification | argmax_{k} v . t_k |

---

## Summary Table: All 6 Deviations

| # | Paper (Equations) | Version 10 (Code) | Notebook Cell |
|---|---|---|---|
| 1 | Eq. 4: first crossing `min{t: z_t > tau}` | argmax among candidates above tau | Cell 48 |
| 2 | Sec. 2.3: start at floor(N/3) | center on detected t*_frame | Cells 41, 49, 51 |
| 3 | (not in paper): no thresholding | 90th percentile zeroing on flow map | Cell 41 |
| 4 | Table 2: ViT-B/32 | ViT-L/14 | Cell 47 |
| 5 | Eq. 7, Table 1: 3 prompts/class | 5 prompts/class | Cell 45 |
| 6 | Eq. 8, Table 2: 4 peak-region frames | 8 peak-region frames | Cell 50 |
